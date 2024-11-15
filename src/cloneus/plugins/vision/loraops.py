import copy
from pathlib import Path


import torch

import peft
import diffusers.utils as diffutils 
from diffusers import FluxTransformer2DModel, FluxPipeline, SD3Transformer2DModel
import safetensors.torch as sft

from tqdm.auto import tqdm

from cloneus.utils.common import release_memory,read_state_dict
from . import sdconvert



def relabel_lora(state_dict:dict[str, torch.Tensor]):
    if isinstance(state_dict, (str,Path)):
        state_dict = sft.load_file(state_dict)
    
    replacements = [
        ('lora_down', 'lora_A'), 
        ('lora_up', 'lora_B'), 
        ('.diff_b', '.bias'),
    ]
    
    prefixes = [
        'model.diffusion_model.',
        'diffusion_model.',
        'model.',
        'transformer.',
    ]
    
    for k in list(state_dict):
        k_init = k

        for pfx in prefixes:
            k = k.removeprefix(pfx)
        for old_key,new_key in replacements:
            k = k.replace(old_key, new_key)
        
        if k != k_init:
            state_dict[k] = state_dict.pop(k_init)
            
    return state_dict

class LoraManager():
    '''Diffusers handles loras well enough 80% of the time. This is for the other 20%'''
    
    def __init__(self, lora_sd_or_path:Path|dict[str,torch.Tensor]) -> None:
        self.grouped_lora_sd = self._group_keys(relabel_lora(lora_sd_or_path))
        
        self.target_params = set()
        
        self.deltas = {}
        self.scale = None

    @property
    def A(self): return self.grouped_lora_sd['A']
    @property
    def B(self): return self.grouped_lora_sd['B']
    @property
    def bias(self): return self.grouped_lora_sd['bias']
    @property
    def extra(self): return self.grouped_lora_sd['extra']

    @A.setter
    def A(self, v): 
        self._key_check(self.grouped_lora_sd['A'], v)
        self.grouped_lora_sd['A'] = v
    @B.setter
    def B(self, v): 
        self._key_check(self.grouped_lora_sd['B'], v)
        self.grouped_lora_sd['B'] = v
    @bias.setter
    def bias(self, v): 
        self._key_check(self.grouped_lora_sd['bias'], v)
        self.grouped_lora_sd['bias'] = v
    @extra.setter
    def extra(self, v): 
        self._key_check(self.grouped_lora_sd['extra'], v)
        self.grouped_lora_sd['extra'] = v
    
    def _key_check(self, cur:dict, new:dict):
        n_old = len(cur.keys())
        n_new = len(new.keys())
        if n_old != n_new:
            raise KeyError(f'Key mismatch. New key count: {n_new} != {n_old}')

    def _group_keys(self, state_dict:dict, set_target:bool=True) -> dict[str, dict[str, torch.Tensor]]:
        sd_grouped = {'A': {}, 'B': {}, 'bias': {}, 'extra': {}}
        
        for k in list(state_dict):
            if 'lora_A' in k: 
                group_label = 'A'
            elif 'lora_B' in k: 
                group_label = 'B'
            elif 'bias' in k: 
                group_label = 'bias'
            else: 
                group_label = 'extra'
                
            key = k.replace('.lora_A.','.').replace('.lora_B.','.')
            sd_grouped[group_label][key] = state_dict[k]
            
            if set_target and group_label != 'extra':
                self.target_params.add(key)
        
        release_memory()
        
        return sd_grouped



    def group(self, state_dict:dict = None) -> dict[str, dict[str, torch.Tensor]]:
        if state_dict is None:
            return self.grouped_lora_sd
        return self._group_keys(state_dict, set_target=False)

    def ungroup(self, grouped_sd:dict=None, only_AB:bool = False) -> dict[str, torch.Tensor]:
        if grouped_sd is None:
            grouped_sd = self.grouped_lora_sd    
        
        sd_ungrouped = {}
        
        sd_ungrouped.update({k.replace('.weight', '.lora_A.weight'): v for k,v in grouped_sd['A'].items()})
        sd_ungrouped.update({k.replace('.weight', '.lora_B.weight'): v for k,v in grouped_sd['B'].items()})
        if only_AB:
            return sd_ungrouped
        
        sd_ungrouped.update(**grouped_sd['bias'], **grouped_sd['extra'])
        release_memory()
        return sd_ungrouped
        

    @torch.inference_mode()
    def get_deltas(self, dtype:torch.dtype=None):
        if self.deltas:
            return self.deltas
        
        #self.grouped_lora_sd = self._group_keys()
        
        if dtype is None:
            dtype = next(iter(self.grouped_lora_sd['A'].values())).dtype
            if dtype == torch.float32:
                dtype = torch.bfloat16
        
        
        with tqdm(self.target_params) as pbar:
            for k in pbar:
                pbar.set_postfix_str(k)
                
                if k.endswith('.bias'):
                    self.grouped_lora_sd['bias'][k] = self.grouped_lora_sd['bias'][k].to('cpu', dtype) # * scale
                    continue

                B = self.grouped_lora_sd['B'][k].to('cuda', torch.float32)
                A = self.grouped_lora_sd['A'][k].to('cuda', torch.float32)

                try:
                    self.deltas[k] = ((B @ A)).to('cpu', dtype)
                except Exception as e:
                    if B.ndim > 2 or A.ndim > 2:
                        # https://github.com/microsoft/LoRA/blob/main/loralib/layers.py#L255
                        # (A.T @ B.T).T) 
                        self.deltas[k] = ((B.flatten(1,) @ A.flatten(1,)).view(*B.shape[:2], *A.shape[2:])).to('cpu', dtype)
                    else:
                        print('Failed:',k,'\n',e)

        release_memory()
        return self.deltas

    @torch.inference_mode()
    def merge_lora(self, original_state_dict:dict[str, torch.Tensor], scale=1.0):
        #self.lora_state_dict = self.grouped()
        self.scale = scale

        if not self.deltas:
            dtype = next(iter(original_state_dict.values())).dtype
            self.deltas = self.get_deltas(dtype)
        
        
        for k in tqdm(list(self.deltas)):
            original_state_dict[k] += self.deltas[k] * scale


        for k in tqdm(list(self.grouped_lora_sd['bias'])):
            original_state_dict[k] += self.grouped_lora_sd['bias'][k] #*scale
        
        return original_state_dict
    
    @torch.inference_mode()
    def unmerge_lora(self, merged_state_dict:dict[str, torch.Tensor]):
        if self.scale is None:
            raise ValueError('merge never called, cannot unmerge')
        
        #self.lora_state_dict = self.grouped()
        
        for k in tqdm(list(self.deltas)):
            merged_state_dict[k] -= self.deltas[k] * self.scale


        for k in tqdm(list(self.grouped_lora_sd['bias'])):
            merged_state_dict[k] -= self.grouped_lora_sd['bias'][k] #*scale
        
        self.scale = 0.0
        return merged_state_dict
    
    def rescale_merged(self, merged_state_dict:dict[str, torch.Tensor], new_scale:float, previous_scale:float=None):
        if self.scale is None:
            raise ValueError('merge never called, cannot rescale')
        
        if previous_scale is None:
            previous_scale = self.scale
        
        for k in tqdm(list(self.deltas)):
            merged_state_dict[k] += self.deltas[k]*(new_scale-previous_scale)
            # merged_state_dict[k] += -previous_scale*self.deltas[k] + new_scale*self.deltas[k]
        
        self.scale = new_scale
        return merged_state_dict
    

def convert_sd3_lora(lora_sd_or_path:Path|dict[str,torch.Tensor], dtype=torch.float16, bias=False):
    base_id = 'stabilityai/stable-diffusion-3.5-large'
    lora = LoraManager(lora_sd_or_path)
    
    lora.A = sdconvert.sd3_to_diffusers(lora.A, dtype=dtype, allow_missing=True, is_lora_A=True, base_model_id=base_id)[0]
    lora.B = sdconvert.sd3_to_diffusers(lora.B, dtype=dtype, allow_missing=True, is_lora_A=False, base_model_id=base_id)[0]
    
    if bias:
        lora.bias = sdconvert.sd3_to_diffusers(lora.bias, dtype=dtype, allow_missing=True, base_model_id=base_id)[0]
        
    return lora.ungroup(only_AB=(not bias))

def convert_flux_lora(lora_sd_or_path:Path|dict[str,torch.Tensor], dtype=torch.bfloat16, bias=False):
    base_id = 'black-forest-labs/FLUX.1-dev'
    lora = LoraManager(lora_sd_or_path)
    
    lora.A = sdconvert.flux_to_diffusers(lora.A, dtype=dtype, allow_missing=True, is_lora_A=True, base_model_id=base_id)[0]
    lora.B = sdconvert.flux_to_diffusers(lora.B, dtype=dtype, allow_missing=True, is_lora_A=False, base_model_id=base_id)[0]
    
    if bias:
        lora.bias = sdconvert.flux_to_diffusers(lora.bias, dtype=dtype, allow_missing=True, base_model_id=base_id)[0]
        
    return lora.ungroup(only_AB=(not bias))


def lora_to_config(lora_state_dict:dict):
    rank = {k: v.shape[1] for k,v in lora_state_dict.items() if "lora_B" in k}
    if not rank:
        if any("lora_up" in k for k in lora_state_dict):
            raise ValueError('LoRA not in diffusers format. Conversion required.')
        raise KeyError('State dict does not appear to be LoRA or is using non-standard format.')
    lora_config_kwargs = diffutils.get_peft_kwargs(rank, network_alpha_dict=None, peft_state_dict=lora_state_dict)
    #lora_config_kwargs.pop('use_dora')
    lora_config = peft.LoraConfig(**lora_config_kwargs)
    return lora_config

@torch.inference_mode()
def flux_inject_lora(lora_sd_or_path:dict[str, torch.Tensor]|Path, transformer:FluxTransformer2DModel, adapter_name:str|None=None):
    #diffutils.convert_all_state_dict_to_peft(converted_state_dict)
    converted_state_dict = FluxPipeline.lora_state_dict(lora_sd_or_path)
    lora_config = lora_to_config(converted_state_dict)

    if adapter_name is None:
        adapter_name = diffutils.get_adapter_name(transformer)
    
    peft.inject_adapter_in_model(lora_config, transformer, adapter_name=adapter_name)
    incompatible_keys = peft.set_peft_model_state_dict(transformer, converted_state_dict, adapter_name=adapter_name)
    
    return transformer, incompatible_keys


@torch.inference_mode()
def state_dict_merge_lora(lora_sd_path:str, model_id="black-forest-labs/FLUX.1-dev", scale = 0.125, strip_prefix = 'transformer.'):
    
    lora_sd = {k.removeprefix(strip_prefix): v.to('cuda') for k,v in read_state_dict(lora_sd_path).items()}
    
    lora_config = lora_to_config(lora_sd)
    target_modules = set([t.removeprefix(strip_prefix) for t in lora_config.target_modules])
    
    model_sd = read_state_dict(model_id, subfolder='transformer')
    
    # cpu -> cuda -> cpu cuts time in half vs all cpu
    for k in target_modules:
        p = model_sd[k+'.weight'].to('cuda')
        p += (lora_sd[k+'.lora_B.weight'] @ lora_sd[k+'.lora_A.weight'] )*scale
        model_sd[k+'.weight'] = p.to('cpu')
        
    del lora_sd
    release_memory()
    return model_sd