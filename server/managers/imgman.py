import os
import gc
import re
import io
import math
import time
import typing
import itertools
import functools
from pathlib import Path
from dataclasses import dataclass, field, asdict, KW_ONLY, InitVar

import discord
from discord.ext import commands

from tqdm.auto import tqdm
import numpy as np
import torch
from diffusers import (
    AutoPipelineForText2Image, 
    AutoPipelineForImage2Image, 
    DiffusionPipeline, 
    StableDiffusionXLPipeline, 
    StableDiffusionXLImg2ImgPipeline, 
    StableDiffusion3Pipeline,
    StableDiffusion3Img2ImgPipeline,
    SD3Transformer2DModel,
    DPMSolverMultistepScheduler, 
    DPMSolverSinglestepScheduler,
    UNet2DConditionModel, 
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline, 
    FluxTransformer2DModel
)
from diffusers.schedulers import AysSchedules
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from diffusers.utils import load_image, make_image_grid
from transformers.image_processing_utils import select_best_resolution
from transformers import T5EncoderModel, BitsAndBytesConfig, QuantoConfig, AutoModelForTextEncoding
from optimum import quanto
from accelerate import cpu_offload
from DeepCache import DeepCacheSDHelper
from PIL import Image
from spandrel import ImageModelDescriptor, ModelLoader
from compel import Compel, ReturnedEmbeddingsType

from cloneus import cpaths

import config.settings as settings
from cmds import flags as cmd_flags, transformers as cmd_tfms
from utils.globthread import async_wrap_thread, stop_global_thread
from utils.image import batched, timed

bot_logger = settings.logging.getLogger('bot')
model_logger = settings.logging.getLogger('model')
cmds_logger = settings.logging.getLogger('cmds')
event_logger = settings.logging.getLogger('event')

# Resource: https://github.com/CyberTimon/Stable-Diffusion-Discord-Bot/blob/main/bot.py

IMG_DIR = settings.SERVER_ROOT/'output'/'imgs'
PROMPT_FILE = IMG_DIR.joinpath('_prompts.txt')
SDXL_DIMS = [(1024,1024), (1152, 896),(896, 1152), (1216, 832),(832, 1216), (1344, 768),(768, 1344), (1536, 640),(640, 1536),] # https://stablediffusionxl.com/sdxl-resolutions-and-aspect-ratios/
# other: [(1280, 768),(768, 1280),]
torch.backends.cuda.matmul.allow_tf32 = True

def release_memory():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()

def print_memstats(label:str):
    print(f'({label}) max_memory_allocated:', torch.cuda.max_memory_allocated()/(1024**2), 'max_memory_reserved:', torch.cuda.max_memory_reserved()/(1024**2))

def torch_compile_flags(restore_defaults=False, verbose=False):
    # https://huggingface.co/docs/diffusers/tutorials/fast_diffusion#use-faster-kernels-with-torchcompile
    torch.set_float32_matmul_precision("high")
    if verbose:
        print('init _inductor config:')
        print('{c.conv_1x1_as_mm}, {c.coordinate_descent_tuning}, {c.epilogue_fusion}, {c.coordinate_descent_check_all_directions}'.format(c=torch._inductor.config))
    if restore_defaults:
        torch._inductor.config.conv_1x1_as_mm = False
        torch._inductor.config.coordinate_descent_tuning = False
        torch._inductor.config.epilogue_fusion = True
        torch._inductor.config.coordinate_descent_check_all_directions = False
    else:
        torch._inductor.config.conv_1x1_as_mm = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.coordinate_descent_check_all_directions = True
        # torch._inductor.config.force_fuse_int_mm_with_mul = True
        # torch._inductor.config.use_mixed_mm = True

# def batched(iterable, n:int):
#     '''https://docs.python.org/3/library/itertools.html#itertools.batched'''
#     # batched('ABCDEFG', 3) → ABC DEF G
#     if n < 1:
#         raise ValueError('n must be at least one')
#     iterator = iter(iterable)
#     while batch := list(itertools.islice(iterator, n)):
#         yield batch

@dataclass
class CfgItem:
    default: typing.Any
    _: KW_ONLY
    locked: bool = False
    bounds: tuple = None
    name: str = None

@dataclass
class DiffusionConfig:
    # model_name: str
    # model_path: str
    steps: int
    guidance_scale: float
    
    strength: float = 0.55
    negative_prompt: str = None
    aspect: typing.Literal['square','portrait','landscape'] = 'square'
    refine_steps: int = CfgItem(0, bounds=(0, 3))
    refine_strength: float = CfgItem(0.3, bounds=(0.2, 0.4))
    denoise_blend: float = None
    
    img_dims: tuple[int, int]|list[tuple[int, int]] = (1024, 1024)
    refine_guidance_scale: float = None
    _ : KW_ONLY
    locked: InitVar[list[str]] = None
    
    def __post_init__(self, locked: list[str]):
        '''Assign field meta data to a CfgItem names <key>_'''
        # This might be a better solution -- https://docs.python.org/3/library/dataclasses.html#descriptor-typed-fields 
        if locked is None:
            locked = []
        
        self_copy = self.__dict__.copy()
        for k,v in self.__dict__.items():
            if not isinstance(v, CfgItem):
                md_item = CfgItem(v, name=k, locked=(k in locked))
            else:
                name = v.name if v.name is not None else k
                is_locked = v.locked or k in locked
                md_item = CfgItem(v.default, locked=is_locked, bounds=v.bounds, name=name)
            
            self_copy[f'{k}_'] = md_item
            self_copy[f'{k}'] = md_item.default
        
        for k, v in self_copy.items():
            setattr(self, f'{k}', v)
        
        # These are the vals that are not passed as arguments in command calls. 
        self.metadata_keys = ['img_dims', 'refine_guidance_scale', 'locked']
    
    def lock(self, *args):
        for k in args:
            getattr(self, f'{k}_').locked = True

    def get_defaults(self, only_usable:bool=True) -> dict:
        defaults = {}
        for k,v in self.to_dict().items():
            if only_usable:
                if k in self.metadata_keys or (v.default is None and v.locked):
                    continue
            defaults[k] = v.default
        return defaults


    def to_dict(self):
        return {v.name: v for k,v in vars(self).items() if k.endswith('_')}

    def to_md(self):
        md_text = ""
        for k,v in self.to_dict().items():
            if k in self.metadata_keys or (v.default is None and v.locked):
                continue
            
            default = v.default
            lb, ub = 0, 0
            if v.bounds is not None:
                lb, ub = v.bounds
            # These values are displayed as 0-100 for end-user simplicity
            if k in ['strength', 'refine_strength', 'denoise_blend']:
                default = int(default*100)
                lb = int(lb*100)
                ub = int(ub*100)
            
            postfix = ''
            if v.locked:
                postfix = '🔒' 
            elif v.bounds is not None:
                postfix = f'*(typical: {lb} - {ub})*'
            
            md_text += f"\n- {k}: {default} {postfix}"
        return md_text

    def get_if_none(self, **kwargs):
        '''Fill with default config value if arg is None'''
        filled_kwargs = {}
        for k,v in kwargs.items():
            if v is None or getattr(self, f'{k}_').locked:
                v = getattr(self, k)
            filled_kwargs[k] = v
        return filled_kwargs
        
    def nearest_dims(self, img_wh:tuple[int,int], dim_choices:list[tuple[int,int]]=None, scale:float=1.0, use_hf_sbr:bool=False):
        # TODO: compare implementation vs transformers select_best_resolution
        if dim_choices is None:
            dim_choices = self.img_dims
        
        if isinstance(dim_choices, tuple):
            return dim_choices

        scale_dims = np.array(dim_choices)
        if scale != 1:
            scale_dims *= scale
            scale_dims -= (scale_dims % 64) #8) # equiv: ((scale_dims//64)*64
        dim_choices = list(map(tuple,scale_dims.astype(int)))

        #dim_choices = list(map(tuple,(np.array(dim_choices)*scale).astype(int)))
        
        #if isinstance(self.img_dims, list):
        w_in, h_in = img_wh
        if use_hf_sbr:
            resolutions = [(h,w) for w,h in dim_choices] # May not matter since symmetric, but a precaution
            dim_out = select_best_resolution((h_in, w_in), possible_resolutions=resolutions)
            dim_out = (dim_out[1],dim_out[0]) # (h,w) -> (w,h)
        else:
            ar_in = w_in/h_in
            dim_out = min(dim_choices, key=lambda wh: abs(ar_in - wh[0]/wh[1]))
        
        return dim_out
    
    def get_dims(self, aspect:typing.Literal['square','portrait', 'landscape']):
        dim_out = self.img_dims
        
        if isinstance(self.img_dims, list):
            if aspect == 'square':
                dim_out = min(self.img_dims, key=lambda wh: abs(1 - wh[0]/wh[1]))
            elif aspect == 'portrait':
                dim_out = min(self.img_dims, key=lambda wh: wh[0]/wh[1])
            elif aspect == 'landscape':
                dim_out = max(self.img_dims, key=lambda wh: wh[0]/wh[1])
            else:
                raise ValueError(f'Unknown aspect {aspect!r}')
        return dim_out

def calc_esteps(num_inference_steps:int, strength:float, min_effective_steps:int):
    if num_inference_steps < 1:
        return math.ceil(min_effective_steps*(1/strength))
    if num_inference_steps*strength < min_effective_steps:
        steps = math.ceil(min_effective_steps/strength)
        print(f'steps ({num_inference_steps}) too low, forcing to {steps}')
        num_inference_steps = steps
    return num_inference_steps

def discretize_strengths(steps:int, nframes:int, start:float=0.3, end:float=0.8, linear:bool=True, step_truncate:bool=False, return_steps:bool=False, ):
    strdist = np.linspace(start, end, num=nframes) if linear else np.geomspace(start, end, num=nframes)
    noisy_steps = steps*strdist
    round_steps = np.floor(noisy_steps) if step_truncate else noisy_steps.round()
    round_steps = round_steps.clip(1, steps) #.floor() for diffusers style 
    if return_steps:
        return round_steps.astype(int)
    strengths = round_steps/steps
    return strengths


def get_scheduler(alias:typing.Literal['DPM++ 2M','DPM++ 2M Karras','DPM++ 2M SDE','DPM++ 2M SDE Karras','DPM++ SDE','DPM++ SDE Karras','Euler','Euler A', 'Euler FM'], init_sched_config:dict, **kwargs):
    # https://huggingface.co/docs/diffusers/main/en/api/schedulers/overview#schedulers
    # partials to avoid warnings for incompatible schedulers
    sched_map = {
        'DPM++ 2M':             functools.partial(DPMSolverMultistepScheduler.from_config, use_karras_sigmas=False, **kwargs),
        'DPM++ 2M Karras':      functools.partial(DPMSolverMultistepScheduler.from_config, use_karras_sigmas=True, **kwargs),
        'DPM++ 2M SDE':         functools.partial(DPMSolverMultistepScheduler.from_config, use_karras_sigmas=False, algorithm_type='sde-dpmsolver++', **kwargs),
        'DPM++ 2M SDE Karras':  functools.partial(DPMSolverMultistepScheduler.from_config, use_karras_sigmas=True, algorithm_type='sde-dpmsolver++', **kwargs),
        
        'DPM++ SDE':            functools.partial(DPMSolverSinglestepScheduler.from_config, use_karras_sigmas=False, algorithm_type='dpmsolver++', **kwargs), 
        'DPM++ SDE Karras':     functools.partial(DPMSolverSinglestepScheduler.from_config, use_karras_sigmas=True, algorithm_type='dpmsolver++', **kwargs),

        'Euler':                functools.partial(EulerDiscreteScheduler.from_config, **kwargs),
        'Euler A':              functools.partial(EulerAncestralDiscreteScheduler.from_config, **kwargs),
        'Euler FM':             functools.partial(FlowMatchEulerDiscreteScheduler.from_config, **kwargs),
    }
    scheduler = sched_map[alias](init_sched_config)
    return scheduler

class OneStageImageGenManager:
    def __init__(self, model_name: str, model_path:str, config:DiffusionConfig, offload=False, scheduler_setup:str|tuple[str, dict]=None, clip_skip:int=None):
        self.is_compiled = False
        self.is_ready = False
        self.dc_enabled = None
        
        self.model_name = model_name
        self.model_path = model_path
        self.config = config
        self.offload = offload
        self.initial_scheduler_config = None
        self._scheduler_setup = (scheduler_setup, {}) if isinstance(scheduler_setup,str) else scheduler_setup
        self.scheduler_kwargs = None
        
        self.clip_skip = clip_skip
        self.global_seed = None
        self.has_adapters = False
        
    def set_seed(self, seed:int|None = None):
        self.global_seed = seed
    
    def load_lora(self, weight_name:str = 'detail-tweaker-xl.safetensors', lora_dirpath:Path=None):
        if lora_dirpath is None:
            lora_dirpath = cpaths.ROOT_DIR/'extras/loras'
        
        adapter_name = weight_name.rsplit('.', maxsplit=1)[0].replace('-','_')
        
        try:
            self.base.load_lora_weights(lora_dirpath, weight_name=weight_name, adapter_name=adapter_name)
            self.has_adapters = True
        except IOError as e:
            if weight_name == 'detail-tweaker-xl.safetensors':
                print('detail-tweaker-xl not found, detail parameter will not function. '
                    'To use, download from: https://civitai.com/models/122359/detail-tweaker-xl')
            else:
                raise NotImplementedError(f'Unsupported lora adapater {adapter_name!r}')
            
    @async_wrap_thread()
    def load_pipeline(self):
        pipeline_loader = (StableDiffusionXLPipeline.from_single_file if self.model_path.endswith('.safetensors') 
                           else AutoPipelineForText2Image.from_pretrained)

        self.base: StableDiffusionXLPipeline = pipeline_loader(
            self.model_path, torch_dtype=torch.bfloat16, variant="fp16", use_safetensors=True, add_watermarker=False,
        )
        
        if not self.offload:
            self.base = self.base.to(0)

        self.initial_scheduler_config = self.base.scheduler.config
        
        if self._scheduler_setup is not None:
            sched_alias, self.scheduler_kwargs = self._scheduler_setup
            self.base.scheduler = get_scheduler(sched_alias, self.initial_scheduler_config, **self.scheduler_kwargs)
        
        self.load_lora(adapter_name='detail_tweaker_xl')
        
        self.compeler = Compel(
            tokenizer=[self.base.tokenizer, self.base.tokenizer_2],
            text_encoder=[self.base.text_encoder, self.base.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True]
        )
        
        self.basei2i: StableDiffusionXLImg2ImgPipeline = AutoPipelineForImage2Image.from_pipe(self.base,)# torch_dtype=torch.bfloat16, use_safetensors=True, add_watermarker=False)#.to(0)

        if self.offload:
            # calling on both pipes seems to make offload more consistent, may not be inherited properly with from_pipe 
            self.base.enable_model_cpu_offload()
            self.basei2i.enable_model_cpu_offload()
        #else:
        #    self.base = self.base.to("cuda")

        #if not self.offload:
        #    self.basei2i = self.basei2i.to("cuda")

        self.upsampler = Upsampler('4xUltrasharp-V10.pth', "BGR", dtype=torch.bfloat16)
        #self.upsampler = Upsampler('4xNMKD-Superscale.pth', "RGB", dtype=torch.bfloat16)
        
        self.is_ready = True
    
    def available_schedulers(self, return_aliases: bool = True) -> list[str]:
        if self.base is None:
            return []
        
        compat = self.base.scheduler.compatibles
        implemented = ['DPMSolverMultistepScheduler', 'DPMSolverSinglestepScheduler', 'EulerDiscreteScheduler', 'EulerAncestralDiscreteScheduler', 'FlowMatchEulerDiscreteScheduler']
        
        if not return_aliases:
            return list(set(implemented) & set([s.__name__ for s in compat]))
        
        schedulers = []
        
        if DPMSolverMultistepScheduler in compat:
            schedulers += ['DPM++ 2M', 'DPM++ 2M Karras', 'DPM++ 2M SDE', 'DPM++ 2M SDE Karras']
        if DPMSolverSinglestepScheduler in compat:
            schedulers += ['DPM++ SDE', 'DPM++ SDE Karras']
        if EulerDiscreteScheduler in compat:
            schedulers += ['Euler']
        if EulerAncestralDiscreteScheduler in compat:
            schedulers += ['Euler A'] 
        if FlowMatchEulerDiscreteScheduler in compat:
            schedulers += ['Euler FM']
        return schedulers

    def set_scheduler(self, alias:typing.Literal['DPM++ 2M','DPM++ 2M Karras','DPM++ 2M SDE','DPM++ 2M SDE Karras','DPM++ SDE','DPM++ SDE Karras','Euler','Euler A', 'Euler FM'], **kwargs) -> str:
        # https://huggingface.co/docs/diffusers/main/en/api/schedulers/overview#schedulers
        assert self.base is not None, 'Model not loaded. Call `load_pipeline()` before proceeding.'
        if alias not in self.available_schedulers(return_aliases=True):
            raise KeyError(f'Scheduler {alias!r} not found or is incompatiable with active scheduler ({self.base.scheduler.__class__.__name__!r})')
        
        # be sure to keep anything initially passed like "timespace trailing" unless explicitly set otherwise
        if self.scheduler_kwargs:
            kwargs = {**self.scheduler_kwargs, **kwargs}
        
        scheduler = get_scheduler(alias, self.initial_scheduler_config, **kwargs)
        self.base.scheduler = scheduler
        self.basei2i.scheduler = scheduler

        return self.base.scheduler.__class__.__name__

    def _setup_dc(self):
        # https://github.com/horseee/DeepCache/blob/master/DeepCache/extension/deepcache.py
        self.dc_base = DeepCacheSDHelper(self.base)
        self.dc_base.set_params(cache_interval=3, cache_branch_id=0)
        
        self.dc_basei2i = DeepCacheSDHelper(self.basei2i)
        self.dc_basei2i.set_params(cache_interval=3, cache_branch_id=0)
        self.dc_enabled = False
        
    def dc_fastmode(self, enable:bool, img2img=False):
        # TODO: This will break under various conditions
        # if min_effective_steps (steps*strength < 5) refine breaks?
        # e.g. refine_strength=0.3, steps < 25 / refine_strength=0.4, steps < 20
        if self.dc_enabled is None:
            self._setup_dc()
        if enable != self.dc_enabled: 
            if enable:
                (self.dc_basei2i if img2img else self.dc_base).enable()
            else:
                (self.dc_basei2i if img2img else self.dc_base).disable()
                
            self.dc_enabled=enable 

    @async_wrap_thread
    def unload_pipeline(self):
        self.base = None
        self.basei2i = None
        self.compeler = None
        self.upsampler = None
        self.initial_scheduler_config = None
        self.scheduler_kwargs = None
        release_memory()
        self.is_ready = False
        if self.is_compiled:
            torch_compile_flags(restore_defaults=True)
            torch._dynamo.reset()
            self.is_compiled = False

    @async_wrap_thread
    def compile_pipeline(self):
        # calling the compiled pipeline on a different image size triggers compilation again which can be expensive.
        # - https://huggingface.co/docs/diffusers/optimization/torch2.0#torchcompile
        torch_compile_flags()
        self.base.unet.to(memory_format=torch.channels_last)
        self.base.vae.to(memory_format=torch.channels_last)
        self.base.unet = torch.compile(self.base.unet, mode="reduce-overhead", fullgraph=True) #  "max-autotune"
        self.base.upcast_vae()
        _ = self.pipeline(prompt='Cat', num_inference_steps=4, guidance_scale=self.config.guidance_scale)
        self.is_compiled = True
    
    @torch.inference_mode()
    def embed_prompts(self, prompt:str, negative_prompt:str = None, batch_size: int = 1):
        if set(prompt) & set('()-+'):
            conditioning, pooled = self.compeler([prompt]*batch_size)

            prompt_encodings = dict(prompt_embeds=conditioning, pooled_prompt_embeds=pooled)
            
            if negative_prompt is not None:
                neg_conditioning, neg_pooled = self.compeler([negative_prompt]*batch_size)
                prompt_encodings.update(negative_prompt_embeds=neg_conditioning, negative_pooled_prompt_embeds=neg_pooled)
        else:
            prompt = [prompt]*batch_size
            if negative_prompt is not None:
                negative_prompt = [negative_prompt]*batch_size
                
            prompt_encodings = dict(prompt=prompt, negative_prompt=negative_prompt)
            
        return prompt_encodings
        
    
    @torch.inference_mode()
    def pipeline(self, prompt, num_inference_steps, negative_prompt=None, guidance_scale=None, detail_weight=None, strength=None, refine_steps=None, refine_strength=None, image=None, target_size=None, seed=None):
        if seed is None:
            seed = self.global_seed
        gseed = torch.Generator(device='cpu').manual_seed(seed) if seed is not None else None
        #meval.seed_everything(self.global_seed)
        #torch.manual_seed(seed)
        #torch.cuda.manual_seed(seed)
            
        t0 = time.perf_counter()
        prompt_encodings = self.embed_prompts(prompt, negative_prompt=negative_prompt)
        lora_kwargs={'cross_attention_kwargs': {"scale": detail_weight}} if detail_weight and self.has_adapters else {}
        
        t_pe = time.perf_counter()
        t_main = t_pe # default in case skip
        mlabel = '_2I'
        # Txt2Img
        if image is None: 
            #h,w is all you need -- https://github.com/huggingface/diffusers/blob/v0.26.3/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L1075
            h,w = target_size
            image = self.base(num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, height=h, width=w, clip_skip=self.clip_skip,
                              **prompt_encodings, **lora_kwargs, generator=gseed).images[0] # num_images_per_prompt=4
            #return make_image_grid(imgs, 2, 2)
            t_main = time.perf_counter()
            print_memstats('draw')
            #release_memory()
            mlabel = 'T2I'
            
        
        # Img2Img - not just else because could pass image with str=0 to just up+refine. But should never be image=None + strength
        elif strength:
            num_inference_steps = calc_esteps(num_inference_steps, strength, min_effective_steps=1)
            image = self.basei2i(image=image, num_inference_steps=num_inference_steps, strength=strength, guidance_scale=guidance_scale, clip_skip=self.clip_skip,
                                 **prompt_encodings, **lora_kwargs, generator=gseed).images[0]
            t_main = time.perf_counter()
            print_memstats('redraw')
            mlabel = 'I2I'
            
        t_up = t_re = t_main  # default in case skip
        # Upscale + Refine - must be up->refine. refine->up does not improve quality
        if refine_steps > 0 and refine_strength: 
            image = self.upsampler.upscale(image, scale=1.5)
            t_up = time.perf_counter()
            print_memstats('upscale')
            #release_memory()
            torch.cuda.empty_cache()
            
            # NOTE: this will use the models default num_steps every time since "steps" isnt a param in HD
            # num_inference_steps = calc_esteps(-1, refine_strength, min_effective_steps=refine_steps)
            num_inference_steps = calc_esteps(num_inference_steps, refine_strength, min_effective_steps=refine_steps)
            image = self.basei2i(image=image, num_inference_steps=num_inference_steps, strength=refine_strength, guidance_scale=guidance_scale, clip_skip=self.clip_skip,
                                 **prompt_encodings, **lora_kwargs, generator=gseed).images[0]
            t_re = time.perf_counter()
            print_memstats('refine')
        
        t_fi = time.perf_counter()
        #release_memory()
        torch.cuda.empty_cache()
        
        print(f'total: {t_fi-t0:.2f}s | prompt_embed: {t_pe-t0:.2f}s | {mlabel}: {t_main-t_pe:.2f}s | upscale: {t_up-t_main:.2f}s | refine: {t_re-t_up:.2f}s')
        return image
    
    @async_wrap_thread
    def generate_image(self, prompt:str, 
                       steps:int=None, 
                       negative_prompt:str=None, 
                       guidance_scale:float=None, 
                       detail_weight:float=0,
                       aspect:typing.Literal['square','portrait','landscape']=None, 
                       refine_steps:int=0, 
                       refine_strength:float=None, 
                       seed:int = None,
                       **kwargs) -> Image.Image:
        fkwg = self.config.get_if_none(steps=steps, negative_prompt=negative_prompt, guidance_scale=guidance_scale, refine_steps=refine_steps, refine_strength=refine_strength, aspect=aspect)
        print(f'kwargs: {kwargs}\nfkwg:{fkwg}')
        img_dims = self.config.get_dims(fkwg['aspect'])
        target_size = (img_dims[1], img_dims[0]) # h,w
        
        # We don't want to run refine unless refine steps passed
        refine_strength = cmd_tfms.percent_transform(fkwg['refine_strength'])
        image = self.pipeline(prompt=prompt, num_inference_steps=fkwg['steps'], negative_prompt=fkwg['negative_prompt'], 
                              guidance_scale=fkwg['guidance_scale'], detail_weight=detail_weight, refine_steps=refine_steps, refine_strength=refine_strength, target_size=target_size, seed=seed)
        if detail_weight:
            fkwg.update(detail_weight=detail_weight)
        if seed is not None:
            fkwg.update(seed=seed)
        return image, fkwg

    @async_wrap_thread
    def regenerate_image(self, prompt: str, image: Image.Image,  
                         steps: int = None, 
                         strength: float = None, 
                         negative_prompt: str = None, 
                         guidance_scale: float = None, 
                         detail_weight: float = 0,
                         aspect: typing.Literal['square','portrait','landscape'] = None, 
                         refine_steps: int = 0, 
                         refine_strength: float = None, 
                         seed: int = None,
                         **kwargs) -> Image.Image:
        
        fkwg = self.config.get_if_none(steps=steps, strength=strength, negative_prompt=negative_prompt, guidance_scale=guidance_scale, refine_steps=refine_steps, refine_strength=refine_strength)
        print(f'kwargs: {kwargs}\nfkwg:{fkwg}')
        
        # Resize to best dim match unless aspect given. don't use fkwg[aspect] because dont want None autofilled
        dim_out = self.config.nearest_dims(image.size, use_hf_sbr=False) if aspect is None else self.config.get_dims(aspect)
        print('regenerate_image input size:', image.size, '->', dim_out)
        image = image.resize(dim_out, resample=Image.Resampling.LANCZOS)
        
        strength = cmd_tfms.percent_transform(fkwg['strength'])
        refine_strength = cmd_tfms.percent_transform(fkwg['refine_strength'])
        image = self.pipeline(prompt=prompt, num_inference_steps=fkwg['steps'], strength=strength, negative_prompt=fkwg['negative_prompt'], 
                              guidance_scale=fkwg['guidance_scale'], detail_weight=detail_weight, refine_steps=refine_steps, refine_strength=refine_strength, image=image, seed=seed) # target_size defaults img_size

        #self.base.disable_vae_tiling()
        # Don't want to fill with default value, but still want it in filledkwargs
        fkwg.update(aspect=aspect)
        if detail_weight:
            fkwg.update(detail_weight=detail_weight)
        if seed is not None:
            fkwg.update(seed=seed)
        return image, fkwg

    @async_wrap_thread
    def regenerate_frames(self, prompt: str, frame_array: np.ndarray, 
                          steps: int = None, 
                          astrength: float = 0.5, 
                          imsize: typing.Literal['small','med','full'] = 'small', 

                          negative_prompt: str = None, 
                          guidance_scale: float = None, 
                          detail_weight: float = 0, 
                          aseed: int = None, 
                          **kwargs):   
        
        gseed = None
        # NOTE: special behavior since having a seed improves results substantially 
        if aseed is None: 
            if self.global_seed is not None:
                aseed = self.global_seed
            else:
                aseed = np.random.randint(1e9, 1e10-1)
        elif aseed < 0:
            aseed = False
            
            
        lora_kwargs={'cross_attention_kwargs': {"scale": detail_weight}} if detail_weight and self.has_adapters else {}

        fkwg = self.config.get_if_none(steps=steps, strength=astrength, negative_prompt=negative_prompt, guidance_scale=guidance_scale)
        steps = fkwg['steps']
        astrength = cmd_tfms.percent_transform(astrength)
        negative_prompt=fkwg['negative_prompt']
        guidance_scale=fkwg['guidance_scale']

        astrength = max(astrength*steps, 1)/steps # clip strength so at least 1 step occurs
        print(f'kwargs: {kwargs}\nfkwg:{fkwg}')
        
        nf,h,w,c = frame_array.shape
        img_wh = (w,h)
        
        dims_opts = {
            'tiny': [(512,512), (512,640), (640,512)], # 1.25
            'small': [(640,640), (512,768), (768,512)], # 1.5
            'med': [(768,768), (640,896), (896,640)], # 1.4
            'full': [(1024,1024), (832, 1216), (1216, 832)] # 1.46
        }
        batch_sizes = {'tiny': 4, 'small': 4, 'med': 3, 'full': 2}
        upsample_bsz = 8
        upsample_px_thresh = 256 # upsample first if less than 1/4th the recommended total pixels 

        dim_choices = dims_opts.get(imsize, None) # SDXL_DIMS
        bsz = batch_sizes.get(imsize, 1) #round(2/scale) # half = b4, full = b2
        
        dim_out = self.config.nearest_dims(img_wh, dim_choices=dim_choices, scale=1, use_hf_sbr=False)
        
        # if img_wh[0] < dim_out[0] or img_wh[1] < dim_out[1]:
        if img_wh[0]*img_wh[1] <= upsample_px_thresh**2: 
            print(f'Upsampling... ({img_wh[0]}*{img_wh[1]}) < {upsample_px_thresh}² < {dim_out}')
            # resized_images = [self.upsampler.upsample(Image.fromarray(imarr, mode='RGB')).resize(dim_out, resample=Image.Resampling.LANCZOS) for imarr in frame_array]
            
            batched_frames = list(batched(frame_array, upsample_bsz))
            resized_images = [img.resize(dim_out, resample=Image.Resampling.LANCZOS) for imbatch in tqdm(batched_frames) for img in self.upsampler.upsample(imbatch)]
            print_memstats('batched upsample')
        else:
            resized_images = [Image.fromarray(imarr, mode='RGB').resize(dim_out, resample=Image.Resampling.LANCZOS) for imarr in frame_array]
        #resized_images = [Image.fromarray(imarr, mode='RGB').resize(dim_out, resample=Image.Resampling.LANCZOS) for imarr in frame_array]
        print('regenerate_frames input size:', img_wh, '->', dim_out)
        
        batched_prompts_encodings = self.embed_prompts(prompt, negative_prompt=negative_prompt, batch_size=bsz)
        
        for imbatch in batched(resized_images, bsz):
            batch_len = len(imbatch)
            # shared common seed helps SIGNIFICANTLY with cohesion
            if aseed:
                # list of generators is very important. Otherwise it does not apply correctly
                gseed = [torch.Generator(device='cuda').manual_seed(aseed) for _ in range(batch_len)]
            
            if batch_len != bsz:
                # it can only be less due to how batched is implemented
                batched_prompts_encodings = {k: v[:batch_len] for k,v in batched_prompts_encodings.items() if v is not None}

            
            images = self.basei2i(image=imbatch, num_inference_steps=steps, strength=astrength, guidance_scale=guidance_scale, clip_skip=self.clip_skip, 
                                  generator=gseed, **batched_prompts_encodings, **lora_kwargs, **kwargs).images
            
            yield images
        release_memory()
        #return out_images, fkwg


    @async_wrap_thread
    def generate_frames(self, prompt: str, 
                            image: Image.Image = None, 
                            nframes: int = 16,
                            steps: int = None, 
                            strength_end: float = 0.80, 
                            strength_start: float = 0.30, 
                            negative_prompt: str = None, 
                            guidance_scale: float = None, 
                            detail_weight: float = 0, 
                            aspect: typing.Literal['square','portrait','landscape'] = None, 
                            seed: int = None, 
                            **kwargs):   
        #from tqdm.auto import tqdm
        if seed is None:
            seed = self.global_seed
            # NOTE: if you attempt to update the seed on each iteration, you get some interesting behavoir
            # you effectively turn it into a coloring book generator. I assume this is a product of how diffusion works
            # since it predicts the noise to remove, when you feed its last prediction autoregressive style, boils it down
            # the minimal representation of the prompt. If you 
        
        gseed = torch.Generator(device='cpu').manual_seed(seed) if seed is not None else None
        prompt_encodings = self.embed_prompts(prompt, negative_prompt=negative_prompt)
        lora_kwargs={'cross_attention_kwargs': {"scale": detail_weight}} if detail_weight and self.has_adapters else {}

        fkwg = self.config.get_if_none(steps=steps, negative_prompt=negative_prompt, guidance_scale=guidance_scale, aspect=aspect)
        negative_prompt=fkwg['negative_prompt']
        guidance_scale=fkwg['guidance_scale']
        strength_end = cmd_tfms.percent_transform(strength_end)
        strength_start = cmd_tfms.percent_transform(strength_start)
        steps = fkwg['steps']
        aspect = fkwg['aspect']
        print(f'kwargs: {kwargs}\nfkwg:{fkwg}')
        
        
        # subtract 1 for first image
        nframes = nframes - 1

        
        if image is None:
            strengths = [strength_end]*nframes
            w,h = self.config.get_dims(aspect)
            # timesteps=AysSchedules["StableDiffusionXLTimesteps"]
            image = self.base(num_inference_steps=steps, guidance_scale=guidance_scale, height=h, width=w, clip_skip=self.clip_skip,
                              **prompt_encodings, **lora_kwargs, generator=gseed).images[0] # num_images_per_prompt=4
        else:
            #strengths = np.linspace(strength_start, strength_end, num=nframes)
            # strengths = np.around(steps*np.linspace(strength_start, strength_end, num=nframes))/steps
            strengths = discretize_strengths(steps, nframes, start=strength_start, end=strength_end)
            # round up strengths since they will be floored in get_timesteps via int()
            # and it makes step distribution more uniform for lightning models
            
            img_wh = image.size
            dim_choices=SDXL_DIMS # None
            dim_out = self.config.nearest_dims(img_wh, dim_choices=dim_choices, scale=1.0, use_hf_sbr=False)
            print('regenerate_frames input size:', img_wh, '->', dim_out)
            image = image.resize(dim_out, resample=Image.Resampling.LANCZOS) 
        
        yield image

        
        steps = calc_esteps(steps, min(strengths), min_effective_steps=1)
        
        
        for i in range(nframes):
            # gseed.manual_seed(seed) # uncommenting this will turn into a coloring book generator
            image = self.basei2i(image=image, num_inference_steps=steps, strength=strengths[i], guidance_scale=guidance_scale, clip_skip=self.clip_skip,
                                    **prompt_encodings, **lora_kwargs, generator=gseed).images[0]
            #image_frames.append(image)
            yield image

        release_memory()
        #return out_images, fkwg
            
    
class Upsampler:#(ImageFormatter):
    def __init__(self, model_name:typing.Literal["4xNMKD-Superscale.pth", "4xUltrasharp-V10.pth","4xRealWebPhoto_v4_dat2.safetensors"], input_mode:typing.Literal['BGR','RGB'], dtype=torch.bfloat16):
        # https://civitai.com/articles/904/settings-recommendations-for-novices-a-guide-for-understanding-the-settings-of-txt2img
        # https://github.com/joeyballentine/ESRGAN-Bot/blob/master/testbot.py#L73

        # https://huggingface.co/uwg/upscaler/blob/main/ESRGAN/8x_NMKD-Superscale_150000_G.pth
        self.input_mode = input_mode
        # NOTE: for some models (e.g. 4xNMKD-Superscale) rgb/bgr does seem to make a difference
        # others are barely perceptible
        # From limited testing: 4xNMKD: RGB ⋙ BGR, 4xRealWeb: RGB ≥ BGR, 4xUltra: BGR ≥ RGB, 
        ext_modeldir = cpaths.ROOT_DIR/'extras/models'
        
        # load a model from disk
        self.model = ModelLoader('cuda').load_from_file(ext_modeldir.joinpath(model_name))
        # make sure it's an image to image model
        assert isinstance(self.model, ImageModelDescriptor)
        
        self.model = self.model.eval().to('cuda', dtype=dtype)
    
    def nppil_to_torch(self, images: np.ndarray|Image.Image|list[Image.Image]) -> torch.FloatTensor:
        if not isinstance(images, np.ndarray):
            if not isinstance(images, list):
                images = [images]
            
            images = np.stack([np.array(img) for img in images]) # .convert("RGB")
            
        if images.ndim == 3:
            images = images[None]
        if self.input_mode == 'BGR':
            images = images[:, :, :, ::-1]  # flip RGB to BGR
        images = np.transpose(images, (0, 3, 1, 2))  # BHWC to BCHW
        images = np.ascontiguousarray(images, dtype=np.float32) / 255.  # Rescale to [0, 1]
        return torch.from_numpy(images)

    def torch_to_pil(self, tensor: torch.Tensor) -> list[Image.Image]:
        arr = tensor.float().cpu().clamp_(0, 1).numpy()
        arr = (arr * 255).round().astype("uint8")
        arr = arr.transpose(0, 2, 3, 1) # BCHW to BHWC
        if self.input_mode == 'BGR':
            arr = arr[:, :, :, ::-1] # BGR -> RGB

        return [Image.fromarray(a, "RGB") for a in arr]
        
    @torch.inference_mode()
    def process(self, img: torch.FloatTensor) -> torch.Tensor:
        img = img.to(self.model.device, dtype=self.model.dtype)
        with torch.autocast(self.model.device.type, self.model.dtype):
            output = self.model(img).detach_()
        
        return output
    
        
    @torch.inference_mode()
    def upsample(self, images: np.ndarray|Image.Image|list[Image.Image]) -> list[Image.Image]:
        output = self.process(self.nppil_to_torch(images))
        return self.torch_to_pil(output)
    
    @torch.inference_mode()
    def upscale(self, images: np.ndarray|Image.Image|list[Image.Image], scale:float=None) -> list[Image.Image]:
        images = self.nppil_to_torch(images)
        if scale is None:
            scale = self.model.scale
        b,c,h,w = images.shape
        dest_w = int((w * scale) // 8 * 8)
        dest_h = int((h * scale) // 8 * 8)

        for _ in range(3):
            b,c,h,w = images.shape
            if w >= dest_w and h >= dest_h:
                break
            
            images = self.process(images)

        images = self.torch_to_pil(images)
        if images[0].width != dest_w or images[0].height != dest_h:
            images = [img.resize((dest_w, dest_h), resample=Image.Resampling.LANCZOS) for img in images]

        return images
    
class SDXLTurboManager(OneStageImageGenManager):
    def __init__(self, offload=False):
        super().__init__(
            model_name='sdxl_turbo', 
            model_path="stabilityai/sdxl-turbo",
            config=DiffusionConfig(
                steps = CfgItem(4, bounds=(1,4)), # 1-4 steps
                guidance_scale = CfgItem(0.0, locked=True),
                strength = CfgItem(0.55, bounds=(0.3, 0.9)),
                img_dims = (512,512),
                #refine_strength = 0.,
                locked=['guidance_scale', 'negative_prompt', 'aspect', 'denoise_blend', 'refine_guidance_scale'] # 'refine_strength'
            ), 
            offload=offload)
        
    
    def dc_fastmode(self, enable:bool, img2img=False):
        pass

    @async_wrap_thread
    def compile_pipeline(self):
        # Currently, not working properly for SDXL Turbo + savings are negligible
        pass


class DreamShaperXLManager(OneStageImageGenManager):
    def __init__(self, offload=True):
        super().__init__(
            model_name = 'dreamshaper_turbo', # bf16 saves ~3gb vram over fp16
            model_path = 'lykon/dreamshaper-xl-v2-turbo', # https://civitai.com/models/112902?modelVersionId=333449
            config = DiffusionConfig(
                steps = CfgItem(8, bounds=(4,8)),
                guidance_scale = CfgItem(2.0, locked=True),
                strength = CfgItem(0.85, bounds=(0.3, 0.95)),#0.55
                img_dims = [(1024,1024), (832,1216), (1216,832)],
                #refine_strength=CfgItem(0.3, bounds=(0.2, 0.4)),
                locked=['guidance_scale', 'denoise_blend',  'refine_guidance_scale'] # 'refine_strength',
            ),
            offload=offload,
            #scheduler_callback = lambda sched_config: DPMSolverSinglestepScheduler.from_config(sched_config, use_karras_sigmas=True)
            scheduler_setup=('DPM++ SDE Karras')
        )

class JuggernautXLLightningManager(OneStageImageGenManager):
    def __init__(self, offload=True):
        super().__init__( # https://huggingface.co/RunDiffusion/Juggernaut-XL-Lightning
            model_name = 'juggernaut_lightning', # https://civitai.com/models/133005/juggernaut-xl?modelVersionId=357609
            model_path = 'https://huggingface.co/RunDiffusion/Juggernaut-XL-Lightning/blob/main/Juggernaut_RunDiffusionPhoto2_Lightning_4Steps.safetensors', 
            
            config = DiffusionConfig(
                steps = CfgItem(6, bounds=(5,7)), # 4-6 step /  5 and 7
                guidance_scale = CfgItem(1.5, bounds=(1.5,2.0)),
                strength = CfgItem(0.85, bounds=(0.3, 0.95)),#0.6
                img_dims = [(1024,1024), (832,1216), (1216,832)],
                aspect='portrait',
                #refine_strength=CfgItem(0.3, bounds=(0.2, 0.4)),
                locked = ['denoise_blend', 'refine_guidance_scale'] # 'refine_strength'
            ),
            offload=offload,
            #scheduler_callback= lambda sched_config: DPMSolverSinglestepScheduler.from_config(sched_config, lower_order_final=True, use_karras_sigmas=False)
            scheduler_setup=('DPM++ SDE', {'lower_order_final':True})
        )
# https://huggingface.co/stabilityai/stable-diffusion-3-medium
# https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers
class SD3MediumManager(OneStageImageGenManager):
    def __init__(self, offload=True):
        super().__init__(
            model_name = 'sd3_medium',
            model_path = 'stabilityai/stable-diffusion-3-medium-diffusers',
            config = DiffusionConfig(
                steps = CfgItem(28, bounds=(24,42)),
                guidance_scale = CfgItem(7.0, bounds=(5,10)), 
                strength = CfgItem(0.65, bounds=(0.3, 0.95)),
                img_dims = [(1024,1024), (832,1216), (1216,832)],
                #refine_strength=CfgItem(0.3, bounds=(0.2, 0.4)),
                locked=['denoise_blend',  'refine_guidance_scale'] # 'refine_strength',
            ),
            offload=offload,
            # scheduler_callback = lambda: DPMSolverMultistepScheduler.from_config(self.base.scheduler.config, use_karras_sigmas=False)
            # scheduler_callback = lambda sched_config: FlowMatchEulerDiscreteScheduler.from_config(sched_config)
           
        )
    
    @async_wrap_thread
    def load_pipeline(self):
        pipeline_loader = (SD3Transformer2DModel.from_single_file if self.model_path.endswith('.safetensors') 
                           else StableDiffusion3Pipeline.from_pretrained)

        self.base: StableDiffusion3Pipeline = pipeline_loader(
            self.model_path, torch_dtype=torch.bfloat16,  use_safetensors=True, #add_watermarker=False,
        )
        self.initial_scheduler_config = self.base.scheduler.config
        if self._scheduler_setup is not None:
            sched_alias, self.scheduler_kwargs = self._scheduler_setup
            self.base.scheduler = get_scheduler(sched_alias, self.initial_scheduler_config, **self.scheduler_kwargs)
        
        if self.offload:
            self.base.enable_model_cpu_offload()
        else:
            self.base = self.base.to("cuda")
        
        self.compeler = None # Unsupported

        self.basei2i: StableDiffusion3Img2ImgPipeline = StableDiffusion3Img2ImgPipeline.from_pipe(self.base,)
        
        if not self.offload:
            self.basei2i = self.basei2i.to("cuda")
        
        
        self.upsampler = Upsampler('4xUltrasharp-V10.pth', "BGR", dtype=torch.bfloat16)
        # self.upsampler = Upsampler('4xNMKD-Superscale.pth', "RGB", dtype=torch.bfloat16)
        
        self.is_ready = True
    
    # Unsupported
    def embed_prompts(self, prompt:str, negative_prompt:str = None):
        return  dict(prompt=prompt, negative_prompt=negative_prompt)
    # Unsupported
    def dc_fastmode(self, enable:bool, img2img=False):
        return
    
    @async_wrap_thread
    def compile_pipeline(self):
        # calling the compiled pipeline on a different image size triggers compilation again which can be expensive.
        # - https://huggingface.co/docs/diffusers/optimization/torch2.0#torchcompile
        # https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_3#using-torch-compile-to-speed-up-inference
        torch_compile_flags()
        self.base.transformer.to(memory_format=torch.channels_last)
        self.base.vae.to(memory_format=torch.channels_last)
        self.base.transformer = torch.compile(self.base.transformer, mode="max-autotune", fullgraph=True)
        self.base.vae.decode = torch.compile(self.base.vae.decode, mode="max-autotune", fullgraph=True)
        
        for _ in range(3):
            _ = self.base("a photo of a cat holding a sign that says hello world")#, num_inference_steps=4, guidance_scale=self.config.guidance_scale)
        self.is_compiled = True
    
class ColorfulXLLightningManager(OneStageImageGenManager):
    def __init__(self, offload=True):
        super().__init__( # https://huggingface.co/recoilme/ColorfulXL-Lightning
            model_name = 'colorfulxl_lightning', # https://civitai.com/models/388913/colorfulxl-lightning
            model_path = 'recoilme/ColorfulXL-Lightning',  
            
            config = DiffusionConfig(
                steps = CfgItem(9, bounds=(4,10)), # https://imgsys.org/rankings
                guidance_scale = CfgItem(1.5, bounds=(0,2.0)),
                strength = CfgItem(0.85, bounds=(0.3, 0.95)),
                img_dims = [(1024,1024), (832,1216), (1216,832)],
                aspect='square',#'portrait',
                #refine_strength=CfgItem(0.3, bounds=(0.2, 0.4)),
                locked = ['denoise_blend', 'refine_guidance_scale'] # 'refine_strength'
            ),
            offload=offload,
            # scheduler_callback= lambda: DPMSolverSinglestepScheduler.from_config(self.base.scheduler.config, lower_order_final=True, use_karras_sigmas=False)
            # scheduler_callback= lambda: EulerAncestralDiscreteScheduler.from_config(self.base.scheduler.config)
            # scheduler_callback= lambda: EulerDiscreteScheduler.from_config(self.base.scheduler.config, timestep_spacing="trailing")
            scheduler_setup = ('Euler A', {'timestep_spacing': "trailing"}),
            #scheduler_callback= lambda sched_config: EulerAncestralDiscreteScheduler.from_config(sched_config, timestep_spacing="trailing"),
            clip_skip=1,
        )

class RealVizXL4Manager(OneStageImageGenManager):
    def __init__(self, offload=True):
        super().__init__( # https://huggingface.co/SG161222/RealVisXL_V4.0
            model_name = 'realvisxl_v4', # https://civitai.com/models/139562?modelVersionId=344487
            model_path = 'SG161222/RealVisXL_V4.0',  
            
            config = DiffusionConfig(
                steps = CfgItem(25, bounds=(15,40)), # https://imgsys.org/rankings
                guidance_scale = CfgItem(7.5, bounds=(6,10)),
                strength = CfgItem(0.55, bounds=(0.3, 0.95)),
                img_dims = [(1024,1024), (832,1216), (1216,832)],
                aspect='square',#'portrait',
                #refine_strength=CfgItem(0.3, bounds=(0.2, 0.4)),
                locked = ['denoise_blend', 'refine_guidance_scale'] # 'refine_strength'
            ),
            offload=offload,
            scheduler_setup='DPM++ 2M Karras'
            #scheduler_callback= lambda sched_config: DPMSolverMultistepScheduler.from_config(sched_config, use_karras_sigmas=True) #, algorithm_type=sde-dpmsolver++
            # scheduler_callback= lambda: DPMSolverSinglestepScheduler.from_config(self.base.scheduler.config, lower_order_final=True, use_karras_sigmas=False)
            # scheduler_callback= lambda: EulerDiscreteScheduler.from_config(self.base.scheduler.config)

            #clip_skip=1,
        )        
# https://civitai.com/models/119229/zavychromaxl

class QuantizedFluxTransformer2DModel(quanto.QuantizedDiffusersModel):
    base_class = FluxTransformer2DModel
    
class QuantizedModelForTextEncoding(quanto.QuantizedTransformersModel):
    auto_class = AutoModelForTextEncoding
    

class FluxBase(OneStageImageGenManager):
    text_enc_model_id = "black-forest-labs/FLUX.1-schnell"
    quant_basedir = cpaths.ROOT_DIR / 'extras/quantized/flux/'
    
    @async_wrap_thread
    def load_pipeline(self):

        self.base: FluxPipeline = self._load_helper(qtype=self.qtype, offload=self.offload, text_encoder2_quant=self.te_2)
        

        self.initial_scheduler_config = self.base.scheduler.config
        
        if self._scheduler_setup is not None:
            sched_alias, self.scheduler_kwargs = self._scheduler_setup
            self.base.scheduler = get_scheduler(sched_alias, self.initial_scheduler_config, **self.scheduler_kwargs)
        
        #self.load_lora(adapter_name='detail_tweaker_xl')
        
        self.compeler = None
        #self.basei2i = self.base#AutoPipelineForImage2Image.from_pipe(self.base,)# torch_dtype=torch.bfloat16, use_safetensors=True, add_watermarker=False)#.to(0)

        self.upsampler = None #Upsampler('4xUltrasharp-V10.pth', "BGR", dtype=torch.bfloat16)
        #self.upsampler = Upsampler('4xNMKD-Superscale.pth', "RGB", dtype=torch.bfloat16)
        
        self.is_ready = True
        release_memory()

    def _load_helper(self, qtype:typing.Literal['qint4','qint8','qfloat8'], text_encoder2_quant:typing.Literal['bnb4','bnb8','bf16', 'qint4','qint8','qfloat8']=None, offload:bool=False,):
        '''Try to load quantized from disk otherwise, quantize on the fly
        
        Args:
            qtype: quantization type for transformer
            offload: enable model cpu off (incompatible with bnb)
            
            text_encoder2_quant: quantization type for T5 encoder. If None, use `qtype`. If bf16, no quantization.
        '''
        if text_encoder2_quant is None:
            text_encoder2_quant = qtype

        use_bnb = text_encoder2_quant.startswith('bnb')
        if use_bnb and offload:
            raise ValueError('Cannot use offloading with bitsandbytes quantized text encoder')

        if qtype=='qint4' and text_encoder2_quant == 'bf16':
            print('WARNING: qtype=qint4 is incompatiable with bfloat16 dtype. Text Encoder will be upcast to full fp32 precision. Expect high vRAM usage.')
        
        return self._load_quantized_nbit(qtype, text_encoder2_quant=text_encoder2_quant, offload=offload,)
        
    def _bnb_text_encoder(self, bnb_bits:typing.Literal['bnb4','bnb8'], torch_dtype:torch.dtype = torch.bfloat16):
        '''
        bnb pros: no quant time cost (~40s), slightly better results, slightly faster
        bnb cons: much more vram (4bit = 18.6 idle, 20.3 peak | 8bit = 20.0 idle, 21.8 peak) vs 17.6 idle. CANT OFFLOAD.
        Note:
            load_in_4bit=True with no args uses same vRAM and has similar if not better results than:
            load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        '''

        if bnb_bits == 'bnb4':
            quant_config = BitsAndBytesConfig(load_in_4bit=True)
        elif bnb_bits == 'bnb8':
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise NotImplementedError('only 4 or 8 bit is supported')
        with timed('BnB Load - text_encoder_2'):    
            text_encoder = T5EncoderModel.from_pretrained(self.text_enc_model_id, subfolder="text_encoder_2", quantization_config=quant_config, torch_dtype=torch_dtype)
        return text_encoder
    
    def load_or_quantize(self, module:typing.Literal["transformer","text_encoder_2"], qtype:typing.Literal['qint4','qint8','qfloat8'], quant_basedir:Path, device:torch.device):
        if qtype == 'qint4':
            torch_dtype = torch.float32
            transformer_exclude = ["proj_out", "x_embedder", "norm_out", "context_embedder"]
        else:
            torch_dtype = torch.bfloat16
            transformer_exclude = None

        transformer_dir = quant_basedir/qtype/self.variant/'transformer/'
        textenc2_dir = quant_basedir/qtype/'text_encoder_2/'

        if module=='text_encoder_2':
            if textenc2_dir.exists():
                with timed('Load - text_encoder_2'):
                    text_encoder_2 = QuantizedModelForTextEncoding.from_pretrained(textenc2_dir)#.to(device)
                    # .to(device) is required even if device=None, otherwise will get
                    # "TypeError: 'QuantizedModelForTextEncoding' object is not callable" since QuantizedTransformersModel doesn't implement __call__
            else:
                with timed('Quantize - text_encoder_2'):
                    text_encoder_2 = T5EncoderModel.from_pretrained(self.text_enc_model_id, subfolder="text_encoder_2", torch_dtype=torch_dtype)
                    quanto.quantize(text_encoder_2, weights=qtype, exclude=None)
                    quanto.freeze(text_encoder_2)
            
            return text_encoder_2.to(device)
        
        if module=='transformer':
            if transformer_dir.exists():
                with timed('Load - transformer'):
                    transformer = QuantizedFluxTransformer2DModel.from_pretrained(transformer_dir)#.to(device) # .to(device) is required to shed the wrapper
            else:
                with timed('Quantize - transformer'):
                    transformer:FluxTransformer2DModel = FluxTransformer2DModel.from_pretrained(self.model_path, subfolder="transformer", torch_dtype=torch_dtype)
                    quanto.quantize(transformer, weights=qtype, exclude=transformer_exclude)
                    quanto.freeze(transformer)
            
            return transformer.to(device)
        
        raise NotImplementedError(f'unknown module {module!r}')



    def _load_quantized_nbit(self, qtype:typing.Literal['qint4','qint8','qfloat8'], text_encoder2_quant:typing.Literal['bnb4','bnb8','bf16', 'qint4','qint8','qfloat8'], offload:bool=False,):
        device = None if offload else 'cuda'
        torch_dtype = torch.bfloat16 
        
        if qtype == 'qint4' or text_encoder2_quant == 'qint4':
            torch_dtype = torch.float32 # if using qint4, need to upcast.
            
        
        use_bnb = text_encoder2_quant.startswith('bnb')
        
        #quant_dir = self.quant_basedir/qtype
        #transformer_dir = quant_dir/self.variant/'transformer/'
        #textenc2_dir = quant_dir/'text_encoder_2/'

        #with timed('load - transformer'):
        transformer = self.load_or_quantize('transformer', qtype=qtype, quant_basedir=self.quant_basedir, device=device)
            #transformer = QuantizedFluxTransformer2DModel.from_pretrained(transformer_dir).to(device) # .to(device) is required to shed the wrapper
        

        text_encoder = None
        if text_encoder2_quant.startswith('q'):
            text_encoder = self.load_or_quantize('text_encoder_2', qtype=text_encoder2_quant, quant_basedir=self.quant_basedir, device=device)
            #with timed('load - text_encoder'):
            #    text_encoder = QuantizedModelForTextEncoding.from_pretrained(textenc2_dir).to(device)
                
        elif text_encoder2_quant=='bf16':
            text_encoder = T5EncoderModel.from_pretrained(self.text_enc_model_id, subfolder="text_encoder_2", torch_dtype=torch_dtype)

                
        with timed('pipe load'):
            pipe: FluxPipeline = FluxPipeline.from_pretrained(self.model_path, transformer=None, text_encoder_2=None, torch_dtype=torch_dtype)
        
        
        # Do not call pipe.to(device) more than once.
        # It will break qint4
        
        pipe.transformer = transformer

        if text_encoder is not None:
            pipe.text_encoder_2 = text_encoder
        
        if torch_dtype == torch.bfloat16:
            # can safely cast all elements of pipe to bf16 
            pipe.to(dtype=torch.bfloat16)
        

        if offload:
            pipe.enable_model_cpu_offload()
        else:
          with timed('pipe to device'):
              pipe.to(device)
        
        # This MUST come after all calls to `.to()`. bnb will yell at you if you try to move it anywhere
        if use_bnb:
            pipe.text_encoder_2 = self._bnb_text_encoder(text_encoder2_quant, torch_dtype=torch_dtype)
            #text_encoder

        print('PIPE:', pipe.dtype, pipe.device)
        print('TRANSFORMER:', pipe.transformer.dtype, pipe.transformer.device)
        print('TEXT ENC 2:', pipe.text_encoder_2.dtype, pipe.text_encoder_2.device)
        
        
        return pipe
        

    


    # @torch.inference_mode()
    @torch.no_grad()
    def pipeline(self, prompt, num_inference_steps, negative_prompt=None, guidance_scale=None, detail_weight=None, strength=None, refine_steps=None, refine_strength=None, image=None, target_size=None, seed=None):
        if image is not None:
            return image
        if seed is None:
            seed = self.global_seed
        gseed = torch.Generator(device='cuda').manual_seed(seed) if seed is not None else None
            
        t0 = time.perf_counter()
        prompt_encodings = self.embed_prompts(prompt, negative_prompt=negative_prompt)
        lora_kwargs={}#{'cross_attention_kwargs': {"scale": detail_weight}} if detail_weight and self.has_adapters else {}
        
        t_pe = time.perf_counter()
        t_main = t_pe # default in case skip
        mlabel = '_2I'
        # Txt2Img

        h,w = target_size
        image = self.base(num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, height=h, width=w, max_sequence_length=self.max_seq_len, #clip_skip=self.clip_skip,
                            **prompt_encodings, **lora_kwargs, generator=gseed).images[0] # num_images_per_prompt=4
        #return make_image_grid(imgs, 2, 2)
        t_main = time.perf_counter()
        print_memstats('draw')
        
        mlabel = 'T2I'
        release_memory()
        t_fi = time.perf_counter()
        print(f'total: {t_fi-t0:.2f}s | prompt_embed: {t_pe-t0:.2f}s | {mlabel}: {t_main-t_pe:.2f}s')# | upscale: {t_up-t_main:.2f}s | refine: {t_re-t_up:.2f}s')
        return image
    
    # Unsupported
    def embed_prompts(self, prompt:str, negative_prompt:str = None):
        return  dict(prompt=prompt) #, negative_prompt=negative_prompt) # neg not supported (offically)
    # Unsupported
    def dc_fastmode(self, enable:bool, img2img=False):
        return
    
    @async_wrap_thread
    def compile_pipeline(self):
        # calling the compiled pipeline on a different image size triggers compilation again which can be expensive.
        # - https://huggingface.co/docs/diffusers/optimization/torch2.0#torchcompile
        # https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_3#using-torch-compile-to-speed-up-inference
        torch_compile_flags()
        self.base.transformer.to(memory_format=torch.channels_last)
        self.base.vae.to(memory_format=torch.channels_last)
        self.base.transformer = torch.compile(self.base.transformer, mode="max-autotune", fullgraph=True)
        self.base.vae.decode = torch.compile(self.base.vae.decode, mode="max-autotune", fullgraph=True)
        
        for _ in range(3):
            _ = self.base("a photo of a cat holding a sign that says hello world")#, num_inference_steps=4, guidance_scale=self.config.guidance_scale)
        self.is_compiled = True

class FluxSchnellManager(FluxBase):
    def __init__(self, offload=True):
        super().__init__(
            model_name = 'flux_schnell',
            model_path = 'black-forest-labs/FLUX.1-schnell',
            config = DiffusionConfig(
                steps = CfgItem(4, bounds=(1,4)),
                guidance_scale = CfgItem(0.0, locked=True), 
                strength = CfgItem(0.65, bounds=(0.3, 0.95)),
                img_dims = [(1024,1024), (832,1216), (1216,832)],
                #refine_strength=CfgItem(0.3, bounds=(0.2, 0.4)),
                locked=['denoise_blend',  'refine_guidance_scale'] # 'refine_strength',
            ),
            offload=offload,
           
        )
        self.variant = 'schnell' if 'schnell' in self.model_name else 'dev'
        self.qtype = 'qfloat8' # ['qint4', 'qint8', 'qfloat8']
        #self.te_2 = 'bf16' if offload else 'bnb4'  # ['bnb4', 'bnb8', 'bf16', None, 'qint4', 'qint8', 'qfloat8']
        self.te_2 = None
        self.quant_basedir = cpaths.ROOT_DIR / 'extras/quantized/flux/'
        self.max_seq_len = 256
        
    

class FluxDevManager(FluxBase):
    def __init__(self, offload=True):
        super().__init__(
            model_name = 'flux_dev',
            model_path = 'black-forest-labs/FLUX.1-dev',
            config = DiffusionConfig(
                steps = CfgItem(50, bounds=(25,60)),
                guidance_scale = CfgItem(3.5, bounds=(1,10)), 
                strength = CfgItem(0.65, bounds=(0.3, 0.95)),
                img_dims = [(1024,1024), (832,1216), (1216,832)],
                #refine_strength=CfgItem(0.3, bounds=(0.2, 0.4)),
                locked=['denoise_blend',  'refine_guidance_scale'] # 'refine_strength',
            ),
            offload=offload,
        )
        self.variant = 'schnell' if 'schnell' in self.model_name else 'dev'
        self.qtype = 'qint4' # ['qint4', 'qint8', 'qfloat8']
        #self.te_2 = 'bf16' if offload else 'bnb4'  # ['bnb4', 'bnb8', 'bf16', None]
        self.te_2 = 'qfloat8'#None
        self.quant_basedir = cpaths.ROOT_DIR / 'extras/quantized/flux/'
        self.max_seq_len = 512
        # self.text_enc_model_id = "black-forest-labs/FLUX.1-schnell"



AVAILABLE_MODELS = {
    'sdxl_turbo': {
        'manager': SDXLTurboManager,
        'desc': 'Turbo SDXL' # (Sm, fast)
    },
    'dreamshaper_turbo': {
        'manager': DreamShaperXLManager,
        'desc': 'DreamShaper XL2 Turbo' #  (M, fast)
    },
    'juggernaut_lightning': {
        'manager': JuggernautXLLightningManager,
        'desc': 'Juggernaut XL Lightning' # (M, fast)
    },
    'sd3_medium': {
        'manager': SD3MediumManager,
        'desc': 'SD3 Medium' #  (Lg, slow)
    },
    'colorfulxl_lightning': {
        'manager': ColorfulXLLightningManager,
        'desc': 'ColorfulXL Lightning' # (M, fast)
    },
    'realvisxl_v4': {
        'manager': RealVizXL4Manager,
        'desc': 'RealVisXL V4.0' # (M, avg)
    },
    'flux_schnell': {
        'manager': FluxSchnellManager,
        'desc': 'Flux Schnell' # (Lg, avg)
    },
    'flux_dev': {
        'manager': FluxDevManager,
        'desc': 'Flux Dev' # (Lg, slow)
    },
}

