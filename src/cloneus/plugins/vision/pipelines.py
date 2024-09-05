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
from abc import abstractmethod
from dataclasses import dataclass, field, asdict, KW_ONLY, InitVar

import discord
from discord.ext import commands

from tqdm.auto import tqdm
import numpy as np
import torch

from cloneus.plugins.vision import quantops
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)

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
    FluxTransformer2DModel,
    AutoencoderKL,
    AnimateDiffSDXLPipeline
)

from diffusers.schedulers import AysSchedules
from huggingface_hub import hf_hub_download, snapshot_download


from safetensors.torch import load_file
from diffusers.utils import make_image_grid

from transformers.image_processing_utils import select_best_resolution
from transformers import T5EncoderModel, BitsAndBytesConfig, QuantoConfig, AutoModelForTextEncoding, CLIPTextModel, T5Config, CLIPTextConfig

from optimum import quanto
from accelerate import cpu_offload,init_empty_weights
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model

from DeepCache import DeepCacheSDHelper
from PIL import Image
from spandrel import ImageModelDescriptor, ModelLoader
from compel import Compel, ReturnedEmbeddingsType

from cloneus import cpaths
from cloneus.plugins.vision import quantops,specialists,loraops
from cloneus.plugins.vision.fluximg2img import FluxImg2ImgPipeline
from cloneus.utils.common import release_memory, batched

# import config.settings as settings
# from cmds import flags as cmd_flags, transformers as cmd_tfms
# from utils.globthread import async_wrap_thread, stop_global_thread
# from utils.image import batched

SDXL_DIMS = [(1024,1024), (1152, 896),(896, 1152), (1216, 832),(832, 1216), (1344, 768),(768, 1344), (1536, 640),(640, 1536),] # https://stablediffusionxl.com/sdxl-resolutions-and-aspect-ratios/
# other: [(1280, 768),(768, 1280),]

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

def list_schedulers(compatible_schedulers:list, return_aliases: bool = True) -> list[str]:
    implemented = ['DPMSolverMultistepScheduler', 'DPMSolverSinglestepScheduler', 'EulerDiscreteScheduler', 'EulerAncestralDiscreteScheduler', 'FlowMatchEulerDiscreteScheduler']
    
    if not return_aliases:
        return list(set(implemented) & set([s.__name__ for s in compatible_schedulers]))
    
    schedulers = []
    
    if DPMSolverMultistepScheduler in compatible_schedulers:
        schedulers += ['DPM++ 2M', 'DPM++ 2M Karras', 'DPM++ 2M SDE', 'DPM++ 2M SDE Karras']
    if DPMSolverSinglestepScheduler in compatible_schedulers:
        schedulers += ['DPM++ SDE', 'DPM++ SDE Karras']
    if EulerDiscreteScheduler in compatible_schedulers:
        schedulers += ['Euler']
    if EulerAncestralDiscreteScheduler in compatible_schedulers:
        schedulers += ['Euler A'] 
    if FlowMatchEulerDiscreteScheduler in compatible_schedulers:
        schedulers += ['Euler FM']
    return schedulers

class DeepCacheMixin:
    def _setup_dc(self):
        # https://github.com/horseee/DeepCache/blob/master/DeepCache/extension/deepcache.py
        # TODO: https://huggingface.co/docs/diffusers/v0.30.0/en/optimization/tgate?pipelines=Stable+Diffusion+XL
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

class SingleStagePipeline:
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

        # specialists
        self.upsampler = None
        self.florence = None

        # processors
        self.compeler = None
        
    def set_seed(self, seed:int|None = None):
        self.global_seed = seed
    
    def load_lora(self, weight_name:str = 'detail-tweaker-xl.safetensors', lora_dirpath:Path=None):
        pass
    
    @abstractmethod
    def load_pipeline(self):
        raise NotImplementedError('Requires subclass override')
    
    def unload_pipeline(self):
        self.base = None
        self.basei2i = None
        self.compeler = None
        self.upsampler = None
        self.florence = None
        self.initial_scheduler_config = None
        self.scheduler_kwargs = None
        release_memory()
        self.is_ready = False
        if self.is_compiled:
            torch_compile_flags(restore_defaults=True)
            torch._dynamo.reset()
            self.is_compiled = False

    def available_schedulers(self, return_aliases: bool = True) -> list[str]:
        if self.base is None:
            return []
        return list_schedulers(self.base.scheduler.compatibles, return_aliases=return_aliases)

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
        
    def dc_fastmode(self, enable:bool, img2img=False):
        pass

    @abstractmethod  
    def compile_pipeline(self):
        raise NotImplementedError('Requires subclass override')
    

    def batch_settings(self, imsize: typing.Literal['tiny','small','med','full'] = 'small', ):
        dims_opts = {
            'tiny': [(512,512), (512,640), (640,512)], # 1.25
            'small': [(640,640), (512,768), (768,512)], # 1.5
            'med': [(768,768), (640,896), (896,640)], # 1.4
            'full': [(1024,1024), (832, 1216), (1216, 832)] # 1.46
        }
        
        #batch_sizes = {'tiny': 4, 'small': 4, 'med': 3, 'full': 2} # With out vae_slicing
        batch_sizes = {'tiny': 24, 'small': 16, 'med': 12, 'full': 4} # With vae_slicing # 12.6gb, 13.5gb, 15gb -- NO BLOCK. (until converting gif)
        #batch_sizes = {'tiny': 32, 'small': 20, 'med': 16, 'full': 6} # With vae_slicing # 14gb, X, X -- 13.5gb, X, X heatbeat blocked small, 
        #batch_sizes = {'tiny': 24, 'small': 32, 'med': 24, 'full': 16} # With vae_slicing # 17gb, 16.5gb, 20gb -- heartbeat blocked all 3
        return (dims_opts[imsize], batch_sizes[imsize])

    @abstractmethod  
    def embed_prompts(self, prompt:str, negative_prompt:str = None, batch_size: int = 1, **kwargs):
        raise NotImplementedError('Requires subclass override')
    
    @torch.inference_mode()    
    def upsample(self, image, scale:float|None=None):
        '''If scale is None, do not resize after upsampling. Out size will be 4x image_dims (if 4xUltra) '''
        if self.upsampler is None:
            self.upsampler = specialists.Upsampler('4xUltrasharp-V10.pth', "BGR", dtype=torch.bfloat16)
        
        if scale is None:
            return self.upsampler.upsample(image)

        return self.upsampler.upscale(image, scale=scale)

    @torch.inference_mode()    
    def caption(self, image:Image.Image, caption_type:typing.Literal['brief', 'detailed', 'verbose']):
        if self.florence is None:
            self.florence = specialists.Florence(offload=True)
           
        return self.florence.caption(image, caption_type)
    
    @torch.inference_mode()
    def pipeline(self, prompt, num_inference_steps, negative_prompt=None, guidance_scale=None, detail_weight=None, strength=None, refine_steps=None, refine_strength=None, image=None, target_size=None, seed=None):
        if seed is None:
            seed = self.global_seed
        gseed = torch.Generator(device='cpu').manual_seed(seed) if seed is not None else None
            
        t0 = time.perf_counter()
        lora_scale = detail_weight if detail_weight and self.has_adapters else None
        if self.has_adapters and detail_weight is not None:
            self.base.set_adapters(self.base.get_active_adapters(), detail_weight)
        
        prompt_encodings = self.embed_prompts(prompt, negative_prompt=negative_prompt, lora_scale=lora_scale, clip_skip=self.clip_skip)

        t_pe = time.perf_counter()
        t_main = t_pe # default in case skip
        mlabel = '_2I'
        # Txt2Img
        if image is None: 
            #h,w is all you need -- https://github.com/huggingface/diffusers/blob/v0.26.3/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L1075
            h,w = target_size
            image = self.base(num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, height=h, width=w, **prompt_encodings, generator=gseed).images[0] # num_images_per_prompt=4
            #return make_image_grid(imgs, 2, 2)
            t_main = time.perf_counter()
            print_memstats('draw')
            #release_memory()
            mlabel = 'T2I'
        
        # Img2Img - not just else because could pass image with str=0 to just up+refine. But should never be image=None + strength
        elif strength:
            w,h = image.size
            #num_inference_steps = calc_esteps(num_inference_steps, strength, min_effective_steps=1)
            strength = np.clip(strength, 1/num_inference_steps, 1)
            image = self.basei2i(image=image, num_inference_steps=num_inference_steps, strength=strength, guidance_scale=guidance_scale, height=h, width=w, **prompt_encodings, generator=gseed).images[0]

            t_main = time.perf_counter()
            print_memstats('redraw')
            mlabel = 'I2I'
            
        t_up = t_re = t_main  # default in case skip
        # Upscale + Refine - must be up->refine. refine->up does not improve quality
        if refine_steps > 0 and refine_strength: 
            image = self.upsample(image, scale=1.5)[0]
            w,h = image.size
            t_up = time.perf_counter()
            print_memstats('upscale')
            #release_memory()
            torch.cuda.empty_cache()
            
            # NOTE: this will use the models default num_steps every time since "steps" isnt a param in HD
            # num_inference_steps = calc_esteps(-1, refine_strength, min_effective_steps=refine_steps)
            #num_inference_steps = calc_esteps(num_inference_steps, refine_strength, min_effective_steps=refine_steps)
            refine_strength = np.clip(refine_strength, 1/num_inference_steps, 1)
            image = self.basei2i(image=image, num_inference_steps=num_inference_steps, strength=refine_strength, guidance_scale=guidance_scale, height=h, width=w, **prompt_encodings, generator=gseed).images[0] 
            t_re = time.perf_counter()
            print_memstats('refine')
        
        t_fi = time.perf_counter()
        #release_memory()
        torch.cuda.empty_cache()
        
        print(f'total: {t_fi-t0:.2f}s | prompt_embed: {t_pe-t0:.2f}s | {mlabel}: {t_main-t_pe:.2f}s | upscale: {t_up-t_main:.2f}s | refine: {t_re-t_up:.2f}s')
        return image
    
    @torch.inference_mode()
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
        
        # TODO: return the ACTUAL kwargs used instead of subset. Do filtering in imagegen cog.

        # We don't want to run refine unless refine steps passed
        call_kwargs = dict(prompt=prompt, num_inference_steps=fkwg['steps'], negative_prompt=fkwg['negative_prompt'], 
                              guidance_scale=fkwg['guidance_scale'], detail_weight=detail_weight, refine_steps=refine_steps, refine_strength=refine_strength, target_size=target_size, seed=seed)
        
        image = self.pipeline(**call_kwargs)
        
        if detail_weight:
            fkwg.update(detail_weight=detail_weight)
        if seed is not None:
            fkwg.update(seed=seed)
        return image, fkwg

    @torch.inference_mode()
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
        target_size = tuple(reversed(image.size))

        call_kwargs = dict(prompt=prompt, num_inference_steps=fkwg['steps'], strength=strength, negative_prompt=fkwg['negative_prompt'], guidance_scale=fkwg['guidance_scale'], 
                              detail_weight=detail_weight, refine_steps=refine_steps, refine_strength=refine_strength, image=image, target_size=target_size, seed=seed,)
        
        image = self.pipeline(**call_kwargs)

        # Don't want to fill with default value, but still want it in filledkwargs
        call_kwargs.update(aspect=aspect)
        fkwg.update(aspect=aspect)
        if detail_weight:
            fkwg.update(detail_weight=detail_weight)
        if seed is not None:
            fkwg.update(seed=seed)
        return image, fkwg


    @torch.inference_mode()
    def regenerate_frames(self, prompt: str, frame_array: np.ndarray, 
                          steps: int = None, 
                          astrength: float = 0.5, 
                          imsize: typing.Literal['tiny','small','med','full'] = 'small', 

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
            
        if self.has_adapters and detail_weight is not None:
            self.base.set_adapters(self.base.get_active_adapters(), detail_weight)
        
        lora_scale = detail_weight if detail_weight and self.has_adapters else None

        fkwg = self.config.get_if_none(steps=steps, strength=astrength, negative_prompt=negative_prompt, guidance_scale=guidance_scale)
        steps = fkwg['steps']
        #astrength = cmd_tfms.percent_transform(astrength)
        negative_prompt=fkwg['negative_prompt']
        guidance_scale=fkwg['guidance_scale']

        astrength = np.clip(astrength, 1/steps, 1) # clip strength so at least 1 step occurs
        #astrength = max(astrength*steps, 1)/steps 
        print(f'kwargs: {kwargs}\nfkwg:{fkwg}')
        
        nf,h,w,c = frame_array.shape
        fstep = (nf//100) + 1 # if more than 100 frames, take every nth frame so we're not here all day
        frame_array = frame_array[::fstep]
        img_wh = (w,h)
        
        # NOTE: walking the latents could be maybe used? This implemention even makes use of slerp 
        # https://old.reddit.com/r/StableDiffusion/comments/1ez6m4q/a_simple_python_notebook_to_render_your_own/
        # https://github.com/rolux/flux-random-walk/blob/main/flux-random-walk.ipynb
        # Another guide: https://huggingface.co/learn/cookbook/en/stable_diffusion_interpolation

        upsample_bsz = 8
        upsample_px_thresh = 256 # upsample first if less than 1/4th the recommended total pixels 
        
        dim_choices,bsz = self.batch_settings(imsize)
        
        dim_out = self.config.nearest_dims(img_wh, dim_choices=dim_choices, scale=1, use_hf_sbr=False)

        if img_wh[0]*img_wh[1] <= upsample_px_thresh**2: 
            print(f'Upsampling... ({img_wh[0]}*{img_wh[1]}) < {upsample_px_thresh}² < {dim_out}')
            # resized_images = [self.upsampler.upsample(Image.fromarray(imarr, mode='RGB')).resize(dim_out, resample=Image.Resampling.LANCZOS) for imarr in frame_array]
            
            batched_frames = list(batched(frame_array, upsample_bsz))
            resized_images = [img.resize(dim_out, resample=Image.Resampling.LANCZOS) for imbatch in tqdm(batched_frames) for img in self.upsample(imbatch, scale=None)]
            print_memstats('batched upsample')
            torch.cuda.empty_cache()
        else:
            resized_images = [Image.fromarray(imarr).resize(dim_out, resample=Image.Resampling.LANCZOS) for imarr in frame_array]
        
        w,h = dim_out
        print('regenerate_frames input size:', img_wh, '->', dim_out)
        
        batched_prompts_encodings = self.embed_prompts(prompt, negative_prompt=negative_prompt, batch_size=bsz, lora_scale=lora_scale, clip_skip=self.clip_skip, partial_offload=True)
        
        latents = []
        t0 = time.perf_counter()
        for imbatch in batched(resized_images, bsz):
            batch_len = len(imbatch)
            # shared common seed helps SIGNIFICANTLY with cohesion
            if aseed:
                # list of generators is very important. Otherwise it does not apply correctly
                gseed = [torch.Generator(device='cuda').manual_seed(aseed) for _ in range(batch_len)]
            
            if batch_len != bsz:
                # it can only be less due to how batched is implemented, so slice is fine
                batched_prompts_encodings = {k: v[:batch_len] for k,v in batched_prompts_encodings.items() if v is not None}

            images = self.basei2i(image=imbatch, num_inference_steps=steps, strength=astrength, guidance_scale=guidance_scale, height=h, width=w, **batched_prompts_encodings, generator=gseed, output_type='latent', **kwargs).images 
            latents.append(images)
            yield batch_len
            
        
        image_frames = self.decode_latents(latents, height=h, width=w)
        yield image_frames

        te = time.perf_counter()
        runtime = te-t0
        print(f'RUN TIME: {runtime:0.2f}s | BSZ: {bsz} | DIM: {dim_out} | N_IMAGE: {len(resized_images)} | IMG/SEC: {len(resized_images)/runtime:0.2f}')
        release_memory()
        
        
        

    @torch.inference_mode()
    def generate_frames(self, prompt: str, 
                            image: Image.Image|None = None, 
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


        fkwg = self.config.get_if_none(steps=steps, negative_prompt=negative_prompt, guidance_scale=guidance_scale, aspect=aspect)
        negative_prompt=fkwg['negative_prompt']
        guidance_scale=fkwg['guidance_scale']
        steps = fkwg['steps']
        aspect = fkwg['aspect']
        print(f'kwargs: {kwargs}\nfkwg:{fkwg}')
        
        lora_scale = detail_weight if detail_weight and self.has_adapters else None
        prompt_encodings = self.embed_prompts(prompt, negative_prompt=negative_prompt, lora_scale=lora_scale, clip_skip=self.clip_skip, partial_offload=True)
        # subtract 1 for first image
        nframes = nframes - 1
        
        latents = []
        image_frames = []
        if image is None:
            strength_end = np.clip(strength_end, 1/steps, 1) 
            strengths = [strength_end]*nframes
            w,h = self.config.get_dims(aspect)
            # timesteps=AysSchedules["StableDiffusionXLTimesteps"]
            # image = self.base(num_inference_steps=steps, guidance_scale=guidance_scale, height=h, width=w,  **prompt_encodings, generator=gseed).images[0]
            image = self.base(num_inference_steps=steps, guidance_scale=guidance_scale, height=h, width=w,  **prompt_encodings, output_type='latent', generator=gseed).images#[0]
                              #**clip_kwargs, **prompt_encodings, **lora_kwargs, generator=gseed).images[0] # num_images_per_prompt=4
            
            latents.append(image)
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
            image_frames.append(image)
            w,h = image.size
        
        #yield image
        yield -1
        
        #steps = calc_esteps(steps, min(strengths), min_effective_steps=1)
        
        for i in range(nframes):
            # gseed.manual_seed(seed) # uncommenting this will turn into a coloring book generator
            image = self.basei2i(image=image, num_inference_steps=steps, strength=strengths[i], guidance_scale=guidance_scale, height=h, width=w, **prompt_encodings, output_type='latent', generator=gseed).images
            latents.append(image)
            yield i
            
            
        image_frames += self.decode_latents(latents, height=h, width=w, )
        
        yield image_frames
        release_memory()
    
    @abstractmethod
    def encode_images(self, images, **kwargs):
        raise NotImplementedError('Requires subclass override')
    
    @abstractmethod
    def decode_latents(self, latents, **kwargs):
        raise NotImplementedError('Requires subclass override')
    




class SDXLBase(SingleStagePipeline, DeepCacheMixin):
    def load_pipeline(self):
        pipeline_loader = (StableDiffusionXLPipeline.from_single_file if self.model_path.endswith('.safetensors') 
                           else AutoPipelineForText2Image.from_pretrained)

        self.base: StableDiffusionXLPipeline = pipeline_loader(
            self.model_path, torch_dtype=torch.bfloat16, variant="fp16", use_safetensors=True, add_watermarker=False, #, device_map=device,
        )
        
        if not self.offload:
            self.base = self.base.to(0)
        
        self.base.vae.enable_slicing()

        self.initial_scheduler_config = self.base.scheduler.config
        
        if self._scheduler_setup is not None:
            sched_alias, self.scheduler_kwargs = self._scheduler_setup
            self.base.scheduler = get_scheduler(sched_alias, self.initial_scheduler_config, **self.scheduler_kwargs)
        
        self.load_lora(weight_name='detail-tweaker-xl.safetensors')
        
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
        
        self.is_ready = True

    @torch.inference_mode()
    def load_lora(self, weight_name:str = 'detail-tweaker-xl.safetensors', lora_dirpath:Path=None):
        if lora_dirpath is None:
            lora_dirpath = cpaths.ROOT_DIR/'extras/loras/sdxl'
        
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
            

    def batch_settings(self, imsize: typing.Literal['tiny','small','med','full'] = 'small', ):
        dims_opts = {
            'tiny': [(512,512), (512,640), (640,512)], # 1.25
            'small': [(640,640), (512,768), (768,512)], # 1.5
            'med': [(768,768), (640,896), (896,640)], # 1.4
            'full': [(1024,1024), (832, 1216), (1216, 832)] # 1.46
        }
        
        #batch_sizes = {'tiny': 4, 'small': 4, 'med': 3, 'full': 2} # With out vae_slicing
        batch_sizes = {'tiny': 24, 'small': 16, 'med': 12, 'full': 4} # With vae_slicing # 12.6gb, 13.5gb, 15gb -- NO BLOCK. (until converting gif)
        #batch_sizes = {'tiny': 32, 'small': 20, 'med': 16, 'full': 6} # With vae_slicing # 14gb, X, X -- 13.5gb, X, X heatbeat blocked small, 
        #batch_sizes = {'tiny': 24, 'small': 32, 'med': 24, 'full': 16} # With vae_slicing # 17gb, 16.5gb, 20gb -- heartbeat blocked all 3
        return (dims_opts[imsize], batch_sizes[imsize])

    @torch.inference_mode()
    def embed_prompts(self, prompt:str, negative_prompt:str = None, batch_size: int = 1, **kwargs):
        prompt = [prompt]#*batch_size
        if negative_prompt is not None:
            negative_prompt = [negative_prompt]#*batch_size
        
        prompt_encodings = {}
        
        if set(prompt) & set('()-+'):
            conditioning, pooled = self.compeler(prompt)
            prompt_encodings.update(prompt_embeds=conditioning, pooled_prompt_embeds=pooled)
            prompt = None

            if negative_prompt is not None:
                neg_conditioning, neg_pooled = self.compeler(negative_prompt)
                prompt_encodings.update(negative_prompt_embeds=neg_conditioning, negative_pooled_prompt_embeds=neg_pooled)
                negative_prompt = None
            

        (p_emb, neg_p_emb, pooled_p_emb, neg_pooled_p_emb) = self.base.encode_prompt(
            prompt=prompt, negative_prompt=negative_prompt, device=0, **prompt_encodings, num_images_per_prompt=batch_size,
             lora_scale=kwargs.get('lora_scale'), clip_skip=kwargs.get('clip_skip'))

        prompt_encodings.update(
            prompt_embeds=p_emb, negative_prompt_embeds=neg_p_emb, 
            pooled_prompt_embeds=pooled_p_emb, negative_pooled_prompt_embeds=neg_pooled_p_emb
        )
            
        return prompt_encodings

    @torch.inference_mode()
    def decode_latents(self, latents, **kwargs):
        # https://github.com/huggingface/diffusers/blob/c977966502b70f4758c83ee5a855b48398042b03/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py#L1444
        if isinstance(latents, list):
            latents = torch.cat(latents, dim=0)
        
        # make sure the VAE is in float32 mode, as it overflows in float16
        needs_upcasting = self.base.vae.dtype == torch.float16 and self.base.vae.config.force_upcast

        if needs_upcasting:
            self.base.upcast_vae()
            latents = latents.to(next(iter(self.base.vae.post_quant_conv.parameters())).dtype)
        elif latents.dtype != self.base.vae.dtype:
            if torch.backends.mps.is_available():
                # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                self.base.vae = self.base.vae.to(latents.dtype)

        # unscale/denormalize the latents
        # denormalize with the mean and std if available and not None
        has_latents_mean = hasattr(self.base.vae.config, "latents_mean") and self.base.vae.config.latents_mean is not None
        has_latents_std = hasattr(self.base.vae.config, "latents_std") and self.base.vae.config.latents_std is not None
        if has_latents_mean and has_latents_std:
            latents_mean = (
                torch.tensor(self.base.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(self.base.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
            )
            latents = latents * latents_std / self.base.vae.config.scaling_factor + latents_mean
        else:
            latents = latents / self.base.vae.config.scaling_factor

        image = self.base.vae.decode(latents, return_dict=False)[0]

        # cast back to fp16 if needed
        if needs_upcasting:
            self.base.vae.to(dtype=torch.float16)

        image = self.base.image_processor.postprocess(image, output_type='pil')
        
        return image
    

class SD3Base(SingleStagePipeline):
    def __init__(self, model_name: str, model_path: str, config: DiffusionConfig, offload=False, scheduler_setup: str | tuple[str, dict] = None, clip_skip: int = None, ):
        super().__init__(model_name, model_path, config, offload, scheduler_setup, clip_skip)
    
    @torch.inference_mode()
    def load_pipeline(self):
        pipeline_loader = (SD3Transformer2DModel.from_single_file if self.model_path.endswith('.safetensors') 
                           else StableDiffusion3Pipeline.from_pretrained)

        self.base: StableDiffusion3Pipeline = pipeline_loader(
            self.model_path, torch_dtype=torch.bfloat16,  use_safetensors=True,
        )

        if not self.offload:
            self.base.to(0)

        self.base.vae.enable_slicing()

        self.initial_scheduler_config = self.base.scheduler.config
        if self._scheduler_setup is not None:
            sched_alias, self.scheduler_kwargs = self._scheduler_setup
            self.base.scheduler = get_scheduler(sched_alias, self.initial_scheduler_config, **self.scheduler_kwargs)
        
        self.basei2i: StableDiffusion3Img2ImgPipeline = StableDiffusion3Img2ImgPipeline.from_pipe(self.base,)
        
        if self.offload:
            self.base.enable_model_cpu_offload()
            self.basei2i.enable_model_cpu_offload()
        
        self.is_ready = True
    
    
    @torch.inference_mode()
    def embed_prompts(self, prompt:str, negative_prompt:str = None, batch_size: int = 1, **kwargs):
        prompt = [prompt]#*batch_size
        if negative_prompt is not None:
            negative_prompt = [negative_prompt]#*batch_size

        if not self.offload:
            self.base.text_encoder_3.to(0)

        (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds) = self.base.encode_prompt(
            prompt=prompt, prompt_2=None, prompt_3=None, negative_prompt=negative_prompt, num_images_per_prompt=batch_size,
            device=self.base.text_encoder_2.device, lora_scale=kwargs.get('lora_scale'), clip_skip=kwargs.get('clip_skip'))
        
        # moving the text encoder is just always a good idea since it's in full bf16 most of the time.
        if not self.offload and kwargs.get('partial_offload'): 
            self.base.text_encoder_3.to('cpu')
            torch.cuda.empty_cache()
        
            print(prompt_embeds.dtype, prompt_embeds.device)
        
        return dict(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds, negative_pooled_prompt_embeds=negative_pooled_prompt_embeds) # .bfloat16()
    
    @torch.inference_mode()
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

    @torch.inference_mode()
    def decode_latents(self, latents, **kwargs):
        # https://github.com/huggingface/diffusers/blob/f63c12633f154c2a1d79c17f4238fb073133652c/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py#L924
        if isinstance(latents, list):
            latents = torch.cat(latents, dim=0)
        
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type='pil')
        
        return image

class FluxBase(SingleStagePipeline):
    def __init__(self, model_name: str, model_path: str, config: DiffusionConfig, offload=False, scheduler_setup: str | tuple[str, dict] = None, clip_skip: int = None, 
                 qtype:typing.Literal['bnb4','qint4','qint8','qfloat8'] = 'bnb4', te2_qtype:typing.Literal['bnb4','bnb8','bf16', 'qint4','qint8','qfloat8'] = 'bf16', quant_basedir:Path = None):
        
        super().__init__(model_name, model_path, config, offload, scheduler_setup, clip_skip)
        
        self.text_enc_model_id = "black-forest-labs/FLUX.1-dev" # use same shared text_encoder_2, avoid 2x-download
        self.qtype = qtype
        self.te2_qtype = te2_qtype
        self.quant_basedir = quant_basedir if quant_basedir is not None else cpaths.ROOT_DIR / 'extras/quantized/flux/'
        
        self.variant = model_name.split('_')[-1] # flux_schell,flux_dev,flux_* -> *
        self.max_seq_len = 256 if self.variant == 'schnell' else 512
        self.text_pipe:FluxPipeline = None
    
    
    def unload_pipeline(self):
        components = ['text_encoder','text_encoder_2','transformer','vae']
        if self.base:
            self.base.to('cpu')
            for comp in components:
                getattr(self.base, comp).to('cpu')
                setattr(self.base, comp, None)
            
        #     self.base.to('cpu')
        #     self.base.text_encoder_2.to('cpu')
        #     self.base.text_encoder.to('cpu')
        #     self.base.transformer.to('cpu')
        #     self.base.vae.to('cpu')
        # self.base.text_encoder_2 = None
        # self.base.text_encoder = None
        # self.base.transformer = None
        # self.base.vae = None
        self.text_pipe = None
        super().unload_pipeline()
        #print('basei2i:',self.basei2i)
        # self.base = None
        # self.basei2i = None
        # self.text_pipe = None
        # self.compeler = None
        # self.upsampler = None
        # self.initial_scheduler_config = None
        # self.scheduler_kwargs = None
        # release_memory()
        # self.is_ready = False
        # if self.is_compiled:
        #     torch_compile_flags(restore_defaults=True)
        #     torch._dynamo.reset()
        #     self.is_compiled = False

    def load_pipeline(self):
        with torch.inference_mode():
            self.base: FluxPipeline = self._load_helper(qtype=self.qtype, offload=self.offload, te2_qtype=self.te2_qtype)
        
        if not self.offload:
            self.base.to(0)
        
        self.base.vae.enable_slicing() # all this does is iterate over batch instead of all at once, no reason to ever disable

        self.initial_scheduler_config = self.base.scheduler.config
        
        if self._scheduler_setup is not None:
            sched_alias, self.scheduler_kwargs = self._scheduler_setup
            self.base.scheduler = get_scheduler(sched_alias, self.initial_scheduler_config, **self.scheduler_kwargs)
        
        self.load_lora('detail-maximizer.safetensors')
        self.load_lora('midjourneyV6_1.safetensors')
        
        with torch.inference_mode():
            self.basei2i: FluxImg2ImgPipeline = FluxImg2ImgPipeline.from_pipe(self.base)

        #self.upsampler = Upsampler('4xUltrasharp-V10.pth', "BGR", dtype=torch.bfloat16)
        #self.upsampler = Upsampler('4xNMKD-Superscale.pth', "RGB", dtype=torch.bfloat16)
        if self.offload:
            # calling on both pipes seems to make offload more consistent, may not be inherited properly with from_pipe 
            self.base.enable_model_cpu_offload()
            self.basei2i.enable_model_cpu_offload()
        
        self.is_ready = True
        release_memory()
    
    @torch.inference_mode()
    def _load_transformer(self, qtype:str, device:torch.device, torch_dtype = torch.bfloat16):
        if qtype == 'bnb4':
            print('load transformer')
            
            if str(self.model_path).endswith('.safetensors'):
                # return bnbops.load_quantized(safetensor_filepath=self.model_path, bnb_quant_config=bnb_qconfig).to(device, dtype=torch_dtype)
                return quantops.load_sd_bnb4_transformer(safetensor_filepath=self.model_path).to(device, dtype=torch_dtype)
                #transformer = bnbops.bnb4_transformer(self.model_path, torch_dtype).to(device, dtype=torch_dtype)
            else:
               
                bnb_qconfig = BnbQuantizationConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', 
                                                     bnb_4bit_use_double_quant=False, bnb_4bit_compute_dtype='bf16', torch_dtype=torch.bfloat16, )
                                                    # bnb_4bit_use_double_quant=False, bnb_4bit_compute_dtype='fp32', torch_dtype=torch.float32, )

                #return  bnbops.create_4bit_transformer(self.model_path, bnb_qconfig).to(device, dtype=torch_dtype)
                return  quantops.bnb4_transformer(self.model_path, torch_dtype, bnb_qconfig).to(device, dtype=torch_dtype)
        
        if not (model_path := self.quant_basedir/qtype/self.variant/'transformer/').exists():
            model_path = self.model_path
        
        return quantops.load_or_quantize(model_path, qtype=qtype, module='transformer', device=device)
    
    @torch.inference_mode()
    def _load_text_encoder_2(self, te2_qtype:str, device:torch.device, torch_dtype = torch.bfloat16):
        if te2_qtype=='bf16':
           return T5EncoderModel.from_pretrained(self.text_enc_model_id, subfolder="text_encoder_2", torch_dtype=torch_dtype, low_cpu_mem_usage=True, device_map='auto')
        
        if te2_qtype.startswith('q'):
            if not (model_path := self.quant_basedir/te2_qtype/'text_encoder_2/').exists():
                model_path = self.model_path
            return quantops.load_or_quantize(model_path, qtype=te2_qtype, module='text_encoder_2', device=device)

        if te2_qtype in ['bnb4','bnb8']:
            # leave text_encoder_2 empty in the pipe line and instead use a seperate pipeline for text encoding so that the transformer can still be offloaded
            bnb_quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type= "nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False)
            
            text_encoder = T5EncoderModel.from_pretrained(self.text_enc_model_id, subfolder="text_encoder_2", quantization_config=bnb_quant_config, torch_dtype=torch_dtype, low_cpu_mem_usage=True, device_map='auto')
            self.text_pipe = DiffusionPipeline.from_pretrained(self.model_path, transformer=None, vae=None, text_encoder_2=text_encoder, torch_dtype=torch_dtype, low_cpu_mem_usage=True) 
            return None

    @torch.inference_mode()
    def _load_helper(self, qtype:typing.Literal['bnb4','qint4','qint8','qfloat8'], te2_qtype:typing.Literal['bnb4','bnb8','bf16', 'qint4','qint8','qfloat8'], offload:bool=False,):
        '''Try to load quantized from disk otherwise, quantize on the fly
        
        Args:
            qtype: quantization type for transformer
            te2_quant: quantization type for T5 encoder. If bf16, no quantization.
            offload: enable model cpu off (incompatible with bnb)
            
        '''

        print(f'transformer quant: {qtype}, text_encoder_2 quant: {te2_qtype}')
        
        device = None if offload else 0 #'cuda'
        torch_dtype = torch.bfloat16 
        
        
        te_kwarg = {'text_encoder':None} if te2_qtype.startswith('bnb') else {}
        pipe: FluxPipeline = FluxPipeline.from_pretrained(self.text_enc_model_id, transformer=None, text_encoder_2=None, **te_kwarg, torch_dtype=torch_dtype)  # 
        # don't move to device yet, loading transformer could use a LOT of vram if quantizing
        pipe.transformer = self._load_transformer(qtype, device, torch_dtype)
        release_memory()
        if pipe.text_encoder_2 is None:
            pipe.text_encoder_2 = self._load_text_encoder_2(te2_qtype, device, torch_dtype).to(device)
        
        pipe.transformer = pipe.transformer.to(device)
        release_memory()
        
        # Do not call pipe.to(device) more than once. It will break things
        pipe = pipe.to(device, dtype=torch_dtype)
        
        
        print('PIPE:', pipe.dtype, pipe.device)
        print('TRANSFORMER:', pipe.transformer.dtype, pipe.transformer.device)
        
        if pipe.text_encoder_2 is None:
            print('text_pipe - TEXT ENC 2:', self.text_pipe.text_encoder_2.dtype, self.text_pipe.text_encoder_2.device)
        else:
            print('TEXT ENC 2:', pipe.text_encoder_2.dtype, pipe.text_encoder_2.device)
        
        release_memory()
        return pipe

    @torch.inference_mode()
    def load_lora(self, weight_name:str = 'detail-maximizer.safetensors', lora_dirpath:Path=None):
        if lora_dirpath is None:
            lora_dirpath = cpaths.ROOT_DIR/'extras/loras/flux'
        
        adapter_name = weight_name.rsplit('.', maxsplit=1)[0].replace('-','_')
        
        try:
            #loraops.set_lora_transformer(self.base, Path(lora_dirpath).joinpath(weight_name), adapter_name)
            lora_sd = loraops.get_lora_state_dict(Path(lora_dirpath).joinpath(weight_name))
            self.base.load_lora_weights(lora_sd, adapter_name=adapter_name)
            print(self.base.get_active_adapters())
            #self.base.load_lora_weights(lora_dirpath, weight_name=weight_name, adapter_name=adapter_name)
            self.has_adapters = True
        except IOError as e:
            if weight_name == 'detail-maximizer.safetensors':
                print('detail-maximizer not found, detail parameter will not function. '
                    'To use, download from: https://civitai.com/models/562866/flux-image-upgrader-detail-maximizer-contrast-fix-for-low-cfg-sdxl-sd-15')
            else:
                raise NotImplementedError(f'Unsupported lora adapater {adapter_name!r}')

    def batch_settings(self, imsize: typing.Literal['tiny','small','med','full'] = 'small', ):
        dims_opts = {
            'tiny': [(512,512), (512,640), (640,512)], # 1.25
            'small': [(640,640), (512,768), (768,512)], # 1.5
            'med': [(768,768), (640,896), (896,640)], # 1.4
            'full': [(1024,1024), (832, 1216), (1216, 832)] # 1.46
        }
        
        batch_sizes = {'tiny': 4, 'small': 4, 'med': 3, 'full': 2} # With out vae_slicing
        #batch_sizes = {'tiny': 8, 'small': 8, 'med': 6, 'full': 4} # With out vae_slicing
        #batch_sizes = {'tiny': 24, 'small': 16, 'med': 12, 'full': 4} 

        return (dims_opts[imsize], batch_sizes[imsize])
    
    @torch.inference_mode()
    def embed_prompts(self, prompt:str, negative_prompt:str = None, batch_size: int = 1, **kwargs):
        prompt = [prompt]#*batch_size
        if self.base.text_encoder_2 is None:
            (prompt_embeds, pooled_prompt_embeds,_,) = self.text_pipe.encode_prompt(
                prompt=prompt, prompt_2=None, max_sequence_length=self.max_seq_len, num_images_per_prompt=batch_size,
                  device=self.text_pipe.text_encoder_2.device, lora_scale=kwargs.get('lora_scale'))
            
        else:
            if not self.offload:
                self.base.text_encoder_2.to(0)

            (prompt_embeds, pooled_prompt_embeds,_,) = self.base.encode_prompt(
                prompt=prompt, prompt_2=None, max_sequence_length=self.max_seq_len, num_images_per_prompt=batch_size, 
                device=self.base.text_encoder_2.device, lora_scale=kwargs.get('lora_scale'))
            
            # moving the text encoder is just always a good idea since it's in full bf16 most of the time.
            if not self.offload and kwargs.get('partial_offload'): 
                self.base.text_encoder_2.to('cpu')
                torch.cuda.empty_cache()
            print(prompt_embeds.dtype, prompt_embeds.device)
        
        return dict(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds) # .bfloat16()
        #return  dict(prompt=prompt) #, negative_prompt=negative_prompt) # neg not supported (offically)
    
    # Unsupported
    def dc_fastmode(self, enable:bool, img2img=False):
        return
    
    @torch.inference_mode()
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

    @torch.inference_mode()
    def decode_latents(self, latents, height:int, width:int):
        # https://github.com/huggingface/diffusers/blob/4cfb2164fb05d54dd594373b4bd1fbb101fef70c/src/diffusers/pipelines/flux/pipeline_flux.py#L759
        if isinstance(latents, list):
            latents = torch.cat(latents, dim=0)
        #print(latents.shape)
        
        # make sure the VAE is in float32 mode, as it overflows in float16
        latents = self.base._unpack_latents(latents, height, width, self.base.vae_scale_factor)
        latents = (latents / self.base.vae.config.scaling_factor) + self.base.vae.config.shift_factor
        image = self.base.vae.decode(latents, return_dict=False)[0]

        image = self.base.image_processor.postprocess(image, output_type='pil')
        
        return image