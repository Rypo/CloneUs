import os
import gc
import re
import io
import math
import time
import random
import typing
import logging
import itertools
import functools
from pathlib import Path
from abc import abstractmethod
from dataclasses import dataclass, field, asdict, KW_ONLY, InitVar


from tqdm.auto import tqdm
import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)

from diffusers import (
    AutoPipelineForText2Image, 
    AutoPipelineForImage2Image, 
    DiffusionPipeline, 
    StableDiffusionXLPipeline, 
    StableDiffusionXLImg2ImgPipeline, 
    StableDiffusionXLPAGPipeline,
    StableDiffusionXLPAGImg2ImgPipeline,
    StableDiffusion3Pipeline,
    StableDiffusion3Img2ImgPipeline,
    SD3Transformer2DModel,
    UNet2DConditionModel, 
    DPMSolverMultistepScheduler, 
    DPMSolverSinglestepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler,
    FluxPipeline, 
    FluxTransformer2DModel,
    AutoencoderKL,
    AnimateDiffSDXLPipeline,
    BitsAndBytesConfig,
    StableDiffusion3PAGPipeline
)

from diffusers.schedulers import AysSchedules
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img import retrieve_latents
from huggingface_hub import hf_hub_download, snapshot_download



from transformers.image_processing_utils import select_best_resolution


from optimum import quanto
from accelerate import cpu_offload,init_empty_weights
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model


from PIL import Image
from spandrel import ImageModelDescriptor, ModelLoader
# from compel import Compel, ReturnedEmbeddingsType

from cloneus import cpaths
from . import quantops,specialists,loraops,interpolation
from .fluxpatch import FluxImg2ImgPipeline
from .lpw_sdxl import get_weighted_text_embeddings_sdxl

from cloneus.utils.common import release_memory, batched

logger = logging.getLogger(__name__)
SDXL_DIMS = [(1024,1024), (1152, 896),(896, 1152), (1216, 832),(832, 1216), (1344, 768),(768, 1344), (1536, 640),(640, 1536),] # https://stablediffusionxl.com/sdxl-resolutions-and-aspect-ratios/
# other: [(1280, 768),(768, 1280),]

def print_memstats(label:str, reset_peak:bool=True):
    print(f'({label}) max_memory_allocated:', torch.cuda.max_memory_allocated()/(1024**2), 'max_memory_reserved:', torch.cuda.max_memory_reserved()/(1024**2))
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()

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
    clip_skip: int = None
    
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

    def to_md(self, keymap:dict=None):
        md_text = ""
        if keymap is None:
            keymap = {}
        for k,v in self.to_dict().items():
            if k in self.metadata_keys or (v.default is None and v.locked):
                continue
            
            default = v.default
            lb, ub = 0, 0
            if v.bounds is not None:
                lb, ub = v.bounds
            # These values are displayed as 0-100 for end-user simplicity
            if k in ['strength', 'refine_strength', ]: # 'denoise_blend'
                default = int(default*100)
                lb = int(lb*100)
                ub = int(ub*100)
            
            postfix = ''
            if v.locked:
                postfix = '🔒' 
            elif v.bounds is not None:
                postfix = f'*(typical: {lb} - {ub})*'
            
            keyname = keymap.get(k, k)
            md_text += f"\n- {keyname}: {default} {postfix}"
        return md_text

    def get_if_none(self, **kwargs):
        '''Fill with default config value if arg is None'''
        filled_kwargs = {}
        for k,v in kwargs.items():
            if v is None or getattr(self, f'{k}_').locked:
                v = getattr(self, k)
            filled_kwargs[k] = v
        return filled_kwargs
        
    def nearest_dims(self, img_wh:tuple[int,int], dim_choices:list[tuple[int,int]]=None, use_hf_sbr:bool=False):
        # TODO: compare implementation vs transformers select_best_resolution
        if dim_choices is None:
            dim_choices = self.img_dims
        
        if isinstance(dim_choices, tuple):
            return dim_choices

        # scale_dims = np.array(dim_choices)
        # if scale != 1:
        #     scale_dims *= scale
        #     scale_dims -= (scale_dims % 64) #8) # equiv: ((scale_dims//64)*64
        # dim_choices = list(map(tuple,scale_dims.astype(int)))

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

def rebatch_prompt_embs(emb_prompts:dict[str, torch.Tensor], batch_size:int):
    # TODO: compare with
    # https://github.com/comfyanonymous/ComfyUI/blob/0dbba9f7516acd6839e2849eca87facf61b5bf1a/comfy/utils.py#L543
    cur_bz = emb_prompts[next(iter(emb_prompts))].size(0)
    if cur_bz == batch_size:
        return emb_prompts
    
    # most common case, no need to expand
    if batch_size < cur_bz: 
        return {k: v[:batch_size,...] for k,v in emb_prompts.items()}
    
    rebatched_prombs = {}
    for k,v in emb_prompts.items():
        non_batch_dims = [-1]*(v.ndim - 1)
        rebatched_prombs[k] = v[[0], ...].expand(batch_size, *non_batch_dims)
            
    return rebatched_prombs

def generator_batch(seed:int|list[int]|None, num_seeds:int, device='cpu', all_unique:bool=False):
    if seed is None:
        return None
    
    if isinstance(seed, int):
        seeds = [seed]*num_seeds
    
    else:
        seeds = seed
        # given the exact same params, you can use 1 generator to reproduce. 
        # But if you change how you iterate over inputs, outputs with differ. This controls for that.
        
        if (unqlen := len(set(seeds))) != num_seeds:
            raise ValueError(f'If `seed` is a list of ints, it must have `num_seeds` ({num_seeds}) unique elements but only ({unqlen}) unique seeds passed')
        #seeds = seed
        #seeds = [seed + i for i in range(batch_size)]


    return [torch.Generator(device=device).manual_seed(s) for s in seeds]

SAMPLER_ALIASES = typing.Literal['DPM++ 2M', 'DPM++ 2M SDE', 'DPM++', 'Euler', 'Euler A', 'Euler FM', 'UniPC']
SCHEDULER_TYPE_ALIASES = typing.Literal['Karras','sgm_uniform','exponential','beta', 'ts_leading']
# https://github.com/comfyanonymous/ComfyUI/blob/f58475827150c2ac610cbb113019276efcd1a733/comfy/samplers.py#L732
def get_scheduler(init_sched_config:dict, sampler_alias:SAMPLER_ALIASES, sched_alias:SCHEDULER_TYPE_ALIASES =None,  **kwargs):
    # alias:typing.Literal['DPM++ 2M','DPM++ 2M Karras','DPM++ 2M SDE','DPM++ 2M SDE Karras','DPM++ SDE','DPM++ SDE Karras','Euler','Euler A', 'Euler FM']
    
    # https://huggingface.co/docs/diffusers/main/en/api/schedulers/overview#noise-schedules-and-schedule-types
    scheduler_type_aliases = {
        'Karras': dict(use_karras_sigmas=True),
        'sgm_uniform': dict(timestep_spacing="trailing"), #'simple': dict(timestep_spacing="trailing"),
        'exponential': dict(timestep_spacing="linspace", use_exponential_sigmas=True),
        'beta': dict(timestep_spacing="linspace", use_beta_sigmas=True),
        # custom aliases
        'ts_leading': dict(timestep_spacing="leading"),
    }

    kwargs.update({} if sched_alias is None else scheduler_type_aliases[sched_alias])
    # https://huggingface.co/docs/diffusers/main/en/api/schedulers/overview#schedulers
    # partials to avoid warnings for incompatible schedulers
    sampler_aliases = {
        'DPM++ 2M':             functools.partial(DPMSolverMultistepScheduler.from_config, **kwargs),
        #'DPM++ 2M Karras':      functools.partial(DPMSolverMultistepScheduler.from_config, use_karras_sigmas=True, **kwargs),
        'DPM++ 2M SDE':         functools.partial(DPMSolverMultistepScheduler.from_config, algorithm_type='sde-dpmsolver++', **kwargs),
        #'DPM++ 2M SDE Karras':  functools.partial(DPMSolverMultistepScheduler.from_config, use_karras_sigmas=True, algorithm_type='sde-dpmsolver++', **kwargs),
        
        'DPM++':                functools.partial(DPMSolverSinglestepScheduler.from_config, algorithm_type='dpmsolver++', **kwargs), 
        #'DPM++ SDE':            # sde-dpmsolver++ not supported
        #'DPM++ SDE Karras':     functools.partial(DPMSolverSinglestepScheduler.from_config, use_karras_sigmas=True, algorithm_type='dpmsolver++', **kwargs),

        'Euler':                functools.partial(EulerDiscreteScheduler.from_config, **kwargs),
        'Euler A':              functools.partial(EulerAncestralDiscreteScheduler.from_config, **kwargs),
        'Euler FM':             functools.partial(FlowMatchEulerDiscreteScheduler.from_config, **kwargs),

        'UniPC':                functools.partial(UniPCMultistepScheduler.from_config, **kwargs), # bh2 (default)
    }

    scheduler = sampler_aliases[sampler_alias](init_sched_config)
    return scheduler

def list_schedulers(compatible_schedulers:list, return_aliases: bool = True) -> list[str]:    
    # NOTE: diffusers has an enum for all compats, _compatibles = [e.name for e in KarrasDiffusionSchedulers] ( diffusers.schedulers.scheduling_utils.KarrasDiffusionSchedulers )
    implemented = ['DPMSolverMultistepScheduler', 'DPMSolverSinglestepScheduler', 
                   'EulerDiscreteScheduler', 'EulerAncestralDiscreteScheduler', 'FlowMatchEulerDiscreteScheduler',
                   'UniPCMultistepScheduler']
    
    if not return_aliases:
        return list(set(implemented) & set([s.__name__ for s in compatible_schedulers]))
    
    # will need to update if this ever changes
    if FlowMatchEulerDiscreteScheduler in compatible_schedulers:
        return ['Euler FM']

    schedulers = ['DPM++ 2M', 'DPM++ 2M SDE',] + ['DPM++', ] + ['Euler'] + ['Euler A'] + ['UniPC']
    
    # if DPMSolverMultistepScheduler in compatible_schedulers:
    #     schedulers += ['DPM++ 2M', 'DPM++ 2M SDE',] # 'DPM++ 2M Karras',  'DPM++ 2M SDE Karras'
    # if DPMSolverSinglestepScheduler in compatible_schedulers:
    #     schedulers += ['DPM++', ] # 'DPM++ SDE', 'DPM++ SDE Karras'
    # if EulerDiscreteScheduler in compatible_schedulers:
    #     schedulers += ['Euler']
    # if EulerAncestralDiscreteScheduler in compatible_schedulers:
    #     schedulers += ['Euler A'] 

    # if UniPCMultistepScheduler in compatible_schedulers:
    #     schedulers += ['UniPC','UniPC BH2']
    return schedulers

class DeepCacheMixin:
    def _setup_dc(self):
        from DeepCache import DeepCacheSDHelper
        # from tgate import TgateSDXLDeepCacheLoader
        # https://github.com/horseee/DeepCache/blob/master/DeepCache/extension/deepcache.py
        # TODO: https://huggingface.co/docs/diffusers/v0.30.0/en/optimization/tgate?pipelines=Stable+Diffusion+XL
        self.dc_base = DeepCacheSDHelper(self.base)
        self.dc_base.set_params(cache_interval=3, cache_branch_id=0)
        #self._dc_base = TgateSDXLDeepCacheLoader(self.base, cache_interval=3, cache_branch_id=0)
        #self.dc_base = functools.partial(self._dc_base.tgate, gate_step=10)
        #self.dc_base.tgate()
        #self.dc_basei2i = DeepCacheSDHelper(self.basei2i)
        #self.dc_basei2i.set_params(cache_interval=3, cache_branch_id=0)
        #self._dc_basei2i = TgateSDXLDeepCacheLoader(self.basei2i, cache_interval=3, cache_branch_id=0)
        #self.dc_basei2i = functools.partial(self._dc_basei2i.tgate, gate_step=10)
        self.dc_base.enable()
        self.dc_enabled = True
        
    def dc_fastmode(self, enable:bool, img2img=False):
        # TODO: This will break under various conditions
        # if min_effective_steps (steps*strength < 5) refine breaks?
        # e.g. refine_strength=0.3, steps < 25 / refine_strength=0.4, steps < 20

        # NOTE: if you call enable when already enabled, it will crash and burn
        if self.dc_enabled is None:
            if enable: self._setup_dc()
        elif self.dc_enabled != enable:
            # it's already been set up and states are in disagreement
            if enable: self.dc_base.enable()
            else: self.dc_base.disable()
            self.dc_enabled = enable

    
class SingleStagePipeline:
    def __init__(self, model_name: str, model_path:str, config:DiffusionConfig, offload=False, scheduler_setup:str|tuple[str, str, dict]=None, dtype=torch.bfloat16, root_name:str='base'):
        self.is_compiled = False
        self.is_ready = False
        self.dc_enabled = None
        
        self.model_name = model_name
        self.model_path = model_path
        self.config = config
        self.offload = offload

        self.initial_scheduler_config = None
        self._scheduler_setup = scheduler_setup #(scheduler_setup, {}) if isinstance(scheduler_setup,str) else scheduler_setup
        self.scheduler_kwargs = None

        self.dtype = dtype
        self.root_name = root_name
        self.clip_skip = self.config.clip_skip

        self.adapter_names = []
        self.adapter_weight = None
        self.lora_dirpath = cpaths.ROOT_DIR/f'extras/loras/{self.root_name}'
        # core
        self.base = None
        self.basei2i = None
        # specialists
        self.upsampler = None
        self.florence = None
        self.interpolator = None
        self.vqa = None

        # processors
        self.compeler = None
        # extra keywords passed to t2i,i2i pipes (e.g. pag_scale if using PAG)
        self.pipe_xkwgs = {}

    def t2i(self, *args, **kwargs):
        return self.base(*args, **kwargs, **self.pipe_xkwgs)
    
    def i2i(self, *args, **kwargs):
        kwargs.pop('height',None)
        kwargs.pop('width', None)
        return self.basei2i(*args, **kwargs, **self.pipe_xkwgs)
    
    def pbar_config(self, **kwargs):
        if self.base is not None:
            self.base.set_progress_bar_config(**kwargs)
        if self.basei2i is not None:
            self.basei2i.set_progress_bar_config(**kwargs)
    
    
    @abstractmethod
    def load_pipeline(self):
        raise NotImplementedError('Requires subclass override')
    
    def unload_pipeline(self):
        self.base = None
        self.basei2i = None
        self.compeler = None
        self.upsampler = None
        self.florence = None
        self.interpolator = None
        self.vqa = None
        self.initial_scheduler_config = None
        self.scheduler_kwargs = None
        release_memory()
        self.is_ready = False
        if self.is_compiled:
            torch_compile_flags(restore_defaults=True)
            torch._dynamo.reset()
            self.is_compiled = False
    
    def set_pag_config(self, pag_scale:float=0.0, pag_applied_layers: str|list[str] = None, disable_on_zero:bool=False):
        raise NotImplementedError('PAG not implemented for this pipeline type')
    
    def load_lora(self, weight_name:str, init_weight:float=None):
        pass


    def _scheduler_init(self) -> None:
        self.initial_scheduler_config = self.base.scheduler.config
        # cases: str, (str, str), (str, dict), (str, str, dict)
        if self._scheduler_setup is not None:
            sch_setup = self._scheduler_setup
            if isinstance(sch_setup, tuple) and len(sch_setup) < 2:
                sch_setup=sch_setup[0]
            
            if isinstance(sch_setup, str):
                self._scheduler_setup = (sch_setup, None, {})
            elif len(self._scheduler_setup) == 2:
                self._scheduler_setup = (*sch_setup, {}) if isinstance(sch_setup[1], str) else (sch_setup[0], None, sch_setup[1])


            sampler_alias, sched_alias, self.scheduler_kwargs = self._scheduler_setup
            self.base.scheduler = get_scheduler(self.initial_scheduler_config, sampler_alias, sched_alias, **self.scheduler_kwargs)

    def available_schedulers(self, return_aliases: bool = True) -> list[str]:
        if self.base is None:
            return []
        return list_schedulers(self.base.scheduler.compatibles, return_aliases=return_aliases)

    def set_scheduler(self, sampler_alias:SAMPLER_ALIASES, schedtype_alias:SCHEDULER_TYPE_ALIASES=None, **kwargs) -> str:
        # https://huggingface.co/docs/diffusers/main/en/api/schedulers/overview#schedulers
        assert self.base is not None, 'Model not loaded. Call `load_pipeline()` before proceeding.'
        if sampler_alias not in self.available_schedulers(return_aliases=True):
            raise KeyError(f'Scheduler {sampler_alias!r} not found or is incompatiable with active scheduler ({self.base.scheduler.__class__.__name__!r})')
        
        # be sure to keep anything initially passed like "timespace trailing" unless explicitly set otherwise
        if self.scheduler_kwargs:
            kwargs = {**self.scheduler_kwargs, **kwargs}
        
        scheduler = get_scheduler(self.initial_scheduler_config, sampler_alias, schedtype_alias, **kwargs)
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

    def preprocess_prompts(self, prompt:str, negative_prompt:str = None):
        if negative_prompt is None:
            negative_prompt = ''
        return prompt, negative_prompt


    @abstractmethod  
    def embed_prompts(self, prompt:str, negative_prompt:str = None, batch_size: int = 1, **kwargs):
        raise NotImplementedError('Requires subclass override')
    
    @torch.inference_mode()    
    def upsample(self, image:np.ndarray | Image.Image | list[Image.Image], scale:float|None=None, out_wh:tuple[int,int]=None):
        '''If scale is None, do not resize after upsampling. Out size will be 4x image_dims (if 4xUltra).
        If out_wh, resize to images exactly (w,h) after upsamping
        '''
        if self.upsampler is None:
            self.upsampler = specialists.Upsampler('4xUltrasharp-V10.pth', "BGR", dtype=torch.bfloat16)
        
        if scale is None:
            images = self.upsampler.upsample(image)
        else:
            images = self.upsampler.upscale(image, scale=scale)
        
        if out_wh is not None:
            images = [img.resize(out_wh, resample=Image.Resampling.LANCZOS) for img in images]
        
        return images

    @torch.inference_mode()    
    def caption(self, image:Image.Image, caption_type:typing.Literal['brief', 'detailed', 'verbose']):
        if self.florence is None:
            self.florence = specialists.Florence(offload=True)
        
        resp = self.florence.caption(image, caption_type)
        release_memory() 
        return resp
    
    @torch.inference_mode()    
    def vqa_chat(self, prompt:str=None, images:np.ndarray=None,  frames_as_video:bool=True, img_metas:list[dict]=None):
        if self.vqa is None:
            self.vqa = specialists.VQA(offload=True)
        
        resp = self.vqa.chat(prompt=prompt, images=images, frames_as_video=frames_as_video, img_metas=img_metas)
        release_memory()   
        return resp
    
    @torch.inference_mode()
    def interpolate(self, images:list[Image.Image], inter_frames:int=4, batch_size=2, allow_resize:bool=True,):
        if self.interpolator is None:
            self.interpolator = specialists.Interpolator()
        if len(images) == 2:
            inter_frames = max(inter_frames, 28)
            batch_size = 1
            if images[0].size != images[1].size:
                images[0] = images[0].resize(images[1].size, resample=Image.Resampling.LANCZOS)
        
        init_wh = images[1].size
        dim_out = self.config.nearest_dims(init_wh, dim_choices=SDXL_DIMS, use_hf_sbr=False)
        if init_wh[0]*init_wh[1] > dim_out[0]*dim_out[1]:
            images = [img.resize(dim_out, resample=Image.Resampling.LANCZOS) for img in images]

        frames = self.interpolator.interpolate_frames(images, inter_frames=inter_frames, batch_size=batch_size, allow_resize=allow_resize)
        release_memory()
        return [Image.fromarray(img) for img in frames]
        
    @torch.inference_mode()
    def toggle_loras(self, detail_weight:float|None = None):
        lora_scale = None
        if self.adapter_names and detail_weight != self.adapter_weight:
            print('Detail weight:', detail_weight, 'Current adapter weight:', self.adapter_weight)
            if detail_weight:
                lora_scale = detail_weight
                self.base.enable_lora()
                #self.base.enable_adapters()
                self.base.set_adapters(self.adapter_names, detail_weight)
            else:
                #self.base.disable_adapters()
                self.base.disable_lora()
            self.adapter_weight = detail_weight
        
            print('Active Adapters:', self.base.get_active_adapters())
            print('All Adapters:', self.base.get_list_adapters())
        return lora_scale


    @torch.inference_mode()
    def _pipe_txt2img(self, prompt_encodings:dict, num_inference_steps:int, guidance_scale:float, target_size:tuple[int,int], seed=None):
        gseed = torch.Generator(device='cpu').manual_seed(seed) if seed is not None else None
        h,w = target_size
        image = self.t2i(num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, height=h, width=w, **prompt_encodings, generator=gseed).images[0] # num_images_per_prompt=4
        
        return image

    @torch.inference_mode()
    def _pipe_img2img(self, prompt_encodings:dict, image:Image.Image, num_inference_steps:int, strength:float, guidance_scale:float, seed=None):
        gseed = torch.Generator(device='cpu').manual_seed(seed) if seed is not None else None
        w,h = image.size
        #num_inference_steps = calc_esteps(num_inference_steps, strength, min_effective_steps=1)
        strength = np.clip(strength, 1/num_inference_steps, 1)
        image = self.i2i(image=image, num_inference_steps=num_inference_steps, strength=strength, guidance_scale=guidance_scale, height=h, width=w, **prompt_encodings, generator=gseed).images[0]
        
        return image

    @torch.inference_mode()
    def _pipe_img2upimg(self, prompt_encodings:dict, image:Image.Image, num_inference_steps:int, refine_strength:float, guidance_scale:float, seed=None, scale=1.5):
        image = self.upsample(image, scale=scale)[0]
        print('upsized (w,h):', image.size)
        torch.cuda.empty_cache()

        return self._pipe_img2img(prompt_encodings, image, num_inference_steps, strength=refine_strength, guidance_scale=guidance_scale, seed=seed)
    
    @torch.inference_mode()
    def _batched_txt2img(self, prompts_encodings:dict[str, torch.Tensor], num_images:int, batch_size:int, num_inference_steps:int, guidance_scale:float, target_size:tuple[int, int], seed:list[int]|None, output_type:str='pil'):
        generators = generator_batch(seed, num_images, device='cpu', all_unique=True) # if seed is set, want a reproducible set of n different images, 
        if generators is None:
            generators = [None]*num_images
        h,w = target_size

        batch_size = min(num_images, batch_size)
        batched_prompts = rebatch_prompt_embs(prompts_encodings, batch_size)


        for gen_batch in batched(generators, batch_size):
            if (batch_len := len(gen_batch)) != batch_size:
               batched_prompts = rebatch_prompt_embs(prompts_encodings, batch_len)
            
            if gen_batch[0] is None:
                gen_batch = None
            
            images = self.t2i(num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, height=h, width=w, **batched_prompts, generator=gen_batch, output_type=output_type).images
            
            if output_type != 'latent':
                torch.cuda.empty_cache()
            
            yield images

    @torch.inference_mode()
    def _batched_img2img(self, prompts_encodings:dict[str, torch.Tensor], image:Image.Image, num_images:int, batch_size:int, num_inference_steps:int, strength:float, guidance_scale:float, seed:list[int]|None, output_type:str='pil'):
        generators = generator_batch(seed, num_images, device='cpu', all_unique=True) # if seed is set, want a reproducible set of n different images, 
        if generators is None:
            generators = [None]*num_images
        w,h = image.size

        batch_size = min(num_images, batch_size)
        batched_prompts = rebatch_prompt_embs(prompts_encodings, batch_size)
        
        # n_full_batches,final_batch_size = divmod(num_images, batch_size)
        # batch_lengths = [batch_size]*n_full_batches
        # if final_batch_size:
        #     batch_lengths.append(final_batch_size)
        
        # gen_batch = generators
        for gen_batch in batched(generators, batch_size):
            if (batch_len := len(gen_batch)) != batch_size:
                batched_prompts = rebatch_prompt_embs(prompts_encodings, batch_len)
            if gen_batch[0] is None:
                gen_batch = None
            #if generators is not None:
            #    gen_batch, generators = generators[:batch_len], generators[batch_len:] # pop slice
            
            images = self.i2i(image=image, num_inference_steps=num_inference_steps, strength=strength, guidance_scale=guidance_scale, height=h, width=w, **batched_prompts, generator=gen_batch, output_type=output_type).images 
            
            if output_type != 'latent':
                torch.cuda.empty_cache()

            yield images

    def _resize_image(self, image:Image.Image, aspect = None, dim_choices = None):
        dim_out = self.config.nearest_dims(image.size, dim_choices=dim_choices, use_hf_sbr=False)
        if aspect is not None:
             dim_out = self.config.get_dims(aspect) 
        #else:
            
        print('_resize_image input size:', image.size, '->', dim_out)
        
        image = image.resize(dim_out, resample=Image.Resampling.LANCZOS)
        return image
    
    @torch.inference_mode()
    def _resize_image_frames(self, frame_array:np.ndarray, dim_choices, max_num_frames:int=100, upsample_px_thresh:int = 256, upsample_bsz:int = 8):
        nf,in_h,in_w,c = frame_array.shape
        fstep = (nf//max_num_frames) + 1 # if more than 100 frames, take every nth frame so we're not here all day
        frame_array = frame_array[::fstep]
        print(f'num frames: {nf} -> {frame_array.shape[0]}')
        init_wh = (in_w, in_h)

        dim_out = self.config.nearest_dims(init_wh, dim_choices=dim_choices, use_hf_sbr=False)

        if in_w*in_h <= upsample_px_thresh**2: 
            print(f'Upsampling... ({in_w}*{in_h}) < {upsample_px_thresh}² < {dim_out}')
            
            batched_frames = list(batched(frame_array, upsample_bsz))
            resized_images = [img for imbatch in tqdm(batched_frames) for img in self.upsample(imbatch, scale=None, out_wh=dim_out)]
            print_memstats('batched upsample')
            torch.cuda.empty_cache()
        else:
            resized_images = [Image.fromarray(imarr).resize(dim_out, resample=Image.Resampling.LANCZOS) for imarr in frame_array]
        
        
        print('_resize_image_frames input size:', init_wh, '->', dim_out)
        return resized_images
    
    def seed_call_kwargs(self, seed:int|None, call_kwargs:dict, n_images:int, ):
        if seed == -1:
            seeds = None
            call_kwargsets = [{**call_kwargs, 'seed':None} for i in range(n_images)]
        else:
            np_rng = np.random.default_rng(call_kwargs.pop('seed', None))
            # could technically fail, but heat death of universe seems more likely
            seeds = list(set(np_rng.integers(1e9, 1e10, 32*n_images).tolist()))[:n_images]
            call_kwargsets = [{**call_kwargs, 'seed':s} for s in seeds]
        
        return call_kwargsets, seeds
    
    @torch.inference_mode()
    def generate_image(self, prompt:str, 
                       n_images:int=1,
                       steps:int=None, 
                       negative_prompt:str=None, 
                       guidance_scale:float=None, 
                       detail_weight:float=0,
                       aspect:typing.Literal['square','portrait','landscape']=None, 
                       refine_strength:float=None, 
                       seed:int = None,
                       **kwargs): 
        
        fkwg = self.config.get_if_none(steps=steps, negative_prompt=negative_prompt, guidance_scale=guidance_scale, aspect=aspect)
        steps = fkwg['steps']
        guidance_scale = fkwg['guidance_scale']
        negative_prompt=fkwg['negative_prompt']
        aspect = fkwg['aspect']
        print(f'unused_kwargs: {kwargs} | fkwg:{fkwg}')

        img_dims = self.config.get_dims(aspect)
        target_size = (img_dims[1], img_dims[0]) # h,w
        

        lora_scale = self.toggle_loras(detail_weight)
        prompt, negative_prompt = self.preprocess_prompts(prompt, negative_prompt)
        prompt_encodings = self.embed_prompts(prompt, negative_prompt=negative_prompt, lora_scale=lora_scale, clip_skip=self.clip_skip)

        call_kwargs = dict(prompt=prompt, steps=steps, negative_prompt=negative_prompt, guidance_scale=guidance_scale, 
                           detail_weight=detail_weight, aspect=aspect, refine_strength=refine_strength, seed=seed)
                
        if n_images > 1:
            _,BSZ = self.batch_settings('full')
            if n_images <= 10:
                BSZ = max(1, BSZ//2) # if a small batch, cut batch size in half
            
            call_kwargsets, seeds = self.seed_call_kwargs(seed, call_kwargs, n_images=n_images)
            batchgen = self._batched_txt2img(prompt_encodings, num_images=n_images, batch_size=BSZ, num_inference_steps=steps, guidance_scale=guidance_scale, target_size=target_size, seed=seeds, output_type='pil')
            
            for imbatch,kwbatch in zip(batchgen, batched(call_kwargsets, BSZ)):
                yield (imbatch, kwbatch)
        else:
            image = self._pipe_txt2img(prompt_encodings, num_inference_steps=steps, guidance_scale=guidance_scale, target_size=target_size, seed=seed)
            if refine_strength:
                image = self._pipe_img2upimg(prompt_encodings, image, num_inference_steps=steps, refine_strength=refine_strength, guidance_scale=guidance_scale, seed=seed, scale=1.5)
            yield ([image], [call_kwargs])
        
        release_memory()

    @torch.inference_mode()
    def regenerate_image(self, prompt: str, image: Image.Image,  
                         n_images: int = 1,
                         steps: int = None, 
                         strength: float = None, 
                         negative_prompt: str = None, 
                         guidance_scale: float = None, 
                         detail_weight: float = 0,
                         aspect: typing.Literal['square','portrait','landscape'] = None, 
                         
                         refine_strength: float = None, 
                         seed: int = None,
                         **kwargs):
        
        fkwg = self.config.get_if_none(steps=steps, strength=strength, negative_prompt=negative_prompt, guidance_scale=guidance_scale, )
        
        steps = fkwg['steps']
        guidance_scale = fkwg['guidance_scale']
        negative_prompt=fkwg['negative_prompt']
        strength = fkwg['strength']
        print(f'unused_kwargs: {kwargs} | fkwg:{fkwg}')
        
        # Resize to best dim match unless aspect given. don't use fkwg[aspect] because dont want None autofilled
        image = self._resize_image(image, aspect)
                

        lora_scale = self.toggle_loras(detail_weight)
        prompt, negative_prompt = self.preprocess_prompts(prompt, negative_prompt)
        prompt_encodings = self.embed_prompts(prompt, negative_prompt=negative_prompt, lora_scale=lora_scale, clip_skip=self.clip_skip)
        
        # don't want to pass around full image, going to replace with image_url in imagegen anyway
        call_kwargs = dict(prompt=prompt, image='PLACEHOLDER', steps=steps, strength=strength, negative_prompt=negative_prompt, guidance_scale=guidance_scale, 
                           detail_weight=detail_weight, aspect=aspect, refine_strength=refine_strength, seed=seed,)
        
        
        if n_images > 1:
            _,BSZ = self.batch_settings('full')
            if n_images <= 10:
                BSZ = max(1, BSZ//2) # if a small batch, cut batch size in half

            call_kwargsets, seeds = self.seed_call_kwargs(seed, call_kwargs, n_images=n_images)
            batchgen = self._batched_img2img(prompt_encodings, image, num_images=n_images, batch_size=BSZ, num_inference_steps=steps, strength=strength, guidance_scale=guidance_scale, seed=seeds, output_type='pil')
            
            for imbatch,kwbatch in zip(batchgen, batched(call_kwargsets, BSZ)):
                yield (imbatch, kwbatch)
        else:
            image = self._pipe_img2img(prompt_encodings, image, num_inference_steps=steps, strength=strength, guidance_scale=guidance_scale, seed=seed)
            if refine_strength:
                image = self._pipe_img2upimg(prompt_encodings, image, num_inference_steps=steps, refine_strength=refine_strength, guidance_scale=guidance_scale, seed=seed, scale=1.5)
            
            yield ([image], [call_kwargs])

        release_memory()
        #return image, call_kwargs
    
    @torch.inference_mode()
    def refine_image(self, image: Image.Image,
                     prompt: str = '', # prompt is optional
                     refine_strength: float = 0.3, 
                     steps: int = None, 
                     negative_prompt: str = None, 
                     guidance_scale: float = None, 
                     detail_weight: float = 0,
                     seed: int = None,
                     **kwargs) -> tuple[Image.Image, dict]:
        
        fkwg = self.config.get_if_none(steps=steps, negative_prompt=negative_prompt, guidance_scale=guidance_scale, )
        
        steps = fkwg['steps']
        guidance_scale = fkwg['guidance_scale']
        negative_prompt=fkwg['negative_prompt']

        print(f'unused_kwargs: {kwargs} | fkwg:{fkwg}')
        image = self._resize_image(image, aspect=None, dim_choices=None) # Resize to best dim match

        lora_scale = self.toggle_loras(detail_weight)
        prompt, negative_prompt = self.preprocess_prompts(prompt, negative_prompt)
        prompt_encodings = self.embed_prompts(prompt, negative_prompt=negative_prompt, lora_scale=lora_scale, clip_skip=self.clip_skip)

        image = self._pipe_img2upimg(prompt_encodings, image, num_inference_steps=steps, refine_strength=refine_strength, guidance_scale=guidance_scale, seed=seed, scale=1.5)
        
        call_kwargs = dict(image=image, prompt=prompt, refine_strength=refine_strength, steps=steps, negative_prompt=negative_prompt, 
                           guidance_scale=guidance_scale, detail_weight=detail_weight, seed=seed,)
        release_memory()
        return image, call_kwargs
    
    @torch.inference_mode()
    def generate_frames(self, prompt: str, 
                            image: Image.Image|None = None, 
                            nframes: int = 11,
                            steps: int = None, 
                            strength_end: float = 0.80, 
                            strength_start: float = 0.30, 
                            negative_prompt: str = None, 
                            guidance_scale: float = None, 
                            detail_weight: float = 0, 
                            aspect: typing.Literal['square','portrait','landscape'] = None, 
                            mid_frames:int=0,
                            seed: int = None, 
                            **kwargs):   
        
        gseed = torch.Generator(device='cpu').manual_seed(seed) if seed is not None else None
        # NOTE: if you attempt to update the seed on each iteration, you get some interesting behavoir
        # you effectively turn it into a coloring book generator. I assume this is a product of how diffusion works
        # since it predicts the noise to remove, when you feed its last prediction autoregressive style, boils it down
        # the minimal representation of the prompt. If you 
        
        # TODO: https://huggingface.co/THUDM/CogVideoX-5b-I2V
        fkwg = self.config.get_if_none(steps=steps, negative_prompt=negative_prompt, guidance_scale=guidance_scale, aspect=aspect)
        
        negative_prompt=fkwg['negative_prompt']
        guidance_scale=fkwg['guidance_scale']
        steps = fkwg['steps']
        aspect = fkwg['aspect']

        print(f'unused_kwargs: {kwargs} | fkwg:{fkwg}')
        
        lora_scale = self.toggle_loras(detail_weight)
        prompt, negative_prompt = self.preprocess_prompts(prompt, negative_prompt)
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
            image = self.t2i(num_inference_steps=steps, guidance_scale=guidance_scale, height=h, width=w,  **prompt_encodings, output_type='latent', generator=gseed).images#[0] # num_images_per_prompt=4
            
            latents.append(image)
        else:
            strengths = discretize_strengths(steps, nframes, start=strength_start, end=strength_end)
            # round up strengths since they will be floored in get_timesteps via int()
            # and it makes step distribution more uniform for lightning models
            image = self._resize_image(image, aspect=aspect, dim_choices=SDXL_DIMS)

            image_frames.append(image)
            w,h = image.size
        
        #yield image
        yield -1
                
        for i in range(nframes):
            # gseed.manual_seed(seed) # uncommenting this will turn into a coloring book generator
            image = self.i2i(image=image, num_inference_steps=steps, strength=strengths[i], guidance_scale=guidance_scale, height=h, width=w, **prompt_encodings, output_type='latent', generator=gseed).images
            latents.append(image)
            yield i
            
        image_frames += self.decode_latents(latents, height=h, width=w, )
        release_memory()
        if mid_frames:
            image_frames = self.interpolate(image_frames, inter_frames=mid_frames, batch_size=2)
            
            #image_frames = interpolate.image_lerp(image_frames, total_frames=33, t0=0, t1=1, loop_back=False, use_slerp=False)
        yield image_frames
        release_memory()
    

    @torch.inference_mode()
    def _batched_imgs2imgs(self, prompts_encodings:dict[str, torch.Tensor], images:list[Image.Image]|torch.Tensor, batch_size:int, steps:int, strength:float, guidance_scale:float, img_wh:tuple[int, int], seed:int, **kwargs):
        batched_prompts = rebatch_prompt_embs(prompts_encodings, batch_size)
        batched_images = torch.split(images, batch_size) if isinstance(images, torch.Tensor) else batched(images, batch_size)
        w,h = img_wh # need wh in case `images` is a batch of latents
        
        for imbatch in batched_images:
            batch_len = len(imbatch)
            # shared common seed helps SIGNIFICANTLY with cohesion
            # list of generators is very important. Otherwise it does not apply correctly
            generators = generator_batch(seed, batch_len, device='cpu')
                
            if batch_len != batch_size:
               batched_prompts = rebatch_prompt_embs(prompts_encodings, batch_len)

            latents = self.i2i(image=imbatch, num_inference_steps=steps, strength=strength, guidance_scale=guidance_scale, height=h, width=w, **batched_prompts, generator=generators, output_type='latent', **kwargs).images 

            yield latents
    

    
    @torch.inference_mode()
    def regenerate_frames(self, prompt: str, frame_array: np.ndarray, 
                          steps: int = None, 
                          astrength: float = 0.5, 
                          imsize: typing.Literal['tiny','small','med','full'] = 'small', 

                          negative_prompt: str = None, 
                          guidance_scale: float = None, 
                          detail_weight: float = 0,
                          two_stage: bool = False, 
                          aseed: int = None, 
                          **kwargs):   
        
       
        # NOTE: special behavior since having a seed improves results substantially 
        if aseed is None: 
            aseed = np.random.randint(1e9, 1e10-1)
        elif aseed < 0:
            aseed = None
            
        # if self.has_adapters and detail_weight is not None:
        #     self.base.set_adapters(self.base.get_active_adapters(), detail_weight)
        
        # lora_scale = detail_weight if detail_weight and self.has_adapters else None

        fkwg = self.config.get_if_none(steps=steps, strength=astrength, negative_prompt=negative_prompt, guidance_scale=guidance_scale)
        steps = fkwg['steps']
        negative_prompt=fkwg['negative_prompt']
        guidance_scale=fkwg['guidance_scale']

        astrength = np.clip(astrength, 1/steps, 1) # clip strength so at least 1 step occurs
        #astrength = max(astrength*steps, 1)/steps 
        print(f'unused_kwargs: {kwargs} | fkwg:{fkwg}')
                
        dim_choices,bsz = self.batch_settings(imsize)
        resized_images = self._resize_image_frames(frame_array, dim_choices, max_num_frames=100, upsample_px_thresh=256, upsample_bsz=8) # upsample first if less 256^2 pixels 
        dim_out = resized_images[0].size
        w,h = dim_out
        
        lora_scale = self.toggle_loras(detail_weight)
        prompt, negative_prompt = self.preprocess_prompts(prompt, negative_prompt)
        prompts_encodings = self.embed_prompts(prompt, negative_prompt=negative_prompt, batch_size=bsz, lora_scale=lora_scale, clip_skip=self.clip_skip, partial_offload=True)
        
        # First, yield the total number of frames since it may have changed from slice
        yield len(resized_images)

        latents = []
        t0 = time.perf_counter()
        for img_lats in self._batched_imgs2imgs(prompts_encodings, resized_images, batch_size=bsz, steps=steps, strength=astrength, guidance_scale=guidance_scale, img_wh=dim_out, seed=aseed, **kwargs):
            latents.append(img_lats)
            yield len(img_lats)
 
        
        if two_stage:
            latents=torch.cat(latents, 0)
            raw_image_latents, latent_soft_mask = self._prepare_raw_latents(resized_images, aseed)
            latent_blend = self._interpolate_latents(raw_image_latents, latents, latent_soft_mask, dim_out, time_blend=True, keep_dims=True)
            
            latents = []
            for img_lats in self._batched_imgs2imgs(prompts_encodings, latent_blend, batch_size=bsz, steps=steps, strength=0.3, guidance_scale=guidance_scale, img_wh=dim_out, seed=aseed, **kwargs):
                latents.append(img_lats)
                yield len(img_lats)
            
        latents = torch.cat(latents, 0)
        
        image_frames = self.decode_latents(latents, height=h, width=w)
        yield image_frames

        te = time.perf_counter()
        runtime = te-t0
        logger.info(f'RUN TIME: {runtime:0.2f}s | BSZ: {bsz} | DIM: {dim_out} | N_IMAGE: {len(resized_images)} | IMG/SEC: {len(resized_images)/runtime:0.2f}')
        release_memory()
    
    @torch.inference_mode()
    def _prepare_raw_latents(self, resized_images, seed):
        raw_image_latents = self.encode_images(resized_images, seed)
        
        mot_mask_tensor = torch.from_numpy(interpolation.motion_mask(resized_images, px_thresh=0.02, qtile=90))
        #if raw_image_latents.ndim == 4:
        mot_mask_tensor = mot_mask_tensor.expand(1, 1, -1, -1)

        latent_soft_mask = torch.nn.functional.interpolate(
            mot_mask_tensor,
            size=raw_image_latents.shape[-2:], #(h // self.pipei2i.vae_scale_factor, w // self.pipei2i.vae_scale_factor)
            mode='area',#'bilinear',
            ).to(raw_image_latents)
        #print(raw_image_latents.shape, latent_soft_mask.shape)
        return raw_image_latents, latent_soft_mask
    
    @torch.inference_mode()
    def _interpolate_latents(self, raw_image_latents, out_latents, latent_soft_mask, img_wh, time_blend=True, keep_dims=True):
        # flux will override this
        blended_latents = interpolation.blend_latents(raw_image_latents, out_latents, latent_soft_mask, time_blend=time_blend, keep_dims=keep_dims)
        return blended_latents

  
    @abstractmethod
    def encode_images(self, images, seed:int=42, **kwargs):
        raise NotImplementedError('Requires subclass override')
    
    @abstractmethod
    def decode_latents(self, latents, **kwargs):
        raise NotImplementedError('Requires subclass override')
    


class SDXLBase(DeepCacheMixin, SingleStagePipeline, ):
    def __init__(self, model_name: str, model_path: str, config: DiffusionConfig, offload=False, scheduler_setup: str | tuple[str, dict] = None, dtype: torch.dtype = torch.bfloat16):
        super().__init__(model_name, model_path, config, offload, scheduler_setup, dtype, root_name='sdxl')

    def load_pipeline(self):
        _pipe_kwargs = dict(torch_dtype=self.dtype, variant="fp16", use_safetensors=True, add_watermarker=False, )
        if str(self.model_path).endswith('.safetensors'):
            self.base: StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_single_file(self.model_path, **_pipe_kwargs)
        else:
            self.base: StableDiffusionXLPipeline = AutoPipelineForText2Image.from_pretrained(self.model_path, **_pipe_kwargs) 
            #custom_pipeline='lpw_stable_diffusion_xl,'#, device_map=device,
        
        if not self.offload:
            self.base = self.base.to(0)
        
        self.base.vae.enable_slicing()

        self._scheduler_init()
        
        self.load_lora(weight_name='detail-tweaker-xl.safetensors')
        # https://github.com/comfyanonymous/ComfyUI/blob/f58475827150c2ac610cbb113019276efcd1a733/comfy/sd1_clip.py#L234
        # self.compeler = Compel(
        #     tokenizer=[self.base.tokenizer, self.base.tokenizer_2],
        #     text_encoder=[self.base.text_encoder, self.base.text_encoder_2],
        #     truncate_long_prompts=False,
        #     returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
        #     requires_pooled=[False, True]
        # )
        
        self.basei2i: StableDiffusionXLImg2ImgPipeline = AutoPipelineForImage2Image.from_pipe(self.base)

        if self.offload:
            # calling on both pipes seems to make offload more consistent, may not be inherited properly with from_pipe 
            # https://huggingface.co/docs/diffusers/using-diffusers/loading?pipelines=specific+pipeline#:~:text=Some%20pipeline%20methods
            self.base.enable_model_cpu_offload()
            self.basei2i.enable_model_cpu_offload()
        
        self.is_ready = True

    
    def set_pag_config(self, pag_scale:float=3.0, pag_applied_layers: str|list[str] = "mid", disable_on_zero:bool=False):
        # https://huggingface.co/docs/diffusers/main/en/using-diffusers/pag?tasks=Text-to-image
        # https://huggingface.co/docs/diffusers/main/en/api/pipelines/pag#perturbed-attention-guidance
        if isinstance(pag_applied_layers, str):
            pag_applied_layers = [pag_applied_layers]
        
        pag_config = {'enabled':False, 'scale':None, 'layers':None}
        if pag_scale:
            if 'page_scale' not in self.pipe_xkwgs:
                pag_kwargs = {'enable_pag':True, 'pag_applied_layers':pag_applied_layers,}
                self.base: StableDiffusionXLPAGPipeline = AutoPipelineForText2Image.from_pipe(self.base, **pag_kwargs,)
                # self.basei2i: StableDiffusionXLPAGImg2ImgPipelinePatch = AutoPipelineForImage2Image.from_pipe(self.basei2i, **pag_kwargs,)
                self.basei2i: StableDiffusionXLPAGImg2ImgPipeline = StableDiffusionXLPAGImg2ImgPipeline.from_pipe(self.basei2i, pag_applied_layers=pag_applied_layers,)
                if self.offload:
                    self.base.enable_model_cpu_offload()
                    self.basei2i.enable_model_cpu_offload()
            
            if self.base.pag_applied_layers != pag_applied_layers:
                self.base.set_pag_applied_layers(pag_applied_layers)
                self.basei2i.set_pag_applied_layers(pag_applied_layers)

            self.pipe_xkwgs.update(pag_scale=pag_scale)
            pag_config.update(enabled=True, scale=pag_scale, layers=pag_applied_layers)

        elif disable_on_zero:
            self.pipe_xkwgs.pop('pag_scale', None)
            self.base = AutoPipelineForText2Image.from_pipe(self.base, enable_pag=False)
            self.basei2i = AutoPipelineForImage2Image.from_pipe(self.basei2i, enable_pag=False,)
            if self.offload:
                self.base.enable_model_cpu_offload()
                self.basei2i.enable_model_cpu_offload()
        else:
            self.pipe_xkwgs.update(pag_scale=0.0)
            pag_config.update(enabled=False, scale=0.0, layers=pag_applied_layers)
        
        print(self.base.__class__,self.base.__class__.__name__)
        print(self.basei2i.__class__, self.basei2i.__class__.__name__)
        return pag_config

        

    @torch.inference_mode()
    def load_lora(self, weight_name:str, init_weight:float=None):        
        adapter_name = weight_name.rsplit('.', maxsplit=1)[0].replace('-','_')
        
        try:
            self.base.load_lora_weights(self.lora_dirpath, weight_name=weight_name, adapter_name=adapter_name)
            self.adapter_names.append(adapter_name)
            if init_weight is None:
                self.base.disable_lora()
            else:
                self.base.set_adapters(adapter_name, init_weight)
        except IOError as e:
            raise NotImplementedError(f'Unsupported lora adapter {adapter_name!r}')
            

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
        if negative_prompt is None:
            negative_prompt = ""

        prompt_encodings = {}
        #compel_tokens = set('()-+')

        # if set(prompt) & compel_tokens:
        #     conditioning, pooled = self.compeler(prompt)
        #     prompt_encodings.update(prompt_embeds=conditioning, pooled_prompt_embeds=pooled)
        #     prompt = None

        # if negative_prompt is not None and set(negative_prompt) & compel_tokens:
        #     neg_conditioning, neg_pooled = self.compeler(negative_prompt)
        #     prompt_encodings.update(negative_prompt_embeds=neg_conditioning, negative_pooled_prompt_embeds=neg_pooled)
        #     negative_prompt = None
        
        # https://github.com/huggingface/diffusers/blob/main/examples/community/lpw_stable_diffusion_xl.py
        # https://github.com/huggingface/diffusers/issues/2136#issuecomment-1514338525
        (p_emb, neg_p_emb, pooled_p_emb, neg_pooled_p_emb) = get_weighted_text_embeddings_sdxl( # self.base.encode_prompt(
            self.base, prompt=prompt, neg_prompt=negative_prompt, device=0, **prompt_encodings, num_images_per_prompt=batch_size,
             lora_scale=kwargs.get('lora_scale'), clip_skip=kwargs.get('clip_skip'))
        
        prompt_encodings.update(
            prompt_embeds=p_emb, negative_prompt_embeds=neg_p_emb, 
            pooled_prompt_embeds=pooled_p_emb, negative_pooled_prompt_embeds=neg_pooled_p_emb
        )
        torch.cuda.empty_cache()    
        return prompt_encodings
    
    @torch.inference_mode()
    def encode_images(self, images:list[Image.Image], seed:int=42):
        nframes = len(images)
        w,h = images[0].size
        gseed = [torch.Generator('cpu').manual_seed(seed) for _ in range(nframes)]
        proc_images = self.basei2i.image_processor.preprocess(images, height=h, width=w)
        latents = self.basei2i.prepare_latents(proc_images, timestep=None, batch_size=1, num_images_per_prompt=nframes, dtype=torch.bfloat16, device=0, generator=gseed, add_noise=False)
        release_memory()
        return latents
    
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
    def __init__(self, model_name: str, model_path: str, config: DiffusionConfig, offload=False, scheduler_setup: str | tuple[str, dict] = None, dtype: torch.dtype = torch.bfloat16, 
                 quantize:bool=True):
        super().__init__(model_name, model_path, config, offload, scheduler_setup, dtype, root_name='sd35')
        self.max_seq_len = 512
        self.quantize = quantize

        
    
    @torch.inference_mode()
    def load_pipeline(self):
        if self.model_path.endswith('.safetensors'):
            self.base = StableDiffusion3Pipeline.from_single_file(self.model_path, torch_dtype=self.dtype,  use_safetensors=True,)
            # SD3Transformer2DModel.from_single_file
        else:
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16) if self.quantize else None
            transformer = SD3Transformer2DModel.from_pretrained(self.model_path, subfolder="transformer", quantization_config = quant_config, torch_dtype=self.dtype)
            
            self.base: StableDiffusion3Pipeline =  StableDiffusion3Pipeline.from_pretrained(
                self.model_path, transformer=transformer, torch_dtype=self.dtype, use_safetensors=True,
            )
        
        # self.load_lora('sd35Fusion_8Steps.safetensors', 1.0)
        device = torch.device('cuda')
        if not self.offload:
            self.base = self.base.to(device)
            self.base.text_encoder_3 = self.base.text_encoder_3.to(device, self.dtype)
            #if self.quantize:
            self.base.text_encoder_3 = cpu_offload(self.base.text_encoder_3, execution_device = device)

        self.base.vae.enable_slicing()

        self._scheduler_init()
        

        self.basei2i: StableDiffusion3Img2ImgPipeline = StableDiffusion3Img2ImgPipeline.from_pipe(self.base,)

        
        if self.offload:
            self.base.enable_model_cpu_offload()
            self.basei2i.enable_model_cpu_offload()
        
        self.is_ready = True
    
    @torch.inference_mode()
    def load_lora(self, weight_name:str, init_weight:float=None):
        
        adapter_name = weight_name.rsplit('.', maxsplit=1)[0].replace('-','_')
        
        try:
            #loraops.set_lora_transformer(self.base, Path(lora_dirpath).joinpath(weight_name), adapter_name)
            #lora_sd = loraops.get_lora_state_dict(Path(lora_dirpath).joinpath(weight_name))
            lora_sd = Path(self.lora_dirpath).joinpath(weight_name)
            #self.base.transformer,skipped_keys = loraops.manual_lora(lora_sd, self.base.transformer, adapter_name=adapter_name)
            self.base.load_lora_weights(lora_sd, adapter_name=adapter_name)
            #self.base.load_lora_into_transformer(self.base.lora_state_dict(lora_sd), transformer=self.base.transformer, adapter_name=adapter_name, low_cpu_mem_usage=True)
            #self.base.load_lora_into_transformer(lora_sd, None, self.base.transformer, adapter_name=adapter_name)
            #self.base.load_lora_weights(lora_sd, adapter_name=adapter_name)
            if init_weight is None:
                self.base.disable_lora()
            else:
                self.base.set_adapters(adapter_name, init_weight)
            #self.base.load_lora_weights(lora_dirpath, weight_name=weight_name, adapter_name=adapter_name)
            self.adapter_names.append(adapter_name)
            
        except IOError as e:
            raise NotImplementedError(f'Unsupported lora adapter {weight_name!r}')
        
    @torch.inference_mode()
    def embed_prompts(self, prompt:str, negative_prompt:str = None, batch_size: int = 1, **kwargs):
        prompt = [prompt]
        if negative_prompt is not None:
            negative_prompt = [negative_prompt]


        device = torch.device('cuda')
        (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds) = self.base.encode_prompt(
            prompt=prompt, prompt_2=None, prompt_3=None, negative_prompt=negative_prompt, num_images_per_prompt=batch_size,
            device=device, lora_scale=kwargs.get('lora_scale'), clip_skip=kwargs.get('clip_skip'), max_sequence_length=self.max_seq_len)
        
        torch.cuda.empty_cache()
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
            _ = self.t2i("a photo of a cat holding a sign that says hello world")#, num_inference_steps=4, guidance_scale=self.config.guidance_scale)
        self.is_compiled = True
        torch.compiler.reset()
    
    @torch.inference_mode()
    def encode_images(self, images:list[Image.Image], seed:int=42):
        nframes = len(images)
        w,h = images[0].size
        
        gseed = generator_batch(seed, nframes, 'cpu') #[torch.Generator('cpu').manual_seed(seed) for _ in range(nframes)]
        proc_images = self.basei2i.image_processor.preprocess(images, height=h, width=w).to(dtype=torch.float32)
        
        #latents = self.basei2i._encode_vae_image(image=proc_images, generator=gseed)
        #img_encodings = self.basei2i.vae.encode(proc_images)
        # NOTE: this might be wrong
        latents = [
            retrieve_latents(self.basei2i.vae.encode(proc_images), generator=gseed[0])# for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0)

        latents = (latents - self.basei2i.vae.config.shift_factor) * self.basei2i.vae.config.scaling_factor
        #latents = self.basei2i.prepare_latents(proc_images, timestep=None, batch_size=1, num_images_per_prompt=nframes, dtype=torch.bfloat16, device=0, generator=gseed, add_noise=False)
        release_memory()
        return latents
    
    @torch.inference_mode()
    def decode_latents(self, latents, **kwargs):
        # https://github.com/huggingface/diffusers/blob/f63c12633f154c2a1d79c17f4238fb073133652c/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py#L924
        if isinstance(latents, list):
            latents = torch.cat(latents, dim=0)
        
        latents = (latents / self.basei2i.vae.config.scaling_factor) + self.basei2i.vae.config.shift_factor
        image = self.basei2i.vae.decode(latents, return_dict=False)[0]
        image = self.basei2i.image_processor.postprocess(image, output_type='pil')
        
        return image


class FluxBase(SingleStagePipeline):
    def __init__(self, model_name: str, model_path: str, config: DiffusionConfig, offload=False, scheduler_setup: str | tuple[str, dict] = None, dtype: torch.dtype = torch.bfloat16,
                 qtype:typing.Literal['bnb4','qint4','qint8','qfloat8'] = 'bnb4', te2_qtype:typing.Literal['bnb4','bnb8','bf16', 'qint4','qint8','qfloat8'] = 'bf16', quant_basedir:Path = None):
        
        super().__init__(model_name, model_path, config, offload, scheduler_setup, dtype, root_name='flux')
        
        self.text_enc_model_id = "black-forest-labs/FLUX.1-dev" # use same shared text_encoder_2, avoid 2x-download
        self.qtype = qtype
        self.te2_qtype = te2_qtype
        self.quant_basedir = quant_basedir if quant_basedir is not None else cpaths.ROOT_DIR / 'extras/quantized/flux/'
        
        self.variant = model_name.split('_')[-1] # flux_schell,flux_dev,flux_* -> *
        self.max_seq_len = 512
        self.text_pipe:FluxPipeline = None
    
    def i2i(self, *args, **kwargs):
        # because latents shape cannot be reliably inferred, height and width are required for flux img2img
        return self.basei2i(*args, **kwargs, **self.pipe_xkwgs)
    
    def unload_pipeline(self):
        components = ['text_encoder','text_encoder_2','transformer','vae']
        if self.is_ready:
            #self.base.to('cpu')
            for comp in components:
                #getattr(self.base, comp).to('cpu')
                setattr(self.base, comp, None)
                setattr(self.basei2i, comp, None)
            
        self.text_pipe = None
        super().unload_pipeline()
    
    @torch.inference_mode()
    def load_pipeline(self):
        bnb_qconfig = BnbQuantizationConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=False, bnb_4bit_compute_dtype='bf16', torch_dtype=self.dtype, )

        if self.model_name == 'flux_hyper':
            transformer = quantops.flux_hyper_transformer_4bit(self.model_path, scale=0.125,dtype=self.dtype, bnb_qconfig=bnb_qconfig).to(0, dtype=self.dtype)
            self.base: FluxPipeline = FluxPipeline.from_pretrained(self.model_path, transformer=transformer, torch_dtype=self.dtype,)
            
        elif self.model_name == 'flux_pixelwave':
            
            
            transformer = FluxTransformer2DModel.from_pretrained(self.model_path, quantization_config=bnb_qconfig, torch_dtype=self.dtype, low_cpu_mem_usage = True)
            self.base = FluxPipeline.from_pretrained(self.text_enc_model_id, transformer=transformer, torch_dtype=self.dtype,)

        elif self.qtype == 'bnb4' and self.te2_qtype == 'bf16':
            if str(self.model_path).endswith('.safetensors'):
                transformer = quantops.load_prequantized_transformer(safetensor_filepath=self.model_path)
            else:
                transformer = quantops.bnb4_transformer(self.model_path, self.dtype, bnb_qconfig)
        else:
            qloader = quantops.FluxLoadHelper(self.model_path, self.quant_basedir, self.text_enc_model_id, self.variant)
            self.base: FluxPipeline = qloader.load_quant(qtype=self.qtype, offload=self.offload, te2_qtype=self.te2_qtype)
            
        if not self.offload: 
            self.base = self.base.to(0)
            # if not offloading everything, just offload text_encoder_2
            self.base.text_encoder_2 = cpu_offload(self.base.text_encoder_2, execution_device = torch.device('cuda:0'))
        
        self.base.vae.enable_slicing() # all this does is iterate over batch instead of all at once, no reason to ever disable
        
        self._scheduler_init()
        
        #self.load_lora('detail-maximizer_v02.safetensors')
        #self.load_lora('midjourneyV61_v02.safetensors')
        
        self.basei2i: FluxImg2ImgPipeline = FluxImg2ImgPipeline.from_pipe(self.base)

        if self.offload:
            # calling on both pipes seems to make offload more consistent, may not be inherited properly with from_pipe 
            self.base.enable_model_cpu_offload()
            self.basei2i.enable_model_cpu_offload()
        
        self.is_ready = True
        release_memory()
    

    @torch.inference_mode()
    def load_lora(self, weight_name:str, init_weight:float=None):
        
        adapter_name = weight_name.rsplit('.', maxsplit=1)[0].replace('-','_')
        # https://civitai.green/user/nakif0968/models?types=LORA&baseModels=Flux.1+D&baseModels=Flux.1+S
        try:
            #loraops.set_lora_transformer(self.base, Path(lora_dirpath).joinpath(weight_name), adapter_name)
            #lora_sd = loraops.get_lora_state_dict(Path(lora_dirpath).joinpath(weight_name))
            lora_sd = Path(self.lora_dirpath).joinpath(weight_name)
            #self.base.transformer,skipped_keys = loraops.manual_lora(lora_sd, self.base.transformer, adapter_name=adapter_name)
            #self.base.load_lora_weights(lora_sd, adapter_name=adapter_name)
            self.base.load_lora_into_transformer(*self.base.lora_state_dict(lora_sd, return_alphas=True), transformer=self.base.transformer, adapter_name=adapter_name, low_cpu_mem_usage=True)
            #self.base.load_lora_into_transformer(lora_sd, None, self.base.transformer, adapter_name=adapter_name)
            #self.base.load_lora_weights(lora_sd, adapter_name=adapter_name)
            if init_weight is None:
                self.base.disable_lora()
            else:
                self.base.set_adapters(adapter_name, init_weight)
            #self.base.load_lora_weights(lora_dirpath, weight_name=weight_name, adapter_name=adapter_name)
            self.adapter_names.append(adapter_name)
            
        except IOError as e:
            raise NotImplementedError(f'Unsupported lora adapter {weight_name!r}')

    def batch_settings(self, imsize: typing.Literal['tiny','small','med','full'] = 'small', ):
        dims_opts = {
            'tiny': [(512,512), (512,640), (640,512)], # 1.25
            'small': [(640,640), (512,768), (768,512)], # 1.5
            'med': [(768,768), (640,896), (896,640)], # 1.4
            'full': [(1024,1024), (832, 1216), (1216, 832)] # 1.46
        }
        
        # batch_sizes = {'tiny': 4, 'small': 4, 'med': 3, 'full': 2} # With out vae_slicing
        batch_sizes = {'tiny': 8, 'small': 8, 'med': 6, 'full': 4} # With out vae_slicing
        #batch_sizes = {'tiny': 24, 'small': 16, 'med': 12, 'full': 4} 

        return (dims_opts[imsize], batch_sizes[imsize])
    
    @torch.inference_mode()
    def embed_prompts(self, prompt:str, negative_prompt:str = None, batch_size: int = 1, **kwargs):
        prompt = [prompt]
        # lora weights for Flux should ONLY be loaded into transformer, not text_encoder
        lora_scale = kwargs.get('lora_scale', None) # None
        device = torch.device('cuda:0')
        if self.text_pipe is None:
            (prompt_embeds, pooled_prompt_embeds,_,) = self.base.encode_prompt(
                prompt=prompt, prompt_2=None, max_sequence_length=self.max_seq_len, num_images_per_prompt=batch_size, 
                device=device, lora_scale=lora_scale)
        else:
            (prompt_embeds, pooled_prompt_embeds,_,) = self.text_pipe.encode_prompt(
                prompt=prompt, prompt_2=None, max_sequence_length=self.max_seq_len, num_images_per_prompt=batch_size,
                  device=self.text_pipe.device, lora_scale=lora_scale)
        
        torch.cuda.empty_cache()
        return dict(prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds) # neg not supported (offically)
    
    
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
            _ = self.t2i("a photo of a cat holding a sign that says hello world")#, num_inference_steps=4, guidance_scale=self.config.guidance_scale)
        self.is_compiled = True
        torch.compiler.reset()
    
    @torch.inference_mode()
    def encode_images(self, images:list[Image.Image], seed:int=42):
        nframes = len(images)
        w,h = images[0].size
        generators = generator_batch(seed, nframes, device='cpu')
        proc_images = self.basei2i.image_processor.preprocess(images, height=h, width=w).to(self.basei2i.device, self.basei2i.dtype)
        latents = self.basei2i._encode_vae_image(image=proc_images, generator=generators)
        
        release_memory()
        return latents
    
    @torch.inference_mode()
    def _repack_latents(self, latents, img_h, img_w): # ex: ([NFRAME, 16, 80, 80]) -> ([NFRAME, 1600, 64]) 
        num_channels_latents = self.basei2i.transformer.config.in_channels // 4
        height = 2 * (int(img_h) // self.basei2i.vae_scale_factor)
        width = 2 * (int(img_w) // self.basei2i.vae_scale_factor)
        latents = self.basei2i._pack_latents(latents, latents.shape[0], num_channels_latents, height, width)
        return latents
            
    @torch.inference_mode()
    def _interpolate_latents(self, raw_image_latents, out_latents, latent_soft_mask, img_wh, time_blend=True, keep_dims=True):
        img_w,img_h = img_wh
        if out_latents.ndim == 3 and out_latents.shape[-1] == 64:
            out_latents = self.basei2i._unpack_latents(out_latents, img_h, img_w, self.basei2i.vae_scale_factor)
        blended_latents = interpolation.blend_latents(raw_image_latents, out_latents, latent_soft_mask, time_blend=time_blend, keep_dims=keep_dims)
        packed_blended_latents = self._repack_latents(blended_latents, img_h, img_w)
        return packed_blended_latents
    
    @torch.inference_mode()
    def decode_latents(self, latents, height:int, width:int):
        # https://github.com/huggingface/diffusers/blob/4cfb2164fb05d54dd594373b4bd1fbb101fef70c/src/diffusers/pipelines/flux/pipeline_flux.py#L759
        if isinstance(latents, list):
            latents = torch.cat(latents, dim=0)
        #print(latents.shape)
        
        # make sure the VAE is in float32 mode, as it overflows in float16
        if latents.ndim == 3 and latents.shape[-1] == 64:
            latents = self.base._unpack_latents(latents, height, width, self.base.vae_scale_factor)
        latents = (latents / self.base.vae.config.scaling_factor) + self.base.vae.config.shift_factor
        image = self.base.vae.decode(latents, return_dict=False)[0]

        image = self.base.image_processor.postprocess(image, output_type='pil')
        
        return image