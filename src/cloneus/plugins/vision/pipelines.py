import os
import gc
import re
import io
import math
import time
import random
import typing
import inspect
import logging
import warnings
import itertools
import functools
import contextlib
from pathlib import Path
from abc import abstractmethod
from dataclasses import dataclass, field, asdict, KW_ONLY, InitVar


from tqdm.auto import tqdm
import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)
from unsloth import FastModel, FastLanguageModel

from diffusers import (
    AutoPipelineForText2Image, 
    AutoPipelineForImage2Image, 
    DiffusionPipeline, 
    StableDiffusionXLPipeline, 
    StableDiffusionXLImg2ImgPipeline, 
    StableDiffusionXLPAGPipeline,
    StableDiffusionXLPAGImg2ImgPipeline,
    UNet2DConditionModel, 
    DPMSolverMultistepScheduler, 
    DPMSolverSinglestepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler,
    FluxPipeline, 
    FluxTransformer2DModel,
    FluxImg2ImgPipeline,
    QwenImagePipeline, 
    QwenImageImg2ImgPipeline, 
    QwenImageTransformer2DModel, 
    QwenImageEditPlusPipeline,
    ZImageTransformer2DModel,
    ZImagePipeline,
    ZImageImg2ImgPipeline
)

from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.quantizers.quantization_config import GGUFQuantizationConfig, QuantoConfig, TorchAoConfig, BitsAndBytesConfig as DiffuBitsAndBytesConfig
from diffusers.schedulers import AysSchedules
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img import retrieve_latents

from diffusers.hooks import apply_group_offloading
from huggingface_hub import hf_hub_download, snapshot_download

from nunchaku import NunchakuFluxTransformer2dModel, NunchakuT5EncoderModel, NunchakuQwenImageTransformer2DModel, NunchakuZImageTransformer2DModel
from nunchaku.lora.flux.compose import compose_lora
from nunchaku.utils import get_precision

from transformers.image_processing_utils import select_best_resolution
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3ForCausalLM, T5Model, T5EncoderModel, T5PreTrainedModel, BitsAndBytesConfig as TransBitsAndBytesConfig

from optimum import quanto
from accelerate import cpu_offload,init_empty_weights
from accelerate.utils import release_memory
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

from transformers import AutoConfig, AutoModelForTextEncoding
import safetensors.torch as sft

# import cache_dit
from PIL import Image
from spandrel import ImageModelDescriptor, ModelLoader
# from compel import Compel, ReturnedEmbeddingsType
from compel import CompelForSDXL, CompelForFlux # alternative: https://github.com/xhinker/sd_embed

from . import specialists,interpolation

from cloneus.utils.common import batched # release_memory

logger = logging.getLogger(__name__)
SDXL_DIMS = [(1024,1024), (1152, 896),(896, 1152), (1216, 832),(832, 1216), (1344, 768),(768, 1344), (1536, 640),(640, 1536),] # https://stablediffusionxl.com/sdxl-resolutions-and-aspect-ratios/
# other: [(1280, 768),(768, 1280),]
# Qwen-Image
_aspect_ratios_qi = { # (width, height)
    "1:1": (1328, 1328), # 1.76m
    "16:9": (1664, 928), # 1.54m
    "9:16": (928, 1664), # 1.54m
    "4:3": (1472, 1140), # 1.68m
    "3:4": (1140, 1472), # 1.68m
    "3:2": (1584, 1056), # 1.67m
    "2:3": (1056, 1584), # 1.67m
}
# multiple-64, ar_exact (other than 9;16)  
_aspect_ratios_64 = { # (width, height)
    "1:1": (1152, 1152), # 1.33m
    "16:9": (1472, 832), # 1.23m
    "9:16": (832, 1472), # 1.23m
    "4:3": (1280, 960),  # 1.23m
    "3:4": (960, 1280),  # 1.23m
    "3:2": (1344, 896),  # 1.20m
    "2:3": (896, 1344),  # 1.20m
}
# https://huggingface.co/docs/diffusers/en/quantization/gguf#using-optimized-cuda-kernels-with-gguf
# os.environ.update({'DIFFUSERS_GGUF_CUDA_KERNELS':'true'})
# - No support for torch > 2.7: https://huggingface.co/Isotr0py/ggml/tree/main/build
# - https://github.com/Isotr0py/ggml-libtorch/tree/main/hf-kernels/ggml-kernels
warnings.filterwarnings(action="ignore", category=FutureWarning, module='diffusers.*', message='.*Passing `image` as torch tensor.*') # Z-Img turbo prints on every iter 

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

def generator_batch(seed: int|list[int]|None, num_seeds:int, device='cpu', all_unique:bool=False):
    if seed is None:
        return None
    
    if isinstance(seed, int):
        seeds = [seed+i for i in range(num_seeds)] if all_unique else [seed]*num_seeds
    else:
        seeds = seed
        # given the exact same params, you can use 1 generator to reproduce. 
        # But if you change how you iterate over inputs, outputs with differ. This controls for that.
        if (unqlen := len(set(seeds))) != num_seeds:
            raise ValueError(f'If `seed` is a list of ints, it must have `num_seeds` ({num_seeds}) unique elements but only ({unqlen}) unique seeds passed')

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

#region Base Class 
class SingleStagePipeline:
    def __init__(self, 
                 model_name: str, 
                 model_path: str, 
                 config: DiffusionConfig, 
                 offload: bool = False, 
                 scheduler_setup: str|tuple[str, dict] = None, 
                 dtype = torch.bfloat16, 
                 init_loras: list[tuple[str,float]] = None, 
                 root_name: str = 'base',):
        
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
        
        self.init_loras = init_loras
        self.adapter_weights = {}

        self.root_name = root_name

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
        return self.basei2i(*args, **kwargs, **self.pipe_xkwgs)
    
    def pbar_config(self, **kwargs):
        for pipe in (self.base, self.basei2i):
            if pipe is not None:
                pipe.set_progress_bar_config(**kwargs)

    @contextlib.contextmanager
    def offload_disabled(self, components: tuple[str, ...] = ('vae', 'transformer', 'unet')):
        if isinstance(components, str):
            components = (components, )
        try:
            on_device = torch.device("cuda:0")
            if self.offload:
                for pipe in (self.base, self.basei2i):
                    pipe.maybe_free_model_hooks()
                    pipe.remove_all_hooks()
                    for comp_name in components:
                        if (component := getattr(pipe, comp_name, None)):
                            component.to(on_device, ) # non_blocking=True)
                    
                    # patch over the class' device property -- https://stackoverflow.com/a/31591589
                    class _FixedDevicePipeline(pipe.__class__): device=on_device
                    pipe.__class__ = _FixedDevicePipeline
                    # pipe.__class__ = type('_FixedDevicePipeline', (pipe.__class__, ), dict(device = on_device)) # -- one-liner, but less obvious
            yield
        finally:
            if self.offload:
                for pipe in (self.base, self.basei2i):
                    if pipe.__class__.__name__ == '_FixedDevicePipeline':
                        pipe.__class__ = pipe.__class__.__base__ # restore original class
                    pipe.remove_all_hooks()
                    pipe.reset_device_map()
                    pipe.enable_model_cpu_offload()
    
    @abstractmethod
    def load_pipeline(self):
        raise NotImplementedError('Requires subclass override')
    
    def unload_pipeline(self):
        if self.base:
            for comp in self.base.components: 
                setattr(self.base, comp, None)
        if self.basei2i:
            for comp in self.basei2i.components: 
                setattr(self.basei2i, comp, None)
        (
            self.base, self.basei2i, 
            self.compeler, self.upsampler, self.florence, self.interpolator, self.vqa, 
            self.initial_scheduler_config, self.scheduler_kwargs
        ) = release_memory(
             self.base, self.basei2i, 
             self.compeler, self.upsampler, self.florence, self.interpolator, self.vqa, 
             self.initial_scheduler_config, self.scheduler_kwargs
        )
        # The ugliest possible way to set a bunch of things to None

        self.is_ready = False
        if self.is_compiled:
            torch_compile_flags(restore_defaults=True)
            torch._dynamo.reset()
            self.is_compiled = False
    
    def set_pag_config(self, pag_scale:float=0.0, pag_applied_layers: str|list[str] = None, disable_on_zero:bool=False):
        raise NotImplementedError('PAG not implemented for this pipeline type')
    
    @torch.inference_mode()
    def load_loras(self, path_weights: list[tuple[str,float]] = None):
        if path_weights is None:
            path_weights = self.init_loras
        
        if not path_weights:
            return
        
        for lora_path,lora_weight in path_weights:
            if isinstance(lora_path, Path):
                lora_path = lora_path.as_posix()
            
            adapter_name = lora_path.split('/')[-1].rsplit('.', maxsplit=1)[0].replace('-','_')

            try:
                self.base.load_lora_weights(lora_path, adapter_name=adapter_name)
                self.adapter_weights[adapter_name] = lora_weight
            except IOError as e:
                raise NotImplementedError(f'Unsupported lora adapter {adapter_name!r}')
        
        if sum(self.adapter_weights.values()) > 0:
            self.base.set_adapters(self.adapter_weights.keys(), self.adapter_weights.values())
        
    @torch.inference_mode()
    def toggle_loras(self):
        if self.base.get_active_adapters():
            self.base.disable_lora()
        elif self.adapter_weights:
            self.base.enable_lora()

        logger.debug(f'Active Adapters: {self.base.get_active_adapters()}\nAll Adapters: {self.base.get_list_adapters()}' )
              
    @torch.inference_mode()
    def set_adapter_weight(self, adapter_name:str, weight: float):            
        try:
            self.adapter_weights[adapter_name] = weight
        except KeyError:
            raise KeyError(f'Unknown adapter_name: {adapter_name!r}')
        
        self.base.set_adapters(self.adapter_weights.keys(), self.adapter_weights.values())

        logger.debug(f'Active Adapters: {self.base.get_active_adapters()}\nAdapter Weights: {self.adapter_weights}')


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
    
    
    @abstractmethod
    def batch_settings(self, imsize: typing.Literal['tiny','small','med','full'] = 'small', ) -> tuple[list[tuple[int, int]], int]:
        raise NotImplementedError('Requires subclass override')

    def preprocess_prompts(self, prompt:str, negative_prompt:str = None):
        if negative_prompt is None:
            negative_prompt = ''
        return prompt, negative_prompt

    @abstractmethod  
    def embed_prompts(self, prompt:str, negative_prompt:str = None, batch_size: int = 1, **kwargs):
        raise NotImplementedError('Requires subclass override')

    @abstractmethod
    def encode_images(self, images, seed:int=42, **kwargs):
        raise NotImplementedError('Requires subclass override')
    
    @abstractmethod
    def decode_latents(self, latents, **kwargs):
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
    def _pipe_txt2img(self, prompt_encodings:dict, num_inference_steps:int, guidance_scale:float, target_size:tuple[int,int], seed=None):
        gseed = torch.Generator(device='cuda').manual_seed(seed) if seed is not None else None
        h,w = target_size
        image = self.t2i(num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, height=h, width=w, **prompt_encodings, generator=gseed).images
        
        return image[0]

    @torch.inference_mode()
    def _pipe_img2img(self, prompt_encodings:dict, image:Image.Image, num_inference_steps:int, strength:float, guidance_scale:float, seed=None):
        gseed = torch.Generator(device='cuda').manual_seed(seed) if seed is not None else None
        w,h = image[0].size if isinstance(image, list) else image.size
        #num_inference_steps = calc_esteps(num_inference_steps, strength, min_effective_steps=1)
        strength = np.clip(strength, 1/num_inference_steps, 1)
        image = self.i2i(image=image, num_inference_steps=num_inference_steps, strength=strength, guidance_scale=guidance_scale, height=h, width=w, **prompt_encodings, generator=gseed).images
        
        return image[0]

    @torch.inference_mode()
    def _pipe_img2upimg(self, prompt_encodings:dict, image:Image.Image, num_inference_steps:int, refine_strength:float, guidance_scale:float, seed=None, scale=1.5):
        image = self.upsample(image, scale=scale)
        logger.debug(f'upsized (w,h): {image[0].size if isinstance(image, list) else image.size}')
        torch.cuda.empty_cache()

        return self._pipe_img2img(prompt_encodings, image, num_inference_steps, strength=refine_strength, guidance_scale=guidance_scale, seed=seed)
    
    @torch.inference_mode()
    def _batched_txt2img(self, prompt_encodings:dict[str, torch.Tensor], num_images:int, batch_size:int, num_inference_steps:int, guidance_scale:float, target_size:tuple[int, int], seed:list[int]|None, output_type:str='pil'):
        generators = generator_batch(seed, num_images, device='cpu', all_unique=True) # if seed is set, want a reproducible set of n different images, 
        if generators is None:
            generators = [None]*num_images
        h,w = target_size

        batch_size = min(num_images, batch_size)
        batched_prompts = rebatch_prompt_embs(prompt_encodings, batch_size)


        for gen_batch in batched(generators, batch_size):
            if (batch_len := len(gen_batch)) != batch_size:
               batched_prompts = rebatch_prompt_embs(prompt_encodings, batch_len)
            
            if gen_batch[0] is None:
                gen_batch = None
            
            images = self.t2i(num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, height=h, width=w, **batched_prompts, generator=gen_batch, output_type=output_type).images
            
            if output_type != 'latent':
                torch.cuda.empty_cache()
            
            yield images

    @torch.inference_mode()
    def _batched_img2img(self, prompt_encodings:dict[str, torch.Tensor], image:Image.Image, num_images:int, batch_size:int, num_inference_steps:int, strength:float, guidance_scale:float, seed:list[int]|None, output_type:str='pil'):
        generators = generator_batch(seed, num_images, device='cpu', all_unique=True) # if seed is set, want a reproducible set of n different images, 
        if generators is None:
            generators = [None]*num_images
        w,h = image[0].size if isinstance(image, list) else image.size

        batch_size = min(num_images, batch_size)
        batched_prompts = rebatch_prompt_embs(prompt_encodings, batch_size)
        
        # n_full_batches,final_batch_size = divmod(num_images, batch_size)
        # batch_lengths = [batch_size]*n_full_batches
        # if final_batch_size:
        #     batch_lengths.append(final_batch_size)
        
        # gen_batch = generators
        for gen_batch in batched(generators, batch_size):
            if (batch_len := len(gen_batch)) != batch_size:
                batched_prompts = rebatch_prompt_embs(prompt_encodings, batch_len)
            if gen_batch[0] is None:
                gen_batch = None
            #if generators is not None:
            #    gen_batch, generators = generators[:batch_len], generators[batch_len:] # pop slice
            
            images = self.i2i(image=image, num_inference_steps=num_inference_steps, strength=strength, guidance_scale=guidance_scale, height=h, width=w, **batched_prompts, generator=gen_batch, output_type=output_type).images 
            
            if output_type != 'latent':
                torch.cuda.empty_cache()

            yield images

    @torch.inference_mode()
    def _batched_imgs2imgs(self, prompt_encodings:dict[str, torch.Tensor], images:list[Image.Image]|torch.Tensor, batch_size:int, steps:int, strength:float, guidance_scale:float, img_wh:tuple[int, int], seed:int, **kwargs):
        batched_prompts = rebatch_prompt_embs(prompt_encodings, batch_size)
        batched_images = torch.split(images, batch_size) if isinstance(images, torch.Tensor) else batched(images, batch_size)
        w,h = img_wh # need wh in case `images` is a batch of latents
        
        for imbatch in batched_images:
            batch_len = len(imbatch)
            # shared common seed helps SIGNIFICANTLY with cohesion
            # list of generators is very important. Otherwise it does not apply correctly
            generators = generator_batch(seed, batch_len, device='cpu')
                
            if batch_len != batch_size:
               batched_prompts = rebatch_prompt_embs(prompt_encodings, batch_len)

            latents = self.i2i(image=imbatch, num_inference_steps=steps, strength=strength, guidance_scale=guidance_scale, height=h, width=w, **batched_prompts, generator=generators, output_type='latent', **kwargs).images 

            yield latents

    def _resize_image(self, images: list[Image.Image], aspect: typing.Literal['square', 'portrait', 'landscape'] | None = None, dim_choices = None) -> list[Image.Image]:
        if not isinstance(images, list):
            images = [images]

        aspect_dims = self.config.get_dims(aspect) if aspect is not None else None
        
        resized_images = []
        for image in images:
            dim_out = self.config.nearest_dims(image.size, dim_choices=dim_choices, use_hf_sbr=False) if aspect_dims is None else aspect_dims
            resized_images.append(image.resize(dim_out, resample=Image.Resampling.LANCZOS))

            logger.debug(f'_resize_image input size: {image.size} -> {dim_out}')
        
        return resized_images
    
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
        

        
        prompt, negative_prompt = self.preprocess_prompts(prompt, negative_prompt)
        prompt_encodings = self.embed_prompts(prompt, negative_prompt=negative_prompt)

        call_kwargs = dict(prompt=prompt, steps=steps, negative_prompt=negative_prompt, guidance_scale=guidance_scale, 
                           aspect=aspect, refine_strength=refine_strength, seed=seed)
                
        if n_images > 1:
            _,BSZ = self.batch_settings('full')
            if n_images <= 10:
                BSZ = max(1, BSZ//2) # if a small batch, cut batch size in half
            
            call_kwargsets, seeds = self.seed_call_kwargs(seed, call_kwargs, n_images=n_images)
            batchgen = self._batched_txt2img(prompt_encodings=prompt_encodings, num_images=n_images, batch_size=BSZ, num_inference_steps=steps, guidance_scale=guidance_scale, target_size=target_size, seed=seeds, output_type='pil')
            
            for imbatch,kwbatch in zip(batchgen, batched(call_kwargsets, BSZ)):
                yield (imbatch, kwbatch)
        else:
            image = self._pipe_txt2img(prompt_encodings=prompt_encodings, num_inference_steps=steps, guidance_scale=guidance_scale, target_size=target_size, seed=seed)
            if refine_strength:
                image = self._pipe_img2upimg(prompt_encodings=prompt_encodings, image=image, num_inference_steps=steps, refine_strength=refine_strength, guidance_scale=guidance_scale, seed=seed, scale=1.5)
            yield ([image], [call_kwargs])
        
        release_memory()

    @torch.inference_mode()
    def regenerate_image(self, prompt: str, image: Image.Image,  
                         n_images: int = 1,
                         steps: int = None, 
                         strength: float = None, 
                         negative_prompt: str = None, 
                         guidance_scale: float = None, 
                         aspect: typing.Literal['square','portrait','landscape'] = None, 
                         
                         refine_strength: float = None, 
                         seed: int = None,
                         **kwargs):
        
        fkwg = self.config.get_if_none(steps=steps, strength=strength, negative_prompt=negative_prompt, guidance_scale=guidance_scale, )
        
        steps = fkwg['steps']
        guidance_scale = fkwg['guidance_scale']
        negative_prompt=fkwg['negative_prompt']
        strength = fkwg['strength']
        logger.debug(f'unused_kwargs: {kwargs} | fkwg:{fkwg}')
        # Resize to best dim match unless aspect given. don't use fkwg[aspect] because dont want None autofilled
        image = self._resize_image(image, aspect)
                

        prompt, negative_prompt = self.preprocess_prompts(prompt, negative_prompt)
        prompt_encodings = self.embed_prompts(prompt, negative_prompt=negative_prompt, image=image)
        
        # don't want to pass around full image, going to replace with image_url in imagegen anyway
        call_kwargs = dict(prompt=prompt, image='PLACEHOLDER', steps=steps, strength=strength, negative_prompt=negative_prompt, guidance_scale=guidance_scale, 
                           aspect=aspect, refine_strength=refine_strength, seed=seed,)
        
        
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
                     seed: int = None,
                     **kwargs) -> tuple[Image.Image, dict]:
        
        fkwg = self.config.get_if_none(steps=steps, negative_prompt=negative_prompt, guidance_scale=guidance_scale, )
        
        steps = fkwg['steps']
        guidance_scale = fkwg['guidance_scale']
        negative_prompt=fkwg['negative_prompt']

        logger.debug(f'unused_kwargs: {kwargs} | fkwg:{fkwg}')
        image = self._resize_image(image, aspect=None, dim_choices=None) # Resize to best dim match

        prompt, negative_prompt = self.preprocess_prompts(prompt, negative_prompt)
        prompt_encodings = self.embed_prompts(prompt, negative_prompt=negative_prompt, image=image)

        image = self._pipe_img2upimg(prompt_encodings, image, num_inference_steps=steps, refine_strength=refine_strength, guidance_scale=guidance_scale, seed=seed, scale=1.5)
        
        call_kwargs = dict(image=image, prompt=prompt, refine_strength=refine_strength, steps=steps, negative_prompt=negative_prompt, 
                           guidance_scale=guidance_scale, seed=seed,)
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
                            aspect: typing.Literal['square','portrait','landscape'] = None, 
                            mid_frames:int=0,
                            seed: int = None, 
                            **kwargs):   
        
        gseed = torch.Generator(device='cpu').manual_seed(seed) if seed is not None else None
        # NOTE: if you attempt to update the seed on each iteration, you get some interesting behavoir
        # you effectively turn it into a coloring book generator. I assume this is a product of how diffusion works
        # since it predicts the noise to remove, when you feed its last prediction autoregressive style, boils it down
        # the minimal representation of the prompt. If you 
        
        fkwg = self.config.get_if_none(steps=steps, negative_prompt=negative_prompt, guidance_scale=guidance_scale, aspect=aspect)
        
        negative_prompt=fkwg['negative_prompt']
        guidance_scale=fkwg['guidance_scale']
        steps = fkwg['steps']
        aspect = fkwg['aspect']

        logger.debug(f'unused_kwargs: {kwargs} | fkwg:{fkwg}')
        
        prompt, negative_prompt = self.preprocess_prompts(prompt, negative_prompt)
        
        # subtract 1 for first image
        nframes = nframes - 1
        
        image_frames = []
        if image is None:
            prompt_encodings = self.embed_prompts(prompt, negative_prompt=negative_prompt, image=None)
            strength_end = np.clip(strength_end, 1/steps, 1-(1/steps)) # make sure strength < 1 or will effectively ignore input image 
            strengths = [strength_end]*nframes
            w,h = self.config.get_dims(aspect)
            # timesteps=AysSchedules["StableDiffusionXLTimesteps"]
            image = self.t2i(num_inference_steps=steps, guidance_scale=guidance_scale, height=h, width=w,  **prompt_encodings, output_type='pil', generator=gseed).images
        else:
            strengths = discretize_strengths(steps, nframes, start=strength_start, end=strength_end)
            # round up strengths since they will be floored in get_timesteps via int() and it makes step distribution more uniform for lightning models
            image = self._resize_image(image, aspect=None, dim_choices=None) #SDXL_DIMS)
            w,h = image[0].size

        image_frames.extend(image)
        prompt_encodings = self.embed_prompts(prompt, negative_prompt=negative_prompt, image=image)

        yield -1
        
        for i in range(nframes):
            # gseed.manual_seed(seed) # uncommenting this will turn into a coloring book generator
            if 'image' in inspect.signature(self.embed_prompts).parameters and i>0: # i=0 was before loop
                prompt_encodings = self.embed_prompts(prompt, negative_prompt=negative_prompt, image=image) # avoid calling this unless actually necessary (e.g. Qwen Edit)
            
            image = self.i2i(image=image, num_inference_steps=steps, strength=strengths[i], guidance_scale=guidance_scale, height=h, width=w, **prompt_encodings, output_type='pil', generator=gseed).images
            image_frames.extend(image)
            yield i
            
        if isinstance(image_frames[-1], torch.Tensor):
            image_frames = self.decode_latents(image_frames, height=h, width=w, )
        
        prompt_encodings = release_memory(prompt_encodings)
        
        if mid_frames:
            image_frames = self.interpolate(image_frames, inter_frames=mid_frames, batch_size=2)
            #image_frames = interpolate.image_lerp(image_frames, total_frames=33, t0=0, t1=1, loop_back=False, use_slerp=False)
        yield image_frames
        
    @torch.inference_mode()
    def regenerate_frames(self, prompt: str, frame_array: np.ndarray, 
                          steps: int = None, 
                          astrength: float = 0.5, 
                          imsize: typing.Literal['tiny','small','med','full'] = 'small', 

                          negative_prompt: str = None, 
                          guidance_scale: float = None, 
                          two_stage: bool = False, 
                          aseed: int = None, 
                          **kwargs):   
        
       
        # NOTE: special behavior since having a seed improves results substantially 
        if aseed is None: 
            aseed = np.random.randint(1e9, 1e10-1)
        elif aseed < 0:
            aseed = None

        fkwg = self.config.get_if_none(steps=steps, strength=astrength, negative_prompt=negative_prompt, guidance_scale=guidance_scale)
        steps = fkwg['steps']
        negative_prompt=fkwg['negative_prompt']
        guidance_scale=fkwg['guidance_scale']

        astrength = np.clip(astrength, 1/steps, 1) # clip strength so at least 1 step occurs
        #astrength = max(astrength*steps, 1)/steps 
        logger.debug(f'unused_kwargs: {kwargs} | fkwg:{fkwg}')
                
        dim_choices,bsz = self.batch_settings(imsize)
        resized_images = self._resize_image_frames(frame_array, dim_choices, max_num_frames=100, upsample_px_thresh=256, upsample_bsz=8) # upsample first if less 256^2 pixels 
        dim_out = resized_images[0].size
        w,h = dim_out
        
        prompt, negative_prompt = self.preprocess_prompts(prompt, negative_prompt)
        prompt_encodings = self.embed_prompts(prompt, negative_prompt=negative_prompt, batch_size=bsz, image=resized_images)
        
        # First, yield the total number of frames since it may have changed from slice
        yield len(resized_images)

        latents = []
        t0 = time.perf_counter()
        for img_lats in self._batched_imgs2imgs(prompt_encodings, resized_images, batch_size=bsz, steps=steps, strength=astrength, guidance_scale=guidance_scale, img_wh=dim_out, seed=aseed, **kwargs):
            latents.append(img_lats)
            yield len(img_lats)

        
        if two_stage:
            raw_image_latents = self.encode_images(resized_images, seed = aseed)
            latent_blend = self._regen_stage2(latents, raw_image_latents, resized_images)
            
            latents = []
            for img_lats in self._batched_imgs2imgs(prompt_encodings, latent_blend, batch_size=bsz, steps=steps, strength=0.3, guidance_scale=guidance_scale, img_wh=dim_out, seed=aseed, **kwargs):
                latents.append(img_lats)
                yield len(img_lats)
            
        latents = torch.cat(latents, 0)
        image_frames = self.decode_latents(latents, height=h, width=w)
        
        prompt_encodings, latents = release_memory(prompt_encodings, latents)
        
        yield image_frames

        te = time.perf_counter()
        runtime = te-t0
        logger.info(f'RUN TIME: {runtime:0.2f}s | BSZ: {bsz} | DIM: {dim_out} | N_IMAGE: {len(resized_images)} | IMG/SEC: {len(resized_images)/runtime:0.2f}')
    
    @torch.inference_mode()
    def _regen_stage2(self, latents:torch.Tensor, raw_image_latents:torch.Tensor, resized_images:list[Image.Image], *, output_type: typing.Literal['pil', 'latent'] = 'pil'):
        img_w,img_h = resized_images[0].size

        if isinstance(latents, list):
            latents = torch.cat(latents, dim=0)
        
        # create a soft mask based on interframe pixel differences
        mot_mask_tensor = torch.from_numpy(interpolation.motion_mask(resized_images, px_thresh=0.02, qtile=90))
        # interpolate to latent shape
        latent_soft_mask = torch.nn.functional.interpolate(mot_mask_tensor.expand(1, 1, -1, -1), size=raw_image_latents.shape[-2:], mode='area',).to(raw_image_latents)
        # slerp between input and output latents guided by soft_mask and then lerp consecutive latent frames pairs
        latent_blend = interpolation.blend_latents(raw_image_latents, latents, latent_soft_mask, time_blend=True, keep_dims=True)
        
        if output_type == 'latent':
            return latent_blend
        # decode back to images by default
        return self.decode_latents(latent_blend, height=img_h, width=img_w)
#endregion

#region Latent Base Class
class LatentOptPipeline(SingleStagePipeline):
    """Optimizations for pipelines that allow latents in place of image input"""
    
    @contextlib.contextmanager
    def no_image_preprocessing(self):
        _image_preprocess_fn = self.basei2i.image_processor.preprocess
        
        try:
            # prevent image processor from modifying the latents
            def noop(image, *args, **kwargs): return image
            self.basei2i.image_processor.preprocess = noop
            yield
        
        finally:
            # restore original image preprocessor
            self.basei2i.image_processor.preprocess = _image_preprocess_fn

    @torch.inference_mode()
    def _batched_imgs2imgs(self, prompt_encodings:dict[str, torch.Tensor], images: torch.Tensor, batch_size:int, steps:int, strength:float, guidance_scale:float, img_wh:tuple[int, int], seed:int, **kwargs):
        batched_prompts = rebatch_prompt_embs(prompt_encodings, batch_size)
        batched_images = torch.split(images, batch_size, dim=0)
        w,h = img_wh # need wh because `images` is a batch of latents
        
        # list of generators with shared common seed is very important. Otherwise it does not apply correctly
        gen_batches = (generator_batch(seed, b.shape[0], device='cuda') for b in batched_images)
        for imbatch in batched_images:                
            if (batch_len := imbatch.shape[0]) != batch_size:
               batched_prompts = rebatch_prompt_embs(prompt_encodings, batch_len)

            generators = next(gen_batches) #generator_batch(seed, batch_len, device='cpu')

            latents = self.i2i(image=imbatch, num_inference_steps=steps, strength=strength, guidance_scale=guidance_scale, height=h, width=w, **batched_prompts, generator=generators, output_type='latent', **kwargs).images 

            yield latents

    @torch.inference_mode()
    def generate_frames(self, prompt: str, 
                            image: Image.Image|None = None, 
                            nframes: int = 11,
                            steps: int = None, 
                            strength_end: float = 0.80, 
                            strength_start: float = 0.30, 
                            negative_prompt: str = None, 
                            guidance_scale: float = None, 
                            aspect: typing.Literal['square','portrait','landscape'] = None, 
                            mid_frames:int=0,
                            seed: int = None, 
                            **kwargs):   
        
        gseed = torch.Generator(device='cpu').manual_seed(seed) if seed is not None else None
        # NOTE: if you attempt to update the seed on each iteration, you get some interesting behavoir
        # you effectively turn it into a coloring book generator. I assume this is a product of how diffusion works
        # since it predicts the noise to remove, when you feed its last prediction autoregressive style, boils it down
        # the minimal representation of the prompt. If you 
        
        fkwg = self.config.get_if_none(steps=steps, negative_prompt=negative_prompt, guidance_scale=guidance_scale, aspect=aspect)
        
        negative_prompt = fkwg['negative_prompt']
        guidance_scale = fkwg['guidance_scale']
        steps = fkwg['steps']
        aspect = fkwg['aspect']

        logger.debug(f'unused_kwargs: {kwargs} | fkwg:{fkwg}')
        
        prompt, negative_prompt = self.preprocess_prompts(prompt, negative_prompt)
        prompt_encodings = self.embed_prompts(prompt, negative_prompt=negative_prompt, image=image)
        
        # subtract 1 for first image
        nframes = nframes - 1
        
        latents = []
        image_frames = []
        latent = None

        if image is None:
            strength_end = np.clip(strength_end, 1/steps, 1-(1/steps)) # make sure strength < 1 or will effectively ignore input image 
            strengths = [strength_end]*nframes
            w,h = self.config.get_dims(aspect)
        else:
            # round up strengths since they will be floored in get_timesteps via int() and it makes step distribution more uniform for lightning models 
            strengths = discretize_strengths(steps, nframes, start=strength_start, end=strength_end)
            image = self._resize_image(image, aspect=None, dim_choices=None)
            w,h = image[0].size

            image_frames.append(image[0])
            latent = self.encode_images(image, seed = seed)
            
        yield -1
                
        # We can massively improve performance by avoiding device transfers and delaying decodes until the end. 
        with (self.offload_disabled(('transformer','unet')), self.no_image_preprocessing()):
            if latent is None:
                latent = self.t2i(num_inference_steps=steps, guidance_scale=guidance_scale, height=h, width=w, **prompt_encodings, output_type='latent', generator=gseed).images
                latents.append(latent)
            
            for i in range(nframes):
                # gseed.manual_seed(seed) # uncommenting this will turn into a coloring book generator
                latent = self.i2i(image=latent, num_inference_steps=steps, strength=strengths[i], guidance_scale=guidance_scale, height=h, width=w, **prompt_encodings, output_type='latent', generator=gseed).images
                latents.append(latent)
                yield i
        
        
        with self.offload_disabled('vae'):    
            image_frames += self.decode_latents(latents, height=h, width=w, )
        
        prompt_encodings, latents = release_memory(prompt_encodings, latents)
        
        if mid_frames:
            image_frames = self.interpolate(image_frames, inter_frames=mid_frames, batch_size=2)
        
        yield image_frames
    
    @torch.inference_mode()
    def regenerate_frames(self, prompt: str, frame_array: np.ndarray, 
                          steps: int = None, 
                          astrength: float = 0.5, 
                          imsize: typing.Literal['tiny','small','med','full'] = 'small', 

                          negative_prompt: str = None, 
                          guidance_scale: float = None, 
                          two_stage: bool = False, 
                          aseed: int = None, 
                          **kwargs):   
        
        # NOTE: special behavior since having a seed improves results substantially 
        if aseed is None: 
            aseed = np.random.randint(1e9, 1e10-1)
        elif aseed < 0:
            aseed = None

        fkwg = self.config.get_if_none(steps=steps, strength=astrength, negative_prompt=negative_prompt, guidance_scale=guidance_scale)
        steps = fkwg['steps']
        negative_prompt=fkwg['negative_prompt']
        guidance_scale=fkwg['guidance_scale']

        astrength = np.clip(astrength, 1/steps, 1) # clip strength so at least 1 step occurs
        logger.debug(f'unused_kwargs: {kwargs} | fkwg:{fkwg}')
        
        dim_choices,bsz = self.batch_settings(imsize)
        resized_images = self._resize_image_frames(frame_array, dim_choices, max_num_frames=100, upsample_px_thresh=256, upsample_bsz=8) # upsample first if less 256^2 pixels 
        dim_out = resized_images[0].size
        w,h = dim_out
        
        prompt, negative_prompt = self.preprocess_prompts(prompt, negative_prompt)
        prompt_encodings = self.embed_prompts(prompt, negative_prompt=negative_prompt, batch_size=bsz, image=resized_images)
        raw_image_latents = self.encode_images(resized_images, seed = aseed)
        
        # First, yield the total number of frames since it may have changed from slice
        yield len(resized_images)
        
        latents = []
        t0 = time.perf_counter()
        
        with (self.offload_disabled(('transformer','unet')), self.no_image_preprocessing()):
            for img_lats in self._batched_imgs2imgs(prompt_encodings, raw_image_latents, batch_size=bsz, steps=steps, strength=astrength, guidance_scale=guidance_scale, img_wh=dim_out, seed=aseed, **kwargs):
                latents.append(img_lats)
                yield len(img_lats)

            if two_stage:
                latent_blend = self._regen_stage2(latents, raw_image_latents, resized_images, output_type='latent')
                
                latents = []
                for img_lats in self._batched_imgs2imgs(prompt_encodings, latent_blend, batch_size=bsz, steps=steps, strength=0.3, guidance_scale=guidance_scale, img_wh=dim_out, seed=aseed, **kwargs):
                    latents.append(img_lats)
                    yield len(img_lats)
        
        latents = torch.cat(latents, 0)

        with self.offload_disabled('vae'):
            image_frames = self.decode_latents(latents, height=h, width=w)
        
        prompt_encodings, latents, raw_image_latents = release_memory(prompt_encodings, latents, raw_image_latents)
        
        yield image_frames

        te = time.perf_counter()
        runtime = te-t0
        logger.info(f'RUN TIME: {runtime:0.2f}s | BSZ: {bsz} | DIM: {dim_out} | N_IMAGE: {len(resized_images)} | IMG/SEC: {len(resized_images)/runtime:0.2f}')
#endregion

#region SDXL
class SDXLBase(DeepCacheMixin, LatentOptPipeline):
    def __init__(self, model_name: str, model_path: str, config: DiffusionConfig, offload=False, scheduler_setup: str | tuple[str, dict] = None, dtype: torch.dtype = torch.bfloat16, init_loras: list[tuple[str,float]] = None,
                 clip_skip: int = None):
        super().__init__(model_name, model_path, config, offload, scheduler_setup, dtype, init_loras, root_name='sdxl')
        
        self.pipe_xkwgs['clip_skip'] = clip_skip # TODO: because of Compel, this has no effect

    def i2i(self, *args, **kwargs):
        kwargs.pop('height',None)
        kwargs.pop('width', None) # sdxl image-to-image does not accept height or width input
        return self.basei2i(*args, **kwargs, **self.pipe_xkwgs)
        
    def load_pipeline(self):
        _pipe_kwargs = dict(torch_dtype=self.dtype, variant="fp16", use_safetensors=True, add_watermarker=False, )
        if str(self.model_path).endswith('.safetensors'):
            self.base: StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_single_file(self.model_path, **_pipe_kwargs)
        else:
            self.base: StableDiffusionXLPipeline = AutoPipelineForText2Image.from_pretrained(self.model_path, **_pipe_kwargs) 
            #custom_pipeline='lpw_stable_diffusion_xl,'#, device_map=device,
        
        if not self.offload:
            self.base = self.base.to(0)

        self._scheduler_init()
        self.load_loras()
        # https://github.com/comfyanonymous/ComfyUI/blob/f58475827150c2ac610cbb113019276efcd1a733/comfy/sd1_clip.py#L234

        self.compeler = CompelForSDXL(self.base)
        
        self.basei2i: StableDiffusionXLImg2ImgPipeline = AutoPipelineForImage2Image.from_pipe(self.base, **_pipe_kwargs)

        # calling on both pipes seems to make offload more consistent, may not be inherited properly with from_pipe 
        # https://huggingface.co/docs/diffusers/v0.36.0/en/using-diffusers/loading#:~:text=reapply%20these%20methods
        for pipe in (self.base, self.basei2i):
            pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
            # Workaround for not accepting attention shape using VAE for Flash Attention
            pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
            pipe.vae.enable_slicing()
            if self.offload:
                pipe.enable_model_cpu_offload()

        self.is_ready = True

    
    def set_pag_config(self, pag_scale:float=3.0, pag_applied_layers: str|list[str] = "mid", disable_on_zero:bool=False):
        # https://huggingface.co/docs/diffusers/main/en/using-diffusers/pag?tasks=Text-to-image
        # https://huggingface.co/docs/diffusers/main/en/api/pipelines/pag#perturbed-attention-guidance
        if isinstance(pag_applied_layers, str):
            pag_applied_layers = [pag_applied_layers]
        
        pag_config = {'enabled':False, 'scale':None, 'layers':None}
        if pag_scale:
            if 'pag_scale' not in self.pipe_xkwgs:
                pag_kwargs = {'enable_pag':True, 'pag_applied_layers':pag_applied_layers,}
                self.base: StableDiffusionXLPAGPipeline = AutoPipelineForText2Image.from_pipe(self.base, **pag_kwargs, torch_dtype=self.dtype)
                self.basei2i: StableDiffusionXLPAGImg2ImgPipeline = StableDiffusionXLPAGImg2ImgPipeline.from_pipe(self.basei2i, pag_applied_layers=pag_applied_layers, torch_dtype=self.dtype,)
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
            self.base = AutoPipelineForText2Image.from_pipe(self.base, enable_pag=False, torch_dtype=self.dtype)
            self.basei2i = AutoPipelineForImage2Image.from_pipe(self.basei2i, enable_pag=False, torch_dtype=self.dtype,)
            if self.offload:
                self.base.enable_model_cpu_offload()
                self.basei2i.enable_model_cpu_offload()
        else:
            self.pipe_xkwgs.update(pag_scale=0.0)
            pag_config.update(enabled=False, scale=0.0, layers=pag_applied_layers)
        
        print(self.base.__class__,self.base.__class__.__name__)
        print(self.basei2i.__class__, self.basei2i.__class__.__name__)
        return pag_config

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
    def compile_pipeline(self):
        # calling the compiled pipeline on a different image size triggers compilation again which can be expensive.
        # Some benefit from arg dynamic = True out of the box, some don't 
        # https://huggingface.co/docs/diffusers/optimization/fp16#dynamic-shape-compilation

        torch_compile_flags()
        torch.fx.experimental._config.use_duck_shape = False
        self.base.unet.to(memory_format=torch.channels_last)
        self.base.vae.to(memory_format=torch.channels_last)
        self.base.unet = torch.compile(self.base.unet, mode="max-autotune", fullgraph=True, dynamic=True)
        self.base.vae.decode = torch.compile(self.base.vae.decode, mode="max-autotune", fullgraph=True, dynamic=True)
        
        for _ in range(3):
            _ = self.t2i("a photo of a cat holding a sign that says hello world")#, num_inference_steps=4, guidance_scale=self.config.guidance_scale)
        self.is_compiled = True
        torch.compiler.reset()

    @torch.inference_mode()
    def embed_prompts(self, prompt:str, negative_prompt:str = None, batch_size: int = 1, **kwargs):
        # compel handles prompt -> [prompt]
        if not negative_prompt:
            negative_prompt = None # compel treats None different than empty string. Save a few steps by ""->None.

        device = torch.device('cuda')
        conditioning = self.compeler(prompt, negative_prompt=negative_prompt)

        prompt_encodings = dict(
            prompt_embeds = conditioning.embeds, 
            pooled_prompt_embeds = conditioning.pooled_embeds,
            negative_prompt_embeds = conditioning.negative_embeds,
            negative_pooled_prompt_embeds = conditioning.negative_pooled_embeds,
        )
        prompt_encodings = {k: v.to(device) for k,v in prompt_encodings.items() if v is not None}

        torch.cuda.empty_cache()    
        return prompt_encodings
    
    @torch.inference_mode()
    def encode_images(self, images:list[Image.Image], seed:int=42):
        nframes = len(images)
        w,h = images[0].size
        gseed = generator_batch(seed, nframes, device='cuda')
        device = torch.device('cuda',0)
        proc_images = self.basei2i.image_processor.preprocess(images, height=h, width=w).to(device, dtype=self.basei2i.vae.dtype)
        latents = self.basei2i.prepare_latents(proc_images, timestep=None, batch_size=1, num_images_per_prompt=nframes, dtype=torch.bfloat16, device='cuda', generator=gseed, add_noise=False)
        proc_images=release_memory(proc_images)
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
#endregion

#region Flux
class FluxBase(SingleStagePipeline):
    def __init__(self, model_name: str, model_path: str, config: DiffusionConfig, offload=False, scheduler_setup: str | tuple[str, dict] = None, dtype: torch.dtype = torch.bfloat16, init_loras: list[tuple[str,float]] = None,):
        super().__init__(model_name, model_path, config, offload, scheduler_setup, dtype, init_loras, root_name='flux')
        
        self.adapter_paths = {}
        self.variant = model_name.split('_')[-1] # flux_schell,flux_dev,flux_* -> *
        self.max_seq_len = 512
        
    @torch.inference_mode()
    def _load_text_encoder2(self, source: typing.Literal['nunchaku','unchained','default','default_nf4'] = 'default_nf4'):
        match source:
            case 'nunchaku':
                text_encoder_2 = NunchakuT5EncoderModel.from_pretrained(
                    "nunchaku-ai/nunchaku-t5/awq-int4-flux.1-t5xxl.safetensors", 
                    torch_dtype=torch.bfloat16, 
                    offload=self.offload
                )
            case 'unchained':
                with init_empty_weights():
                    config = AutoConfig.from_pretrained('Kaoru8/T5XXL-Unchained')
                    text_encoder_2 = AutoModelForTextEncoding.from_config(config, dtype=torch.float16)
                    sd_path = hf_hub_download("Kaoru8/T5XXL-Unchained", "t5xxl-unchained-f16.safetensors")
        
                text_encoder_2.load_state_dict(sft.load_file(sd_path), assign=True)
                text_encoder_2 = text_encoder_2.to(dtype=torch.bfloat16)
            case 'default_nf4':
                quant_config = TransBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16)
                text_encoder_2 = AutoModelForTextEncoding.from_pretrained(
                    'diffusers/t5-nf4', low_cpu_mem_usage = True, dtype=torch.bfloat16, quantization_config = quant_config,
                )
            case 'default':
                text_encoder_2 = AutoModelForTextEncoding.from_pretrained(
                    self.model_path, subfolder = 'text_encoder_2', low_cpu_mem_usage = True, dtype=torch.bfloat16,
                )
            case _:
                raise ValueError(f'unknown source: {source!r}')
        
        text_encoder_2 = text_encoder_2.eval().requires_grad_(False)
        return text_encoder_2

    @torch.inference_mode()
    def load_pipeline(self):
        transformer_model_path = f"nunchaku-ai/nunchaku-flux.1-dev/svdq-{get_precision()}_r32-flux.1-dev.safetensors"
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(transformer_model_path, torch_dtype=torch.bfloat16, offload=self.offload) # nunchaku forces device='cuda'

        self.base = FluxPipeline.from_pretrained(
            self.model_path, 
            transformer = transformer,
            text_encoder_2 = self._load_text_encoder2('default_nf4'),
            torch_dtype = torch.bfloat16,
            dtype = torch.bfloat16,
            low_cpu_mem_usage = True,
        )
        
        if not self.offload:
            self.base = self.base.to(0)
            
        
        self._scheduler_init()
        self.load_loras()

        self.compeler = CompelForFlux(self.base) 
        self.basei2i: FluxImg2ImgPipeline = FluxImg2ImgPipeline.from_pipe(self.base, transformer=self.base.transformer, torch_dtype=torch.bfloat16) # Do NOT forget torch_dtype or will 2x Vram

        
        for pipe in (self.base, self.basei2i):
            # pipe.enable_xformers_memory_efficient_attention()
            pipe.vae.enable_slicing() # all this does is iterate over batch instead of all at once, no reason to ever disable
            if self.offload:
                pipe.enable_model_cpu_offload()
            elif not getattr(pipe.text_encoder_2, 'is_quantized', False): 
                # if not offloading everything, just offload text_encoder_2 if in full precision
                apply_group_offloading(pipe.text_encoder_2, onload_device=torch.device('cuda'), offload_type="block_level", 
                                        num_blocks_per_group=1, use_stream=True, record_stream=True, non_blocking=True)

        self.is_ready = True
        release_memory()
    
    @torch.inference_mode()
    def load_loras(self, path_weights: list[tuple[str,float]] = None):
        if path_weights is None:
            path_weights = self.init_loras
        
        if not path_weights:
            return
        
        lora_path_weights = []
        for path,weight in path_weights:
            if isinstance(path, Path):
                path = path.as_posix()
            
            lora_path_weights.append((path, weight))
            adapter_name = path.split('/')[-1].rsplit('.', maxsplit=1)[0].replace('-','_')
            
            self.adapter_weights[adapter_name] = weight
            self.adapter_paths[adapter_name] = path
            
        self.base.transformer.update_lora_params(compose_lora(lora_path_weights))
        
    
    @torch.inference_mode()
    def set_adapter_weight(self, adapter_name:str, weight: float):            
        try:
            self.adapter_weights[adapter_name] = weight
        except KeyError:
            raise KeyError(f'Unknown adapter_name: {adapter_name!r}')
        
        self.base.transformer.update_lora_params(compose_lora([
            (self.adapter_paths[name], self.adapter_weights[name]) for name in self.adapter_weights
        ]))
        
        logger.debug(f'Active Adapters: {self.base.get_active_adapters()}\nAll Adapters: {self.base.get_list_adapters()}\nAdapter Weights: {self.adapter_weights}' )

    def batch_settings(self, imsize: typing.Literal['tiny','small','med','full'] = 'small', ):
        dims_opts = {
            'tiny': [(512,512), (512,640), (640,512)], # 1.25
            'small': [(640,640), (512,768), (768,512)], # 1.5
            'med': [(768,768), (640,896), (896,640)], # 1.4
            'full': [(1024,1024), (832, 1216), (1216, 832)] # 1.46
        }
        
        # Max batch size is 8 due to awq_gemv kernel restriction
        batch_sizes = {'tiny': 8, 'small': 8, 'med': 6, 'full': 4} # With out vae_slicing

        return (dims_opts[imsize], batch_sizes[imsize])
    
    @torch.inference_mode()
    def embed_prompts(self, prompt:str, negative_prompt:str = None, batch_size: int = 1, **kwargs):
        # compel handles prompt -> [prompt]
        if not negative_prompt:
            negative_prompt = None # compel treats None different than empty string. Save a few steps by ""->None.
        
        device = torch.device('cuda')
        
        conditioning = self.compeler(prompt, style_prompt=None, negative_prompt=negative_prompt, negative_style_prompt=None)
                
        prompt_encodings = dict(
            prompt_embeds = conditioning.embeds, 
            pooled_prompt_embeds = conditioning.pooled_embeds,
            negative_prompt_embeds = conditioning.negative_embeds,
            negative_pooled_prompt_embeds = conditioning.negative_pooled_embeds,
        )
        # required for text_encoder_2 independent offload
        prompt_encodings = {k: v.to(device) for k,v in prompt_encodings.items() if v is not None}
            
        torch.cuda.empty_cache()
        return prompt_encodings
    
    
    @torch.inference_mode()
    def encode_images(self, images:list[Image.Image], seed:int=42):
        nframes = len(images)
        w,h = images[0].size
        device = torch.device('cuda', 0)
        generators = generator_batch(seed, nframes, device='cuda')
        proc_images = self.basei2i.image_processor.preprocess(images, height=h, width=w).to(device, self.basei2i.vae.dtype)
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
    def decode_latents(self, latents, height:int, width:int):
        # https://github.com/huggingface/diffusers/blob/4cfb2164fb05d54dd594373b4bd1fbb101fef70c/src/diffusers/pipelines/flux/pipeline_flux.py#L759
        if isinstance(latents, list):
            latents = torch.cat(latents, dim=0)
        #print(latents.shape)
        
        # make sure the VAE is in float32 mode, as it overflows in float16
        if latents.ndim == 3 and latents.shape[-1] == self.basei2i.transformer.config.in_channels: # 64
            latents = self.base._unpack_latents(latents, height, width, self.base.vae_scale_factor)
        latents = (latents / self.base.vae.config.scaling_factor) + self.base.vae.config.shift_factor
        image = self.base.vae.decode(latents, return_dict=False)[0]

        image = self.base.image_processor.postprocess(image, output_type='pil')
        
        return image
    
    @torch.inference_mode()
    def _regen_stage2(self, latents:torch.Tensor, raw_image_latents:torch.Tensor, resized_images:list[Image.Image], *, output_type: typing.Literal['pil', 'latent'] = 'pil'):
        img_w,img_h = resized_images[0].size
        
        if isinstance(latents, list):
            latents = torch.cat(latents, dim=0)
        
        if latents.ndim == 3 and latents.shape[-1] == self.basei2i.transformer.config.in_channels: # 64
            latents = self.basei2i._unpack_latents(latents, img_h, img_w, self.basei2i.vae_scale_factor)
        
        latent_blend = super()._regen_stage2(latents=latents, raw_image_latents=raw_image_latents, resized_images=resized_images, output_type='latent')
        if output_type == 'latent':
            return latent_blend
        
        return self.decode_latents(latent_blend, img_h, img_w)
#endregion

#region Qwen Image
class QwenImageBase(LatentOptPipeline):
    def __init__(self, model_name: str, model_path: str, config: DiffusionConfig, offload=False, scheduler_setup: str | tuple[str, dict] = None, dtype: torch.dtype = torch.bfloat16, init_loras: list[tuple[str,float]] = None, 
                 num_inference_steps = 4, rank=32):
        super().__init__(model_name, model_path, config, offload, scheduler_setup, dtype, init_loras, root_name='qwen')
        
        self.num_inference_steps = num_inference_steps #8
        self.rank = rank

        self.max_seq_len = 1024
        self.pipe_xkwgs['true_cfg_scale'] = 1.0

    def t2i(self, *args, **kwargs):
        out = self.base(*args, **kwargs, **self.pipe_xkwgs)
        if kwargs.get('output_type') == 'latent':
            # squeeze in necessary to pass image_preprocessor filter which checks ndim ∈ {3,4}. decode_latents calls .unsqueeze(2) to correct for this
            out.images = self.base._unpack_latents(out.images, kwargs.get('height'), kwargs.get('width'), self.base.vae_scale_factor).squeeze(2) # [B,C,1,H',W']->[B,C,H',W']
        return out
    
    def i2i(self, *args, **kwargs):
        out = self.basei2i(*args, **kwargs, **self.pipe_xkwgs)
        if kwargs.get('output_type') == 'latent':
            out.images = self.basei2i._unpack_latents(out.images, kwargs.get('height'), kwargs.get('width'), self.basei2i.vae_scale_factor).squeeze(2) # [B,C,1,H',W']->[B,C,H',W']
        return out
    
    @torch.inference_mode()
    def _scheduler_get(self):
        #  From https://github.com/ModelTC/Qwen-Image-Lightning/blob/342260e8f5468d2f24d084ce04f55e101007118b/generate_with_diffusers.py#L82C9-L97C10
        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),  # We use shift=3 in distillation
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),  # We use shift=3 in distillation
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,  # set shift_terminal to None
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        return FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)


    @torch.inference_mode()
    def load_pipeline(self):
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            'unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit',
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            low_cpu_mem_usage = True,
        )

        text_encoder: Qwen2_5_VLForConditionalGeneration = FastModel.for_inference(text_encoder)
        text_encoder = text_encoder.eval().requires_grad_(False)
        text_encoder.is_loaded_in_8bit = False # unsloth loaded 4bit models set this=True which prevents offloading : ValueError: `.to` is not supported for `8-bit` bitsandbytes models.

        scheduler = self._scheduler_get()

        transformer_model_path = f"nunchaku-ai/nunchaku-qwen-image/svdq-int4_r{self.rank}-qwen-image-lightningv1.0-{self.num_inference_steps}steps.safetensors"
        transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(transformer_model_path, torch_dtype=torch.bfloat16, device=("cpu" if self.offload else "cuda")) #offload=self.offload)#, device_map="auto", low_cpu_mem_usage = True, )
        transformer = transformer.eval().requires_grad_(False)

        self.base: QwenImagePipeline = QwenImagePipeline.from_pretrained(
            self.model_path, transformer=transformer, text_encoder=text_encoder, scheduler=scheduler, torch_dtype=torch.bfloat16, low_cpu_mem_usage = True,
        )
            
        self.basei2i: QwenImageImg2ImgPipeline = QwenImageImg2ImgPipeline.from_pipe(self.base, transformer=None, torch_dtype=torch.bfloat16)
        self.basei2i.transformer = self.base.transformer

        for pipe in (self.base, self.basei2i):
            # pipe.enable_xformers_memory_efficient_attention() # appears to use more vram?
            
            pipe.vae.enable_slicing()
            if self.offload:
                pipe.enable_model_cpu_offload()

        self.is_ready = True
    

    def batch_settings(self, imsize: typing.Literal['tiny','small','med','full'] = 'small', ):
        dims_opts = {
            'tiny': [(512,512), (512,640), (640,512)], # 1.25
            'small': [(640,640), (512,768), (768,512)], # 1.5
            'med': [(768,768), (640,896), (896,640)], # 1.4
            'full': [(1024,1024), (832, 1216), (1216, 832)] # 1.46
        }
        
        batch_sizes = {'tiny': 2, 'small': 2, 'med': 1, 'full': 1} 

        return (dims_opts[imsize], batch_sizes[imsize])    

    @torch.inference_mode()
    def embed_prompts(self, prompt:str, negative_prompt:str = None, batch_size: int = 1, **kwargs):
        device = torch.device('cuda:0')
        
        # prompt += ", Ultra HD, 4K, cinematic composition."
        
        prompt_embeds, prompt_embeds_mask = self.base.encode_prompt(prompt=prompt, num_images_per_prompt=batch_size, device=device, max_sequence_length=self.max_seq_len)
        
        prompt_encodings = dict(prompt_embeds=prompt_embeds, prompt_embeds_mask=prompt_embeds_mask)

        if negative_prompt:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.base.encode_prompt(prompt=negative_prompt, num_images_per_prompt=batch_size, device=device, max_sequence_length=self.max_seq_len)
            prompt_encodings.update(negative_prompt_embeds=negative_prompt_embeds, negative_prompt_embeds_mask=negative_prompt_embeds_mask)
        
        torch.cuda.empty_cache()
        return prompt_encodings

    @torch.inference_mode()
    def encode_images(self, images:list[Image.Image], seed:int=42):
        nframes = len(images)
        w,h = images[0].size
        generators = generator_batch(seed, nframes, device='cuda')
        device = torch.device('cuda:0')
        proc_images = self.basei2i.image_processor.preprocess(images, height=h, width=w).to(device, dtype=self.basei2i.vae.dtype)
        
        # If image is [B,C,H,W] -> add T=1. If it's already [B,C,T,H,W], leave it.
        if proc_images.dim() == 4:
            proc_images = proc_images.unsqueeze(2)
        
        image_latents = self.basei2i._encode_vae_image(image=proc_images, generator=generators)
        image_latents = torch.cat([image_latents], dim=0)
        # image_latents = image_latents.transpose(1, 2)  # -> [B,1,z,H',W'] 
        release_memory()
        return image_latents.squeeze(2) # [B,z,1,H',W'] -> [B,z,H',W']

    @torch.inference_mode()
    def decode_latents(self, latents, height, width, **kwargs):
        # https://github.com/huggingface/diffusers/blob/6f1042e36cd588a7b66498f45c3bb7085e4fa395/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_img2img.py#L852
        if isinstance(latents, list):
            latents = torch.cat(latents, dim=0)
        
        if latents.ndim == 3 and latents.shape[-1] == self.basei2i.transformer.config.in_channels: # 64
            latents = self.basei2i._unpack_latents(latents, height, width, self.basei2i.vae_scale_factor).to(self.basei2i.vae.dtype)
        elif latents.ndim == 4 and latents.shape[1] == self.basei2i.vae.config.z_dim:
            latents = latents.unsqueeze(2) # [B,z,H',W'] -> [B,z,1,H',W']
        
        latents_mean = (
            torch.tensor(self.basei2i.vae.config.latents_mean)
            .view(1, self.basei2i.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.basei2i.vae.config.latents_std).view(1, self.basei2i.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )

        latents = latents / latents_std + latents_mean
        image = self.basei2i.vae.decode(latents, return_dict=False)[0][:, :, 0]
        image = self.basei2i.image_processor.postprocess(image, output_type='pil')
        
        return image
    
    @torch.inference_mode()
    def _regen_stage2(self, latents:torch.Tensor, raw_image_latents:torch.Tensor, resized_images:list[Image.Image], *, output_type: typing.Literal['pil', 'latent'] = 'pil'):
        img_w,img_h = resized_images[0].size

        if isinstance(latents, list):
            latents = torch.cat(latents, dim=0)
        if latents.ndim == 3 and latents.shape[-1] == self.basei2i.transformer.config.in_channels: # 64
            latents = self.basei2i._unpack_latents(latents, img_h, img_w, self.basei2i.vae_scale_factor)

        latents = latents.squeeze() # [B,z,1,H',W'] -> # [B,z,H',W']
        raw_image_latents = raw_image_latents.squeeze() # [B,1,z,H',W'] -> [B,z,H',W']
        # create a soft mask based on interframe pixel differences
        mot_mask_tensor = torch.from_numpy(interpolation.motion_mask(resized_images, px_thresh=0.02, qtile=90))
        # interpolate to latent shape
        xdims = [1]*(raw_image_latents.ndim - 2) # we only care about last two dims, broadcast all others
        latent_soft_mask = torch.nn.functional.interpolate(mot_mask_tensor.expand(*xdims, -1, -1), size=raw_image_latents.shape[-2:], mode='area',).to(raw_image_latents)
        # slerp between input and output latents guided by soft_mask and then lerp consecutive latent frames pairs
        
        latent_blend = interpolation.blend_latents(raw_image_latents, latents, latent_soft_mask, time_blend=True, keep_dims=True)
        
        #latent_blend = latent_blend.unsqueeze(2) # [B,z,H',W'] -> [B,z,1,H',W']
        if output_type == 'latent':
            return latent_blend
        
        return self.decode_latents(latent_blend, img_h, img_w)
#endregion

#region Qwen Edit
class QwenEditBase(QwenImageBase):
    CONDITION_IMAGE_SIZE = 384 * 384
    VAE_IMAGE_SIZE = 1024 * 1024

    @staticmethod
    def calculate_dimensions(target_area, ratio):
        width = math.sqrt(target_area * ratio)
        height = width / ratio

        width = round(width / 32) * 32
        height = round(height / 32) * 32

        return width, height
    
    def i2i(self, *args, **kwargs):
        kwargs.pop('strength', None)
        return self.basei2i(*args, **kwargs, **self.pipe_xkwgs)

    def t2i(self, *args, **kwargs):
        return self.base(*args, **kwargs, **self.pipe_xkwgs)
    

    @torch.inference_mode()
    def load_pipeline(self):
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            'unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit',
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            low_cpu_mem_usage = True,
        )

        text_encoder: Qwen2_5_VLForConditionalGeneration = FastModel.for_inference(text_encoder)
        text_encoder = text_encoder.eval().requires_grad_(False)
        text_encoder.is_loaded_in_8bit = False # unsloth loaded 4bit models set this=True which prevents offloading : ValueError: `.to` is not supported for `8-bit` bitsandbytes models.
        
        scheduler = self._scheduler_get()

        transformer_model_path = (f"nunchaku-ai/nunchaku-qwen-image-edit-2509/lightning-251115/svdq-{get_precision()}_r{self.rank}-qwen-image-edit-2509-lightning-{self.num_inference_steps}steps-251115.safetensors")
        # transformer_model_path = (f"QuantFunc/Nunchaku-Qwen-Image-EDIT-2511/nunchaku_qwen_image_edit_2511_balance_{get_precision()}.safetensors")
        transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(transformer_model_path, torch_dtype=torch.bfloat16, device=("cpu" if self.offload else "cuda")) # offload=self.offload) -- offload here = gen faster by ~3-4s, but memory cannot be reclaimed with unload_pipeline
        transformer = transformer.eval().requires_grad_(False)

        self.basei2i: QwenImageEditPlusPipeline = QwenImageEditPlusPipeline.from_pretrained(
            self.model_path, transformer=transformer, text_encoder=text_encoder, scheduler=scheduler, torch_dtype=torch.bfloat16, low_cpu_mem_usage = True,
        )
        
        self.base = QwenImagePipeline.from_pipe(self.basei2i, transformer=None, text_encoder=self.basei2i.text_encoder, scheduler=self.basei2i.scheduler, torch_dtype=torch.bfloat16,)
        self.base.transformer = self.basei2i.transformer # late bind or will get "ValueError: Casting a quantized model to a new `dtype` is unsupported." 
        
        for pipe in (self.base, self.basei2i):
            # pipe.enable_xformers_memory_efficient_attention() # appears to use more vram?
            pipe.vae.enable_slicing()
            if self.offload:
                pipe.enable_model_cpu_offload()

        self.is_ready = True
        
        self.pipe_xkwgs['true_cfg_scale'] = 1.0
        
    
    def batch_settings(self, imsize: typing.Literal['tiny','small','med','full'] = 'small', ):
        dims_opts = {
            'tiny': [(768,768), (640,896), (896,640)], # 1.4
            'small': [(768,768), (640,896), (896,640)], # 1.4
            'med': [(768,768), (640,896), (896,640)], # 1.4
            'full': [(1024,1024), (832, 1216), (1216, 832)] # 1.46
        }
        
        batch_sizes = {'tiny': 1, 'small': 1, 'med': 1, 'full': 1} # Batch > 1 is consistently slower regardless of image size

        return (dims_opts[imsize], batch_sizes[imsize])  
    
    @torch.inference_mode()
    def preprocess_image(self, image: Image.Image | list[Image.Image] | None):
        if image is None or (isinstance(image, torch.Tensor) and image.size(1) == self.basei2i.latent_channels): # 16
            return image
        
        if isinstance(image, Image.Image):
            image = [image]

        condition_images = []
        #vae_images = []

        for img in image:
            image_width, image_height = img.size
            condition_width, condition_height = self.calculate_dimensions(self.CONDITION_IMAGE_SIZE, image_width / image_height)
            #vae_width, vae_height = self.calculate_dimensions(self.VAE_IMAGE_SIZE, image_width / image_height)
            condition_images.append(self.basei2i.image_processor.resize(img, condition_height, condition_width))
            #vae_images.append(self.image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2))
        return condition_images
    
    @torch.inference_mode()
    def _rebatch_embeds(self, prompt_embeds:torch.Tensor, prompt_embeds_mask:torch.Tensor, num_images_per_prompt:int, max_sequence_length:int|None = None):
        if num_images_per_prompt == 1:
            return prompt_embeds, prompt_embeds_mask
        
        # Carefully adjust batch since usually >1 img != batch
        batch_size = 1
        if max_sequence_length:
            prompt_embeds = prompt_embeds[:, :max_sequence_length]
            prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]
        
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)
        return prompt_embeds, prompt_embeds_mask

    @torch.inference_mode()
    def _get_qwen_prompt_embeds(self, prompt:str | list[str], image: list[Image.Image] | None, device: torch.device, dtype: torch.dtype = None):
        if image is None:
            return self.base._get_qwen_prompt_embeds(prompt, device, dtype)
        return self.basei2i._get_qwen_prompt_embeds(prompt, image, device, dtype)

    @torch.inference_mode()
    def embed_prompts(self, prompt:str, negative_prompt:str = None, batch_size: int = 1, image: Image.Image|list[Image.Image]|None = None, **kwargs):
        # https://github.com/huggingface/diffusers/blob/6f1042e36cd588a7b66498f45c3bb7085e4fa395/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py#L287
        device = torch.device('cuda:0')
        max_seq_length = self.max_seq_len if image is None else None # edit++ doesn't use max_seq_len to truncate embeds

        if isinstance(prompt, str):
            prompt = [prompt]
        
        if not isinstance(image, list):
            image = [image]
        
        if isinstance(image[0], list):
            image = [self.preprocess_image(img) for img in image] # batch of images:  list[list[Image]] e.g. [[i1],[i2],[i3]]
        elif image[0] is not None:
            image = [self.preprocess_image(image)] # multi-image: list[Image], e.g. [[i1, i2, i3]]
        
        
        prompt_embeds, prompt_embeds_mask = tuple(zip(*[self._get_qwen_prompt_embeds(prompt, img, device) for img in image]))
        prompt_embeds, prompt_embeds_mask = torch.cat(prompt_embeds,0), torch.cat(prompt_embeds_mask,0)
        prompt_embeds, prompt_embeds_mask = self._rebatch_embeds(prompt_embeds, prompt_embeds_mask, num_images_per_prompt=batch_size, max_sequence_length=max_seq_length)
        
        prompt_encodings = {
            'prompt_embeds': prompt_embeds, 
            'prompt_embeds_mask': prompt_embeds_mask,
        }
        
        if negative_prompt:
            negative_prompt_embeds, negative_prompt_embeds_mask = tuple(zip(*[self._get_qwen_prompt_embeds(negative_prompt, img, device) for img in image]))
            negative_prompt_embeds, negative_prompt_embeds_mask = torch.cat(negative_prompt_embeds,0), torch.cat(negative_prompt_embeds_mask,0)
            negative_prompt_embeds, negative_prompt_embeds_mask = self._rebatch_embeds(negative_prompt_embeds, negative_prompt_embeds_mask, num_images_per_prompt=batch_size, max_sequence_length=max_seq_length)
            prompt_encodings.update({
                'negative_prompt_embeds': negative_prompt_embeds, 
                'negative_prompt_embeds_mask': negative_prompt_embeds_mask
            })
        
        torch.cuda.empty_cache()
        return prompt_encodings

    @torch.inference_mode()
    def _batched_imgs2imgs(self, prompt_encodings:dict[str, torch.Tensor], images:list[Image.Image]|torch.Tensor, batch_size:int, steps:int, strength:float, guidance_scale:float, img_wh:tuple[int, int], seed:int, **kwargs):
        #batched_prompts = rebatch_prompt_embs(prompts_encodings, batch_size)
        batched_prompts = {k: torch.split(v, batch_size) for k,v in prompt_encodings.items()}
        batched_images = torch.split(images, batch_size) if isinstance(images, torch.Tensor) else batched(images, batch_size)
        w,h = img_wh # need wh in case `images` is a batch of latents
        
        
        for i,imbatch in enumerate(batched_images):
            #batch_len = 
            embed_batch = {k: v[i] for k,v in batched_prompts.items()}
            # shared common seed helps SIGNIFICANTLY with cohesion
            # list of generators is very important. Otherwise it does not apply correctly
            generators = generator_batch(seed, len(imbatch), device='cuda')

            latents = self.i2i(image=imbatch, num_inference_steps=steps, strength=strength, guidance_scale=guidance_scale, height=h, width=w, **embed_batch, generator=generators, output_type='latent', **kwargs).images 

            yield latents

    @torch.inference_mode()
    def regenerate_frames(self, prompt: str, frame_array: np.ndarray, 
                          steps: int = None, 
                          astrength: float = 0.5, 
                          imsize: typing.Literal['tiny','small','med','full'] = 'small', 

                          negative_prompt: str = None, 
                          guidance_scale: float = None, 
                          two_stage: bool = False, 
                          aseed: int = None, 
                          **kwargs):   
        
       
        # NOTE: special behavior since having a seed improves results substantially 
        if aseed is None: 
            aseed = np.random.randint(1e9, 1e10-1)
        elif aseed < 0:
            aseed = None
            

        fkwg = self.config.get_if_none(steps=steps, strength=astrength, negative_prompt=negative_prompt, guidance_scale=guidance_scale)
        steps = fkwg['steps']
        strength = fkwg['strength']
        negative_prompt=fkwg['negative_prompt']
        guidance_scale=fkwg['guidance_scale']

        # astrength = np.clip(astrength, 1/steps, 1) # clip strength so at least 1 step occurs
        #astrength = max(astrength*steps, 1)/steps 
        logger.debug(f'unused_kwargs: {kwargs} | fkwg:{fkwg}')
                
        dim_choices,_ = self.batch_settings(imsize)
        bsz = 1
        resized_images = self._resize_image_frames(frame_array, dim_choices, max_num_frames=30, upsample_px_thresh=256, upsample_bsz=8) # upsample first if less 256^2 pixels 
        dim_out = resized_images[0].size
        
        prompt, negative_prompt = self.preprocess_prompts(prompt, negative_prompt)
        
        #condition_images = np.stack(self.preprocess_image(resized_images), 0) 
        #prompts_encodings = self.embed_prompts(prompt, negative_prompt=negative_prompt, batch_size=1, lora_scale=lora_scale, clip_skip=self.clip_skip, partial_offload=True, image=condition_images)
        
        # First, yield the total number of frames since it may have changed from slice
        yield len(resized_images)
        #batch_iter = self._batched_imgs2imgs(prompts_encodings, resized_images, bsz, steps, strength=strength, guidance_scale=guidance_scale, img_wh=dim_out, seed=aseed, **kwargs)
        #image_prev = self._pipe_img2img(prompts_encodings, img_init, num_inference_steps=steps, strength=strength, guidance_scale=guidance_scale, seed=aseed, )
        
        t0 = time.perf_counter()
        latents = []

        
        latents = []
        w,h = dim_out
        # image_frames = [image_prev]
        # # prompt = prompt + '. Continue on from image 2 though it was the previous frame in an animated GIF.' # '. Use image 2 as the starting point for the transformation.'
        prompt_encs = []

        for image_next in resized_images:
            # image_next = resized_images[i]
            prompt_encs.append(self.embed_prompts(prompt, negative_prompt=negative_prompt, batch_size=1, image=image_next))
        
        for img,embeds in zip(resized_images, prompt_encs):
            generator = torch.Generator('cpu').manual_seed(aseed)
            #img_input = image_next # [image_next, image_prev]
            # prompts_encodings = self.embed_prompts(prompt, negative_prompt=negative_prompt, batch_size=bsz, lora_scale=lora_scale, clip_skip=self.clip_skip, partial_offload=True, image=image_next)
            lat = self.i2i(image=img, **embeds, num_inference_steps=steps, guidance_scale=guidance_scale, num_images_per_prompt=1, 
                           generator=generator, height=h, width=w, max_sequence_length=self.max_seq_len, output_type='latent',).images
            # lat = self._pipe_img2img(prompts_encodings, image_next, num_inference_steps=steps, strength=strength, guidance_scale=guidance_scale, seed=aseed)
            latents.append(lat)
            yield bsz
         
        # latents = torch.cat(latents, 0)
        
        image_frames = self.decode_latents(latents, height=h, width=w)
        yield image_frames

        te = time.perf_counter()
        runtime = te-t0
        logger.info(f'RUN TIME: {runtime:0.2f}s | BSZ: {bsz} | DIM: {dim_out} | N_IMAGE: {len(resized_images)} | IMG/SEC: {len(resized_images)/runtime:0.2f}')
        release_memory()
    
    @torch.inference_mode()
    def generate_frames(self, prompt: str, 
                            image: Image.Image|None = None, 
                            nframes: int = 11,
                            steps: int = None, 
                            strength_end: float = 0.80, 
                            strength_start: float = 0.30, 
                            negative_prompt: str = None, 
                            guidance_scale: float = None, 
                            aspect: typing.Literal['square','portrait','landscape'] = None, 
                            mid_frames:int=0,
                            seed: int = None, 
                            **kwargs): 
        
        gseed = torch.Generator(device='cpu').manual_seed(seed) if seed is not None else None
        
        fkwg = self.config.get_if_none(steps=steps, negative_prompt=negative_prompt, guidance_scale=guidance_scale, aspect=aspect)
        
        negative_prompt=fkwg['negative_prompt']
        guidance_scale=fkwg['guidance_scale']
        steps = fkwg['steps']
        aspect = fkwg['aspect']

        logger.debug(f'unused_kwargs: {kwargs} | fkwg:{fkwg}')
        
        prompt, negative_prompt = self.preprocess_prompts(prompt=prompt, negative_prompt=negative_prompt)

        image_frames = []

        if image is None:
            prompt_encodings = self.embed_prompts(prompt, negative_prompt=negative_prompt, image=None)
            w,h = self.config.get_dims(aspect)
            image = self.t2i(num_inference_steps=steps, guidance_scale=guidance_scale, height=h, width=w, **prompt_encodings, output_type='pil', generator=gseed).images
        else:
            image = self._resize_image(image, aspect=None, dim_choices=None)
            w,h = image[0].size

        image_frames.extend(image)

        # subtract 1 for first image
        nframes = nframes - 1
        
        yield -1
        
        for i in range(nframes):
            prompt_encodings = self.embed_prompts(prompt, negative_prompt=negative_prompt, image=image)
            image = self.i2i(image=image, num_inference_steps=steps, guidance_scale=guidance_scale, height=h, width=w, **prompt_encodings, output_type='pil', generator=gseed).images
            image_frames.extend(image)
            yield i

        if isinstance(image_frames[-1], torch.Tensor):    
            image_frames = self.decode_latents(image_frames, height=h, width=w, )
        
        prompt_encodings = release_memory(prompt_encodings)
        
        if mid_frames:
            image_frames = self.interpolate(image_frames, inter_frames=mid_frames, batch_size=2)
            
        yield image_frames
#endregion

#region Z-Image
class ZImageBase(LatentOptPipeline):
    def __init__(self, model_name: str, model_path: str, config: DiffusionConfig, offload=False, scheduler_setup: str | tuple[str, dict] = None, dtype: torch.dtype = torch.bfloat16, init_loras: list[tuple[str,float]] = None,
                 rank = 256):
        super().__init__(model_name, model_path, config, offload, scheduler_setup, dtype, init_loras, root_name='zimg')
        self.rank = rank
        self.max_seq_len = 512
    
    @torch.inference_mode()
    def load_pipeline(self):
        pipeline_quant_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16, 'bnb_4bit_use_double_quant': False},
            components_to_quantize=["text_encoder", ], #"transformer",],
        )
        
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            "unsloth/Qwen3-4B-unsloth-bnb-4bit",
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            low_cpu_mem_usage = True,
            # quantization_config = TransBitsAndBytesConfig(**pipeline_quant_config.quant_kwargs, ),
        )
        
        text_encoder: Qwen3ForCausalLM = FastLanguageModel.for_inference(text_encoder)
        text_encoder = text_encoder.eval().requires_grad_(False)
        text_encoder.is_loaded_in_8bit = False # unsloth loaded 4bit models set this=True which prevents offloading : ValueError: `.to` is not supported for `8-bit` bitsandbytes models.
        
        transformer_model_path = f"nunchaku-ai/nunchaku-z-image-turbo/svdq-{get_precision()}_r{self.rank}-z-image-turbo.safetensors"
        transformer = NunchakuZImageTransformer2DModel.from_pretrained(transformer_model_path, torch_dtype=torch.bfloat16)
        transformer = transformer.eval().requires_grad_(False)
            
        self.base: ZImagePipeline =  ZImagePipeline.from_pretrained(self.model_path, transformer=transformer, text_encoder=text_encoder, torch_dtype=self.dtype, low_cpu_mem_usage = False,)
        
        if not self.offload:
            self.base = self.base.to(0)

        self._scheduler_init()
        self.load_loras()

        self.basei2i: ZImageImg2ImgPipeline = ZImageImg2ImgPipeline.from_pipe(self.base, transformer=self.base.transformer, text_encoder=self.base.text_encoder, torch_dtype=self.dtype)
        
        for pipe in (self.base, self.basei2i):
            pipe.enable_xformers_memory_efficient_attention()
            # pipe.transformer.set_attention_backend("xformers") # "flash"
            # pipe.text_encoder.set_attn_implementation('flash_attention_2')
            pipe.vae.enable_slicing()
            if self.offload:
                pipe.enable_model_cpu_offload()
        
        self.is_ready = True

    def batch_settings(self, imsize: typing.Literal['tiny','small','med','full'] = 'small', ):
        dims_opts = {
            'tiny': [(768,768), (640,896), (896,640)], # 1.4
            'small': [(768,768), (640,896), (896,640)], # 1.4
            'med': [(768,768), (640,896), (896,640)], # 1.4
            'full': [(1024,1024), (832, 1216), (1216, 832)] # 1.46
        }
        
        batch_sizes = {'tiny': 1, 'small': 1, 'med': 1, 'full': 1} # Batch > 1 crashes with "Assertion `rotary_emb.shape[0] * rotary_emb.shape[1] == M'"

        return (dims_opts[imsize], batch_sizes[imsize])
    


    @torch.inference_mode()
    def embed_prompts(self, prompt:str, negative_prompt:str = None, batch_size: int = 1, **kwargs):
        prompt = [prompt]
        if negative_prompt is not None:
            negative_prompt = [negative_prompt]

        device = torch.device('cuda')
        prompt_embeds, negative_prompt_embeds = self.base.encode_prompt(prompt=prompt, device=device, negative_prompt=negative_prompt, max_sequence_length=self.max_seq_len,)#num_images_per_prompt=batch_size,
        
        prompt_embeds = [pe for pe in prompt_embeds for _ in range(batch_size)]
        if negative_prompt_embeds:
            negative_prompt_embeds = [npe for npe in negative_prompt_embeds for _ in range(batch_size)]
        
        prompt_encodings = dict(prompt_embeds=torch.stack(prompt_embeds, 0), negative_prompt_embeds=torch.stack(negative_prompt_embeds, 0))
        torch.cuda.empty_cache()
        return prompt_encodings
    
    @torch.inference_mode()
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = torch.cat([
                retrieve_latents(self.basei2i.vae.encode(image[i : i + 1]), generator=generator[i]) 
                for i in range(image.shape[0])], dim = 0)
        else:
            image_latents = retrieve_latents(self.basei2i.vae.encode(image), generator=generator)

        # Apply scaling (inverse of decoding: decode does latents/scaling_factor + shift_factor)
        image_latents = (image_latents - self.basei2i.vae.config.shift_factor) * self.basei2i.vae.config.scaling_factor
        return image_latents

    @torch.inference_mode()
    def encode_images(self, images:list[Image.Image], seed:int=42):
        nframes = len(images)
        w,h = images[0].size
        
        generator = generator_batch(seed, nframes, 'cuda')
        device = torch.device('cuda')

        batch_size = 1 * nframes # actual_batch_size = batch_size * nframes

        image = self.basei2i.image_processor.preprocess(images, height=h, width=w).to(device=device, dtype=self.dtype)
        height,width = image.shape[-2:]
        # Encode the input image
        if image.shape[1] == self.basei2i.transformer.in_channels:
            image_latents = image
        else:
            image_latents = self._encode_vae_image(image, generator)

        # Handle batch size expansion
        if batch_size > image_latents.shape[0]: 
            if batch_size % image_latents.shape[0] == 0:
                image_latents = torch.cat([image_latents] * (batch_size // image_latents.shape[0]), dim=0) # additional_image_per_prompt = batch_size//image_latents.shape[0]
            else:
                raise ValueError(f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts.")
        
        image=release_memory(image)
        return image_latents
    
    @torch.inference_mode()
    def decode_latents(self, latents, **kwargs):
        # https://github.com/huggingface/diffusers/blob/f6b6a7181eb44f0120b29cd897c129275f366c2a/src/diffusers/pipelines/z_image/pipeline_z_image_img2img.py#L696
        if isinstance(latents, list):
            latents = torch.cat(latents, dim=0)
        
        latents = latents.to(self.basei2i.vae.dtype)
        latents = (latents / self.basei2i.vae.config.scaling_factor) + self.basei2i.vae.config.shift_factor
        image = self.basei2i.vae.decode(latents, return_dict=False)[0]
        image = self.basei2i.image_processor.postprocess(image, output_type='pil')
        
        return image
#endregion