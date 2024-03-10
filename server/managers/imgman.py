import os
import gc
import re
import io
import math
import time
import typing
from dataclasses import dataclass, field, asdict, KW_ONLY, InitVar

import discord
from discord.ext import commands

import numpy as np
import torch
from diffusers import (
    AutoPipelineForText2Image, 
    AutoPipelineForImage2Image, 
    DiffusionPipeline, 
    StableDiffusionXLPipeline, 
    StableDiffusionXLImg2ImgPipeline, 
    DPMSolverMultistepScheduler, 
    DPMSolverSinglestepScheduler,
    UNet2DConditionModel, 
    EulerDiscreteScheduler
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from diffusers.utils import load_image, make_image_grid
from DeepCache import DeepCacheSDHelper
from PIL import Image

from cloneus import cpaths
import config.settings as settings
from cmds import flags as cmd_flags, transformers as cmd_tfms
from utils.globthread import async_wrap_thread, stop_global_thread
from views import redrawui

bot_logger = settings.logging.getLogger('bot')
model_logger = settings.logging.getLogger('model')
cmds_logger = settings.logging.getLogger('cmds')
event_logger = settings.logging.getLogger('event')

# Resource: https://github.com/CyberTimon/Stable-Diffusion-Discord-Bot/blob/main/bot.py

IMG_DIR = settings.SERVER_ROOT/'output'/'imgs'
PROMPT_FILE = IMG_DIR.joinpath('_prompts.txt')

torch.backends.cuda.matmul.allow_tf32 = True

def release_memory():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()

def torch_compile_flags(restore_defaults=False, verbose=False):
    # https://huggingface.co/docs/diffusers/tutorials/fast_diffusion#use-faster-kernels-with-torchcompile
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
    orient: typing.Literal['square','portrait'] = 'square'
    img_dims: tuple[int, int]|list[tuple[int, int]] = (1024, 1024)
    denoise_blend: float = None
    refine_strength: float = None
    refine_guidance_scale: float = None
    _ : KW_ONLY
    locked: InitVar[list[str]] = None
    
    def __post_init__(self, locked):
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
        
    
    def lock(self, *args):
        for k in args:
            getattr(self, f'{k}_').locked = True

    def to_dict(self):
        return {v.name: v for k,v in vars(self).items() if k.endswith('_')}#{'default':v.default, 'locked':v.locked, 'bounds':v.bounds} 

    def to_md(self):
        md_text = ""
        for k,v in self.to_dict().items():
            if v.default is None and v.locked:
                continue
            
            default = v.default
            lb, ub = 0, 0
            if v.bounds is not None:
                lb, ub = v.bounds
            # These values are displayed as 0-100 for end-user simplicity
            if k in ['strength', 'refine_strength', 'stage_mix']:
                default = int(default*100)
                lb = int(lb*100)
                ub = int(ub*100)
            
            postfix = ''
            if v.locked:
                postfix = '🔒' 
            elif v.bounds is not None:
                postfix = f'*(common: {lb} - {ub})*'
            
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
        
    def nearest_dims(self, img_wh):
        dim_out = self.img_dims
        
        if isinstance(self.img_dims, list):
            w_in, h_in = img_wh
            ar_in = w_in/h_in
            dim_out = min(self.img_dims, key=lambda wh: abs(ar_in - wh[0]/wh[1]))
        
        return dim_out
    
    def get_dims(self, orient:typing.Literal['square','portrait']=None):
        if orient is None:
            orient = self.orient
        
        dim_out = self.img_dims
        
        if isinstance(self.img_dims, list):
            if orient == 'square':
                dim_out = min(self.img_dims, key=lambda wh: abs(1 - wh[0]/wh[1]))
            elif orient == 'portrait':
                dim_out = min(self.img_dims, key=lambda wh: wh[0]/wh[1])

        return dim_out

class OneStageImageGenManager:
    def __init__(self, model_name: str, model_path:str, config:DiffusionConfig, offload=False):
        self.is_compiled = False
        self.is_ready = False
        self.dc_enabled = None
        
        self.model_name = model_name
        self.model_path = model_path
        self.config = config
        self.offload = offload
    

    @async_wrap_thread
    def load_pipeline(self, scheduler_callback:typing.Callable=None):
        self.base: StableDiffusionXLPipeline = AutoPipelineForText2Image.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16, variant="fp16", use_safetensors=True, add_watermarker=False,
        )
        if scheduler_callback is not None:
            self.base.scheduler = scheduler_callback()
        if self.offload:
            self.base.enable_model_cpu_offload()
        else:
            self.base = self.base.to("cuda")
            
        self.basei2i = AutoPipelineForImage2Image.from_pipe(self.base, torch_dtype=torch.bfloat16, use_safetensors=True, add_watermarker=False)
        if not self.offload:
            self.basei2i = self.basei2i.to("cuda")
       
        self.is_ready = True
    
    def _setup_dc(self):
        # https://github.com/horseee/DeepCache/blob/master/DeepCache/extension/deepcache.py
        self.dc_base = DeepCacheSDHelper(self.base)
        self.dc_base.set_params(cache_interval=3, cache_branch_id=0)
        
        self.dc_basei2i = DeepCacheSDHelper(self.basei2i)
        self.dc_basei2i.set_params(cache_interval=3, cache_branch_id=0)
        self.dc_enabled = False
        
    def dc_fastmode(self, enable:bool, img2img=False):
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
    def pipeline(self, prompt, num_inference_steps, negative_prompt=None, guidance_scale=None, strength=None, image=None, target_size=None):
        if image is None:
            #h,w is all you need -- https://github.com/huggingface/diffusers/blob/v0.26.3/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L1075
            h,w = target_size
            return self.base(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, negative_prompt=negative_prompt, height=h, width=w).images[0]
            #imgs = self.base(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, negative_prompt=negative_prompt, num_images_per_prompt=1, **kwargs).images#[0]
            #return make_image_grid(imgs, 2, 2)
        if strength>0:  
            if num_inference_steps*strength < 1:
                steps = math.ceil(1/strength)
                print(f'steps ({num_inference_steps}) too low, forcing to {steps}')
                num_inference_steps = steps
                
            out_image = self.basei2i(prompt=prompt, image=image, num_inference_steps=num_inference_steps, strength=strength, guidance_scale=guidance_scale, target_size=target_size).images[0]
        else:
            out_image=image
        return out_image
    
    @async_wrap_thread
    def generate_image(self, prompt:str, steps:int=None, negative_prompt:str=None, guidance_scale:float=None, orient:typing.Literal['square','portrait']=None, **kwargs) -> Image.Image:
        fkwg = self.config.get_if_none(steps=steps, guidance_scale=guidance_scale, negative_prompt=negative_prompt)
        print(f'kwargs: {kwargs}\nfkwg:{fkwg}')
        dim_out = self.config.get_dims(orient)
        target_size = (dim_out[1], dim_out[0]) # h,w
            
        image = self.pipeline(prompt=prompt, num_inference_steps=fkwg['steps'], negative_prompt=fkwg['negative_prompt'], guidance_scale=fkwg['guidance_scale'], target_size=target_size)
        return image

    @async_wrap_thread
    def regenerate_image(self, image:Image.Image, prompt:str, steps:int=None, strength:float=None, negative_prompt:str=None, guidance_scale:float=None, **kwargs) -> Image.Image:
        fkwg = self.config.get_if_none(steps=steps, guidance_scale=guidance_scale, negative_prompt=negative_prompt, strength=strength)
        print(f'kwargs: {kwargs}\nfkwg:{fkwg}')
        
        dim_out = self.config.nearest_dims(image.size)
        print('regenerate_image input size:', image.size, '->', dim_out)
        image = image.resize(dim_out)
        # target_size defaults img_size 
        # - https://github.com/huggingface/diffusers/blob/v0.26.3/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py#L1325
        strength = cmd_tfms.percent_transform(fkwg['strength'])
        out_image = self.pipeline(prompt=prompt, num_inference_steps=fkwg['steps'], negative_prompt=fkwg['negative_prompt'], strength=strength, guidance_scale=fkwg['guidance_scale'], image=image)

        #self.base.disable_vae_tiling()
        return out_image    


class TwoStageImageGenManager:
    def __init__(self, model_name: str, model_path:str, refiner_path:str, config:DiffusionConfig, offload=False):
        self.is_ready = False
        self.is_compiled = False
        self.dc_enabled = None
        
        self.model_name = model_name
        self.model_path = model_path
        self.refiner_path = refiner_path
        self.config = config
        self.offload = offload
        # https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output
    
    @async_wrap_thread
    def load_pipeline(self, scheduler_callback:typing.Callable=None):
        self.base: StableDiffusionXLPipeline = DiffusionPipeline.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16, variant="fp16", use_safetensors=True, add_watermarker=False,
        )
        if scheduler_callback is not None:
            self.base.scheduler = scheduler_callback()
        
        self.refiner: StableDiffusionXLImg2ImgPipeline = DiffusionPipeline.from_pretrained(
            self.refiner_path, text_encoder_2=self.base.text_encoder_2, vae=self.base.vae,
            torch_dtype=torch.bfloat16, variant="fp16", use_safetensors=True, add_watermarker=False,
        )
        if self.offload:
            self.base.enable_model_cpu_offload()
            self.refiner.enable_model_cpu_offload()
        else:
            self.base = self.base.to("cuda")
            self.refiner = self.refiner.to("cuda")

        self.basei2i = AutoPipelineForImage2Image.from_pipe(self.base, torch_dtype=torch.bfloat16, use_safetensors=True, add_watermarker=False)
        if not self.offload:
            self.basei2i = self.basei2i.to("cuda")
        
        self.is_ready=True

    def _setup_dc(self):
        # https://github.com/horseee/DeepCache/blob/master/DeepCache/extension/deepcache.py
        self.dc_base = DeepCacheSDHelper(self.base)
        self.dc_base.set_params(cache_interval=3, cache_branch_id=0)
        
        self.dc_basei2i = DeepCacheSDHelper(self.basei2i)
        self.dc_basei2i.set_params(cache_interval=3, cache_branch_id=0)

        self.dc_refiner = DeepCacheSDHelper(self.refiner)
        self.dc_refiner.set_params(cache_interval=3, cache_branch_id=0)
        self.dc_enabled = False
        
    def dc_fastmode(self, enable:bool, img2img=False):
        if self.dc_enabled is None:
            self._setup_dc()
        if enable != self.dc_enabled: 
            if enable:
                (self.dc_basei2i if img2img else self.dc_base).enable()
                self.dc_refiner.enable()
            else:
                (self.dc_basei2i if img2img else self.dc_base).disable() 
                self.dc_refiner.disable()
            
            self.dc_enabled=enable 


    @async_wrap_thread
    def unload_pipeline(self):
        self.base = None
        self.basei2i = None
        self.refiner = None

        release_memory()
        self.is_ready=False

        if self.is_compiled:
            torch._dynamo.reset() # @async_wrap_thread might be needed since compile_pipeline uses it
            # https://github.com/pytorch/pytorch/blob/main/torch/_inductor/cudagraph_trees.py#L286
            # uses local thread to store vars? So, can't reset on a different thread
            torch_compile_flags(restore_defaults=True)
            self.is_compiled = False

    @async_wrap_thread
    def compile_pipeline(self):
        torch_compile_flags()
        # https://huggingface.co/docs/diffusers/tutorials/fast_diffusion#use-faster-kernels-with-torchcompile
        self.base.unet.to(memory_format=torch.channels_last)
        self.base.vae.to(memory_format=torch.channels_last)
        self.refiner.unet.to(memory_format=torch.channels_last)
        self.refiner.vae.to(memory_format=torch.channels_last)

        self.base.unet = torch.compile(self.base.unet, mode="max-autotune", fullgraph=True)
        #self.base.vae.decode = torch.compile(self.base.vae.decode, mode="max-autotune", fullgraph=True)

        self.refiner.unet = torch.compile(self.refiner.unet, mode="max-autotune", fullgraph=True,)
        #self.base.unet = torch.compile(self.base.unet, mode="reduce-overhead", fullgraph=True)
        #self.refiner.unet = torch.compile(self.refiner.unet, mode="reduce-overhead", fullgraph=True)
        _ = self.pipeline(prompt='Cat', num_inference_steps=10, negative_prompt=None, guidance_scale=self.config.guidance_scale, denoise_blend=None, refine_strength=self.config.refine_strength, image=None,)
        self.is_compiled = True
        release_memory()

    @torch.inference_mode()
    def pipeline(self, prompt, num_inference_steps, negative_prompt, guidance_scale, denoise_blend, refine_strength, strength=None, image=None, target_size=None):
        output_type = 'latent'
        if denoise_blend is None:
            output_type = 'pil'
        elif denoise_blend == 1:
            denoise_blend = None
        
        t0 = time.perf_counter()
        if image is None:
            h,w = target_size
            image = self.base(prompt=prompt, num_inference_steps=num_inference_steps, denoising_end=denoise_blend, 
                                guidance_scale=guidance_scale, negative_prompt=negative_prompt, output_type=output_type, height=h, width=w).images[0]
            
        elif strength*num_inference_steps >= 1:
            image = self.basei2i(prompt=prompt, num_inference_steps=num_inference_steps, denoising_end=denoise_blend, 
                                guidance_scale=guidance_scale, negative_prompt=negative_prompt, image=image, strength=strength, output_type=output_type).images[0]
        
        t_base = time.perf_counter()
        if refine_strength*num_inference_steps >= 1:
            if (refine_guidance_scale := self.config.refine_guidance_scale) is None:
                refine_guidance_scale = guidance_scale*1.5
            image = self.refiner(prompt=prompt, num_inference_steps=num_inference_steps, denoising_start=denoise_blend,
                                    guidance_scale=refine_guidance_scale, negative_prompt=negative_prompt, image=[image], strength=refine_strength).images[0]
        
        t_refine = time.perf_counter()
        print(f'ImageGen Time - total: {t_refine-t0:0.2f}s | base: {t_base-t0:0.2f}s | refine: {t_refine-t_base:0.2f}s' )
        return image

    @async_wrap_thread
    def generate_image(self, prompt:str, 
                       steps: int = None, 
                       negative_prompt: str = None, 
                       guidance_scale: float = None, 
                       orient: typing.Literal['square','portrait'] = None, 
                       denoise_blend: float|None = None, 
                       refine_strength: float = None, 
                       **kwargs) -> Image.Image:
        
        fkwg = self.config.get_if_none(steps=steps, negative_prompt=negative_prompt, guidance_scale=guidance_scale, denoise_blend=denoise_blend, refine_strength=refine_strength)
        print(f'kwargs: {kwargs}\nfkwg:{fkwg}')
        
        dim_out = self.config.get_dims(orient)
        target_size = (dim_out[1], dim_out[0]) # h,w

        denoise_blend = cmd_tfms.percent_transform(fkwg['denoise_blend'])
        refine_strength = cmd_tfms.percent_transform(fkwg['refine_strength'])
        
        image = self.pipeline(prompt=prompt, num_inference_steps=fkwg['steps'], negative_prompt=fkwg['negative_prompt'], guidance_scale=fkwg['guidance_scale'], 
                              denoise_blend=denoise_blend, refine_strength=refine_strength, target_size=target_size)
        return image

    @async_wrap_thread
    def regenerate_image(self, image:Image.Image, prompt:str, 
                         steps: int = None, 
                         strength: float = None, 
                         negative_prompt: str = None, 
                         guidance_scale: float = None, 
                         denoise_blend: float|None = None, 
                         refine_strength: float = None, 
                         **kwargs) -> Image.Image:
        
        fkwg = self.config.get_if_none(steps=steps, strength=strength, guidance_scale=guidance_scale, negative_prompt=negative_prompt, denoise_blend=denoise_blend, refine_strength=refine_strength)
        print(f'kwargs: {kwargs}\nfkwg:{fkwg}')

        dim_out = self.config.nearest_dims(image.size)
        print('regenerate_image input size:', image.size, '->', dim_out)
        image = image.resize(dim_out)
        
        strength = cmd_tfms.percent_transform(fkwg['strength'])
        denoise_blend = cmd_tfms.percent_transform(fkwg['denoise_blend'])
        refine_strength = cmd_tfms.percent_transform(fkwg['refine_strength'])
        
        image = self.pipeline(prompt=prompt, num_inference_steps=fkwg['steps'], negative_prompt=fkwg['negative_prompt'], guidance_scale=fkwg['guidance_scale'], 
                              denoise_blend=denoise_blend, refine_strength=refine_strength, strength=strength, image=image)
        return image


# https://huggingface.co/ByteDance/SDXL-Lightning
# https://civitai.com/models/133005?modelVersionId=357609
    
class SDXLTurboManager(OneStageImageGenManager):
    def __init__(self):
        super().__init__(
            model_name='sdxl_turbo', 
            model_path="stabilityai/sdxl-turbo",
            config=DiffusionConfig(
                steps = CfgItem(4, bounds=(1,4)), # 1-4 steps
                guidance_scale = CfgItem(0.0, locked=True),
                strength = CfgItem(0.55, bounds=(0.3, 0.9)),
                img_dims = (512,512),

                locked=['guidance_scale', 'negative_prompt', 'orient', 'denoise_blend', 'refine_strength', 'refine_guidance_scale']
            ), 
            offload=True)
        
    
    def dc_fastmode(self, enable:bool, img2img=False):
        pass

    @async_wrap_thread
    def compile_pipeline(self):
        # Currently, not working properly for SDXL Turbo + savings are negligible
        pass

class SDXLManager(TwoStageImageGenManager):
    def __init__(self):
        super().__init__(
            model_name = "sdxl",
            model_path = "stabilityai/stable-diffusion-xl-base-1.0",
            refiner_path="stabilityai/stable-diffusion-xl-refiner-1.0",
            config = DiffusionConfig(
                steps = CfgItem(50, bounds=(40,60)),
                guidance_scale = CfgItem(7.0, bounds=(5,15)), 
                strength = CfgItem(0.55, bounds=(0.3, 0.9)),
                orient = CfgItem('square', locked=True),
                img_dims = (1024,1024),
                denoise_blend= CfgItem(None, bounds=(0.7, 0.9)),
                refine_strength = CfgItem(0.3, bounds=(0.2, 0.5)),

                locked=['orient', 'refine_guidance_scale']
            ),
            
            offload=True
        )

class DreamShaperXLManager(OneStageImageGenManager):
    def __init__(self):
        super().__init__(
            model_name = 'dreamshaper_turbo', # bf16 saves ~3gb vram over fp16
            model_path = 'lykon/dreamshaper-xl-v2-turbo', # https://civitai.com/models/112902?modelVersionId=333449
            config = DiffusionConfig(
                steps = CfgItem(8, bounds=(4,8)),
                guidance_scale = CfgItem(2.0, locked=True),
                strength = CfgItem(0.55, bounds=(0.3, 0.9)),
                img_dims = [(1024,1024), (832,1216)],
                
                locked=['guidance_scale', 'denoise_blend', 'refine_strength', 'refine_guidance_scale']
            ),
            offload=True,
        )

    #@async_wrap_thread
    async def load_pipeline(self):
        def lazy_scheduler():
            return DPMSolverSinglestepScheduler.from_config(self.base.scheduler.config, use_karras_sigmas=True)
            #return DPMSolverMultistepScheduler.from_config(self.base.scheduler.config)
        await super().load_pipeline(lazy_scheduler)

class JuggernautXLLightningManager(OneStageImageGenManager):
    def __init__(self):
        super().__init__( # https://huggingface.co/RunDiffusion/Juggernaut-XL-Lightning
            model_name = 'juggernaut_lightning', # https://civitai.com/models/133005/juggernaut-xl?modelVersionId=357609
            model_path = 'https://huggingface.co/RunDiffusion/Juggernaut-XL-Lightning/blob/main/Juggernaut_RunDiffusionPhoto2_Lightning_4Steps.safetensors', 
            
            config = DiffusionConfig(
                steps = CfgItem(6, bounds=(5,7)), # 4-6 step /  5 and 7
                guidance_scale = CfgItem(1.5, bounds=(1.5,2.0)),
                strength = CfgItem(0.6, bounds=(0.3, 0.9)),
                img_dims = [(1024,1024), (832,1216)],

                locked = ['denoise_blend', 'refine_strength', 'refine_guidance_scale']
            ),
            offload=True,
        )
    
    @async_wrap_thread
    def load_pipeline(self):
        self.base = StableDiffusionXLPipeline.from_single_file(
            self.model_path, torch_dtype=torch.bfloat16, variant="fp16", use_safetensors=True, add_watermarker=False,
        )
        # https://huggingface.co/docs/diffusers/v0.26.3/en/api/schedulers/overview#schedulers
        # DPM++ SDE == DPMSolverSinglestepScheduler
        # DPM++ SDE Karras == DPMSolverSinglestepScheduler(use_karras_sigmas=True)
        self.base.scheduler = DPMSolverSinglestepScheduler.from_config(self.base.scheduler.config, use_karras_sigmas=False)
        #self.base.scheduler = DPMSolverMultistepScheduler.from_config(self.base.scheduler.config, use_karras_sigmas=False, algorithm_type='sde-dpmsolver++', solver_order=2)
        # use_karras_sigmas=True, euler_at_final=True --- https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_xl
        if self.offload:
            self.base.enable_model_cpu_offload()
        else:
            self.base = self.base.to("cuda")
            
        self.basei2i = AutoPipelineForImage2Image.from_pipe(self.base, torch_dtype=torch.bfloat16, use_safetensors=True, add_watermarker=False)
        if not self.offload:
            self.basei2i = self.basei2i.to("cuda")
       
        self.is_ready = True

AVAILABLE_MODELS = {
    'sdxl_turbo': {
        'manager': SDXLTurboManager(),
        'desc': 'Turbo SDXL (fast)'
    },
    'sdxl': {
        'manager': SDXLManager(),
        'desc':'SDXL (big)'
    },
    'dreamshaper_turbo': {
        'manager': DreamShaperXLManager(),
        'desc': 'DreamShaper XL2 Turbo (Med)'
    },
    'juggernaut_lightning': {
        'manager': JuggernautXLLightningManager(),
        'desc': 'Juggernaut XL Lightning'
    },
}

