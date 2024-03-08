import os
import gc
import re
import io
import math
import time
import typing

from dataclasses import dataclass, field



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
class DiffusionConfig:
    # model_name: str
    # model_path: str
    steps: int
    i2i_steps: int 
    guidance_scale: float
    img_dims: tuple[int]

    strength: float = 0.55
    negative_prompt: str = None
    denoise_blend: float = None
    refine_strength: float = 0.3
    refine_guidance_scale: float = None

    locked: list[str] = field(default_factory=list, kw_only=True)

    def lock(self, *args):
        self.locked += args

    def get_if_none(self, **kwargs):
        '''Fill with default config value if arg is None'''
        filled_kwargs = {}
        for k,v in kwargs.items():
            if v is None or k in self.locked:
                v = getattr(self, k)
            filled_kwargs[k] = v
        return filled_kwargs
        #return {k: (getattr(self, k) if (v is None or k in self.locked) else v) for k,v in kwargs.items()}
        

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
    def pipeline(self, prompt, num_inference_steps, negative_prompt=None, guidance_scale=None, strength=None, image=None):
        if image is None:
            return self.base(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, negative_prompt=negative_prompt).images[0]
            #imgs = self.base(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, negative_prompt=negative_prompt, num_images_per_prompt=1, **kwargs).images#[0]
            #return make_image_grid(imgs, 2, 2)
        if strength>0:  
            if num_inference_steps*strength < 1:
                steps = math.ceil(1/strength)
                print(f'steps ({num_inference_steps}) too low, forcing to {steps}')
                num_inference_steps = steps
                
            out_image = self.basei2i(prompt=prompt, image=image, num_inference_steps=num_inference_steps, strength=strength, guidance_scale=guidance_scale).images[0]
        else:
            out_image=image
        return out_image
    
    @async_wrap_thread
    def generate_image(self, prompt:str, steps:int=None, negative_prompt:str=None, guidance_scale:float=None, **kwargs) -> Image.Image:
        fkwg = self.config.get_if_none(steps=steps, guidance_scale=guidance_scale, negative_prompt=negative_prompt)
        print(kwargs)
        print('fkwg:', fkwg)
        image = self.pipeline(prompt=prompt, num_inference_steps=fkwg['steps'], negative_prompt=fkwg['negative_prompt'], guidance_scale=fkwg['guidance_scale'])
        return image

    @async_wrap_thread
    def regenerate_image(self, image:Image.Image, prompt:str, steps:int=None, strength:float=None, negative_prompt:str=None, guidance_scale:float=None, **kwargs) -> Image.Image:
        fkwg = self.config.get_if_none(i2i_steps=steps, guidance_scale=guidance_scale, negative_prompt=negative_prompt, strength=strength)
        print('regenerate_image input size:',image.size)
        print(kwargs)
        print('fkwg:', fkwg)
        strength = cmd_tfms.percent_transform(fkwg['strength'])
        out_image = self.pipeline(prompt=prompt, num_inference_steps=fkwg['i2i_steps'], negative_prompt=fkwg['negative_prompt'], strength=strength, guidance_scale=fkwg['guidance_scale'], image=image.resize(self.config.img_dims))
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
    def pipeline(self, prompt, num_inference_steps, negative_prompt, guidance_scale, denoise_blend, refine_strength, strength=None, image=None):
        output_type = 'latent'
        if denoise_blend is None:
            output_type = 'pil'
        elif denoise_blend == 1:
            denoise_blend = None
        
        t0 = time.perf_counter()
        if image is None:
            image = self.base(prompt=prompt, num_inference_steps=num_inference_steps, denoising_end=denoise_blend, 
                                guidance_scale=guidance_scale, negative_prompt=negative_prompt, output_type=output_type).images[0]
            
        elif strength*num_inference_steps >= 1:
            image = self.basei2i(prompt=prompt, num_inference_steps=num_inference_steps, denoising_end=denoise_blend, 
                                guidance_scale=guidance_scale, negative_prompt=negative_prompt, image=image, strength=strength, output_type=output_type).images[0]
        
        t_base = time.perf_counter()
        if refine_strength*num_inference_steps >= 1:
            refine_guidance_scale = self.config.refine_guidance_scale
            if refine_guidance_scale is None:
                refine_guidance_scale = guidance_scale*1.5
            image = self.refiner(prompt=prompt, num_inference_steps=num_inference_steps, denoising_start=denoise_blend,
                                    guidance_scale=refine_guidance_scale, negative_prompt=negative_prompt, image=[image], strength=refine_strength).images[0]
        
        t_refine = time.perf_counter()
        print(f'ImageGen Time - total: {t_refine-t0:0.2f}s | base: {t_base-t0:0.2f}s | refine: {t_refine-t_base:0.2f}s' )
        return image

    @async_wrap_thread
    def generate_image(self, prompt:str, steps:int=None, negative_prompt:str=None, guidance_scale:float=None, denoise_blend:float|None=None, refine_strength:float=None, **kwargs) -> Image.Image:
        print(kwargs)
        fkwg = self.config.get_if_none(steps=steps, negative_prompt=negative_prompt, guidance_scale=guidance_scale, denoise_blend=denoise_blend, refine_strength=refine_strength)
        print('fkwg:', fkwg)
        
        denoise_blend = cmd_tfms.percent_transform(fkwg['denoise_blend'])
        refine_strength = cmd_tfms.percent_transform(fkwg['refine_strength'])
        
        image = self.pipeline(prompt=prompt, num_inference_steps=fkwg['steps'], negative_prompt=fkwg['negative_prompt'], guidance_scale=fkwg['guidance_scale'], denoise_blend=denoise_blend, refine_strength=refine_strength)
        return image

    @async_wrap_thread
    def regenerate_image(self, image:Image.Image, prompt:str, steps:int=None, strength:float=None, negative_prompt:str=None, guidance_scale:float=None, denoise_blend:float|None=None, refine_strength:float=None, **kwargs) -> Image.Image:
        print('regenerate_image input size:',image.size)
        
        print(kwargs)
        fkwg = self.config.get_if_none(i2i_steps=steps, strength=strength, guidance_scale=guidance_scale, negative_prompt=negative_prompt, denoise_blend=denoise_blend, refine_strength=refine_strength)
        print('fkwg:', fkwg)
        
        strength = cmd_tfms.percent_transform(fkwg['strength'])
        denoise_blend = cmd_tfms.percent_transform(fkwg['denoise_blend'])
        refine_strength = cmd_tfms.percent_transform(fkwg['refine_strength'])
        
        image = self.pipeline(prompt=prompt, num_inference_steps=fkwg['i2i_steps'], negative_prompt=fkwg['negative_prompt'], guidance_scale=fkwg['guidance_scale'], 
                              denoise_blend=denoise_blend, refine_strength=refine_strength, strength=strength, image=image.resize(self.config.img_dims))
        return image


# https://huggingface.co/ByteDance/SDXL-Lightning
# https://civitai.com/models/133005?modelVersionId=357609
    
class SDXLTurboManager(OneStageImageGenManager):
    def __init__(self):
        super().__init__(
            model_name='sdxl_turbo', 
            model_path="stabilityai/sdxl-turbo",
            config=DiffusionConfig(
                steps = 2, # 1-4 steps
                i2i_steps = 4,
                guidance_scale = 0.0,
                img_dims = (512,512),
                locked=['guidance_scale', 'negative_prompt', 'denoise_blend', 'refine_strength']
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
                steps = 40,
                i2i_steps = 50,
                guidance_scale = 7.0,
                img_dims = (1024,1024),
                refine_strength=0.3,
            ),
            offload=True
        )

class DreamShaperXLManager(OneStageImageGenManager):
    def __init__(self):
        super().__init__(
            model_name = 'dreamshaper_turbo', # bf16 saves ~3gb vram over fp16
            model_path = 'lykon/dreamshaper-xl-v2-turbo', # https://civitai.com/models/112902?modelVersionId=333449
            config = DiffusionConfig(
                steps = 8, # 4-8 steps
                i2i_steps = 8,
                guidance_scale = 2.0, # cfg=2.0, 
                img_dims = (1024,1024),
                locked=['guidance_scale', 'denoise_blend', 'refine_strength']
            ),
            offload=True,
        )

    #@async_wrap_thread
    async def load_pipeline(self):
        def lazy_scheduler():
            return DPMSolverMultistepScheduler.from_config(self.base.scheduler.config)
        await super().load_pipeline(lazy_scheduler)

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
}

