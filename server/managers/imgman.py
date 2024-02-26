import os
import gc
import re
import io
import math
import time
import datetime

import functools
from pathlib import Path

import typing

import discord
from discord.ext import commands

import numpy as np
import torch
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image, DiffusionPipeline, StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler
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

def torch_compile_flags(restore_defaults=False):
    # https://huggingface.co/docs/diffusers/tutorials/fast_diffusion#use-faster-kernels-with-torchcompile
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



class BaseImageGenManager:
    def __init__(self):
        self.is_compiled = False
        self.is_ready = False
        self.model_name = None
    
    @async_wrap_thread
    def load_pipeline(self):
        self.base: StableDiffusionXLPipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", 
            torch_dtype=torch.bfloat16,
            variant="fp16", 
            use_safetensors=True, 
            add_watermarker=False, 
            #device_map="auto",
        ).to("cuda")
        
        self.is_ready = True

    @async_wrap_thread
    def unload_pipeline(self):
        self.base = None
        
        release_memory()
        torch_compile_flags(restore_defaults=True)
        torch._dynamo.reset()
        self.is_ready = False
        self.is_compiled = False

    @async_wrap_thread
    def compile_pipeline(self):
        torch_compile_flags()
        self.base.unet = torch.compile(self.base.unet, mode="max-autotune", fullgraph=True)
        #self.base.unet = torch.compile(self.base.unet, mode="reduce-overhead", fullgraph=True)
        self.base.upcast_vae()
        _ = self.pipeline(prompt='Cat')
        self.is_compiled = True
    
    @torch.inference_mode()
    def pipeline(self, prompt, num_inference_steps=2, **kwargs):
        image = self.base(prompt=prompt, guidance_scale=0.0, num_inference_steps=num_inference_steps, negative_prompt=None, **kwargs).images[0]
        return image
    
    @async_wrap_thread
    def generate_image(self, prompt:str, steps=2, negative_prompt=None, guidance=None, stage_mix=None, **kwargs) -> Image.Image:
        
        image = self.pipeline(prompt=prompt, num_inference_steps=steps)
        return image
    


class FastImageGenManager:
    def __init__(self):
        self.is_compiled = False
        self.is_ready = False
        self.dc_enabled = False
        self.model_name = 'sdxl_turbo'
        self.guidance_scale = 0.0
        self.img_size = (512,512)
    
    @async_wrap_thread
    def load_pipeline(self):
        self.base: StableDiffusionXLPipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", 
            torch_dtype=torch.bfloat16,
            #variant="fp16", 
            use_safetensors=True, 
            add_watermarker=False, 
            #device_map="auto",
        ).to("cuda")
        self.basei2i = AutoPipelineForImage2Image.from_pipe(self.base, torch_dtype=torch.bfloat16, use_safetensors=True, add_watermarker=False,).to("cuda")
        self.is_ready = True
    
    def dc_fastmode(self, enable:bool, img2img=False):
        pass

    @async_wrap_thread
    def unload_pipeline(self):
        self.base = None
        self.basei2i = None
        release_memory()
        torch_compile_flags(restore_defaults=True)
        torch._dynamo.reset()
        self.is_ready = False
        self.is_compiled = False

    @async_wrap_thread
    def compile_pipeline(self):
        torch_compile_flags()
        self.base.unet = torch.compile(self.base.unet, mode="max-autotune", fullgraph=True)
        #self.base.unet = torch.compile(self.base.unet, mode="reduce-overhead", fullgraph=True)
        self.base.upcast_vae()
        _ = self.pipeline(prompt='Cat')
        self.is_compiled = True
    
    @torch.inference_mode()
    def pipeline(self, prompt, num_inference_steps, strength=0.3, image=None, **kwargs):
        if image is None:
            out_image = self.base(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=self.guidance_scale, **kwargs).images[0]
        else:
            if strength>0:
                if num_inference_steps*strength < 1:
                    steps = math.ceil(1/strength)
                    print(f'steps ({num_inference_steps}) too low, forcing to {steps}')
                    num_inference_steps = steps
                    
                out_image = self.basei2i(prompt=prompt, image=image, num_inference_steps=num_inference_steps, strength=strength, guidance_scale=self.guidance_scale, **kwargs).images[0]
            else:
                out_image=image
        return out_image
    
    @async_wrap_thread
    def generate_image(self, prompt:str, steps=2, negative_prompt=None, guidance=None, stage_mix=None, **kwargs) -> Image.Image:
        out_image = self.pipeline(prompt=prompt, num_inference_steps=steps) # num_images_per_prompt=1
        return out_image

    @async_wrap_thread
    def regenerate_image(self, image:Image.Image, prompt:str, steps=4, strength=0.3, neg_prompt=None, guidance=None, stage_mix=None, refine_strength=None, **kwargs) -> Image.Image:
        print('regenerate_image size:',image.size)
        strength = cmd_tfms.percent_transform(strength)
        out_image = self.pipeline(prompt=prompt, num_inference_steps=steps, image=image.resize(self.img_size), strength=strength, **kwargs)
        return out_image

class MedImageGenManager(FastImageGenManager):
    def __init__(self):
        self.is_compiled = False
        self.is_ready = False
        self.dc_enabled = False
        self.model_name = 'dreamshaper_turbo'
        self.guidance_scale = 2.0
        self.img_size = (1024,1024)
    
    @async_wrap_thread
    def load_pipeline(self):
        self.base: StableDiffusionXLPipeline = AutoPipelineForText2Image.from_pretrained(
            'lykon/dreamshaper-xl-v2-turbo', # 4-8 steps, cfg=2.0, # https://civitai.com/models/112902?modelVersionId=333449
            torch_dtype=torch.bfloat16,
            variant="fp16", 
            use_safetensors=True, 
            add_watermarker=False, 
            #device_map="auto",
        )
        self.base.scheduler = DPMSolverMultistepScheduler.from_config(self.base.scheduler.config)
        self.base.to("cuda")
        self.basei2i = AutoPipelineForImage2Image.from_pipe(self.base, torch_dtype=torch.bfloat16, use_safetensors=True, add_watermarker=False,).to("cuda")
        self.is_ready = True

class ImageGenManager:
    def __init__(self):
        self.is_ready = False
        self.is_compiled = False
        self.dc_enabled = None
        self.model_name = 'sdxl'
        # https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output

    def _setup_dc(self):
        # https://github.com/horseee/DeepCache/blob/master/DeepCache/extension/deepcache.py
        #if self.dc_enabled is None:
        self.dc_base = DeepCacheSDHelper(self.base)
        self.dc_base.set_params(cache_interval=3, cache_branch_id=0)
        
        self.dc_basei2i = DeepCacheSDHelper(self.basei2i)
        self.dc_basei2i.set_params(cache_interval=3, cache_branch_id=0)

        self.dc_refiner = DeepCacheSDHelper(self.refiner)
        self.dc_refiner.set_params(cache_interval=3, cache_branch_id=0)
        self.dc_enabled = False
        
    
    def dc_fastmode(self, enable:bool, img2img=False):
        #print(self.dc_enabled, enable)
        if enable: 
            if not self.dc_enabled: #if not self.dc_enabled:
                (self.dc_basei2i if img2img else self.dc_base).enable()
                self.dc_refiner.enable()
        elif self.dc_enabled:
            (self.dc_basei2i if img2img else self.dc_base).disable() 
            self.dc_refiner.disable()
            
        self.dc_enabled=enable    

    @async_wrap_thread
    def load_pipeline(self):
        self.base: StableDiffusionXLPipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=torch.bfloat16, 
            #variant="fp16", 
            use_safetensors=True, 
            add_watermarker=False, 
            #device_map="auto",
        ).to("cuda")
        #self.base.enable_model_cpu_offload()
        
        #self.base.vae.decode = torch.compile(self.base.vae.decode, mode="max-autotune", fullgraph=True)
        self.basei2i = AutoPipelineForImage2Image.from_pipe(self.base, torch_dtype=torch.bfloat16, use_safetensors=True, add_watermarker=False,).to("cuda")
        self.refiner: StableDiffusionXLImg2ImgPipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0", text_encoder_2=self.base.text_encoder_2, vae=self.base.vae,
            torch_dtype=torch.bfloat16,
            #variant="fp16",
            use_safetensors=True,
            add_watermarker=False,
            #device_map="auto", 
        ).to("cuda")
        #self.refiner.enable_model_cpu_offload()
        self._setup_dc()
        #self._enable_dc()
        self.is_ready=True

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
        _ = self.pipeline(prompt='Cat', num_inference_steps=10, negative_prompt=None, guidance_scale=10.0, denoise_blend=None, refine_strength=0.3, strength=0.3, image=None,)
        self.is_compiled = True
        release_memory()

    @torch.inference_mode()
    def pipeline(self, prompt, num_inference_steps, negative_prompt, guidance_scale, denoise_blend, refine_strength, strength=0.3, image=None, **kwargs):
        output_type = 'latent'
        if denoise_blend is None:
            output_type = 'pil'
        elif denoise_blend == 1:
            denoise_blend = None
        
        t0 = time.perf_counter()
        if image is None:
            image = self.base(prompt=prompt, num_inference_steps=num_inference_steps, denoising_end=denoise_blend, 
                                guidance_scale=guidance_scale, negative_prompt=negative_prompt, output_type=output_type, **kwargs).images[0]
            
        elif strength*num_inference_steps >= 1:
            image = self.basei2i(prompt=prompt, num_inference_steps=num_inference_steps, denoising_end=denoise_blend, 
                                guidance_scale=guidance_scale, negative_prompt=negative_prompt, image=image, strength=strength, output_type=output_type, **kwargs).images[0]
        
        t_base = time.perf_counter()
        if refine_strength*num_inference_steps >= 1:
            image = self.refiner(prompt=prompt, num_inference_steps=num_inference_steps, denoising_start=denoise_blend,
                                    guidance_scale=guidance_scale*1.5, negative_prompt=negative_prompt, image=[image], strength=refine_strength, **kwargs).images[0]
        
        t_refine = time.perf_counter()
        print(f'ImageGen Time - total: {t_refine-t0:0.2f}s | base: {t_base-t0:0.2f}s | refine: {t_refine-t_base:0.2f}s' )
        return image

    @async_wrap_thread
    def generate_image(self, prompt:str, steps=40, neg_prompt=None, guidance=7.0, stage_mix=None, refine_strength=0.3, **kwargs) -> Image.Image:
        #with torch.inference_mode():
        stage_mix = cmd_tfms.percent_transform(stage_mix)
        refine_strength = cmd_tfms.percent_transform(refine_strength)
        
        image = self.pipeline(prompt=prompt, num_inference_steps=steps, negative_prompt=neg_prompt, guidance_scale=guidance, denoise_blend=stage_mix, refine_strength=refine_strength, **kwargs)
        return image

    @async_wrap_thread
    def regenerate_image(self, image:Image.Image, prompt:str, steps=50, strength=0.3, neg_prompt=None, guidance=10.0, stage_mix=None, refine_strength=0.3, **kwargs) -> Image.Image:
        strength = cmd_tfms.percent_transform(strength)
        stage_mix = cmd_tfms.percent_transform(stage_mix)
        refine_strength = cmd_tfms.percent_transform(refine_strength)
        print('regenerate_image size:',image.size)
        #image = image.resize((1024,1024))
        image = self.pipeline(prompt=prompt, num_inference_steps=steps, strength=strength, negative_prompt=neg_prompt, 
                              guidance_scale=guidance, denoise_blend=stage_mix, refine_strength=refine_strength, image=image.resize((1024,1024)), **kwargs)
        return image