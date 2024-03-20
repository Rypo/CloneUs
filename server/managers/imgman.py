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
from accelerate import cpu_offload
from DeepCache import DeepCacheSDHelper
from PIL import Image
from spandrel import ImageModelDescriptor, ModelLoader
from compel import Compel, ReturnedEmbeddingsType

from cloneus import cpaths

import config.settings as settings
from cmds import flags as cmd_flags, transformers as cmd_tfms
from utils.globthread import async_wrap_thread, stop_global_thread
from views import imageui

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
        
    def nearest_dims(self, img_wh):
        dim_out = self.img_dims
        
        if isinstance(self.img_dims, list):
            w_in, h_in = img_wh
            ar_in = w_in/h_in
            dim_out = min(self.img_dims, key=lambda wh: abs(ar_in - wh[0]/wh[1]))
        
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

class OneStageImageGenManager:
    def __init__(self, model_name: str, model_path:str, config:DiffusionConfig, offload=False, scheduler_callback:typing.Callable=None):
        self.is_compiled = False
        self.is_ready = False
        self.dc_enabled = None
        
        self.model_name = model_name
        self.model_path = model_path
        self.config = config
        self.offload = offload
        self._scheduler_callback = scheduler_callback
        # https://huggingface.co/docs/diffusers/v0.26.3/en/api/schedulers/overview#schedulers
        # DPM++ SDE == DPMSolverSinglestepScheduler
        # DPM++ SDE Karras == DPMSolverSinglestepScheduler(use_karras_sigmas=True)
        self.global_seed = None
    
    def set_seed(self, seed:int|None = None):
        self.global_seed = seed
    
    @property
    def generator(self):
        return torch.Generator('cuda').manual_seed(self.global_seed) if self.global_seed is not None else None


    @async_wrap_thread
    def load_pipeline(self):
        pipeline_loader = (StableDiffusionXLPipeline.from_single_file if self.model_path.endswith('.safetensors') 
                           else AutoPipelineForText2Image.from_pretrained)

        self.base: StableDiffusionXLPipeline = pipeline_loader(
            self.model_path, torch_dtype=torch.bfloat16, variant="fp16", use_safetensors=True, add_watermarker=False,
        )
        if self._scheduler_callback is not None:
            self.base.scheduler = self._scheduler_callback()
        
        if self.offload:
            self.base.enable_model_cpu_offload()
        else:
            self.base = self.base.to("cuda")
        
        self.compeler = Compel(
            tokenizer=[self.base.tokenizer, self.base.tokenizer_2],
            text_encoder=[self.base.text_encoder, self.base.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True]
        )

        self.basei2i: StableDiffusionXLImg2ImgPipeline = AutoPipelineForImage2Image.from_pipe(self.base,)# torch_dtype=torch.bfloat16, use_safetensors=True, add_watermarker=False)#.to(0)
        
        if not self.offload:
            self.basei2i = self.basei2i.to("cuda")
        
        
        self.upsampler = Upsampler('4xUltrasharp-V10.pth', dtype=torch.bfloat16)
        #self.upsampler = Upsampler('4xNMKD-Superscale.pth', dtype=torch.bfloat16)
        
        self.is_ready = True
    
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
    def embed_prompts(self, prompt:str, negative_prompt:str = None):
        
        if set(prompt) & set('()-+'):
            conditioning, pooled = self.compeler(prompt)
            prompt_encodings = dict(prompt_embeds=conditioning, pooled_prompt_embeds=pooled)
            if negative_prompt is not None:
                neg_conditioning, neg_pooled = self.compeler(negative_prompt)
                prompt_encodings.update(negative_prompt_embeds=neg_conditioning, negative_pooled_prompt_embeds=neg_pooled)
        else:
            prompt_encodings = dict(prompt=prompt, negative_prompt=negative_prompt)
            
        return prompt_encodings
        
    
    @torch.inference_mode()
    def pipeline(self, prompt, num_inference_steps, negative_prompt=None, guidance_scale=None, refine_steps=None, strength=None, image=None, target_size=None):
        if self.global_seed is not None:
            #meval.seed_everything(self.global_seed)
            torch.manual_seed(self.global_seed)
            torch.cuda.manual_seed(self.global_seed)
            
        t0 = time.perf_counter()
        prompt_encodings = self.embed_prompts(prompt, negative_prompt=negative_prompt)
        t1 = time.perf_counter()
        
        t2 = t1 # default in case skip
        if image is None:
            #h,w is all you need -- https://github.com/huggingface/diffusers/blob/v0.26.3/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L1075
            h,w = target_size
            image = self.base(num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, height=h, width=w, **prompt_encodings).images[0] # , generator=self.generator
            #imgs = self.base(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, negative_prompt=negative_prompt, num_images_per_prompt=4, **kwargs).images#[0]
            #return make_image_grid(imgs, 2, 2)
            print('(draw) max_memory_allocated:', torch.cuda.max_memory_allocated()/(1024**2), 'max_memory_reserved:', torch.cuda.max_memory_reserved()/(1024**2))
            #release_memory()
            
            t2 = time.perf_counter()
        t3 = t4 = t2  # default in case skip
        if strength:
            min_effective_steps = 1
            if refine_steps > 0:
                min_effective_steps = refine_steps
                
                print('Upscaling..')
                image = self.upsampler.upscale(img=image, scale=1.5)
                t3 = time.perf_counter()
                print('(upscale) max_memory_allocated:', torch.cuda.max_memory_allocated()/(1024**2), 'max_memory_reserved:', torch.cuda.max_memory_reserved()/(1024**2))
                release_memory()
                
            if num_inference_steps*strength < min_effective_steps:
                steps = math.ceil(min_effective_steps/strength)
                print(f'steps ({num_inference_steps}) too low, forcing to {steps}')
                num_inference_steps = steps
            # target_size defaults img_size
            image = self.basei2i(image=image, num_inference_steps=num_inference_steps, strength=strength, guidance_scale=guidance_scale, **prompt_encodings).images[0]
            print('(refine) max_memory_allocated:', torch.cuda.max_memory_allocated()/(1024**2), 'max_memory_reserved:', torch.cuda.max_memory_reserved()/(1024**2))
            
            t4 = time.perf_counter()
        
        release_memory()
        print(f'total: {t4-t0:.2f}s | prompt_embed: {t1-t0:.2f}s | T2I: {t2-t1:.2f}s | upscale: {t3-t2:.2f}s | refine: {t4-t3:.2f}s')
        return image
    
    @async_wrap_thread
    def generate_image(self, prompt:str, steps:int=None, negative_prompt:str=None, guidance_scale:float=None, aspect:typing.Literal['square','portrait','landscape']=None, refine_steps:int=0, refine_strength:float=None, **kwargs) -> Image.Image:
        fkwg = self.config.get_if_none(steps=steps, negative_prompt=negative_prompt, guidance_scale=guidance_scale, refine_steps=refine_steps, refine_strength=refine_strength, aspect=aspect)
        print(f'kwargs: {kwargs}\nfkwg:{fkwg}')
        img_dims = self.config.get_dims(fkwg['aspect'])
        target_size = (img_dims[1], img_dims[0]) # h,w
        
        refine_strength = cmd_tfms.percent_transform(fkwg['refine_strength'])
        image = self.pipeline(prompt=prompt, num_inference_steps=fkwg['steps'], negative_prompt=fkwg['negative_prompt'], guidance_scale=fkwg['guidance_scale'], 
                              refine_steps=refine_steps, strength=refine_strength, target_size=target_size, )
        
        #call_kwargs = {'prompt':prompt, **fkwg, 'refine_steps':refine_steps, 'refine_strength':refine_strength}
        return image, fkwg

    @async_wrap_thread
    def regenerate_image(self, image:Image.Image, prompt:str, steps:int=None, strength:float=None, negative_prompt:str=None, guidance_scale:float=None, refine_steps:int=0, **kwargs) -> Image.Image:
        fkwg = self.config.get_if_none(steps=steps, strength=strength, negative_prompt=negative_prompt, guidance_scale=guidance_scale, refine_steps=refine_steps)
        print(f'kwargs: {kwargs}\nfkwg:{fkwg}')
        
        dim_out = self.config.nearest_dims(image.size)
        print('regenerate_image input size:', image.size, '->', dim_out)
        image = image.resize(dim_out, resample=Image.Resampling.LANCZOS)
        
        strength = cmd_tfms.percent_transform(fkwg['strength'])
        image = self.pipeline(prompt=prompt, num_inference_steps=fkwg['steps'], strength=strength, 
                              negative_prompt=fkwg['negative_prompt'], guidance_scale=fkwg['guidance_scale'], refine_steps=refine_steps, image=image)

        #self.base.disable_vae_tiling()
        #call_kwargs = {'prompt':prompt, **fkwg}
        return image, fkwg


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
                       aspect: typing.Literal['square','portrait', 'landscape'] = None, 
                       denoise_blend: float|None = None, 
                       refine_strength: float = None, 
                       **kwargs) -> Image.Image:
        
        fkwg = self.config.get_if_none(steps=steps, negative_prompt=negative_prompt, guidance_scale=guidance_scale, denoise_blend=denoise_blend, refine_strength=refine_strength, aspect=aspect)
        print(f'kwargs: {kwargs}\nfkwg:{fkwg}')
        
        dim_out = self.config.get_dims(aspect)
        target_size = (dim_out[1], dim_out[0]) # h,w

        denoise_blend = cmd_tfms.percent_transform(fkwg['denoise_blend'])
        refine_strength = cmd_tfms.percent_transform(fkwg['refine_strength'])
        
        image = self.pipeline(prompt=prompt, num_inference_steps=fkwg['steps'], negative_prompt=fkwg['negative_prompt'], guidance_scale=fkwg['guidance_scale'], 
                              denoise_blend=denoise_blend, refine_strength=refine_strength, target_size=target_size)
        return image, fkwg

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
        return image, fkwg

class Upsampler:
    def __init__(self, model_name=typing.Literal["4xNMKD-Superscale.pth", "4xUltrasharp-V10.pth"], dtype=torch.bfloat16):
        # https://civitai.com/articles/904/settings-recommendations-for-novices-a-guide-for-understanding-the-settings-of-txt2img
        ext_modeldir = cpaths.ROOT_DIR/'extras/models'
        
        # load a model from disk
        self.model = ModelLoader().load_from_file(ext_modeldir.joinpath(model_name))
        # make sure it's an image to image model
        assert isinstance(self.model, ImageModelDescriptor)
        
        #self.model = cpu_offload(self.model.eval().to(dtype=dtype))
        self.model.eval().to('cuda', dtype=dtype)
        #cpu_offload(self.model.model)
    
    def pil_image_to_torch_bgr(self, img: Image.Image) -> torch.Tensor:
        # https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/bef51aed032c0aaa5cfd80445bc4cf0d85b408b5/modules/upscaler_utils.py#L14C1-L19C33
        img = np.array(img.convert("RGB"))
        img = img[:, :, ::-1]  # flip RGB to BGR
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img) / 255  # Rescale to [0, 1]
        return torch.from_numpy(img)
    
    def torch_bgr_to_pil_image(self, tensor: torch.Tensor) -> Image.Image:
        # https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/bef51aed032c0aaa5cfd80445bc4cf0d85b408b5/modules/upscaler_utils.py#L22C1-L35C39
        if tensor.ndim == 4:
            # If we're given a tensor with a batch dimension, squeeze it out
            # (but only if it's a batch of size 1).
            if tensor.shape[0] != 1:
                raise ValueError(f"{tensor.shape} does not describe a BCHW tensor")
            tensor = tensor.squeeze(0)
        assert tensor.ndim == 3, f"{tensor.shape} does not describe a CHW tensor"
        # TODO: is `tensor.float().cpu()...numpy()` the most efficient idiom?
        arr = tensor.float().cpu().clamp_(0, 1).numpy()  # clamp
        arr = 255.0 * np.moveaxis(arr, 0, 2)  # CHW to HWC, rescale
        arr = arr.round().astype(np.uint8)
        arr = arr[:, :, ::-1]  # flip BGR to RGB
        return Image.fromarray(arr, "RGB")
        
    
    @torch.inference_mode()
    def process(self, img: torch.FloatTensor) -> torch.Tensor:
        # https://github.com/joeyballentine/ESRGAN-Bot/blob/master/testbot.py#L73
        img = img.to(self.model.device, dtype=self.model.dtype)    
        output = self.model(img).detach_()#.squeeze(0).float().cpu().clamp_(0, 1).numpy()
        
        return output
    
    @torch.inference_mode()
    def upsample(self, image: Image.Image) -> Image.Image:
        img_tens = self.pil_image_to_torch_bgr(image).unsqueeze(0)
        output = self.process(img_tens)
        img_ups = self.torch_bgr_to_pil_image(output)
        return img_ups
    
    @torch.inference_mode()
    def upscale(self, img: Image.Image, scale:float=1.5):
        dest_w = int((img.width * scale) // 8 * 8)
        dest_h = int((img.height * scale) // 8 * 8)

        for _ in range(3):
            if img.width >= dest_w and img.height >= dest_h:
                break

            shape = (img.width, img.height)

            img = self.upsample(img)

            if shape == (img.width, img.height):
                break

        if img.width != dest_w or img.height != dest_h:
            img = img.resize((int(dest_w), int(dest_h)), resample=Image.Resampling.LANCZOS)

        return img
    
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
                refine_strength = 0.,
                locked=['guidance_scale', 'negative_prompt', 'aspect', 'denoise_blend', 'refine_guidance_scale'] # 'refine_strength'
            ), 
            offload=offload)
        
    
    def dc_fastmode(self, enable:bool, img2img=False):
        pass

    @async_wrap_thread
    def compile_pipeline(self):
        # Currently, not working properly for SDXL Turbo + savings are negligible
        pass

class SDXLManager(TwoStageImageGenManager):
    def __init__(self, offload=True):
        super().__init__(
            model_name = "sdxl",
            model_path = "stabilityai/stable-diffusion-xl-base-1.0",
            refiner_path="stabilityai/stable-diffusion-xl-refiner-1.0",
            config = DiffusionConfig(
                steps = CfgItem(50, bounds=(40,60)),
                guidance_scale = CfgItem(7.0, bounds=(5,15)), 
                strength = CfgItem(0.55, bounds=(0.3, 0.9)),
                aspect = CfgItem('square', locked=True),
                img_dims = (1024,1024),
                denoise_blend= CfgItem(None, bounds=(0.7, 0.9)),
                #refine_strength = CfgItem(0.3, bounds=(0.2, 0.4)),

                locked=['aspect', 'refine_guidance_scale']
            ),
            
            offload=offload
        )

class DreamShaperXLManager(OneStageImageGenManager):
    def __init__(self, offload=False):
        super().__init__(
            model_name = 'dreamshaper_turbo', # bf16 saves ~3gb vram over fp16
            model_path = 'lykon/dreamshaper-xl-v2-turbo', # https://civitai.com/models/112902?modelVersionId=333449
            config = DiffusionConfig(
                steps = CfgItem(8, bounds=(4,8)),
                guidance_scale = CfgItem(2.0, locked=True),
                strength = CfgItem(0.55, bounds=(0.3, 0.9)),
                img_dims = [(1024,1024), (832,1216)],
                #refine_strength=CfgItem(0.3, bounds=(0.2, 0.4)),
                locked=['guidance_scale', 'denoise_blend',  'refine_guidance_scale'] # 'refine_strength',
            ),
            offload=offload,
            scheduler_callback = lambda: DPMSolverSinglestepScheduler.from_config(self.base.scheduler.config, use_karras_sigmas=True)
        )

class JuggernautXLLightningManager(OneStageImageGenManager):
    def __init__(self, offload=True):
        super().__init__( # https://huggingface.co/RunDiffusion/Juggernaut-XL-Lightning
            model_name = 'juggernaut_lightning', # https://civitai.com/models/133005/juggernaut-xl?modelVersionId=357609
            model_path = 'https://huggingface.co/RunDiffusion/Juggernaut-XL-Lightning/blob/main/Juggernaut_RunDiffusionPhoto2_Lightning_4Steps.safetensors', 
            
            config = DiffusionConfig(
                steps = CfgItem(6, bounds=(5,7)), # 4-6 step /  5 and 7
                guidance_scale = CfgItem(1.5, bounds=(1.5,2.0)),
                strength = CfgItem(0.6, bounds=(0.3, 0.9)),
                img_dims = [(1024,1024), (832,1216), (1216,832)],
                aspect='portrait',
                #refine_strength=CfgItem(0.3, bounds=(0.2, 0.4)),
                locked = ['denoise_blend', 'refine_guidance_scale'] # 'refine_strength'
            ),
            offload=offload,
            scheduler_callback= lambda: DPMSolverSinglestepScheduler.from_config(self.base.scheduler.config, lower_order_final=True, use_karras_sigmas=False)
        )
    
# https://huggingface.co/ByteDance/SDXL-Lightning

AVAILABLE_MODELS = {
    'sdxl_turbo': {
        'manager': SDXLTurboManager(),
        'desc': 'Turbo SDXL (Sm, fast)'
    },
    'sdxl': {
        'manager': SDXLManager(),
        'desc':'SDXL (Lg, slow)'
    },
    'dreamshaper_turbo': {
        'manager': DreamShaperXLManager(),
        'desc': 'DreamShaper XL2 Turbo (M, fast)'
    },
    'juggernaut_lightning': {
        'manager': JuggernautXLLightningManager(offload=True),
        'desc': 'Juggernaut XL Lightning (M, fast)'
    },
}

