
import gc
from pathlib import Path

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)


from cloneus import cpaths
from cloneus.plugins.vision import pipelines
from cloneus.plugins.vision.pipelines import DiffusionConfig, CfgItem


import config.settings as settings
from utils.globthread import wrap_async_executor, stop_global_executors

bot_logger = settings.logging.getLogger('bot')
model_logger = settings.logging.getLogger('model')
cmds_logger = settings.logging.getLogger('cmds')
event_logger = settings.logging.getLogger('event')

# Resource: https://github.com/CyberTimon/Stable-Diffusion-Discord-Bot/blob/main/bot.py

IMG_DIR = settings.SERVER_ROOT/'output'/'imgs'
PROMPT_FILE = IMG_DIR.joinpath('_prompts.txt')
#SDXL_DIMS = [(1024,1024), (1152, 896),(896, 1152), (1216, 832),(832, 1216), (1344, 768),(768, 1344), (1536, 640),(640, 1536),] # https://stablediffusionxl.com/sdxl-resolutions-and-aspect-ratios/
# other: [(1280, 768),(768, 1280),]

class ImageGenManager:#(pipelines.SingleStagePipeline):
    _async_exec_methods = [
        'load_pipeline', 'unload_pipeline', 'compile_pipeline', 
        'generate_image', 'regenerate_image','refine_image',
        # 'decode_latents', 'embed_prompts', 'upsample', 'caption', 
    ]
    # These methods yield from generators. We want to reserve the primary executor for the generator. 
    # So we'll use an alternate executor to wrap the outer function call.
    # Using 2 executors may not be necessary, but it could safeguard against processing steps that 
    # occur prior to entering the generator from blocking the main event loop 
    _async_generator_exec_methods = [
        'generate_frames', 'regenerate_frames',
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._wrap_methods_athread()

    def _wrap_methods_athread(self):
        for meth in self._async_exec_methods:
            func = getattr(self, meth)
            setattr(self, meth, wrap_async_executor(func))
        
        for meth in self._async_generator_exec_methods:
            func = getattr(self, meth)
            setattr(self, meth, wrap_async_executor(func, use_alternate_executor=True))

class BaseFluxManager(ImageGenManager, pipelines.FluxBase):
    def __init__(self, model_name: str, model_path: str, config: DiffusionConfig, offload: bool = False, scheduler_setup: str | tuple[str, dict] = None, 
                qtype = 'bnb4', te2_qtype = 'bf16', quant_basedir = None):
        super().__init__(model_name, model_path, config, offload, scheduler_setup, qtype=qtype, te2_qtype=te2_qtype, quant_basedir=quant_basedir)
        
class BaseSDXLManager(ImageGenManager, pipelines.SDXLBase):
    def __init__(self, model_name: str, model_path: str, config: DiffusionConfig, offload: bool = False, scheduler_setup: str | tuple[str, dict] = None,):
        super().__init__(model_name, model_path, config, offload, scheduler_setup)   

class BaseSD3Manager(ImageGenManager, pipelines.SD3Base):
    def __init__(self, model_name: str, model_path: str, config: DiffusionConfig, offload: bool = False, scheduler_setup: str | tuple[str, dict] = None):
        super().__init__(model_name, model_path, config, offload, scheduler_setup)   

    
class SDXLTurboManager(BaseSDXLManager):
    def __init__(self, offload=False):
        super().__init__(
            model_name='sdxl_turbo', 
            model_path="stabilityai/sdxl-turbo",
            config=DiffusionConfig(
                steps = CfgItem(4, bounds=(1,4)), # 1-4 steps
                guidance_scale = CfgItem(0.0, locked=True),
                strength = CfgItem(0.55, bounds=(0.3, 0.9)),
                img_dims = (512,512),
                locked=['guidance_scale', 'negative_prompt', 'aspect',  'refine_guidance_scale']
            ), 
            offload=offload)
        
    
    def dc_fastmode(self, enable:bool, img2img=False):
        pass


class SD3MediumManager(BaseSD3Manager):
    def __init__(self, offload=False):
        super().__init__(
            model_name = 'sd3_medium',
            model_path = 'stabilityai/stable-diffusion-3-medium-diffusers',
            config = DiffusionConfig(
                steps = CfgItem(28, bounds=(24,42)),
                guidance_scale = CfgItem(7.0, bounds=(5,10)), 
                strength = CfgItem(0.65, bounds=(0.3, 0.95)),
                img_dims = [(1024,1024), (832,1216), (1216,832)],
                locked=[  'refine_guidance_scale']
            ),
            offload=offload,
        )
    
class ColorfulXLLightningManager(BaseSDXLManager):
    def __init__(self, offload=False):
        super().__init__( # https://huggingface.co/recoilme/ColorfulXL-Lightning
            model_name = 'colorfulxl_lightning', # https://civitai.com/models/388913/colorfulxl-lightning
            model_path = 'recoilme/ColorfulXL-Lightning',  
            
            config = DiffusionConfig(
                steps = CfgItem(9, bounds=(4,10)), # https://imgsys.org/rankings
                guidance_scale = CfgItem(1.0, bounds=(0,2.0)),
                strength = CfgItem(0.85, bounds=(0.3, 0.95)),
                img_dims = [(1024,1024), (832,1216), (1216,832)],
                aspect='square',#'portrait',
                clip_skip=1,
                locked = [ 'refine_guidance_scale']
            ),
            offload=offload,
            scheduler_setup = ('Euler A', {'timestep_spacing': "trailing"}),
            
        )
     
# https://civitai.com/models/119229/zavychromaxl

class JuggernautXIManager(BaseSDXLManager):
    def __init__(self, offload=False):
        super().__init__( # https://huggingface.co/RunDiffusion/Juggernaut-XI-v11
            model_name = 'juggernaut_xi', # https://civitai.com/models/133005?modelVersionId=782002
            model_path = 'https://huggingface.co/RunDiffusion/Juggernaut-XI-v11/blob/main/Juggernaut-XI-byRunDiffusion.safetensors',#'RunDiffusion/Juggernaut-XI-v11', 
            
            config = DiffusionConfig(
                steps = CfgItem(30, bounds=(30,40)), 
                guidance_scale = CfgItem(4.5, bounds=(3.0,6.0)),
                strength = CfgItem(0.55, bounds=(0.3, 0.95)),
                #negative_prompt='bad eyes, cgi, airbrushed, plastic, deformed, watermark'
                img_dims = [(1024,1024), (832,1216), (1216,832)],
                aspect='portrait',
                clip_skip=2,
                locked = ['refine_guidance_scale']
            ),
            offload=offload,
            scheduler_setup=('DPM++ 2M SDE'),#, {'lower_order_final':True})
            
        )

class RealVizXL5Manager(BaseSDXLManager):
    def __init__(self, offload=False):
        super().__init__( # https://huggingface.co/SG161222/RealVisXL_V5.0
            model_name = 'realvisxl_v5', # https://civitai.com/models/139562?modelVersionId=789646
            model_path = 'https://huggingface.co/SG161222/RealVisXL_V5.0/blob/main/RealVisXL_V5.0_fp16.safetensors', #'SG161222/RealVisXL_V5.0',  
            
            config = DiffusionConfig(
                steps = CfgItem(30, bounds=(20,50)), 
                guidance_scale = CfgItem(5, bounds=(3,10)),
                strength = CfgItem(0.60, bounds=(0.3, 0.95)),
                #negative_prompt='bad hands, bad anatomy, ugly, deformed, (face asymmetry, eyes asymmetry, deformed eyes, deformed mouth, open mouth)',
                negative_prompt='(bad hands, bad anatomy, deformed eyes, deformed mouth)+',
                img_dims = [(1024,1024), (896,1152), (1152,896)],
                aspect='portrait',
                clip_skip=1, # 2
                locked = ['refine_guidance_scale']
            ),
            offload=offload,
            scheduler_setup=('DPM++ 2M SDE')#, {'lower_order_final':True})
            #scheduler_setup=('Euler A', {'timestep_spacing': "trailing"}),
            #scheduler_setup='Euler A',
            
        )

class FluxSchnellManager(BaseFluxManager):
    def __init__(self, offload=False):
        super().__init__(
            model_name = 'flux_schnell',
            model_path = 'black-forest-labs/FLUX.1-schnell',
            config = DiffusionConfig(
                steps = CfgItem(4, bounds=(1,4)),
                guidance_scale = CfgItem(0.0, locked=True), 
                strength = CfgItem(0, locked=True),#CfgItem(0.65, bounds=(0.3, 0.95)),
                img_dims = [(1024,1024), (832,1216), (1216,832)],
                #refine_strength=CfgItem(0, locked=True),
                locked=['refine_guidance_scale','negative_prompt'] # 'refine_strength',
            ),
            offload=offload,
            #scheduler_setup='Euler',
            qtype = 'bnb4',
            te2_qtype = 'bf16',
            quant_basedir = cpaths.ROOT_DIR / 'extras/quantized/flux/',
           
        )

        
class FluxDevManager(BaseFluxManager):
    def __init__(self, offload=False):
        super().__init__(
            model_name = 'flux_dev',
            model_path = 'black-forest-labs/FLUX.1-dev',
            config = DiffusionConfig(
                steps = CfgItem(30, bounds=(25,60)),
                guidance_scale = CfgItem(3.5, bounds=(1,5)), 
                strength = CfgItem(0.75, bounds=(0.6, 0.95)),#CfgItem(0.65, bounds=(0.3, 0.95)),
                #img_dims = [(1024,1024), (832,1216), (1216,832)],
                img_dims = [(1024,1024), (896,1152), (1152,896)],
                #refine_strength=CfgItem(0, locked=True),
                locked=['refine_guidance_scale','negative_prompt'] # 'refine_strength',
            ),
            offload=offload,
            #scheduler_setup='Euler',
            qtype = 'bnb4',
            te2_qtype = 'bf16',
            quant_basedir = cpaths.ROOT_DIR / 'extras/quantized/flux/',
        )

# https://old.reddit.com/r/StableDiffusion/comments/1f83d0t/new_vitl14_clipl_text_encoder_finetune_for_flux1/
# https://huggingface.co/zer0int/CLIP-GmP-ViT-L-14/tree/main
class FluxSchnevManager(BaseFluxManager):
    def __init__(self, offload=False):
        super().__init__(
            model_name = 'flux_schnev',
            # model_path = cpaths.ROOT_DIR / 'extras/models/flux/flux-merged',
            model_path = cpaths.ROOT_DIR / 'extras/quantized/flux/nf4/diffusion_pytorch_model.safetensors',
            # model_path = cpaths.ROOT_DIR / 'extras/quantized/flux/nf4_bnb/diffusion_pytorch_model.safetensors',
            config = DiffusionConfig(
                steps = CfgItem(10, bounds=(4,16)),
                guidance_scale = CfgItem(2.5, bounds=(0,5)), 
                strength = CfgItem(0.70, bounds=(0.6, 0.9)),
                #img_dims = [(1024,1024), (832,1216), (1216,832)],
                img_dims = [(1024,1024), (896,1152), (1152,896)],
                #refine_strength=CfgItem(0.3, bounds=(0.2, 0.4)),
                #refine_steps=CfgItem(0, locked=True),
                #refine_strength=CfgItem(0, locked=True),
                locked=['refine_guidance_scale','negative_prompt'] # 'refine_strength',
            ),
            offload=offload,
            #scheduler_setup=('Euler FM', dict(shift=1.8, use_dynamic_shifting=False)),
            qtype = 'bnb4',
            te2_qtype = 'bf16',
            quant_basedir = cpaths.ROOT_DIR / 'extras/quantized/flux/',
        )





AVAILABLE_MODELS = {
    'sdxl_turbo': {
        'manager': SDXLTurboManager,
        'desc': 'Turbo SDXL' # (Sm, fast)
    },
    'sd3_medium': {
        'manager': SD3MediumManager,
        'desc': 'SD3 Medium' #  (Lg, slow)
    },
    'colorfulxl_lightning': {
        'manager': ColorfulXLLightningManager,
        'desc': 'ColorfulXL Lightning' # (M, fast)
    },
    # 'flux_schnell': {
    #     'manager': FluxSchnellManager,
    #     'desc': 'Flux Schnell' # (XLg, avg)
    # },
    'juggernaut_xi': {
        'manager': JuggernautXIManager,
        'desc': 'Juggernaut XI' # (M, avg)
    },
    'realvisxl_v5': {
        'manager': RealVizXL5Manager,
        'desc': 'RealVisXL V5' # (M, avg)
    },
    'flux_dev': {
        'manager': FluxDevManager,
        'desc': 'Flux Dev' # (XLg, slow)
    },
    'flux_schnev': {
        'manager': FluxSchnevManager,
        'desc': 'Flux Schnev' # (XLg, avg)
    },


}

