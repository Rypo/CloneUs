
import gc
from pathlib import Path

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)


from cloneus import cpaths
from cloneus.plugins.vision import pipelines
from cloneus.plugins.vision.pipelines import DiffusionConfig, CfgItem


from utils.globthread import wrap_async_executor, stop_global_executors

LORA_DIR = cpaths.ROOT_DIR/f'extras/loras/'

# Resource: https://github.com/CyberTimon/Stable-Diffusion-Discord-Bot/blob/main/bot.py


#SDXL_DIMS = [(1024,1024), (1152, 896),(896, 1152), (1216, 832),(832, 1216), (1344, 768),(768, 1344), (1536, 640),(640, 1536),] # https://stablediffusionxl.com/sdxl-resolutions-and-aspect-ratios/
# other: [(1280, 768),(768, 1280),]

class ImageGenManager:#(pipelines.SingleStagePipeline):
    _async_exec_methods = [
        'load_pipeline', 'unload_pipeline', 'compile_pipeline', 
        'generate_image', 'regenerate_image','refine_image',
        'vqa_chat'
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
    def __init__(self, model_name: str, model_path: str, config: DiffusionConfig, offload: bool = False, scheduler_setup: str | tuple[str, str|dict] = None, dtype:torch.dtype = torch.bfloat16, init_loras: list[tuple[str,float]] = None,):
        super().__init__(model_name, model_path, config, offload, scheduler_setup, dtype, init_loras)
        
class BaseSDXLManager(ImageGenManager, pipelines.SDXLBase):
    def __init__(self, model_name: str, model_path: str, config: DiffusionConfig, offload: bool = False, scheduler_setup: str | tuple[str, str|dict] = None, dtype:torch.dtype = torch.bfloat16, init_loras: list[tuple[str,float]] = None):
        super().__init__(model_name, model_path, config, offload, scheduler_setup, dtype, init_loras)

class BaseSD3Manager(ImageGenManager, pipelines.SD3Base):
    def __init__(self, model_name: str, model_path: str, config: DiffusionConfig, offload: bool = False, scheduler_setup: str | tuple[str, str|dict] = None, dtype:torch.dtype = torch.bfloat16, init_loras: list[tuple[str,float]] = None):
        super().__init__(model_name, model_path, config, offload, scheduler_setup, dtype, init_loras)   



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
            offload=offload,
            init_loras = [(LORA_DIR.joinpath('sdxl','detail-tweaker-xl.safetensors').as_posix(), 0.0)],
        )
        
    
    def dc_fastmode(self, enable:bool, img2img=False):
        pass


    
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
            scheduler_setup = ('Euler A', 'sgm_uniform'), # {'timestep_spacing': "trailing"}
            init_loras = [(LORA_DIR.joinpath('sdxl','detail-tweaker-xl.safetensors').as_posix(), 0.0)],
        )
     
class JuggernautRagnarokManager(BaseSDXLManager):
    def __init__(self, offload=False):
        super().__init__( # https://huggingface.co/RunDiffusion/
            model_name = 'juggernaut_ragnarok', # https://civitai.com/models/133005?modelVersionId=1759168
            model_path = cpaths.ROOT_DIR / 'extras/models/sdxl/juggernautXL_ragnarok.safetensors',  
            
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
            scheduler_setup=('DPM++ 2M SDE', {'euler_at_final':True}),#, {'lower_order_final':True}) # https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/stable_diffusion_xl#tips
            init_loras = [(LORA_DIR.joinpath('sdxl','detail-tweaker-xl.safetensors').as_posix(), 0.0)],
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
                locked=['refine_guidance_scale'] # 'refine_strength',
            ),
            offload=offload,
            init_loras=[
                (LORA_DIR.joinpath('flux','detail-maximizer_v02.safetensors').as_posix(), 0.0),
                (LORA_DIR.joinpath('flux','midjourneyV61_v02.safetensors').as_posix(), 0.0),
            ],
        )

class FluxTurboManager(BaseFluxManager):
    def __init__(self, offload=True):
        super().__init__(
            model_name = 'flux_turbo',
            model_path = 'black-forest-labs/FLUX.1-dev',
            config = DiffusionConfig(
                steps = CfgItem(8, bounds=(6,12)),
                guidance_scale = CfgItem(3.5, bounds=(0,5)), 
                strength = CfgItem(0.70, bounds=(0.6, 0.9)),
                #img_dims = [(1024,1024), (832,1216), (1216,832)],
                img_dims = [(1024,1024), (896,1152), (1152,896)],
                locked=['refine_guidance_scale'] # 'refine_strength',
            ),
            offload=offload,
            init_loras = [("alimama-creative/FLUX.1-Turbo-Alpha/diffusion_pytorch_model.safetensors", 1.0),]
            #scheduler_setup=('Euler FM', dict(shift=1.8, use_dynamic_shifting=False)),
        )



AVAILABLE_MODELS = {
    'sdxl_turbo': {
        'manager': SDXLTurboManager,
        'desc': 'Turbo SDXL' # (Sm, fast)
    },
    'colorfulxl_lightning': {
        'manager': ColorfulXLLightningManager,
        'desc': 'ColorfulXL Lightning' # (M, fast)
    },
    'juggernaut_ragnarok': {
        'manager': JuggernautRagnarokManager,
        'desc': 'Juggernaut Ragnarok' # (M, avg)
    },
    'flux_dev': {
        'manager': FluxDevManager,
        'desc': 'Flux Dev' # (XLg, slow)
    },
    'flux_turbo': {
        'manager': FluxTurboManager,
        'desc': 'Flux Turbo' # (XLg, avg)
    },
}