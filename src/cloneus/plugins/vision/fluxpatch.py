import functools
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch

from PIL.Image import Image

from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from diffusers.pipelines.flux.pipeline_flux_img2img import FluxImg2ImgPipeline as _FluxImg2ImgPipeline
from diffusers.pipelines.flux.pipeline_flux_inpaint import FluxInpaintPipeline as _FluxInpaintPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast



def _is_packed_latent(image, batch_size, num_channels_latents):
    #num_channels_latents = self.transformer.config.in_channels // 4

    # 3. Preprocess image
    # ADDED (swap 2,3 for edge case handling)
    # check if image is latent tensor
    return (
        isinstance(image, torch.Tensor) 
        and image.ndim == 3 
        and image.shape[2] == num_channels_latents * 4
        # check batch_size for edge case input: batchless image tensor shape (3, H, 64) looks like batch=3 packed latent 
        and image.shape[0] == batch_size
    )
        
def _maybe_unpack_latent(self, image, batch_size, height, width, num_channels_latents):
    #num_channels_latents = self.transformer.config.in_channels // 4
    
    # 3. Preprocess image
    # ADDED (swap 2,3 for edge case handling)
    # check if image is latent tensor
    if _is_packed_latent(image, batch_size, num_channels_latents):
        if height is None or width is None:
            raise ValueError(
                #f'Cannot infer `height` and `width` from input shape {image.shape}'
                #'`height` and `width` must be specified when passing latent tensor as `image`'
                '`height` and `width` cannot be infered when passing latents as `image`. '
            )
        # Returned latents are packed. They must be unpacked for the image_processor to compare channel against vae_latent_channels
        image = self._unpack_latents(image, height=height, width=width, vae_scale_factor=self.vae_scale_factor)
        #print('unpack image shape:', image.shape)
    else:
        height, width = self.image_processor.get_default_height_width(image, height, width)
    # END ADDED



class FluxImg2ImgPipeline(_FluxImg2ImgPipeline):
    r"""
    The Flux pipeline for image-to-image generation.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        transformer ([`FluxTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`T5TokenizerFast`):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]
    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
    ):
        # <CHANGE (revert)
        DiffusionPipeline.__init__(self)
        #super().__init__()
        # CHANGE/>

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels)) if hasattr(self, "vae") and self.vae is not None else 16
        )
        vae_latent_channels = (
            self.vae.config.latent_channels if hasattr(self, "vae") and self.vae is not None else 4
        )
        
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, vae_latent_channels=vae_latent_channels)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = 64

    def prepare_latents(
        self,
        image,
        timestep,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        height = 2 * (int(height) // self.vae_scale_factor)
        width = 2 * (int(width) // self.vae_scale_factor)

        shape = (batch_size, num_channels_latents, height, width)
        latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)

        if latents is not None:
            return latents.to(device=device, dtype=dtype), latent_image_ids

        image = image.to(device=device, dtype=dtype)
        # <CHANGE 
        if image.shape[1] == self.vae.config.latent_channels:
            image_latents = image
        else:
            image_latents = self._encode_vae_image(image=image, generator=generator)
        # image_latents = self._encode_vae_image(image=image, generator=generator)
        # CHANGE/> 

        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = torch.cat([image_latents], dim=0)



        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self.scheduler.scale_noise(image_latents, timestep, noise)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        return latents, latent_image_ids
    
    def _post_call_hook(self, output, height:int, width:int, output_type:str, return_dict:bool):
        if output_type == "latent":
            image = self._unpack_latents(output, height, width, self.vae_scale_factor)
            if not return_dict:
                return (image,)
            return FluxPipelineOutput(images=image)
        return output

    @torch.inference_mode()
    def __call__(self, prompt: str | List[str] = None, prompt_2: str | List[str] | None = None, image: Image | np.ndarray | torch.Tensor | List[Image] | List[np.ndarray] | List[torch.Tensor] = None, height: int | None = None, width: int | None = None, strength: float = 0.6, num_inference_steps: int = 28, timesteps: List[int] = None, guidance_scale: float = 7, num_images_per_prompt: int | None = 1, generator: torch.Generator | List[torch.Generator] | None = None, latents: torch.FloatTensor | None = None, prompt_embeds: torch.FloatTensor | None = None, pooled_prompt_embeds: torch.FloatTensor | None = None, output_type: str | None = "pil", return_dict: bool = True, joint_attention_kwargs: Dict[str, Any] | None = None, callback_on_step_end: Callable[[int, int, Dict], None] | None = None, callback_on_step_end_tensor_inputs: List[str] = ["latents"], max_sequence_length: int = 512):
        output = super().__call__(prompt, prompt_2, image, height, width, strength, num_inference_steps, timesteps, guidance_scale, num_images_per_prompt, generator, latents, prompt_embeds, pooled_prompt_embeds, output_type, return_dict, joint_attention_kwargs, callback_on_step_end, callback_on_step_end_tensor_inputs, max_sequence_length)
        return self._post_call_hook(output, height=height, width=width, output_type=output_type, return_dict=return_dict)
    
class FluxInpaintPipeline(_FluxInpaintPipeline):
    r"""
    The Flux pipeline for image inpainting.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        transformer ([`FluxTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`T5TokenizerFast`):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
    ):
        # <CHANGE (revert)
        DiffusionPipeline.__init__(self)
        #super().__init__()
        # />

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels)) if hasattr(self, "vae") and self.vae is not None else 16
        )
        # <CHANGE 
        vae_latent_channels = (
            self.vae.config.latent_channels if hasattr(self, "vae") and self.vae is not None else 16
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, vae_latent_channels=vae_latent_channels)
        # CHANGE/>
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            vae_latent_channels=self.vae.config.latent_channels,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = 64


    def prepare_latents(
        self,
        image,
        timestep,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        height = 2 * (int(height) // self.vae_scale_factor)
        width = 2 * (int(width) // self.vae_scale_factor)

        shape = (batch_size, num_channels_latents, height, width)
        latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)

        image = image.to(device=device, dtype=dtype)
        # <CHANGE 
        if image.shape[1] == self.vae.config.latent_channels:
            image_latents = image
        else:
            image_latents = self._encode_vae_image(image=image, generator=generator)
        # image_latents = self._encode_vae_image(image=image, generator=generator)
        # CHANGE />

        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = torch.cat([image_latents], dim=0)

        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self.scheduler.scale_noise(image_latents, timestep, noise)
        else:
            noise = latents.to(device)
            latents = noise

        noise = self._pack_latents(noise, batch_size, num_channels_latents, height, width)
        image_latents = self._pack_latents(image_latents, batch_size, num_channels_latents, height, width)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        return latents, noise, image_latents, latent_image_ids
    
    def _post_call_hook(self, output, height:int, width:int, output_type:str, return_dict:bool):
        if output_type == "latent":
            image = self._unpack_latents(output, height, width, self.vae_scale_factor)
            if not return_dict:
                return (image,)
            return FluxPipelineOutput(images=image)
        return output
    
    @torch.inference_mode()
    def __call__(self, prompt: str | List[str] = None, prompt_2: str | List[str] | None = None, image: Image | np.ndarray | torch.Tensor | List[Image] | List[np.ndarray] | List[torch.Tensor] = None, mask_image: Image | np.ndarray | torch.Tensor | List[Image] | List[np.ndarray] | List[torch.Tensor] = None, masked_image_latents: Image | np.ndarray | torch.Tensor | List[Image] | List[np.ndarray] | List[torch.Tensor] = None, height: int | None = None, width: int | None = None, padding_mask_crop: int | None = None, strength: float = 0.6, num_inference_steps: int = 28, timesteps: List[int] = None, guidance_scale: float = 7, num_images_per_prompt: int | None = 1, generator: torch.Generator | List[torch.Generator] | None = None, latents: torch.FloatTensor | None = None, prompt_embeds: torch.FloatTensor | None = None, pooled_prompt_embeds: torch.FloatTensor | None = None, output_type: str | None = "pil", return_dict: bool = True, joint_attention_kwargs: Dict[str, Any] | None = None, callback_on_step_end: Callable[[int, int, Dict], None] | None = None, callback_on_step_end_tensor_inputs: List[str] = ["latents"], max_sequence_length: int = 512):
        output = super().__call__(prompt, prompt_2, image, mask_image, masked_image_latents, height, width, padding_mask_crop, strength, num_inference_steps, timesteps, guidance_scale, num_images_per_prompt, generator, latents, prompt_embeds, pooled_prompt_embeds, output_type, return_dict, joint_attention_kwargs, callback_on_step_end, callback_on_step_end_tensor_inputs, max_sequence_length)
        return self._post_call_hook(output, height=height, width=width, output_type=output_type, return_dict=return_dict)
    