from warnings import warn
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
from diffusers.models.transformers.transformer_qwenimage import compute_text_seq_len_from_mask # PATCH

from nunchaku import NunchakuZImageTransformer2DModel as _NunchakuZImageTransformer2DModel, NunchakuQwenImageTransformer2DModel as _NunchakuQwenImageTransformer2DModel
from nunchaku.models.transformers.transformer_zimage import NunchakuZImageRopeHook


class NunchakuQwenImageTransformer2DModel(_NunchakuQwenImageTransformer2DModel):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        Forward pass for the Nunchaku QwenImage transformer model with ControlNet support.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Image stream input of shape `(batch_size, image_sequence_length, in_channels)`.
        encoder_hidden_states : torch.Tensor, optional
            Text stream input of shape `(batch_size, text_sequence_length, joint_attention_dim)`.
        encoder_hidden_states_mask : torch.Tensor, optional
            Mask for encoder hidden states of shape `(batch_size, text_sequence_length)`.
        timestep : torch.LongTensor, optional
            Timestep for temporal embedding.
        img_shapes : list of tuple, optional
            Image shapes for rotary embedding.
        txt_seq_lens : list of int, optional
            Text sequence lengths.
        guidance : torch.Tensor, optional
            Guidance tensor (for classifier-free guidance).
        attention_kwargs : dict, optional
            Additional attention arguments. A kwargs dictionary that if specified is passed along to the `AttentionProcessor`.
        controlnet_block_samples : optional
            ControlNet block samples for residual connections.
        return_dict : bool, default=True
            Whether to return a dict or tuple.

        Returns
        -------
        torch.Tensor or Transformer2DModelOutput
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        device = hidden_states.device
        if self.offload:
            self.offload_manager.set_device(device)

        hidden_states = self.img_in(hidden_states)

        timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states)
        )
        
        # <PATCH_START>
        # # image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)
        
        # Use the encoder_hidden_states sequence length for RoPE computation and normalize mask
        text_seq_len, _, encoder_hidden_states_mask = compute_text_seq_len_from_mask(
            encoder_hidden_states, encoder_hidden_states_mask
        )

        # if encoder_hidden_states is not None:
        #     text_seq_len = encoder_hidden_states.shape[1]
        # else:
        #     raise ValueError("encoder_hidden_states must be provided to compute text sequence length")

        image_rotary_emb = self.pos_embed(img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device)
        
        block_attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
        if encoder_hidden_states_mask is not None:
            # Build joint mask: [text_mask, all_ones_for_image]
            batch_size, image_seq_len = hidden_states.shape[:2]
            image_mask = torch.ones((batch_size, image_seq_len), dtype=torch.bool, device=hidden_states.device)
            joint_attention_mask = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)
            block_attention_kwargs["attention_mask"] = joint_attention_mask

        # </PATCH_END>
        compute_stream = torch.cuda.current_stream()
        if self.offload:
            self.offload_manager.initialize(compute_stream)
        for block_idx, block in enumerate(self.transformer_blocks):
            with torch.cuda.stream(compute_stream):
                if self.offload:
                    block = self.offload_manager.get_block(block_idx)

                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        # encoder_hidden_states_mask,
                        None,  # Don't pass encoder_hidden_states_mask (using attention_mask instead) # PATCH
                        temb,
                        image_rotary_emb,
                        block_attention_kwargs, # PATCH
                    )
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        # encoder_hidden_states_mask=encoder_hidden_states_mask,
                        encoder_hidden_states_mask=None,  # Don't pass (using attention_mask instead) # PATCH
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        # joint_attention_kwargs=attention_kwargs,
                        joint_attention_kwargs=block_attention_kwargs, # PATCH
                    )

                # controlnet residual - same logic as in diffusers QwenImageTransformer2DModel
                if controlnet_block_samples is not None:
                    interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    hidden_states = hidden_states + controlnet_block_samples[block_idx // interval_control]

            if self.offload:
                self.offload_manager.step(compute_stream)

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        torch.cuda.empty_cache()

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
    
    def to(self, *args, **kwargs):
        """
        Override the default ``.to()`` method.

        If offload is enabled, prevents moving the model to GPU.
        Prevents changing dtype after quantization.

        Parameters
        ----------
        *args
            Positional arguments for ``.to()``.
        **kwargs
            Keyword arguments for ``.to()``.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If attempting to change dtype after quantization.
        """
        device_arg_or_kwarg_present = any(isinstance(arg, torch.device) for arg in args) or "device" in kwargs
        dtype_present_in_args = "dtype" in kwargs

        # Try converting arguments to torch.device in case they are passed as strings
        for arg in args:
            if not isinstance(arg, str):
                continue
            try:
                torch.device(arg)
                device_arg_or_kwarg_present = True
            except RuntimeError:
                pass

        if not dtype_present_in_args:
            for arg in args:
                if isinstance(arg, torch.dtype):
                    dtype_present_in_args = True
                    break

        if dtype_present_in_args and self._is_initialized:
            raise ValueError(
                "Casting a quantized model to a new `dtype` is unsupported. To set the dtype of unquantized layers, please "
                "use the `torch_dtype` argument when loading the model using `from_pretrained` or `from_single_file`."
            )
        if self.offload:
            if device_arg_or_kwarg_present:
                warn("Skipping moving the model to GPU as offload is enabled", UserWarning)
                return self
        # <PATCH_START> avoid inf recursion from inheritance
        # return super(type(self), self).to(*args, **kwargs)
        return super(_NunchakuQwenImageTransformer2DModel, self).to(*args, **kwargs)
        # </PATCH_END>

class NunchakuZImageTransformer2DModel(_NunchakuZImageTransformer2DModel):
    def forward(
        self,
        x: List[torch.Tensor],
        t,
        cap_feats: List[torch.Tensor],
        patch_size=2,
        f_patch_size=1,
        return_dict: bool = True,
    ):
        """
        Adapted from diffusers.models.transformers.transformer_z_image.ZImageTransformer2DModel#forward

        Register pre-forward hooks for caching and substitution of packed `freqs_cis` tensor for all attention submodules and unregister after forwarding is done.
        """
        rope_hook = NunchakuZImageRopeHook()
        self.register_rope_hook(rope_hook)
        try:
            # <PATCH_START>
            # return super().forward(x, t, cap_feats, patch_size, f_patch_size, return_dict)
            # return ZImageTransformer2DModel.forward(self, x, t, cap_feats, return_dict=return_dict, patch_size=patch_size, f_patch_size=f_patch_size, )
            return super(_NunchakuZImageTransformer2DModel, self).forward(x, t, cap_feats, return_dict=return_dict, patch_size=patch_size, f_patch_size=f_patch_size, )
            # </PATCH_END>
        finally:
            self.unregister_rope_hook()
            del rope_hook

