from pathlib import Path

import torch
from diffusers import FluxTransformer2DModel, FluxPipeline, SD3Transformer2DModel
import safetensors.torch as sft

from tqdm.auto import tqdm

from cloneus.utils.common import release_memory,read_state_dict

# in SD3 original implementation of AdaLayerNormContinuous, it split linear projection output into shift, scale;
# while in diffusers it split into scale, shift. Here we swap the linear projection weights in order to be able to use diffusers implementation
def swap_scale_shift(weight, dim=0):
    if weight is None:
        return None
    shift, scale = weight.chunk(2, dim=dim)
    new_weight = torch.cat([scale, shift], dim=dim)
    return new_weight

class popdict(dict):
    def pop(self, k, d=None):
        return super().pop(k, d)

def load_original_checkpoint(ckpt_path):
    original_state_dict = sft.load_file(ckpt_path)
    keys = list(original_state_dict.keys())
    for k in keys:
        if "model.diffusion_model." in k:
            original_state_dict[k.replace("model.diffusion_model.", "")] = original_state_dict.pop(k)
        elif "diffusion_model." in k:
            original_state_dict[k.replace("diffusion_model.", "")] = original_state_dict.pop(k)

    return original_state_dict

def chunk_or_repeat(input:torch.Tensor, chunks:int, dim=0, is_lora_A:bool=False):
    # diffusers likes to split up qkv, so if doing lora conversion, chunk lora_B, and copy lora_A `chunks` times 
    if is_lora_A:
        return input.expand(chunks, *input.shape)
    return torch.chunk(input, chunks, dim)

def split_or_repeat(tensor:torch.Tensor, split_size_or_sections: int | list[int], dim: int = 0, is_lora_A:bool=False):
    if is_lora_A:
        if not isinstance(split_size_or_sections, int):
            split_size_or_sections = len(split_size_or_sections)
        return tensor.expand(split_size_or_sections, *tensor.shape)
    return torch.split(tensor, split_size_or_sections, dim)



# Adapted from: https://github.com/huggingface/diffusers/blob/main/scripts/convert_sd3_to_diffusers.py
def convert_sd3_transformer_checkpoint_to_diffusers(original_state_dict, num_layers, caption_projection_dim, dual_attention_layers, has_qk_norm, is_lora_A=False):
    
    converted_state_dict = {}

    # Positional and patch embeddings.
    converted_state_dict["pos_embed.pos_embed"] = original_state_dict.pop("pos_embed")
    converted_state_dict["pos_embed.proj.weight"] = original_state_dict.pop("x_embedder.proj.weight")
    converted_state_dict["pos_embed.proj.bias"] = original_state_dict.pop("x_embedder.proj.bias")

    # Timestep embeddings.
    converted_state_dict["time_text_embed.timestep_embedder.linear_1.weight"] = original_state_dict.pop("t_embedder.mlp.0.weight")
    converted_state_dict["time_text_embed.timestep_embedder.linear_1.bias"] = original_state_dict.pop("t_embedder.mlp.0.bias")
    converted_state_dict["time_text_embed.timestep_embedder.linear_2.weight"] = original_state_dict.pop("t_embedder.mlp.2.weight")
    converted_state_dict["time_text_embed.timestep_embedder.linear_2.bias"] = original_state_dict.pop("t_embedder.mlp.2.bias")

    # Context projections.
    converted_state_dict["context_embedder.weight"] = original_state_dict.pop("context_embedder.weight")
    converted_state_dict["context_embedder.bias"] = original_state_dict.pop("context_embedder.bias")

    # Pooled context projection.
    converted_state_dict["time_text_embed.text_embedder.linear_1.weight"] = original_state_dict.pop("y_embedder.mlp.0.weight")
    converted_state_dict["time_text_embed.text_embedder.linear_1.bias"] = original_state_dict.pop("y_embedder.mlp.0.bias")
    converted_state_dict["time_text_embed.text_embedder.linear_2.weight"] = original_state_dict.pop("y_embedder.mlp.2.weight")
    converted_state_dict["time_text_embed.text_embedder.linear_2.bias"] = original_state_dict.pop("y_embedder.mlp.2.bias")

    # Transformer blocks 🎸.
    for i in range(num_layers):
        # Q, K, V
        
        if (x_block_qkv := original_state_dict.pop(f"joint_blocks.{i}.x_block.attn.qkv.weight")) is not None:
            sample_q, sample_k, sample_v = chunk_or_repeat(x_block_qkv, 3, dim=0, is_lora_A=is_lora_A)
            converted_state_dict[f"transformer_blocks.{i}.attn.to_q.weight"] = torch.cat([sample_q])
            converted_state_dict[f"transformer_blocks.{i}.attn.to_k.weight"] = torch.cat([sample_k])
            converted_state_dict[f"transformer_blocks.{i}.attn.to_v.weight"] = torch.cat([sample_v])

        if (ctx_block_qkv := original_state_dict.pop(f"joint_blocks.{i}.context_block.attn.qkv.weight"))  is not None:
            context_q, context_k, context_v = chunk_or_repeat(ctx_block_qkv, 3, dim=0, is_lora_A=is_lora_A)
            converted_state_dict[f"transformer_blocks.{i}.attn.add_q_proj.weight"] = torch.cat([context_q])
            converted_state_dict[f"transformer_blocks.{i}.attn.add_k_proj.weight"] = torch.cat([context_k])
            converted_state_dict[f"transformer_blocks.{i}.attn.add_v_proj.weight"] = torch.cat([context_v])


        if (x_block_qkv_bias := original_state_dict.pop(f"joint_blocks.{i}.x_block.attn.qkv.bias")) is not None:
            sample_q_bias, sample_k_bias, sample_v_bias = chunk_or_repeat(x_block_qkv_bias, 3, dim=0, is_lora_A=is_lora_A)
            converted_state_dict[f"transformer_blocks.{i}.attn.to_q.bias"] = torch.cat([sample_q_bias])
            converted_state_dict[f"transformer_blocks.{i}.attn.to_k.bias"] = torch.cat([sample_k_bias])
            converted_state_dict[f"transformer_blocks.{i}.attn.to_v.bias"] = torch.cat([sample_v_bias])
        
        if (ctx_block_qkv_bias := original_state_dict.pop(f"joint_blocks.{i}.context_block.attn.qkv.bias"))  is not None:
            context_q_bias, context_k_bias, context_v_bias = chunk_or_repeat(ctx_block_qkv_bias, 3, dim=0, is_lora_A=is_lora_A)
            converted_state_dict[f"transformer_blocks.{i}.attn.add_q_proj.bias"] = torch.cat([context_q_bias])
            converted_state_dict[f"transformer_blocks.{i}.attn.add_k_proj.bias"] = torch.cat([context_k_bias])
            converted_state_dict[f"transformer_blocks.{i}.attn.add_v_proj.bias"] = torch.cat([context_v_bias])

        # qk norm
        if has_qk_norm:
            converted_state_dict[f"transformer_blocks.{i}.attn.norm_q.weight"] = original_state_dict.pop(f"joint_blocks.{i}.x_block.attn.ln_q.weight")
            converted_state_dict[f"transformer_blocks.{i}.attn.norm_k.weight"] = original_state_dict.pop(f"joint_blocks.{i}.x_block.attn.ln_k.weight")
            converted_state_dict[f"transformer_blocks.{i}.attn.norm_added_q.weight"] = original_state_dict.pop(f"joint_blocks.{i}.context_block.attn.ln_q.weight")
            converted_state_dict[f"transformer_blocks.{i}.attn.norm_added_k.weight"] = original_state_dict.pop(f"joint_blocks.{i}.context_block.attn.ln_k.weight")

        # output projections.
        converted_state_dict[f"transformer_blocks.{i}.attn.to_out.0.weight"] = original_state_dict.pop(f"joint_blocks.{i}.x_block.attn.proj.weight")
        converted_state_dict[f"transformer_blocks.{i}.attn.to_out.0.bias"] = original_state_dict.pop(f"joint_blocks.{i}.x_block.attn.proj.bias")
        if not (i == num_layers - 1):
            converted_state_dict[f"transformer_blocks.{i}.attn.to_add_out.weight"] = original_state_dict.pop(f"joint_blocks.{i}.context_block.attn.proj.weight")
            converted_state_dict[f"transformer_blocks.{i}.attn.to_add_out.bias"] = original_state_dict.pop(f"joint_blocks.{i}.context_block.attn.proj.bias")

        # attn2
        if i in dual_attention_layers:
            # Q, K, V
            if (x_block_attn2qkv:=original_state_dict.pop(f"joint_blocks.{i}.x_block.attn2.qkv.weight")) is not None:
                sample_q2, sample_k2, sample_v2 = chunk_or_repeat(x_block_attn2qkv, 3, dim=0, is_lora_A=is_lora_A)

                converted_state_dict[f"transformer_blocks.{i}.attn2.to_q.weight"] = torch.cat([sample_q2])
                converted_state_dict[f"transformer_blocks.{i}.attn2.to_k.weight"] = torch.cat([sample_k2])
                converted_state_dict[f"transformer_blocks.{i}.attn2.to_v.weight"] = torch.cat([sample_v2])
            
            if (x_block_attn2qkv_bias:=original_state_dict.pop(f"joint_blocks.{i}.x_block.attn2.qkv.bias")) is not None:
                sample_q2_bias, sample_k2_bias, sample_v2_bias = chunk_or_repeat(x_block_attn2qkv_bias, 3, dim=0, is_lora_A=is_lora_A)

                converted_state_dict[f"transformer_blocks.{i}.attn2.to_q.bias"] = torch.cat([sample_q2_bias])
                converted_state_dict[f"transformer_blocks.{i}.attn2.to_k.bias"] = torch.cat([sample_k2_bias])
                converted_state_dict[f"transformer_blocks.{i}.attn2.to_v.bias"] = torch.cat([sample_v2_bias])

            # qk norm
            if has_qk_norm:
                converted_state_dict[f"transformer_blocks.{i}.attn2.norm_q.weight"] = original_state_dict.pop(f"joint_blocks.{i}.x_block.attn2.ln_q.weight")
                converted_state_dict[f"transformer_blocks.{i}.attn2.norm_k.weight"] = original_state_dict.pop(f"joint_blocks.{i}.x_block.attn2.ln_k.weight")

            # output projections.
            converted_state_dict[f"transformer_blocks.{i}.attn2.to_out.0.weight"] = original_state_dict.pop(f"joint_blocks.{i}.x_block.attn2.proj.weight")
            converted_state_dict[f"transformer_blocks.{i}.attn2.to_out.0.bias"] = original_state_dict.pop(f"joint_blocks.{i}.x_block.attn2.proj.bias")

        # norms.
        converted_state_dict[f"transformer_blocks.{i}.norm1.linear.weight"] = original_state_dict.pop(f"joint_blocks.{i}.x_block.adaLN_modulation.1.weight")
        converted_state_dict[f"transformer_blocks.{i}.norm1.linear.bias"] = original_state_dict.pop(f"joint_blocks.{i}.x_block.adaLN_modulation.1.bias")
        if not (i == num_layers - 1):
            converted_state_dict[f"transformer_blocks.{i}.norm1_context.linear.weight"] = original_state_dict.pop(f"joint_blocks.{i}.context_block.adaLN_modulation.1.weight")
            converted_state_dict[f"transformer_blocks.{i}.norm1_context.linear.bias"] = original_state_dict.pop(f"joint_blocks.{i}.context_block.adaLN_modulation.1.bias")
        else:
            converted_state_dict[f"transformer_blocks.{i}.norm1_context.linear.weight"] = swap_scale_shift(
                original_state_dict.pop(f"joint_blocks.{i}.context_block.adaLN_modulation.1.weight"), dim=caption_projection_dim,)
            converted_state_dict[f"transformer_blocks.{i}.norm1_context.linear.bias"] = swap_scale_shift(
                original_state_dict.pop(f"joint_blocks.{i}.context_block.adaLN_modulation.1.bias"), dim=caption_projection_dim,)

        # ffs.
        converted_state_dict[f"transformer_blocks.{i}.ff.net.0.proj.weight"] = original_state_dict.pop(f"joint_blocks.{i}.x_block.mlp.fc1.weight")
        converted_state_dict[f"transformer_blocks.{i}.ff.net.0.proj.bias"] = original_state_dict.pop(f"joint_blocks.{i}.x_block.mlp.fc1.bias")
        converted_state_dict[f"transformer_blocks.{i}.ff.net.2.weight"] = original_state_dict.pop(f"joint_blocks.{i}.x_block.mlp.fc2.weight")
        converted_state_dict[f"transformer_blocks.{i}.ff.net.2.bias"] = original_state_dict.pop(f"joint_blocks.{i}.x_block.mlp.fc2.bias")
        if not (i == num_layers - 1):
            converted_state_dict[f"transformer_blocks.{i}.ff_context.net.0.proj.weight"] = original_state_dict.pop(f"joint_blocks.{i}.context_block.mlp.fc1.weight")
            converted_state_dict[f"transformer_blocks.{i}.ff_context.net.0.proj.bias"] = original_state_dict.pop(f"joint_blocks.{i}.context_block.mlp.fc1.bias")
            converted_state_dict[f"transformer_blocks.{i}.ff_context.net.2.weight"] = original_state_dict.pop(f"joint_blocks.{i}.context_block.mlp.fc2.weight")
            converted_state_dict[f"transformer_blocks.{i}.ff_context.net.2.bias"] = original_state_dict.pop(f"joint_blocks.{i}.context_block.mlp.fc2.bias")

    # Final blocks.
    converted_state_dict["proj_out.weight"] = original_state_dict.pop("final_layer.linear.weight")
    converted_state_dict["proj_out.bias"] = original_state_dict.pop("final_layer.linear.bias")
    converted_state_dict["norm_out.linear.weight"] = swap_scale_shift(original_state_dict.pop("final_layer.adaLN_modulation.1.weight"), dim=caption_projection_dim)
    converted_state_dict["norm_out.linear.bias"] = swap_scale_shift(original_state_dict.pop("final_layer.adaLN_modulation.1.bias"), dim=caption_projection_dim)

    return converted_state_dict


def sd3_to_diffusers(original_ckpt:dict|str, dtype=torch.float16, allow_missing=True, is_lora_A=False, base_model_id:str|None='stabilityai/stable-diffusion-3.5-large'):
    config = SD3Transformer2DModel.load_config(base_model_id, subfolder='transformer') if base_model_id else {}
    
    def is_vae_in_checkpoint(original_state_dict):
        return all(w in original_state_dict for w in ["first_stage_model.decoder.conv_in.weight","first_stage_model.encoder.conv_in.weight"])

    def get_attn2_layers(state_dict):
        attn2_layers = []
        for key in state_dict.keys():
            if "attn2." in key:
                # Extract the layer number from the key
                layer_num = int(key.split(".")[1])
                attn2_layers.append(layer_num)
        return tuple(sorted(set(attn2_layers)))

    def get_pos_embed_max_size(state_dict):
        num_patches = state_dict["pos_embed"].shape[1]
        pos_embed_max_size = int(num_patches**0.5)
        return pos_embed_max_size

    def get_caption_projection_dim(state_dict):
        caption_projection_dim = state_dict["context_embedder.weight"].shape[0]
        return caption_projection_dim

    
    # -------------------------------------------------------
    
    if not isinstance(original_ckpt, dict):
        original_ckpt = load_original_checkpoint(original_ckpt)
    
    if allow_missing:
        original_ckpt = popdict(original_ckpt)

    original_dtype = next(iter(original_ckpt.values())).dtype

    # Initialize dtype with a default value
    #dtype = None

    if dtype != original_dtype:
        print(f"Checkpoint dtype {original_dtype} does not match requested dtype {dtype}. "
              "This can lead to unexpected results, proceed with caution.")

    if config:
        num_layers = config.get('num_layers')
        caption_projection_dim = config.get('caption_projection_dim')
        has_qk_norm = 'qk_norm' in config
        #pos_embed_max_size = config.get('pos_embed_max_size')
    else:
        num_layers = list(set(int(k.split(".", 2)[1]) for k in original_ckpt if "joint_blocks" in k))[-1] + 1  # noqa: C401
        caption_projection_dim = get_caption_projection_dim(original_ckpt)
        # sd3.5 use qk norm("rms_norm")
        has_qk_norm = any("ln_q" in key for key in original_ckpt.keys())
        # sd3.5 2b use pox_embed_max_size=384 and sd3.0 and sd3.5 8b use 192
        # pos_embed_max_size = get_pos_embed_max_size(original_ckpt)


    # () for sd3.0; (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) for sd3.5
    attn2_layers = get_attn2_layers(original_ckpt)

    converted_transformer_state_dict = convert_sd3_transformer_checkpoint_to_diffusers(
        original_ckpt, num_layers, caption_projection_dim, attn2_layers, has_qk_norm, is_lora_A
    )

    missing_keys = []
    for k in converted_transformer_state_dict:
        if converted_transformer_state_dict[k] is None:
            missing_keys.append(k)
            
        
    converted_transformer_state_dict = {k: v for k,v in converted_transformer_state_dict.items() if v is not None}
    
    return converted_transformer_state_dict, missing_keys


# Adapted from: https://github.com/huggingface/diffusers/blob/main/scripts/convert_flux_to_diffusers.py
def convert_flux_transformer_checkpoint_to_diffusers(original_state_dict, num_layers, num_single_layers, inner_dim, mlp_ratio, is_lora_A:bool = False):
    DIM = 1 if is_lora_A else 0

    conv_sd = {}

    ## time_text_embed.timestep_embedder <-  time_in
    conv_sd["time_text_embed.timestep_embedder.linear_1.weight"] = original_state_dict.pop("time_in.in_layer.weight")
    conv_sd["time_text_embed.timestep_embedder.linear_2.weight"] = original_state_dict.pop("time_in.out_layer.weight")
    conv_sd["time_text_embed.timestep_embedder.linear_1.bias"] = original_state_dict.pop("time_in.in_layer.bias")
    conv_sd["time_text_embed.timestep_embedder.linear_2.bias"] = original_state_dict.pop("time_in.out_layer.bias")

    ## time_text_embed.text_embedder <- vector_in
    conv_sd["time_text_embed.text_embedder.linear_1.weight"] = original_state_dict.pop("vector_in.in_layer.weight")
    conv_sd["time_text_embed.text_embedder.linear_2.weight"] = original_state_dict.pop("vector_in.out_layer.weight")
    conv_sd["time_text_embed.text_embedder.linear_1.bias"] = original_state_dict.pop("vector_in.in_layer.bias")
    conv_sd["time_text_embed.text_embedder.linear_2.bias"] = original_state_dict.pop("vector_in.out_layer.bias")

    # guidance
    
    if (has_guidance := any("guidance" in k for k in original_state_dict)):
        conv_sd["time_text_embed.guidance_embedder.linear_1.weight"] = original_state_dict.pop("guidance_in.in_layer.weight")
        conv_sd["time_text_embed.guidance_embedder.linear_2.weight"] = original_state_dict.pop("guidance_in.out_layer.weight")
        conv_sd["time_text_embed.guidance_embedder.linear_1.bias"] = original_state_dict.pop("guidance_in.in_layer.bias")
        conv_sd["time_text_embed.guidance_embedder.linear_2.bias"] = original_state_dict.pop("guidance_in.out_layer.bias")

    # context_embedder
    conv_sd["context_embedder.weight"] = original_state_dict.pop("txt_in.weight")
    conv_sd["context_embedder.bias"] = original_state_dict.pop("txt_in.bias")

    # x_embedder
    conv_sd["x_embedder.weight"] = original_state_dict.pop("img_in.weight")
    conv_sd["x_embedder.bias"] = original_state_dict.pop("img_in.bias")

    # double transformer blocks
    for i in range(num_layers):
        block_prefix = f"transformer_blocks.{i}."
        in_block_prefix = f"double_blocks.{i}."
        # norms.
        
        ## norm1
        conv_sd[f"{block_prefix}norm1.linear.weight"] = original_state_dict.pop(f"{in_block_prefix}img_mod.lin.weight")
        conv_sd[f"{block_prefix}norm1.linear.bias"] = original_state_dict.pop(f"{in_block_prefix}img_mod.lin.bias")
        
        ## norm1_context
        conv_sd[f"{block_prefix}norm1_context.linear.weight"] = original_state_dict.pop(f"{in_block_prefix}txt_mod.lin.weight")
        conv_sd[f"{block_prefix}norm1_context.linear.bias"] = original_state_dict.pop(f"{in_block_prefix}txt_mod.lin.bias")
        
        # Q, K, V
        if (sample_qkv := original_state_dict.pop(f"{in_block_prefix}img_attn.qkv.weight")) is not None:
            sample_q, sample_k, sample_v = chunk_or_repeat(sample_qkv, 3, dim=0, is_lora_A=is_lora_A)
            conv_sd[f"{block_prefix}attn.to_q.weight"] = torch.cat([sample_q])
            conv_sd[f"{block_prefix}attn.to_k.weight"] = torch.cat([sample_k])
            conv_sd[f"{block_prefix}attn.to_v.weight"] = torch.cat([sample_v])
        
        if (context_qkv := original_state_dict.pop(f"{in_block_prefix}txt_attn.qkv.weight")) is not None:
            context_q, context_k, context_v = chunk_or_repeat(context_qkv, 3, dim=0, is_lora_A=is_lora_A)
            conv_sd[f"{block_prefix}attn.add_q_proj.weight"] = torch.cat([context_q])
            conv_sd[f"{block_prefix}attn.add_k_proj.weight"] = torch.cat([context_k])
            conv_sd[f"{block_prefix}attn.add_v_proj.weight"] = torch.cat([context_v])
        
        if (sample_qkv_bias := original_state_dict.pop(f"{in_block_prefix}img_attn.qkv.bias")) is not None:
            sample_q_bias, sample_k_bias, sample_v_bias = chunk_or_repeat(sample_qkv_bias, 3, dim=0, is_lora_A=is_lora_A)
            conv_sd[f"{block_prefix}attn.to_q.bias"] = torch.cat([sample_q_bias])
            conv_sd[f"{block_prefix}attn.to_k.bias"] = torch.cat([sample_k_bias])
            conv_sd[f"{block_prefix}attn.to_v.bias"] = torch.cat([sample_v_bias])

        if (context_qkv_bias := original_state_dict.pop(f"{in_block_prefix}txt_attn.qkv.bias")) is not None:
            context_q_bias, context_k_bias, context_v_bias = chunk_or_repeat(context_qkv_bias, 3, dim=0, is_lora_A=is_lora_A)
            conv_sd[f"{block_prefix}attn.add_q_proj.bias"] = torch.cat([context_q_bias])
            conv_sd[f"{block_prefix}attn.add_k_proj.bias"] = torch.cat([context_k_bias])
            conv_sd[f"{block_prefix}attn.add_v_proj.bias"] = torch.cat([context_v_bias])
            
        
        # qk_norm
        conv_sd[f"{block_prefix}attn.norm_q.weight"] = original_state_dict.pop(f"{in_block_prefix}img_attn.norm.query_norm.scale")
        conv_sd[f"{block_prefix}attn.norm_k.weight"] = original_state_dict.pop(f"{in_block_prefix}img_attn.norm.key_norm.scale")
        conv_sd[f"{block_prefix}attn.norm_added_q.weight"] = original_state_dict.pop(f"{in_block_prefix}txt_attn.norm.query_norm.scale")
        conv_sd[f"{block_prefix}attn.norm_added_k.weight"] = original_state_dict.pop(f"{in_block_prefix}txt_attn.norm.key_norm.scale")
        
        # ff img_mlp
        conv_sd[f"{block_prefix}ff.net.0.proj.weight"] = original_state_dict.pop(f"{in_block_prefix}img_mlp.0.weight")
        conv_sd[f"{block_prefix}ff.net.2.weight"] = original_state_dict.pop(f"{in_block_prefix}img_mlp.2.weight")
        conv_sd[f"{block_prefix}ff_context.net.0.proj.weight"] = original_state_dict.pop(f"{in_block_prefix}txt_mlp.0.weight")
        conv_sd[f"{block_prefix}ff_context.net.2.weight"] = original_state_dict.pop(f"{in_block_prefix}txt_mlp.2.weight")

        conv_sd[f"{block_prefix}ff.net.0.proj.bias"] = original_state_dict.pop(f"{in_block_prefix}img_mlp.0.bias")
        conv_sd[f"{block_prefix}ff.net.2.bias"] = original_state_dict.pop(f"{in_block_prefix}img_mlp.2.bias")
        conv_sd[f"{block_prefix}ff_context.net.0.proj.bias"] = original_state_dict.pop(f"{in_block_prefix}txt_mlp.0.bias")
        conv_sd[f"{block_prefix}ff_context.net.2.bias"] = original_state_dict.pop(f"{in_block_prefix}txt_mlp.2.bias")
        
        # output projections.
        conv_sd[f"{block_prefix}attn.to_out.0.weight"] = original_state_dict.pop(f"{in_block_prefix}img_attn.proj.weight")
        conv_sd[f"{block_prefix}attn.to_add_out.weight"] = original_state_dict.pop(f"{in_block_prefix}txt_attn.proj.weight")
        conv_sd[f"{block_prefix}attn.to_out.0.bias"] = original_state_dict.pop(f"{in_block_prefix}img_attn.proj.bias")
        conv_sd[f"{block_prefix}attn.to_add_out.bias"] = original_state_dict.pop(f"{in_block_prefix}txt_attn.proj.bias")

    # single transfomer blocks
    for i in range(num_single_layers):
        block_prefix = f"single_transformer_blocks.{i}."
        in_block_prefix = f"single_blocks.{i}."
        
        # norm.linear  <- single_blocks.0.modulation.lin
        
        conv_sd[f"{block_prefix}norm.linear.weight"] = original_state_dict.pop(f"{in_block_prefix}modulation.lin.weight")
        conv_sd[f"{block_prefix}norm.linear.bias"] = original_state_dict.pop(f"{in_block_prefix}modulation.lin.bias")
        
        # Q, K, V, mlp
        mlp_hidden_dim = int(inner_dim * mlp_ratio)
        split_size = (inner_dim, inner_dim, inner_dim, mlp_hidden_dim)
        
        if (linear1 := original_state_dict.pop(f"{in_block_prefix}linear1.weight")) is not None:
            q, k, v, mlp = split_or_repeat(linear1, split_size, dim=0, is_lora_A=is_lora_A)
            conv_sd[f"{block_prefix}attn.to_q.weight"] = q
            conv_sd[f"{block_prefix}attn.to_k.weight"] = k
            conv_sd[f"{block_prefix}attn.to_v.weight"] = v
            conv_sd[f"{block_prefix}proj_mlp.weight"] = mlp

        if (linear1_bias := original_state_dict.pop(f"{in_block_prefix}linear1.bias")) is not None:
            q_bias, k_bias, v_bias, mlp_bias = split_or_repeat(linear1_bias, split_size, dim=0, is_lora_A=is_lora_A)
            conv_sd[f"{block_prefix}attn.to_q.bias"] = torch.cat([q_bias])
            conv_sd[f"{block_prefix}attn.to_k.bias"] = torch.cat([k_bias])
            conv_sd[f"{block_prefix}attn.to_v.bias"] = torch.cat([v_bias])
            conv_sd[f"{block_prefix}proj_mlp.bias"] = torch.cat([mlp_bias])
                        
        # qk norm
        conv_sd[f"{block_prefix}attn.norm_q.weight"] = original_state_dict.pop(f"{in_block_prefix}norm.query_norm.scale")
        conv_sd[f"{block_prefix}attn.norm_k.weight"] = original_state_dict.pop(f"{in_block_prefix}norm.key_norm.scale")
        
        # output projections.
        conv_sd[f"{block_prefix}proj_out.weight"] = original_state_dict.pop(f"{in_block_prefix}linear2.weight")
        conv_sd[f"{block_prefix}proj_out.bias"] = original_state_dict.pop(f"{in_block_prefix}linear2.bias")

    conv_sd["proj_out.weight"] = original_state_dict.pop("final_layer.linear.weight")
    conv_sd["proj_out.bias"] = original_state_dict.pop("final_layer.linear.bias")
    
    conv_sd["norm_out.linear.weight"] = swap_scale_shift(original_state_dict.pop("final_layer.adaLN_modulation.1.weight"), dim=DIM)
    conv_sd["norm_out.linear.bias"] = swap_scale_shift(original_state_dict.pop("final_layer.adaLN_modulation.1.bias"))


    conv_sd = {k: v for k,v in conv_sd.items() if v is not None}

    return conv_sd


def flux_to_diffusers(original_ckpt:str|dict, dtype:torch.dtype = torch.bfloat16, allow_missing:bool=True, is_lora_A:bool = False,  base_model_id:str|None='black-forest-labs/FLUX.1-dev'):
    config = FluxTransformer2DModel.load_config(base_model_id, subfolder='transformer') if base_model_id else {}
    
    num_layers = config.get('num_layers', 19)
    num_single_layers = config.get('num_single_layers', 38)
    inner_dim = 3072 # config.get('attention_head_dim', 128) * config.get('num_attention_heads', 24)
    mlp_ratio = 4.0

    if not isinstance(original_ckpt, dict):
        original_ckpt = load_original_checkpoint(original_ckpt)

    if allow_missing:
        original_ckpt = popdict(original_ckpt)
    
    converted_transformer_state_dict = convert_flux_transformer_checkpoint_to_diffusers(
        original_ckpt, num_layers, num_single_layers, inner_dim, mlp_ratio, is_lora_A = is_lora_A,
    )

    missing_keys = []
    for k in converted_transformer_state_dict:
        if converted_transformer_state_dict[k] is None:
            missing_keys.append(k)
            
        
    converted_transformer_state_dict = {k: v for k,v in converted_transformer_state_dict.items() if v is not None}
    
    return converted_transformer_state_dict, missing_keys