import copy
from pathlib import Path

import torch

import peft
import diffusers.utils as diffutils 
from diffusers import FluxTransformer2DModel
import safetensors.torch as sft

# in SD3 original implementation of AdaLayerNormContinuous, it split linear projection output into shift, scale;
# while in diffusers it split into scale, shift. Here we swap the linear projection weights in order to be able to use diffusers implementation
def swap_scale_shift(weight, dim=0):
    shift, scale = weight.chunk(2, dim=dim)
    new_weight = torch.cat([scale, shift], dim=dim)
    return new_weight



class popdict(dict):
    def pop(self, k, d=None):
        return super().pop(k, d)

# SRC: https://github.com/huggingface/diffusers/blob/main/scripts/convert_flux_to_diffusers.py
def flux_lora_to_diffusers(original_state_dict, is_lora_up:bool, num_layers=19, num_single_layers=38, inner_dim=3072, mlp_ratio=4.0):
    DIM = 0 if is_lora_up else 1
    conv_sd = {}
    original_state_dict = popdict(copy.deepcopy(original_state_dict))
    
    ## time_text_embed.timestep_embedder <-  time_in
    conv_sd["time_text_embed.timestep_embedder.linear_1.weight"] = original_state_dict.pop("time_in.in_layer.weight")
    conv_sd["time_text_embed.timestep_embedder.linear_2.weight"] = original_state_dict.pop("time_in.out_layer.weight")
    # conv_sd["time_text_embed.timestep_embedder.linear_1.bias"] = original_state_dict.pop("time_in.in_layer.bias")
    # conv_sd["time_text_embed.timestep_embedder.linear_2.bias"] = original_state_dict.pop("time_in.out_layer.bias")

    ## time_text_embed.text_embedder <- vector_in
    conv_sd["time_text_embed.text_embedder.linear_1.weight"] = original_state_dict.pop("vector_in.in_layer.weight")
    conv_sd["time_text_embed.text_embedder.linear_2.weight"] = original_state_dict.pop("vector_in.out_layer.weight")
    # conv_sd["time_text_embed.text_embedder.linear_1.bias"] = original_state_dict.pop("vector_in.in_layer.bias")
    # conv_sd["time_text_embed.text_embedder.linear_2.bias"] = original_state_dict.pop("vector_in.out_layer.bias")

    # guidance
    has_guidance = any("guidance" in k for k in original_state_dict)
    if has_guidance:
        conv_sd["time_text_embed.guidance_embedder.linear_1.weight"] = original_state_dict.pop("guidance_in.in_layer.weight")
        conv_sd["time_text_embed.guidance_embedder.linear_2.weight"] = original_state_dict.pop("guidance_in.out_layer.weight")
        # conv_sd["time_text_embed.guidance_embedder.linear_1.bias"] = original_state_dict.pop("guidance_in.in_layer.bias")
        # conv_sd["time_text_embed.guidance_embedder.linear_2.bias"] = original_state_dict.pop("guidance_in.out_layer.bias")

    # context_embedder
    conv_sd["context_embedder.weight"] = original_state_dict.pop("txt_in.weight")
    # conv_sd["context_embedder.bias"] = original_state_dict.pop("txt_in.bias")

    # x_embedder
    conv_sd["x_embedder.weight"] = original_state_dict.pop("img_in.weight")
    # conv_sd["x_embedder.bias"] = original_state_dict.pop("img_in.bias")

    # double transformer blocks
    for i in range(num_layers):
        block_prefix = f"transformer_blocks.{i}."
        in_block_prefix = f"double_blocks.{i}."
        # norms.
        
        ## norm1
        conv_sd[f"{block_prefix}norm1.linear.weight"] = original_state_dict.pop(f"{in_block_prefix}img_mod.lin.weight")
        # conv_sd[f"{block_prefix}norm1.linear.bias"] = original_state_dict.pop(f"{in_block_prefix}img_mod.lin.bias")
        
        ## norm1_context
        conv_sd[f"{block_prefix}norm1_context.linear.weight"] = original_state_dict.pop(f"{in_block_prefix}txt_mod.lin.weight")
        # conv_sd[f"{block_prefix}norm1_context.linear.bias"] = original_state_dict.pop(f"{in_block_prefix}txt_mod.lin.bias")
        
        # Q, K, V
        sample_qkv = original_state_dict.pop(f"{in_block_prefix}img_attn.qkv.weight")
        context_qkv = original_state_dict.pop(f"{in_block_prefix}txt_attn.qkv.weight")
        
        # sample_q_bias, sample_k_bias, sample_v_bias = torch.chunk(original_state_dict.pop(f"{in_block_prefix}img_attn.qkv.bias"), 3, dim=0)
        # context_q_bias, context_k_bias, context_v_bias = torch.chunk(original_state_dict.pop(f"{in_block_prefix}txt_attn.qkv.bias"), 3, dim=0)

        if is_lora_up:
            if sample_qkv is not None:
                sample_q, sample_k, sample_v = torch.chunk(sample_qkv, 3, dim=0)
                conv_sd[f"{block_prefix}attn.to_q.weight"] = torch.cat([sample_q])
                conv_sd[f"{block_prefix}attn.to_k.weight"] = torch.cat([sample_k])
                conv_sd[f"{block_prefix}attn.to_v.weight"] = torch.cat([sample_v])
                # converted_state_dict[f"{block_prefix}attn.to_q.bias"] = torch.cat([sample_q_bias])
                # converted_state_dict[f"{block_prefix}attn.to_k.bias"] = torch.cat([sample_k_bias])
                # converted_state_dict[f"{block_prefix}attn.to_v.bias"] = torch.cat([sample_v_bias])

            
            if context_qkv is not None:
                context_q, context_k, context_v = torch.chunk(context_qkv, 3, dim=0)
                conv_sd[f"{block_prefix}attn.add_q_proj.weight"] = torch.cat([context_q])
                conv_sd[f"{block_prefix}attn.add_k_proj.weight"] = torch.cat([context_k])
                conv_sd[f"{block_prefix}attn.add_v_proj.weight"] = torch.cat([context_v])
                # converted_state_dict[f"{block_prefix}attn.add_q_proj.bias"] = torch.cat([context_q_bias])
                # converted_state_dict[f"{block_prefix}attn.add_k_proj.bias"] = torch.cat([context_k_bias])
                # converted_state_dict[f"{block_prefix}attn.add_v_proj.bias"] = torch.cat([context_v_bias])
        else:
            if sample_qkv is not None:
                for k in ['q','k','v']:
                    conv_sd[f"{block_prefix}attn.to_{k}.weight"] = sample_qkv
            
            if context_qkv is not None:
                for k in ['q','k','v']:
                    conv_sd[f"{block_prefix}attn.add_{k}_proj.weight"] = context_qkv
        
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
        # conv_sd[f"{block_prefix}ff.net.0.proj.bias"] = original_state_dict.pop(f"{in_block_prefix}img_mlp.0.bias")
        # conv_sd[f"{block_prefix}ff.net.2.bias"] = original_state_dict.pop(f"{in_block_prefix}img_mlp.2.bias")
        # conv_sd[f"{block_prefix}ff_context.net.0.proj.bias"] = original_state_dict.pop(f"{in_block_prefix}txt_mlp.0.bias")
        # conv_sd[f"{block_prefix}ff_context.net.2.bias"] = original_state_dict.pop(f"{in_block_prefix}txt_mlp.2.bias")
        
        # output projections.
        conv_sd[f"{block_prefix}attn.to_out.0.weight"] = original_state_dict.pop(f"{in_block_prefix}img_attn.proj.weight")
        conv_sd[f"{block_prefix}attn.to_add_out.weight"] = original_state_dict.pop(f"{in_block_prefix}txt_attn.proj.weight")
        # conv_sd[f"{block_prefix}attn.to_out.0.bias"] = original_state_dict.pop(f"{in_block_prefix}img_attn.proj.bias")
        # conv_sd[f"{block_prefix}attn.to_add_out.bias"] = original_state_dict.pop(f"{in_block_prefix}txt_attn.proj.bias")

    # single transfomer blocks
    for i in range(num_single_layers):
        block_prefix = f"single_transformer_blocks.{i}."
        in_block_prefix = f"single_blocks.{i}."
        
        # norm.linear  <- single_blocks.0.modulation.lin
        
        conv_sd[f"{block_prefix}norm.linear.weight"] = original_state_dict.pop(f"{in_block_prefix}modulation.lin.weight")
        # conv_sd[f"{block_prefix}norm.linear.bias"] = original_state_dict.pop(f"{in_block_prefix}modulation.lin.bias")
        
        # Q, K, V, mlp
        mlp_hidden_dim = int(inner_dim * mlp_ratio)
        split_size = (inner_dim, inner_dim, inner_dim, mlp_hidden_dim)
        
        linear1 = original_state_dict.pop(f"{in_block_prefix}linear1.weight")
        if linear1 is not None:
            if is_lora_up:
            
                q, k, v, mlp = torch.split(linear1, split_size, dim=0)
                conv_sd[f"{block_prefix}attn.to_q.weight"] = q
                conv_sd[f"{block_prefix}attn.to_k.weight"] = k
                conv_sd[f"{block_prefix}attn.to_v.weight"] = v
                conv_sd[f"{block_prefix}proj_mlp.weight"] = mlp
                # q_bias, k_bias, v_bias, mlp_bias = torch.split(original_state_dict.pop(f"{in_block_prefix}linear1.bias"), split_size, dim=0)
                # converted_state_dict[f"{block_prefix}attn.to_q.bias"] = torch.cat([q_bias])
                # converted_state_dict[f"{block_prefix}attn.to_k.bias"] = torch.cat([k_bias])
                # converted_state_dict[f"{block_prefix}attn.to_v.bias"] = torch.cat([v_bias])
                # converted_state_dict[f"{block_prefix}proj_mlp.bias"] = torch.cat([mlp_bias])
            else:
                for k in ['q','k','v']:
                    conv_sd[f"{block_prefix}attn.to_{k}.weight"] = linear1
                conv_sd[f"{block_prefix}proj_mlp.weight"] = linear1

                        
        # qk norm
        conv_sd[f"{block_prefix}attn.norm_q.weight"] = original_state_dict.pop(f"{in_block_prefix}norm.query_norm.scale")
        conv_sd[f"{block_prefix}attn.norm_k.weight"] = original_state_dict.pop(f"{in_block_prefix}norm.key_norm.scale")
        
        # output projections.
        conv_sd[f"{block_prefix}proj_out.weight"] = original_state_dict.pop(f"{in_block_prefix}linear2.weight")
        # conv_sd[f"{block_prefix}proj_out.bias"] = original_state_dict.pop(f"{in_block_prefix}linear2.bias")

    conv_sd["proj_out.weight"] = original_state_dict.pop("final_layer.linear.weight")
    # conv_sd["proj_out.bias"] = original_state_dict.pop("final_layer.linear.bias")
    
    adaLN_modulation = original_state_dict.pop("final_layer.adaLN_modulation.1.weight")
    if adaLN_modulation is not None:
        conv_sd["norm_out.linear.weight"] = swap_scale_shift(adaLN_modulation, dim=DIM)
    
    # converted_state_dict["norm_out.linear.bias"] = swap_scale_shift(original_state_dict.pop("final_layer.adaLN_modulation.1.bias"))

    return conv_sd, original_state_dict


def to_flux_sd(lora_sd):
    key_map_up = {}
    key_map_down = {}
    no_split = [
        'double_blocks',
        'single_blocks',

        'img_mlp',
        'txt_mlp',

        'img_attn',
        'txt_attn',

        'img_mod',
        'txt_mod',
    ]

    for key in lora_sd:
        if key.endswith('.alpha'):
            continue
            
        k = key.replace('lora_unet_', '')
        k,lora,weight_ext = k.split('.')
        #print(k)
        for n in no_split:
            k = k.replace(n, n.replace('_','+'))
        
        k = k.replace('_','.')
        k = k.replace('+','_')
        
        if lora == 'lora_up':
            key_map_up[key] = k + '.weight'
        elif lora == 'lora_down':
            key_map_down[key] = k + '.weight'
        else:
            print('BAD KEY', k, key)

    return key_map_up, key_map_down



def get_alpha_rank(state_dict:dict[str, torch.Tensor]):
    alpha=None
    rank=None
    for k,v in state_dict.items():
        if "lora_up" in k or 'lora_B' in k:
            rank=v.shape[1]
        if k.endswith('.alpha'):
            alpha = v#.item()
        if alpha is not None and rank is not None:
            break
    assert alpha is not None and rank is not None, 'FAILED: unable to determine alpha and rank. Did you use the unconverted state_dict?'
    
    return alpha,rank


def convert_sd(state_dict_or_path:str|Path|dict):
    # Some else's attempt: https://github.com/ostris/ai-toolkit/commit/99f24cfb0c8de876cf7366089298385aea0066fe
    # lora_down = lora_A = (sm, large) = (rank, dim1) -- e.g ([2, 3072]), ([2, 12288]) 
    # lora_up = lora_B  = (large, sm)  = (dim0, rank) -- e.g.([3074, 2]), ([12288, 2])
    if isinstance(state_dict_or_path, dict):
        orig_state_dict = state_dict_or_path
    else:
        orig_state_dict = sft.load_file(state_dict_or_path)
    
    alpha,rank = get_alpha_rank(orig_state_dict)
    
    # scaling should only be applied to either up or down since W+B@A would square otherwise
    # To apply to all, take sqrt first # torch.sqrt(w_scale)
    w_scale = alpha/rank
    print(f'LoRa weight multiplier: {w_scale}')

    keymap_up,keymap_down = to_flux_sd(orig_state_dict)

    up_weights = {keymap_up[k]: orig_state_dict[k] for k in keymap_up if not k.endswith('.alpha')}
    down_weights = {keymap_down[k]: orig_state_dict[k] for k in keymap_down if not k.endswith('.alpha')}

    conv_up_w, up_unused = flux_lora_to_diffusers(up_weights, True)
    conv_down_w, down_unused = flux_lora_to_diffusers(down_weights, False)

    lora_B_weights = {k.replace('.weight','.lora_B.weight') : v*w_scale for k,v in conv_up_w.items() if v is not None}
    lora_A_weights = {k.replace('.weight','.lora_A.weight') : v for k,v in conv_down_w.items() if v is not None}
    
    converted_lora_sd = {**lora_A_weights, **lora_B_weights}
    converted_sd = {k: v for k,v in converted_lora_sd.items()}

    return converted_sd, (up_unused, down_unused)


def manual_lora(converted_state_dict:dict[str, torch.Tensor], transformer:FluxTransformer2DModel, adapter_name:str|None=None):
    #diffutils.convert_all_state_dict_to_peft(converted_state_dict)
    rank = {k: v.shape[1] for k,v in converted_state_dict.items() if "lora_B" in k}
    lora_config_kwargs = diffutils.get_peft_kwargs(rank, network_alpha_dict=None, peft_state_dict=converted_state_dict)
    #lora_config_kwargs.pop('use_dora')
    lora_config = peft.LoraConfig(**lora_config_kwargs)

    if adapter_name is None:
        adapter_name = diffutils.get_adapter_name(transformer)
    
    peft.inject_adapter_in_model(lora_config, transformer, adapter_name=adapter_name)
    incompatible_keys = peft.set_peft_model_state_dict(transformer, converted_state_dict, adapter_name=adapter_name)
    
    return transformer, incompatible_keys