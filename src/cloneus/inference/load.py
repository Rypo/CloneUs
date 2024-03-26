import os
import gc
import re
import time
import typing
import random
import itertools
import functools
import warnings
from pathlib import Path
from threading import Thread
from collections import namedtuple
from contextlib import contextmanager
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import datasets
from omegaconf import OmegaConf, DictConfig

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback,
    GenerationConfig,
    BitsAndBytesConfig,
    GPTQConfig,
    StoppingCriteria,
    #TextStreamer, 
    TextIteratorStreamer
)
import transformers

from peft import PeftModel, LoraConfig, get_peft_model, AutoPeftModelForCausalLM, PeftConfig, PeftModelForCausalLM

from safetensors.torch import load_model as load_model_safetensors, save_model as save_model_safetensors

from unsloth import FastLanguageModel

from cloneus.data import roles, dataset
from cloneus.plugins import youtube as youtube

from cloneus.core import paths as cpaths
from . import genconfig



def bnb_readme_config(readme_path):
    # There *must* be a better way to go about this. This is hacky and I hate it
    with open(readme_path,'r') as f:
        readme = f.read()

    config_items = re.findall('^- (.+)', readme, re.MULTILINE)[:-1]
    bnbconf = {}
    for k,v in dict([item.split(': ') for item in config_items]).items():
        if v == 'None':
            bnbconf[k] = None
        elif v[0].isnumeric():
            bnbconf[k] = float(v)
        elif v in ['True', 'False']:
            bnbconf[k] = bool(v)
        else:
            bnbconf[k] = v

    return BitsAndBytesConfig.from_dict(bnbconf)


def warn_tokenizer(tokenizer, flash_attn):
    print(f'{tokenizer.pad_token=}, {tokenizer.unk_token=}, {tokenizer.eos_token=}')
    if flash_attn:
        if tokenizer.padding_side != 'left':
            warnings.warn('CAUTION: detected padding_side!=left. It may break flash attention. MONITOR')

def cleanup(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            torch.cuda.empty_cache()
            gc.collect()
    return wrapper    

def load_full_model(checkpoint_path, full_model_path, model_id=None, flash=False):
    peft_config = LoraConfig.from_pretrained(checkpoint_path)
    if model_id is None:
        model_id = peft_config.base_model_name_or_path

    if 'gptq' in model_id.lower():
        quant_config = GPTQConfig(bits=4, use_exllama=False) # , use_cuda_fp16=True
    else: 
        quant_config = bnb_readme_config(os.path.join(checkpoint_path,'README.md'))
        #BitsAndBytesConfig.from_dict(OmegaConf.load('config/model/bnb_default.yaml'))
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_safetensors=True,
        use_flash_attention_2=flash,
    )

    if tokenizer.additional_special_tokens:
        model.resize_token_embeddings(len(tokenizer))

    # model.config.use_cache = True
    model = get_peft_model(model, peft_config)
    model.load_state_dict(torch.load(full_model_path))
    
    return model, tokenizer



        

@cleanup
def load_awq(awq_dirpath, dtype=torch.float16, attn_implementation:typing.Literal["eager", "sdpa", "flash_attention_2"]="flash_attention_2",  freeze_eval=False):
    '''Load AWQ model
    Initial testing revealed that float16, without flash is far supperior in both speed and quality.
    Going to ignore those args for now.
    '''
    #from awq import AutoAWQForCausalLM
    #model = AutoAWQForCausalLM.from_quantized(awq_dirpath, fuse_layers=fuse_layers, safetensors=safetensors, batch_size=batch_size, offload_folder='tmp')
    # max_new_tokens=512, breaks it
            
    # quantization_config=quant_config,
    # load_in_4bit=True,
        
    model: transformers.PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        awq_dirpath, 
        low_cpu_mem_usage=True,
        #safetensors=True,
        #torch_dtype=dtype, #defaults to torch.float16, seems much faster, but worse. At least in some modes it's worse. It's faster in all modes. 
        #use_flash_attention_2=use_flash, 
        #device_map="cuda:0",
        dtype=None,
        attn_implementation=attn_implementation,
        device_map="auto",
    )
    print(model.dtype, model.training)
    tokenizer = AutoTokenizer.from_pretrained(awq_dirpath, trust_remote_code=True)

    if freeze_eval:
        model = model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    
    return model, tokenizer

# @cleanup
# def load_awq(awq_dirpath, fuse_layers=True, safetensors=True, batch_size=1, freeze_eval=False):
#     '''Load AWQ model
#     `awq_dirpath`: Path to folder containing model files.
#     `max_new_tokens`: The max sequence length, used to allocate kv-cache for fused models.
#     `fuse_layers`: Whether or not to use fused layers.
#     `batch_size`: The batch size to initialize the AWQ model with.'''
#     from awq import AutoAWQForCausalLM
#     model = AutoAWQForCausalLM.from_quantized(awq_dirpath, fuse_layers=fuse_layers, safetensors=safetensors, batch_size=batch_size, offload_folder='tmp')
#     # max_new_tokens=512, breaks it
#     tokenizer = AutoTokenizer.from_pretrained(awq_dirpath, trust_remote_code=True)

#     if freeze_eval:
#         model = model.eval()
#         for p in model.parameters():
#             p.requires_grad_(False)
    
#     return model, tokenizer

@cleanup
def load_gptq(ckpt_dirpath, quant_config=None, dtype=torch.bfloat16, attn_implementation:typing.Literal["eager", "sdpa", "flash_attention_2"]="flash_attention_2"):
    if quant_config is None:
        # disable exllama kernel because training is unstable. This will overwrite the value stored in the config of the model.
        quant_config = GPTQConfig(bits=4, use_exllama=True, exllama_config={'version':2})#use_cuda_fp16=True)
        #quant_config = GPTQConfig(bits=4, disable_exllama=True)
    
    # model = AutoPeftModelForCausalLM.from_pretrained(
    #     ckpt_dirpath, 
    #     is_trainable=False,
    #     quantization_config=quant_config, 
    #     # use_cache=False, 
    #     device_map="auto",
    #     #torch_dtype=dtype, # bf16 appears to not work
    #     low_cpu_mem_usage=True,
    #     attn_implementation=attn_implementation,
    # )
    model: transformers.PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        ckpt_dirpath,
        low_cpu_mem_usage=True,
        quantization_config=quant_config,
        #load_in_4bit=True,
        device_map="auto",
        # CANNOT USE: use_safetensors=True, since I stopped saving the full model
        #torch_dtype=dtype,
        attn_implementation=attn_implementation, # ["eager", "sdpa", "flash_attention_2"]
        #use_flash_attention_2=use_flash,
    )
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dirpath)
    warn_tokenizer(tokenizer, attn_implementation)

    return model, tokenizer


def load_unsloth(checkpoint_dirpath):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dirpath)
    warnings.warn('As of patch 2024.2, unsloth inference is incompatible with contrastive search and will throw an IndexError. Use with caution.')
    
    # can't use unsloths tokenizer without overiding chat_template, padding side, etc.
    model, _tknzr = FastLanguageModel.from_pretrained(
        model_name = checkpoint_dirpath,
        max_seq_length = tokenizer.model_max_length,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    return model, tokenizer


@cleanup
def load_any_inference(model_savedir, **kwargs):
    '''Attempt to load any type of model based on model dir structure
    
    if "merged/awq" in model_savedir -> awq
    if "merged" in model_savdir -> merged model
    (eventually)
    if "merged/gptq" -> gptq
    if "merged/gguf" -> ctransfromers
    '''
    dirstr = str(model_savedir)

    if 'merged/awq' in dirstr:
        #defaults = dict(fuse_layers=True, safetensors=True, batch_size=1, freeze_eval=False)
        defaults = dict(dtype=torch.bfloat16, attn_implementation="flash_attention_2", freeze_eval=False)
        kargs = {k: kwargs.get(k, defaults[k]) for k in defaults}
        return load_awq(model_savedir, **kargs)
    elif 'merged' in dirstr:
        defaults = dict(quant_config=None, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        kargs = {k: kwargs.get(k, defaults[k]) for k in defaults}
        return load_merged(model_savedir, **kargs)
    elif 'gptq' in dirstr:
        defaults = dict(quant_config=None, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        kargs = {k: kwargs.get(k, defaults[k]) for k in defaults}
        return load_gptq(model_savedir, **kargs)
    else:
        quant_method = 'aqlm' if 'aqlm' in dirstr else 'bnb4'
        defaults = dict(quant_method=quant_method, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        kargs = {k: kwargs.get(k, defaults[k]) for k in defaults}
        return load_peft(model_savedir, **kargs)
    # else:
    #     defaults = dict(quant_method='bnb4', dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    #     kargs = {k: kwargs.get(k, defaults[k]) for k in defaults}
    #     return load_unmerged(model_savedir, **kargs)
        #raise ValueError('Unable to determine model type from model_savedir path')

def _overide_embeddings(model, checkpoint_dirpath):
    emb_path = Path(checkpoint_dirpath).joinpath('../embeddings/')

    in_emb = model.get_input_embeddings()
    in_emb.weight.data = torch.load(emb_path/'input_embed_weights.bin')

    out_emb = model.get_output_embeddings()
    out_emb.weight.data = torch.load(emb_path/'output_embed_weights.bin')
    
    return model


def load_unmerged_customtoks(tokenizer, checkpoint_dirpath,  quant_config, dtype=torch.bfloat16, attn_implementation:typing.Literal["eager", "sdpa", "flash_attention_2"]="flash_attention_2"):
    lora_config = PeftConfig.from_pretrained(checkpoint_dirpath)
    model_id = lora_config.base_model_name_or_path

    model: transformers.PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        quantization_config=quant_config,
        device_map="auto",
        # CANNOT USE: use_safetensors=True, since I stopped saving the full model
        torch_dtype=dtype,
        attn_implementation=attn_implementation, 
    )

    model.resize_token_embeddings(len(tokenizer))
    model = _overide_embeddings(model, checkpoint_dirpath)

    lora_config = PeftConfig.from_pretrained(checkpoint_dirpath)
    model.add_adapter(lora_config)

    warn_tokenizer(tokenizer, attn_implementation)

    return model, tokenizer

def load_peft(checkpoint_dirpath, quant_method='bnb4', dtype=torch.bfloat16, attn_implementation:typing.Literal["eager", "sdpa", "flash_attention_2"]="flash_attention_2") -> tuple[PeftModelForCausalLM, transformers.PreTrainedTokenizerFast]:
    t0=time.perf_counter()
    quant_config = None
    if quant_method=='bnb4':
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    pt_kwargs = dict(
        low_cpu_mem_usage=True,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
    )
    if pt_kwargs['quantization_config'] is None:
        pt_kwargs.pop('quantization_config')
    
    model = AutoPeftModelForCausalLM.from_pretrained(checkpoint_dirpath, **pt_kwargs,)
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dirpath)
    print(f'load_peft: {time.perf_counter()-t0:0.2f}s')
    return model, tokenizer

def load_unmerged(checkpoint_dirpath, quant_method='bnb4', dtype=torch.bfloat16, attn_implementation:typing.Literal["eager", "sdpa", "flash_attention_2"]="flash_attention_2") -> typing.Tuple[
    (transformers.LlamaForCausalLM | transformers.MistralForCausalLM), transformers.PreTrainedTokenizerFast]:
    t0=time.perf_counter()
    
    if not torch.cuda.is_bf16_supported():
        return load_unmerged_lowrsc(checkpoint_dirpath, quant_config=quant_method, dtype=dtype, attn_implementation=attn_implementation)
    
    quant_config = None
    
    if quant_method=='bnb4':
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dirpath)

    pt_kwargs = dict(
        low_cpu_mem_usage=True,
        quantization_config=quant_config,
        device_map="auto",
        # CANNOT USE: use_safetensors=True, since I stopped saving the full model
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
    )
    if quant_config is None:
        pt_kwargs.pop('quantization_config')
    
    model: transformers.PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        checkpoint_dirpath,
        **pt_kwargs
    )
    #.to_bettertransformer()
    
    if not model.active_adapters():
        print('No active adapters auto loaded. Attempting manual')
        lora_config = PeftConfig.from_pretrained(checkpoint_dirpath)
        model.add_adapter(lora_config)
        #model = PeftModel.from_pretrained(model, checkpoint_dirpath)
    
    warn_tokenizer(tokenizer, attn_implementation)
    print(f'load_unmerged: {time.perf_counter()-t0:0.2f}s')
    return model, tokenizer

def load_unmerged_lowrsc(checkpoint_dirpath, quant_config=None, dtype=None, attn_implementation:typing.Literal["eager", "sdpa", "flash_attention_2"]="sdpa") -> typing.Tuple[
    (transformers.LlamaForCausalLM | transformers.MistralForCausalLM), transformers.PreTrainedTokenizerFast]:
    
    print('USING SUBOPTIMAL MODEL LOAD FOR LOW RESOURCE DEV')
    
    if attn_implementation == 'flash_attention_2':
        attn_implementation = 'eager'
        print('Flash2 not supported. Setting attn_implementation = eager')
        
    if dtype == torch.bfloat16:
        print('bfloat16 not supported. Setting dype = None')
        dtype=None
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dirpath)

    #if len(tokenizer) > 32000:
    #    return load_unmerged_customtoks(tokenizer, checkpoint_dirpath, quant_config, dtype=dtype, attn_implementation=attn_implementation)
        
    model: transformers.PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        checkpoint_dirpath,
        low_cpu_mem_usage=True,
        quantization_config=quant_config,
        #load_in_4bit=True,
        device_map="auto",
        # CANNOT USE: use_safetensors=True, since I stopped saving the full model
        torch_dtype=dtype,
        attn_implementation=attn_implementation, # ["eager", "sdpa", "flash_attention_2"]
    )
    #model = model.to_bettertransformer()
    
    if not model.active_adapters():
        lora_config = PeftConfig.from_pretrained(checkpoint_dirpath)
        model.add_adapter(lora_config)

    warn_tokenizer(tokenizer, attn_implementation)
    
    return model, tokenizer

def load_merged(merged_savedir, quant_config=None, dtype=torch.bfloat16, attn_implementation:typing.Literal["eager", "sdpa", "flash_attention_2"]="flash_attention_2") -> typing.Tuple[
    (transformers.LlamaForCausalLM | transformers.MistralForCausalLM), transformers.PreTrainedTokenizerFast]:
    if quant_config is None:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    model = AutoModelForCausalLM.from_pretrained(
        merged_savedir,
        low_cpu_mem_usage=True,
        quantization_config=quant_config,
        device_map="auto",
        use_safetensors=True,
        torch_dtype=dtype,
        attn_implementation=attn_implementation, # ["eager", "sdpa", "flash_attention_2"]
        #use_flash_attention_2=use_flash,
    )
    #.to_bettertransformer()
    
    # NOTE: Unless use dutil.get_tokenizer with **NOT** set the pad id, or set padside=right
    #tokenizer = dutil.get_tokenizer(model.config._name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(merged_savedir)
    warn_tokenizer(tokenizer, attn_implementation)
    
    return model, tokenizer

def _model_to_merged(ckpt_dir_path, low_mem=True):
    config = PeftConfig.from_pretrained(ckpt_dir_path)
    
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, low_cpu_mem_usage=low_mem)
    model = PeftModel.from_pretrained(model, ckpt_dir_path, config=config)
    
    merged_model = model.merge_and_unload()
    return merged_model

@cleanup
def load_merge_save(ckpt_dir_path:str, merge_outdir='merged', low_mem=True):
    merge_outpath = os.path.join(ckpt_dir_path, merge_outdir)

    merged_model = _model_to_merged(ckpt_dir_path, low_mem=low_mem)
    tokenizer = dataset.get_tokenizer(merged_model.config._name_or_path)
    #AutoTokenizer.from_pretrained(ckpt_dir_path)
    
    
    merged_model.save_pretrained(merge_outpath, safe_serialization=True)
    tokenizer.save_pretrained(merge_outpath)

    print(f"Saved merged model to: {merge_outpath}")
    return merge_outpath



