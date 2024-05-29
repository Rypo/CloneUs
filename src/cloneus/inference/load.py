import os
import gc
import time
import typing
import functools
import warnings
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GPTQConfig,
    AwqConfig,
)
import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import PeftModel, LoraConfig, get_peft_model, AutoPeftModelForCausalLM, PeftConfig, PeftModelForCausalLM
from safetensors.torch import load_model as load_model_safetensors, save_model as save_model_safetensors


def cleanup(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            torch.cuda.empty_cache()
            gc.collect()
    return wrapper  


def auto_inference_tokenizer(pretrained_model_name_or_path: str | Path, refix_tokenizer=False, *inputs, **kwargs):
    '''AutoTokenizer.from_pretrained but force padding_side=left, pad_tok=eos_tok'''
    # Fixes issues with some tokenizers special token spacing. If trained with unsloth, should have fixed+saved already. 
    if refix_tokenizer:
        from unsloth import load_correct_tokenizer
        tokenizer = load_correct_tokenizer(pretrained_model_name_or_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs, trust_remote_code=True)

    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    #tokenizer.chat_template = tokenization.bos_chat_template(tokenizer)
    return tokenizer

        
@cleanup
def load_awq(awq_dirpath, dtype=torch.float16, attn_implementation:typing.Literal["eager", "sdpa", "flash_attention_2"]="flash_attention_2"):
    '''Load AWQ model - Broken for Exllama v2
    Initial testing revealed that float16, without flash is far supperior in both speed and quality.
    Going to ignore those args for now.
    '''
    # this is broken at the momement. Passing quant_config will try to access exllama_config, but it is None
    #quant_config = AwqConfig(version="exllama", exllama_config={"version":2, "max_input_len": 8192, "max_batch_size": 8})

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        awq_dirpath, 
        low_cpu_mem_usage=True,
        #quantization_config=quant_config,
        torch_dtype=dtype, #defaults to torch.float16, seems much faster, but worse. At least in some modes it's worse. It's faster in all modes. 
        attn_implementation=attn_implementation,
        device_map="auto",
    )
    print(model.dtype, model.training)
    tokenizer = auto_inference_tokenizer(awq_dirpath, trust_remote_code=True)
    
    return model, tokenizer

@cleanup
def load_awq_exl2(awq_dirpath, max_seq_len=8192, batch_size=1, fuse_layers=False, attn_implementation:typing.Literal["eager", "sdpa", "flash_attention_2"]="flash_attention_2"):
    '''Load AWQ model
    `awq_dirpath`: Path to folder containing model files.
    `max_seq_len`: The max sequence length, used to allocate kv-cache for fused models.
    `batch_size`: The batch size to initialize the AWQ model with.
    `use_exllama_v2`: use exllamav2 inplace of GEMM 
    `fuse_layers`: Whether or not to use fused layers. Incompat with flash-attn.
    '''
    from awq import AutoAWQForCausalLM
    model = AutoAWQForCausalLM.from_quantized(
        awq_dirpath, 
        max_seq_len=max_seq_len,
        fuse_layers=fuse_layers, 
        batch_size=batch_size,
        use_exllama_v2=True,
        device_map="auto",
        attn_implementation=attn_implementation, 
    )
    tokenizer = auto_inference_tokenizer(awq_dirpath, trust_remote_code=True)

    model = model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    
    return model, tokenizer

@cleanup
def load_gptq(ckpt_dirpath, quant_config=None, dtype="auto", attn_implementation:typing.Literal["eager", "sdpa", "flash_attention_2"]="flash_attention_2"):
    if quant_config is None:
        # disable exllama kernel because training is unstable. This will overwrite the value stored in the config of the model.
        quant_config = GPTQConfig(bits=4, use_exllama=True, exllama_config={'version':2})#use_cuda_fp16=True)
    
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        ckpt_dirpath,
        low_cpu_mem_usage=True,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
    )
    tokenizer = auto_inference_tokenizer(ckpt_dirpath)

    return model, tokenizer

@cleanup
def load_gguf(gguf_filepath:str|Path, n_gpu_layers=-1, n_ctx=8192):
    from llama_cpp import Llama
    llm = Llama(str(gguf_filepath), n_gpu_layers=n_gpu_layers, n_ctx=n_ctx)
    #llm.create_chat_completion
    return llm


def load_unsloth(checkpoint_dirpath, dtype=None, attn_implementation:typing.Literal["eager", "sdpa", "flash_attention_2"]="flash_attention_2"):
    from unsloth import FastLanguageModel
    tokenizer = auto_inference_tokenizer(checkpoint_dirpath)
    warnings.warn('As of patch 2024.4, unsloth inference is incompatible with contrastive search and will throw an IndexError. Use with caution.')
    # Appears fixed: ~~can't use unsloths tokenizer without overiding chat_template, padding side, etc.~~
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = checkpoint_dirpath,
        max_seq_length = tokenizer.model_max_length,
        dtype = dtype,
        load_in_4bit = True,
        trust_remote_code = True,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,

    )
    
    FastLanguageModel.for_inference(model)

    return model, tokenizer


@cleanup
def load_any_inference(checkpoint_dirpath, quant_method:typing.Literal['awq','merged','gptq','unsloth','gguf','aqlm','bnb4', ]=None, **kwargs):
    '''Attempt to load any type of model based on quant_method or model dir structure
    
    if "awq" in model_savedir -> awq
    if "merged" in model_savdir -> merged model
    if "gptq" -> gptq
    (eventually)
    if "gguf" -> llama_cpp
    '''
    
    quant_load_methods = ['awq','merged','gptq','unsloth','gguf','aqlm','bnb4', ]
    dirstr = str(checkpoint_dirpath)
    
    if quant_method is None:        
        quant_method = next(filter(lambda q: q in dirstr, quant_load_methods), 'bnb4')

    match quant_method:
        case 'awq':
            defaults = dict(max_seq_len=8192, batch_size=1, fuse_layers=False, attn_implementation="flash_attention_2")
            #defaults = dict(dtype=torch.bfloat16, attn_implementation="flash_attention_2", freeze_eval=False)
            kargs = {k: kwargs.get(k, defaults[k]) for k in defaults}
            return load_awq_exl2(checkpoint_dirpath, **kargs)
        case 'merged':
            defaults = dict(quant_config=None, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
            kargs = {k: kwargs.get(k, defaults[k]) for k in defaults}
            return load_merged(checkpoint_dirpath, **kargs)
        case 'gptq':
            defaults = dict(quant_config=None, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
            kargs = {k: kwargs.get(k, defaults[k]) for k in defaults}
            return load_gptq(checkpoint_dirpath, **kargs)
        case 'unsloth':
            defaults = dict(dtype=torch.bfloat16, attn_implementation="flash_attention_2")
            kargs = {k: kwargs.get(k, defaults[k]) for k in defaults}
            return load_unsloth(checkpoint_dirpath, **kargs)
        case 'gguf':
            defaults = dict(n_gpu_layers=-1, n_ctx=8192)
            kargs = {k: kwargs.get(k, defaults[k]) for k in defaults}
            raise NotImplementedError('GGUF is not (yet) supported')
            #return load_gguf(model_savedir, **kargs)
        case _: # 'aqlm' , 'bnb4'
            defaults = dict(quant_method=quant_method, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
            kargs = {k: kwargs.get(k, defaults[k]) for k in defaults}
            return load_peft(checkpoint_dirpath, **kargs) # return load_unmerged(model_savedir, **kargs)
        
        # else:
            #raise ValueError('Unable to determine model type from model_savedir path')


def load_peft(checkpoint_dirpath, quant_method='bnb4', dtype=torch.bfloat16, attn_implementation:typing.Literal["eager", "sdpa", "flash_attention_2"]="flash_attention_2") -> tuple[PeftModelForCausalLM, PreTrainedTokenizer]:
    t0=time.perf_counter()
    quant_config = None
    if quant_method=='bnb4':
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    
    pt_kwargs = dict(
        low_cpu_mem_usage=True,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        use_cache=True,
        trust_remote_code=True
    )
    if pt_kwargs['quantization_config'] is None:
        pt_kwargs.pop('quantization_config')
    
    model = AutoPeftModelForCausalLM.from_pretrained(checkpoint_dirpath, **pt_kwargs,)
    
    tokenizer = auto_inference_tokenizer(checkpoint_dirpath)
    print(f'load_peft: {time.perf_counter()-t0:0.2f}s')
    return model, tokenizer

def load_unmerged(checkpoint_dirpath, quant_method='bnb4', dtype=torch.bfloat16, attn_implementation:typing.Literal["eager", "sdpa", "flash_attention_2"]="flash_attention_2") -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    t0=time.perf_counter()
    
    quant_config = None
    if quant_method=='bnb4':
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    
    if not torch.cuda.is_bf16_supported():
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        if attn_implementation == 'flash_attention_2':
            print('Flash2 not supported. Setting attn_implementation = sdpa')
            attn_implementation = 'sdpa'
        
        if dtype == torch.bfloat16:
            print('bfloat16 not supported. Setting dype = None')
            dtype=None
    

    tokenizer = auto_inference_tokenizer(checkpoint_dirpath)

    pt_kwargs = dict(
        low_cpu_mem_usage=True,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        use_cache=True,
        trust_remote_code=True
    )
    if quant_config is None:
        pt_kwargs.pop('quantization_config')
    
    peft_config = PeftConfig.from_pretrained(checkpoint_dirpath)
    base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, **pt_kwargs) #.to_bettertransformer()
    model = PeftModel.from_pretrained(base_model, model_id=checkpoint_dirpath, is_trainable=False, config=peft_config)
    
    if not model.active_adapters():
        print('No active adapters auto loaded. Attempting manual')
        lora_config = PeftConfig.from_pretrained(checkpoint_dirpath)
        model.add_adapter(lora_config)
    
    
    print(f'load_unmerged: {time.perf_counter()-t0:0.2f}s')
    return model, tokenizer


def load_merged(merged_savedir, quant_config=None, dtype=torch.bfloat16, attn_implementation:typing.Literal["eager", "sdpa", "flash_attention_2"]="flash_attention_2") -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    if quant_config is None:
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    
    model = AutoModelForCausalLM.from_pretrained(
        merged_savedir,
        low_cpu_mem_usage=True,
        quantization_config=quant_config,
        device_map="auto",
        use_safetensors=True,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    #.to_bettertransformer()
    
    tokenizer = auto_inference_tokenizer(merged_savedir)
    
    return model, tokenizer


def _model_to_merged(checkpoint_dirpath, low_cpu_mem_usage=True):
    config = PeftConfig.from_pretrained(checkpoint_dirpath)
    
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, low_cpu_mem_usage=low_cpu_mem_usage)
    model = PeftModel.from_pretrained(model, checkpoint_dirpath, config=config)
    
    merged_model = model.merge_and_unload()
    return merged_model

@cleanup
def load_merge_save(checkpoint_dirpath:str, merge_outdir='merged', low_cpu_mem_usage=True):
    merge_outpath = os.path.join(checkpoint_dirpath, merge_outdir)

    merged_model = _model_to_merged(checkpoint_dirpath, low_cpu_mem_usage=low_cpu_mem_usage)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dirpath) # merged_model.config._name_or_path
    
    merged_model.save_pretrained(merge_outpath, safe_serialization=True)
    tokenizer.save_pretrained(merge_outpath)

    print(f"Saved merged model to: {merge_outpath}")
    return merge_outpath