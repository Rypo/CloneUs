import os
import gc
import time
import typing
from typing import Literal
import functools
import warnings
from pathlib import Path

from omegaconf import DictConfig
import torch
from unsloth import FastLanguageModel, FastModel

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GPTQConfig,
    AwqConfig,
)
import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerBase, AutoModelForImageTextToText#, AutoModelForMultimodalLM
from peft import PeftModel, LoraConfig, get_peft_model, AutoPeftModelForCausalLM, PeftConfig, PeftModelForCausalLM
from safetensors.torch import load_model as load_model_safetensors, save_model as save_model_safetensors
from accelerate.utils import release_memory

from cloneus.data import tokenization
from cloneus.plugins import sampler_hijack

def cleanup(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            torch.cuda.empty_cache()
            gc.collect()
    return wrapper


def auto_inference_tokenizer(pretrained_model_name_or_path: str | Path, refix_tokenizer:bool=False, uncomment_chat_template_bos:bool = True, *inputs, **kwargs):
    '''AutoTokenizer.from_pretrained but force padding_side=left and if needed pad_tok=eos_tok, add bos to chat template'''
    processor = None
    # Fixes issues with some tokenizers special token spacing. If trained with unsloth, should have fixed+saved already. 
    if refix_tokenizer:
        from unsloth import load_correct_tokenizer
        tokenizer = load_correct_tokenizer(pretrained_model_name_or_path, trust_remote_code=True)
    else:
        try: 
            processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs, trust_remote_code=True)
            if not hasattr(processor, 'tokenizer'):
                raise ValueError
            tie_chat_template = (processor.chat_template == processor.tokenizer.chat_template)
            processor.tokenizer = tokenization.set_tokenizer_inference(processor.tokenizer, uncomment_chat_template_bos=uncomment_chat_template_bos)
            if tie_chat_template:
                processor.chat_template = processor.tokenizer.chat_template
            
            sampler_hijack.hijack_samplers(processor) # this enables XTC, DRY sampling methods
        except ValueError:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs, trust_remote_code=True)

            tokenizer = tokenization.set_tokenizer_inference(tokenizer, uncomment_chat_template_bos=uncomment_chat_template_bos)
            
            sampler_hijack.hijack_samplers(tokenizer) # this enables XTC, DRY sampling methods
    if processor:
        return processor
    return tokenizer

        
@cleanup
def load_awq(awq_dirpath, dtype=torch.float16, attn_implementation:Literal["eager", "sdpa", "flash_attention_2"]="flash_attention_2"):
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
def load_awq_exl2(awq_dirpath, max_seq_len=8192, batch_size=1, fuse_layers=False, attn_implementation:Literal["eager", "sdpa", "flash_attention_2"]="flash_attention_2"):
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
def load_gptq(ckpt_dirpath, quant_config=None, dtype="auto", attn_implementation:Literal["eager", "sdpa", "flash_attention_2"]="flash_attention_2"):
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

def load_gguf(model_id:str|Path, gguf_file:str = None, dtype=torch.bfloat16, attn_implementation:Literal["eager", "sdpa", "flash_attention_2"]="flash_attention_2"):
    if gguf_file is None:
        if isinstance(model_id, Path):
            gguf_file = next(model_id.rglob('*.gguf')).relative_to(model_id).as_posix()
        elif model_id.endswith('.gguf'):
            parts = model_id.split('/')
            gguf_file = parts[-1]
            model_id = '/'.join(parts[:-1])
        else:
            raise ValueError(f'Unable to parse gguf_file from model_id={model_id!r}')

    pt_kwargs = dict(
        pretrained_model_name_or_path = model_id,
        gguf_file=gguf_file,
        # quantization_config = ..., # not supported with gguf loading
        device_map = "auto",
        dtype = dtype,
        attn_implementation = attn_implementation, # "kernels-community/flash-attn2"
    )
    model = AutoModelForCausalLM.from_pretrained(**pt_kwargs)
    tokenizer = auto_inference_tokenizer(model_id, uncomment_chat_template_bos=True, force_bos_chat_template=False, gguf_file=gguf_file,)

    model = FastModel.for_inference(model).eval().requires_grad_(False)
    release_memory()

    return model, tokenizer

@torch.inference_mode()
def load_unsloth(checkpoint_dirpath:Path, quant_config=None, dtype=torch.bfloat16, attn_implementation:Literal["eager", "sdpa", "flash_attention_2"]="flash_attention_2"):
    tokenizer = auto_inference_tokenizer(checkpoint_dirpath, uncomment_chat_template_bos=True, force_bos_chat_template=False)

    peft_config = PeftConfig.from_pretrained(checkpoint_dirpath)

    pt_kwargs = dict(
        # pretrained_model_name_or_path = checkpoint_dirpath,
        pretrained_model_name_or_path = peft_config.base_model_name_or_path,
        low_cpu_mem_usage = True,
        device_map = "auto",
        dtype = dtype,
        attn_implementation = attn_implementation,
    )
    if quant_config:
        pt_kwargs.update(quantization_config=quant_config)
    
    peft_kwargs = dict(
         adapter_name = 'default', 
         is_trainable = False, 
         config = peft_config, 
         autocast_adapter_dtype = True, 
         low_cpu_mem_usage = True,
    )
    # Explictly using PeftModel.from_pretrained rather than directly loading adapter with AutoModel give access to peft methods
    if hasattr(tokenizer, 'tokenizer'):
        model = AutoModelForImageTextToText.from_pretrained(**pt_kwargs)
        model = PeftModel.from_pretrained(model, checkpoint_dirpath, **peft_kwargs)
        model = FastModel.for_inference(model)#.to(dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(**pt_kwargs)
        model = PeftModel.from_pretrained(model, checkpoint_dirpath, **peft_kwargs)
        model = FastLanguageModel.for_inference(model)#.to(dtype)
        
    model = model.eval().requires_grad_(False)
    # model.is_loaded_in_8bit = False
    release_memory()
    return model, tokenizer

@torch.inference_mode()
def load_any_inference(checkpoint_dirpath: Path, 
                       quant_method: Literal['awq','gptq','gguf','aqlm','bnb4','bnb8','bf16'] = None, 
                       load_strategy: Literal['unsloth','huggingface','merged'] = 'unsloth', 
                       attn_implementation: Literal["eager", "sdpa", "flash_attention_2"]|dict = "flash_attention_2", 
                       **kwargs):
    '''Attempt to load any type of model based on quant_method or model dir structure
    
    if "awq" in model_savedir -> awq
    if "merged" in model_savdir -> merged model
    if "gptq" -> gptq
    (eventually)
    if "gguf" -> llama_cpp
    '''
    
    quant_load_methods = ['awq','gptq','gguf','aqlm','bnb', ]
    dirstr = str(checkpoint_dirpath)
    
    if quant_method is None:        
        quant_method = next(filter(lambda q: q in dirstr, quant_load_methods), 'bnb4')
    
    # https://huggingface.co/docs/transformers/v5.2.0/en/attention_interface#backbone-specific-attention
    attn_implementation = dict(attn_implementation) if isinstance(attn_implementation, DictConfig) else attn_implementation
    
    match quant_method:
        case 'awq':
            defaults = dict(max_seq_len=8192, batch_size=1, fuse_layers=False, attn_implementation=attn_implementation)
            kargs = {k: kwargs.get(k, defaults[k]) for k in defaults}
            return load_awq_exl2(checkpoint_dirpath, **kargs)
        case 'gptq':
            defaults = dict(quant_config=None, dtype=torch.bfloat16, attn_implementation=attn_implementation)
            kargs = {k: kwargs.get(k, defaults[k]) for k in defaults}
            return load_gptq(checkpoint_dirpath, **kargs)
        case 'gguf':
            defaults = dict(dtype=torch.bfloat16, attn_implementation=attn_implementation)
            kargs = {k: kwargs.get(k, defaults[k]) for k in defaults}
            return load_gguf(checkpoint_dirpath, gguf_file = None, **kargs)
        case _: # 'aqlm' , 'bnb', bf16
            qconfigs = {
                'bnb4': BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype = torch.bfloat16, bnb_4bit_quant_type = 'nf4', bnb_4bit_use_double_quant=True),
                'bnb8': BitsAndBytesConfig(load_in_8bit=True,),
                'bf16': None,
                'aqlm': None,
            }
            
            defaults = dict(
                quant_config = qconfigs.get(quant_method, None), 
                dtype = torch.bfloat16, 
                attn_implementation = attn_implementation
            )
            
            kargs = {k: kwargs.get(k, defaults[k]) for k in defaults}

            if load_strategy == 'unsloth':
                return load_unsloth(checkpoint_dirpath, **kargs)
            elif load_strategy == 'merged':
                return load_merged(checkpoint_dirpath, **kargs)
            else:
                return load_peft(checkpoint_dirpath, **kargs) # return load_unmerged(model_savedir, **kargs)


def load_peft(checkpoint_dirpath, quant_config=None, dtype=torch.bfloat16, attn_implementation:Literal["eager", "sdpa", "flash_attention_2"]="flash_attention_2") -> tuple[PeftModelForCausalLM, PreTrainedTokenizerBase]:
    t0=time.perf_counter()
    
    pt_kwargs = dict(
        low_cpu_mem_usage=True,
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        use_cache=True,
        trust_remote_code=True
    )
    if quant_config:
        pt_kwargs.update(quantization_config=quant_config)
    
    model = AutoPeftModelForCausalLM.from_pretrained(checkpoint_dirpath, **pt_kwargs,)
    
    tokenizer = auto_inference_tokenizer(checkpoint_dirpath)
    print(f'load_peft: {time.perf_counter()-t0:0.2f}s')
    return model, tokenizer

def load_unmerged(checkpoint_dirpath, quant_method='bnb4', dtype=torch.bfloat16, attn_implementation:Literal["eager", "sdpa", "flash_attention_2"]="flash_attention_2") -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
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
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        use_cache=True,
        trust_remote_code=True
    )
    if quant_config:
        pt_kwargs.update(quantization_config=quant_config)
    
    peft_config = PeftConfig.from_pretrained(checkpoint_dirpath)
    base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, **pt_kwargs) #.to_bettertransformer()
    model = PeftModel.from_pretrained(base_model, model_id=checkpoint_dirpath, is_trainable=False, config=peft_config)
    
    if not model.active_adapters():
        print('No active adapters auto loaded. Attempting manual')
        lora_config = PeftConfig.from_pretrained(checkpoint_dirpath)
        model.add_adapter(lora_config)
    
    
    print(f'load_unmerged: {time.perf_counter()-t0:0.2f}s')
    return model, tokenizer


def load_merged(merged_savedir, quant_config=None, dtype=torch.bfloat16, attn_implementation:Literal["eager", "sdpa", "flash_attention_2"]="flash_attention_2") -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
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