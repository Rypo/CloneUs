import typing
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GPTQConfig,
    TrainingArguments
)
from transformers.utils.quantization_config import  QuantizationConfigMixin
import transformers
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from unsloth import FastLanguageModel
from trl import setup_chat_format

from ..utils import misc
from ..data import tokenization

def adjust_chat_format(model, tokenizer, chat_template_format:typing.Literal['chatml']=None):
    if chat_template_format is None:
        return model, tokenizer
    
    if chat_template_format == 'chatml': 
        if any(v not in tokenizer.get_added_vocab() for v in ['<|im_start|>', '<|im_end|>']):
            print('NOTE: chat_template_format=chatml but detected non-chatml format. Missing chatml tokens will be added.')
            # tweaked based on Hermes models
            model, tokenizer = misc.setup_chat_format_patched(model, tokenizer, format='chatmlX')
    else:
        raise ValueError(f'Unsupported chat_template_format: {chat_template_format!r}')
    
    return model, tokenizer

def get_unsloth(model_id, peft_config: LoraConfig, max_seq_length=4096, chat_template_format=None, padding_side=None, custom_chat_template=None,):
    # pip install -e "git+https://github.com/unslothai/unsloth.git#egg=unsloth
    # https://pip.pypa.io/en/stable/cli/pip_install/

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length = max_seq_length,
        dtype = None,
        fix_tokenizer=True,
        load_in_4bit = (peft_config.init_lora_weights != 'loftq'),
        device_map = "sequential",
        use_gradient_checkpointing = True,
        # use_gradient_checkpointing = "unsloth",
    )
    if Path(model_id).exists():
        return model,tokenizer
    # https://old.reddit.com/r/LocalLLaMA/comments/1cc7gtr/llama3_8b_finetuning_2x_faster_fixed_endless/
    model, tokenizer = adjust_chat_format(model, tokenizer, chat_template_format)
    tokenizer = tokenization.configure_tokenizer(tokenizer, padding_side, custom_chat_template)

    # Do model patching and add fast LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        r = peft_config.r,
        target_modules = peft_config.target_modules,
        lora_alpha = peft_config.lora_alpha,
        lora_dropout = 0, # Currently only supports dropout = 0
        bias = "none",    # Currently only supports bias = "none"
        use_gradient_checkpointing = True,
        # use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        max_seq_length = max_seq_length,
        use_rslora=peft_config.use_rslora,
        init_lora_weights = peft_config.init_lora_weights,
        loftq_config=peft_config.loftq_config,
    )
    
    return model, tokenizer

def get_awq(model_id, peft_config, max_seq_len:int, batch_size:int, chat_template_format=None, padding_side=None, custom_chat_template=None, attn_implementation:typing.Literal["eager", "sdpa", "flash_attention_2"]="flash_attention_2"):
    # TODO: awq peft training?
    # - https://github.com/casper-hansen/AutoAWQ/blob/main/examples/train.py
    # test on : https://huggingface.co/solidrust/Nous-Hermes-2-Mistral-7B-DPO-AWQ/tree/main
    
    # ref1: https://github.com/huggingface/transformers/blob/096f304695f7e7b169b031f7814352e900ad71c4/src/transformers/quantizers/quantizer_awq.py#L111
    # ref2: https://github.com/huggingface/transformers/blob/096f304695f7e7b169b031f7814352e900ad71c4/src/transformers/quantizers/quantizer_awq.py#L115C29-L115C84
    from awq import AutoAWQForCausalLM
    model = AutoAWQForCausalLM.from_quantized(model_id, 
                                              fuse_layers=False, # True and False work. False was in train example  -- You cannot save an AWQ model that uses fused modules! - ref1
                                              device_map='auto', 
                                              #use_exllama_v2=True, # can't peft with it -- You cannot save an AWQ model that uses Exllama backend! - ref2
                                              safetensors=True, 
                                              max_seq_len=max_seq_len, 
                                              batch_size=batch_size,
                                              attn_implementation=attn_implementation,
                                              #torch_dtype=None, # will not load in bf16
                                              offload_folder='_tmp'
                                              )
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model, tokenizer = adjust_chat_format(model, tokenizer, chat_template_format)
    tokenizer = tokenization.configure_tokenizer(tokenizer, padding_side, custom_chat_template)
    
    # NOTE: model.model is *required*, just model will error out
    model = prepare_model_for_kbit_training(model.model, use_gradient_checkpointing=True)
    model = get_peft_model(model, peft_config)
    #model = get_peft_model(model.model, peft_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    
    #model.half()
    return model, tokenizer

def get_model(model_id, 
              peft_config: LoraConfig, 
              quant_config:typing.Literal['bnb4','aqlm', 'gptq']|QuantizationConfigMixin='bnb4', 
              attn_implementation:typing.Literal["eager", "sdpa", "flash_attention_2"]="flash_attention_2", 
              chat_template_format=None,
              padding_side=None,
              custom_chat_template=None,
              custom_tokens_map=None):
    
    if isinstance(quant_config, QuantizationConfigMixin):
        quant_config = quant_config
    elif quant_config=='bnb4':
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    elif quant_config=='gptq':
        # https://huggingface.co/docs/optimum/llm_quantization/usage_guides/quantization
        quant_config = GPTQConfig(bits=4, use_exllama=False) # , use_cuda_fp16=True
    elif quant_config=='aqlm':
        quant_config = None
    elif quant_config=='awq':
        raise NotImplementedError('use: get_awq()')
    
    pretrain_kwargs = dict(
        quantization_config=quant_config,
        use_cache=False, 
        attn_implementation=attn_implementation,
        torch_dtype='auto', 
        low_cpu_mem_usage=True,
        device_map="auto", 
    )
    if quant_config is None:
        pretrain_kwargs.pop('quantization_config') # passing as None will error as of transformers 4.39.2 (during .to_dict() call)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    if Path(model_id).exists():
        # TODO: determine if vRAM spike is because of eval step
        print('Resuming ... Skipping tokenizer configuration, loading via AutoPeftModelForCausalLM')
        model : PeftModel = AutoPeftModelForCausalLM.from_pretrained(model_id, is_trainable=True, config=peft_config, **pretrain_kwargs,)
        model.enable_input_require_grads() # required or will say does not require grad
        model.print_trainable_parameters()
        return model, tokenizer


    model = AutoModelForCausalLM.from_pretrained(model_id, **pretrain_kwargs, trust_remote_code=True)

    model, tokenizer = adjust_chat_format(model, tokenizer, chat_template_format)
    tokenizer = tokenization.configure_tokenizer(tokenizer, padding_side, custom_chat_template)


    if custom_tokens_map is not None:
        tokenization.smart_tokenizer_and_embedding_resize(custom_tokens_map, tokenizer, model)
        #model.resize_token_embeddings(len(tokenizer))

    # hf transformers handles bnb/aqlm/gptq already
    #if isinstance(quant_config, BitsAndBytesConfig):
        #model = misc.prepare_model_for_kbit_training(model)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True) # this is the default: gradient_checkpointing_kwargs=dict(use_reentrant=True))
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    
    model.bfloat16()
    #model = model.to(torch.bfloat16)
    return model, tokenizer


def model_tokenizer_from_config(peft_config, cfg, custom_token_map=None):
    name_or_path = cfg.model_id
    if cfg.resume_from_checkpoint is not None:
        name_or_path = cfg.resume_from_checkpoint
        print('Getting model from:', name_or_path)
        
        
    # TODO: Resuming from a path should work. But it will break if tokens have been added.
    # AutoPeftModelForCausalLM resizes embedding automatically, but it's unclear how to make this work with unsloth
    # Resuming using model id does work.
    # But, at least for unsloth, it is slower and uses more vRAM compared to init
    # It may have to do with Trainer's _load_from_checkpoint overwriting pieces of unsloth's patch, but just speculation. 
    # TODO: Investigate. Trainer/Unsloth/Peft checkpoint resume behavior

    if cfg.flashattn_lib=='huggingface':
        if cfg.quant_method =='awq':
            model, tokenizer = get_awq(name_or_path, 
                                        peft_config, 
                                        max_seq_len=cfg.chunk_size, 
                                        batch_size=cfg.batch_size,
                                        
                                        chat_template_format=cfg.chat_template_format, 
                                        padding_side=cfg.padding_side, 
                                        custom_chat_template=cfg.custom_chat_template,
                                        attn_implementation=cfg.attn_implementation, ) #custom_token_map
        else:
            model, tokenizer = get_model(name_or_path, 
                                        peft_config, 
                                        quant_config=cfg.quant_method, 
                                        attn_implementation=cfg.attn_implementation, 
                                        chat_template_format=cfg.chat_template_format, 
                                        padding_side=cfg.padding_side, 
                                        custom_chat_template=cfg.custom_chat_template) #custom_token_map
        

            
    elif cfg.flashattn_lib=='unsloth':
        #print('peft before:',peft_config)
        model, tokenizer = get_unsloth(name_or_path, 
                                       peft_config, 
                                       max_seq_length=cfg.chunk_size, 
                                       chat_template_format = cfg.chat_template_format, 
                                       padding_side=cfg.padding_side, 
                                       custom_chat_template=cfg.custom_chat_template)
        # TODO: look into ~4-7gb higher vRAM usage after changing padding_side=right -> padding_side=left --- https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side
        
    else:
        raise NotImplementedError(f'Unknown flashattn_lib: {cfg.flashattn_lib!r}')
    
    return model, tokenizer
