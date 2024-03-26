import typing
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GPTQConfig,
    TrainingArguments
)
from transformers.utils.quantization_config import  QuantizationConfigMixin
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model
from unsloth import FastLanguageModel
from trl import setup_chat_format

from ..utils import misc
from ..data import tokenization

def adjust_tune_format(model, tokenizer, tune_type:str, padding_side:str, custom_chat_template:str):
    if tune_type == 'chatml' and any(v not in tokenizer.get_added_vocab() for v in ['<|im_start|>', '<|im_end|>']):
        print('Warning: tune_type=chatml but detected non-chatml format. Missing chatml tokens will be added.')
        model, tokenizer = setup_chat_format(model, tokenizer)
        if tokenizer.pad_token != '</s>':
            print(f'Using </s> for pad instead of {tokenizer.pad_token}.')
            tokenizer.pad_token = '</s>'
    
    tokenizer = tokenization.configure_tokenizer(tokenizer, padding_side, custom_chat_template)
    
    return model, tokenizer

def get_unsloth(model_id, peft_config: LoraConfig, max_seq_length=4096, tune_type='chatml', padding_side=None, custom_chat_template=None,):
    # pip install -e "git+https://github.com/unslothai/unsloth.git#egg=unsloth
    # https://pip.pypa.io/en/stable/cli/pip_install/

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = (peft_config.init_lora_weights != 'loftq'),
        device_map = "sequential",
    )
    
    model, tokenizer = adjust_tune_format(model, tokenizer, tune_type, padding_side, custom_chat_template)
    
    # Do model patching and add fast LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        r = peft_config.r,
        target_modules = peft_config.target_modules,
        lora_alpha = peft_config.lora_alpha,
        lora_dropout = 0, # Currently only supports dropout = 0
        bias = "none",    # Currently only supports bias = "none"
        use_gradient_checkpointing = True,
        random_state = 3407,
        max_seq_length = max_seq_length,
        use_rslora=peft_config.use_rslora,
        init_lora_weights = peft_config.init_lora_weights,
        loftq_config=peft_config.loftq_config,
    )
    
    model.config.pretraining_tp = 1  # should already be =1,  https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/configuration_llama.py#L79
    return model, tokenizer

def get_model(model_id, 
              peft_config: LoraConfig, 
              quant_config:typing.Literal['bnb4','aqlm', 'gptq']|QuantizationConfigMixin='bnb4', 
              attn_implementation:typing.Literal["eager", "sdpa", "flash_attention_2"]="flash_attention_2", 
              tune_type='chatml',
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

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    pretrain_kwargs = dict(
        quantization_config=quant_config,
        use_cache=False, 
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16, # 'auto'=31.6gb  (WORSE)
        low_cpu_mem_usage=True,
        device_map="auto", # batch:(2,2): sequential= 26.9gb , auto=26.0/25.9gb
    )
    if quant_config is None:
        pretrain_kwargs.pop('quantization_config') # passing as None will error as of transformers 4.39.1 
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **pretrain_kwargs
        # quantization_config=quant_config,
        # use_cache=False, 
        # attn_implementation=attn_implementation,
        # torch_dtype=torch.bfloat16,
        # low_cpu_mem_usage=True,
    )

    model, tokenizer = adjust_tune_format(model, tokenizer, tune_type, padding_side, custom_chat_template)
    
    model.config.pretraining_tp = 1
    #model.gradient_checkpointing_enable()

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
    

    #model = model.to(torch.bfloat16)
    return model, tokenizer


def model_tokenizer_from_config(peft_config, cfg, custom_token_map=None):
        #tokenizer = tokenization.get_tokenizer(cfg.model_id, padding_side=cfg.padding_side)
    if cfg.flashattn_lib=='huggingface':
        model, tokenizer = get_model(cfg.model_id, 
                                     peft_config, 
                                     quant_config=cfg.quant_method, 
                                     attn_implementation=cfg.attn_implementation, 
                                     tune_type=cfg.tune_type, 
                                     padding_side=cfg.padding_side, 
                                     custom_chat_template=cfg.custom_chat_template) #custom_token_map
            
    elif cfg.flashattn_lib=='unsloth':
        # TODO: investigate why PAD = <unk> again (should be </s> if chat_ml) and if that is causing problems
        print('peft before:',peft_config)
        model, tokenizer = get_unsloth(cfg.model_id, 
                                       peft_config, 
                                       max_seq_length=cfg.chunk_size, 
                                       tune_type = cfg.tune_type, 
                                       padding_side=cfg.padding_side, 
                                       custom_chat_template=cfg.custom_chat_template)
        # TODO: look into ~4-7gb higher vRAM usage after changing padding_side=right -> padding_side=left --- https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side
        
    else:
        raise NotImplementedError(f'Unknown flashattn_lib: {cfg.flashattn_lib!r}')
    
    return model, tokenizer
