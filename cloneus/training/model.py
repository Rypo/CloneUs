import typing
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GPTQConfig,
    TrainingArguments
)
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model
from unsloth import FastLanguageModel

from ..utils import misc
from ..data import dataset


def get_unsloth(model_id, peft_config: LoraConfig, max_seq_length=4096, dtype=torch.bfloat16):
    # pip install -e "git+https://github.com/unslothai/unsloth.git#egg=unsloth
    # https://pip.pypa.io/en/stable/cli/pip_install/

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit= True,
        device_map = "sequential",
    )
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
    )
    model.config.pretraining_tp = 1  # should already be =1,  https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/configuration_llama.py#L79
    return model, tokenizer


def get_model(model_id, peft_config: LoraConfig, quant_config=None, tokenizer=None, custom_tokens_map=None, attn_implementation:typing.Literal["eager", "sdpa", "flash_attention_2"]=None, lora_target_linear=None):
    # mistralai/Mistral-7B-v0.1
    # BitsAndBytesConfig int-4 config
    if quant_config is None:
        if 'gptq' in model_id.lower():
            quant_config = GPTQConfig(bits=4, disable_exllama=True)
        else:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id, #,'pytorch_model.bin'), 
        quantization_config=quant_config, 
        use_cache=False, 
        load_in_4bit=True,
        # use_cache=False, 
        #device_map="auto",
        #device_map={"":0},
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    model.config.pretraining_tp = 1
    #model.gradient_checkpointing_enable()

    if custom_tokens_map is not None:
        dataset.smart_tokenizer_and_embedding_resize(custom_tokens_map, tokenizer, model)
        #model.resize_token_embeddings(len(tokenizer))

    # prepare model for training
    model = misc.prepare_model_for_kbit_training(model)
    #model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=dict(use_reentrant=True))
    model = get_peft_model(model, peft_config)
    model.config.use_cache = False
    model.print_trainable_parameters()

    model = model.to(torch.bfloat16)
    return model, peft_config