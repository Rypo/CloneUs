import os
import gc
import json
import warnings
from typing import  Optional, Any

import numpy as np
import torch

import bitsandbytes as bnb
from peft.tuners.lora import QuantLinear

warnings.filterwarnings(action = "ignore", category = UserWarning, module = "torch")


# SRC: https://github.com/unslothai/unsloth/blob/dev.2023.12/unsloth/models/_utils.py#L24
def prepare_model_for_kbit_training(
    model                      : Any,
    use_gradient_checkpointing : bool = True,
    use_reentrant              : Optional[bool] = True,
) -> Any:
    """
    Calculates where to place the gradient checkpoints given n_layers.
    We also freeze all other layers's gradients

    Args:
        model: Any LlamaModel with layers.
        use_gradient_checkpointing (`bool`, *optional*):
            Default enabled. Provides memory savings by not saving all activations,
            but only some.
        use_reentrant (`bool`, *optional*):
            https://github.com/pytorch/pytorch/blob/main/torch/utils/checkpoint.py#L354
            Optimal gradient checkpointing algorithm which will be the default in
            future Pytorch versions.
    """

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad_(False)

    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # If use_reentrant = True which is the Pytorch default, we just make the input requires_grad.
    if use_reentrant:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    return model

def patch_tokenizer(model, tokenizer):
    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        # Fixes https://github.com/unslothai/unsloth/issues/5
        if hasattr(tokenizer, "unk_token"):
            tokenizer.add_special_tokens({"pad_token" : tokenizer.unk_token})
            tokenizer.pad_token = tokenizer.unk_token
        else:
            # logger.warning_one(
            #     f"{model.config._name_or_path} does not have a padding or unknown token!\n"\
            #     f"Will use the EOS token of id {tokenizer.eos_token_id} as padding."
            # )
            assert(hasattr(tokenizer, "eos_token"))
            tokenizer.add_special_tokens({"pad_token" : tokenizer.eos_token})
            tokenizer.pad_token = tokenizer.eos_token
        config = model.config.update({"pad_token_id" : tokenizer.eos_token_id})

    return model, tokenizer

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    print(f"trainable model parameters: {trainable_model_params}. All model parameters: {all_model_params} ")
    return trainable_model_params

def print_trainable_parameters(model, use_4bit=False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )


def save_way_too_much(trainer, subdir='manualsave'):
    args = trainer.args
    accdir = os.path.join(args.output_dir, subdir)
    
    trainer.save_model(accdir)
    trainer.save_state()
    
    torch.save(trainer.optimizer.state, os.path.join(accdir,'optimizer.state'))
    torch.save(trainer.optimizer.state_dict(), os.path.join(accdir,'optimizer_statedict.pt'))
    torch.save(trainer.optimizer.param_groups, os.path.join(accdir,'optimizer_paramgroups.pt'))
    
    with open(os.path.join(accdir,'lr_scheduler_statedict.json'),'w') as f:
        json.dump(trainer.lr_scheduler.state_dict(),f)
    
    acceldir = os.path.join(accdir,'accelsave')
    trainer.accelerator.save_state(acceldir)
    trainer.accelerator.save_model(trainer.model, acceldir)


def chunk_to_batchga(chunk_size, target_batch_size=16, chunklen_upperbound=8192):
    '''' Get batch size and gradient accumulation steps for a given chunk size
        
    Args:
        chunk_size: int, size of chunks to be batched together
        target_batch_size: int, target batch size, will be used to determine number of gradient accumulation steps
        chunklen_upperbound: int, maximum length chunk that will fit in memory (used to determine batch size)
    '''
    #max_log2 = int(np.log2(mem_max_chunk)) # 13vvwwfww 
    #batch_size = {128: 64, 256: 32, 512: 16, 1024: 8, 2048: 4, 4096: 2, 8192: 1}[chunk_size]
    #batch_size = 2**int(max_log2-np.log2(chunk_size))
    batch_size = chunklen_upperbound//chunk_size
    ga_steps = max(1, target_batch_size//batch_size)

    return batch_size, ga_steps


# SOURCE:
# https://github.com/OpenAccess-AI-Collective/axolotl/blob/332984db186d097be92fa690e931d8895c05d589/src/axolotl/utils/models.py#L507


def find_all_linear_names(model):
    cls = (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt, torch.nn.Linear, QuantLinear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            or "Linear" in module.__class__.__name__
            and module.__class__.__name__ not in ("LlamaLinearScalingRotaryEmbedding",)
        ):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")

    return list(lora_module_names)


def load_lora(model, cfg, inference=False):
    # type: (PreTrainedModel, DictDefault, bool) -> Tuple[PreTrainedModel, Optional[PeftConfig]]

    from peft import LoraConfig, PeftModel, get_peft_model

    lora_target_modules = list(cfg.lora_target_modules or [])

    if cfg.lora_target_linear:
        linear_names = find_all_linear_names(model)
        LOG.info(f"found linear modules: {repr(linear_names)}")
        lora_target_modules = list(set(lora_target_modules + linear_names))

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        fan_in_fan_out=cfg.lora_fan_in_fan_out,
        modules_to_save=cfg.lora_modules_to_save if cfg.lora_modules_to_save else None,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if cfg.lora_model_dir:
        LOG.debug("Loading pretained PEFT - LoRA")
        model = PeftModel.from_pretrained(
            model,
            cfg.lora_model_dir,
            is_trainable=(not inference),
        )
    else:
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    return model, lora_config



def load_lora_custom(model, peft_config, lora_target_linear=True):
    # type: (PreTrainedModel, DictDefault, bool) -> Tuple[PreTrainedModel, Optional[PeftConfig]]

    from peft import  get_peft_model

    lora_target_modules = list(peft_config.target_modules or [])

    if lora_target_linear:
        linear_names = find_all_linear_names(model)
        print(f"found linear modules: {repr(linear_names)}")
        lora_target_modules = list(set(lora_target_modules + linear_names))


    peft_config.target_modules = lora_target_modules


    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    return model, peft_config