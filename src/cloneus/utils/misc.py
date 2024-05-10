import os
import gc
import typing
import json
import warnings
from dataclasses import dataclass

import torch
import bitsandbytes as bnb
from transformers import PreTrainedModel, PreTrainedTokenizer

warnings.filterwarnings(action = "ignore", category = UserWarning, module = "torch")


# SRC: https://github.com/unslothai/unsloth/blob/dev.2023.12/unsloth/models/_utils.py#L24
def prepare_model_for_kbit_training(
    model                      : typing.Any,
    use_gradient_checkpointing : bool = True,
    use_reentrant              : typing.Optional[bool] = True,
) -> typing.Any:
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



def save_embeddings(model, outdir):
    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data
    emb_dir = os.path.join(outdir, 'embeddings')
    os.makedirs(emb_dir, exist_ok=True)
    
    torch.save(input_embeddings, os.path.join(emb_dir, 'input_embed_weights.bin'))
    torch.save(output_embeddings, os.path.join(emb_dir, 'output_embed_weights.bin'))
    print('saved embeddings to:', emb_dir)

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



# SOURCE:
# https://github.com/OpenAccess-AI-Collective/axolotl/blob/332984db186d097be92fa690e931d8895c05d589/src/axolotl/utils/models.py#L507


def find_all_linear_names(model):
    from peft.tuners.lora import QuantLinear
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



# from trl
@dataclass
class ChatMlXSpecialTokens:
    """Dataclass for special tokens used in ChatML, including system, user, assistant, bos, eos, and pad tokens."""

    bos_token: str = "<s>"
    bosx_token: str = "<|im_start|>"
    eos_token: str = "<|im_end|>"
    pad_token: str = "</s>"

    @property
    def system(self):
        return f"{self.bosx_token}system"

    @property
    def user(self):
        return f"{self.bosx_token}user"

    @property
    def assistant(self):
        return f"{self.bosx_token}assistant"

    @property
    def chat_template(self):
        return (
            "{% for message in messages %}"
            f"{{{{'{self.bosx_token}' + message['role'] + '\n' + message['content'] + '{self.eos_token}' + '\n'}}}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            f"{{{{ '{self.assistant}\n' }}}}"
            "{% endif %}"
        )


FORMAT_MAPPING = {"chatmlX": ChatMlXSpecialTokens}


def setup_chat_format_patched(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    format: typing.Optional[typing.Literal["chatmlX"]] = "chatmlX",
    resize_to_multiple_of: typing.Optional[int] = None,
) -> typing.Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Setup chat format by adding special tokens to the tokenizer, setting the correct format, and extending the embedding layer of the model based on the new special tokens.

    Args:
      model (`~transformers.PreTrainedModel`): The model to be modified.
      tokenizer (`~transformers.PreTrainedTokenizer`): The tokenizer to be modified.
      format (`Optional[Literal["chatml"]]`): The format to be set. Defaults to "chatml".
      resize_to_multiple_of (`Optional[int]`): Number to resize the embedding layer to. Defaults to None.
    Returns:
      model (`~transformers.PreTrainedModel`): The modified model.
      tokenizer (`~transformers.PreTrainedTokenizer`): The modified tokenizer.
    """
    # check if format available and retrieve
    if format not in FORMAT_MAPPING:
        raise ValueError(f"Format {format} not available. Please use one of {FORMAT_MAPPING.keys()}")

    chat_format = FORMAT_MAPPING[format]()

    # set special tokens and them
    tokenizer.eos_token = chat_format.eos_token
    tokenizer.pad_token = chat_format.pad_token
    tokenizer.bos_token = chat_format.bos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [chat_format.bosx_token, chat_format.eos_token]})
    # set chat format for tokenizer
    tokenizer.chat_template = chat_format.chat_template

    # resize embedding layer to a multiple of 64, https://x.com/karpathy/status/1621578354024677377
    model.resize_token_embeddings(
        len(tokenizer), pad_to_multiple_of=resize_to_multiple_of if resize_to_multiple_of is not None else None
    )
    # Make sure to update the generation config to use the new eos & bos token
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer