import os
import re
import typing
from dataclasses import dataclass
from contextlib import contextmanager
import torch
import transformers
from trl.models.utils import ChatMlSpecialTokens

def check_if_system(tokenizer):
    from jinja2.exceptions import TemplateError
    
    if not tokenizer.chat_template:
        return False
    
    try:
        # Phi-3 just ignores the system without error, need to test that it is actually used.
        output = tokenizer.apply_chat_template([{'role':'system', 'content':'sentinel'}], tokenize=False)
        has_system = 'sentinel' in output
        
    except TemplateError:
        has_system = False

    return has_system


def to_jinja_template(tag_sep:str, postfix:str):
    # role will be the pre-formated author tag
    template=(
        "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
        "{{ bos_token }}"
        "{% for message in messages %}"
        "{{ message['role'] + '__TAG_SEP__' + message['content'] + '__POSTFIX__' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '' }}"
        "{% endif %}"
    )
    template = template.replace('__TAG_SEP__',tag_sep).replace('__POSTFIX__', postfix)
    return template


# inspired by: https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B/blob/main/tokenizer_config.json
@dataclass
class ChatMlHybridSpecialTokens(ChatMlSpecialTokens):
    """Dataclass for special tokens used in ChatML that reuses existing BOS and PAD if available"""

    bos_token: str | None = None
    pad_token: str | None = None
    bot_token: str = "<|im_start|>"
    eos_token: str = "<|im_end|>"
    custom_roles: bool = True
    
    @property
    def chat_template(self):
        return (
            (f"{{{{'{self.bos_token}'}}}}" if self.bos_token is not None else '')+ # Qwen = no bos
            "{% for message in messages %}"
            f"{{{{'{self.bot_token}' + message['role'] + '\n' + message['content'] + '{self.eos_token}' + '\n'}}}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            +(f"{{{{ '{self.bot_token}' }}}}" if self.custom_roles else f"{{{{ '{self.assistant}\n' }}}}")+
            "{% endif %}"
        )

@dataclass
class ChatMlXSpecialTokens(ChatMlHybridSpecialTokens):
    """Dataclass for special tokens used in ChatML - OpenHermes2.5 flavor, including system, user, assistant, bos, eos, and pad tokens."""

    bos_token: str = "<s>"
    pad_token: str = "</s>"
    
    @property
    def chat_template(self):
        return (
            "{% for message in messages %}"
            f"{{{{'{self.bot_token}' + message['role'] + '\n' + message['content'] + '{self.eos_token}' + '\n'}}}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            +(f"{{{{ '{self.bot_token}' }}}}" if self.custom_roles else f"{{{{ '{self.assistant}\n' }}}}")+
            "{% endif %}"
        )

FORMAT_MAPPING = {"chatml": ChatMlSpecialTokens, "chatmlX": ChatMlXSpecialTokens, "chatmlH": ChatMlHybridSpecialTokens}


def setup_chat_format_patched(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    format: typing.Optional[typing.Literal["chatmlH","chatmlX"]] = "chatmlH",
    custom_roles: bool = True,
    resize_to_multiple_of: typing.Optional[int] = None,
) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:
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

    if format == "chatmlH":
        if tokenizer.pad_token is not None:
            pad_token = tokenizer.pad_token
        else:
            pad_token = tokenizer.eos_token # use the model's existing eos *not* <|im_end|>
        chat_format = ChatMlHybridSpecialTokens(bos_token = tokenizer.bos_token, pad_token=pad_token, custom_roles=custom_roles)
    elif format == "chatmlX":
        chat_format = ChatMlXSpecialTokens(custom_roles=custom_roles)
    else:
        chat_format = FORMAT_MAPPING[format]()

    # set special tokens and them
    tokenizer.eos_token = chat_format.eos_token
    tokenizer.pad_token = chat_format.pad_token
    tokenizer.bos_token = chat_format.bos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [chat_format.bot_token, chat_format.eos_token]})
    # set chat format for tokenizer
    tokenizer.chat_template = chat_format.chat_template

    # resize embedding layer to a multiple of 64, https://x.com/karpathy/status/1621578354024677377
    model.resize_token_embeddings(
        len(tokenizer), pad_to_multiple_of=resize_to_multiple_of if resize_to_multiple_of is not None else None
    )
    # Update the model config to use the new eos & bos tokens
    if getattr(model, "config", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
    # Make sure to update the generation config to use the new eos & bos token
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: typing.Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    add_random_init_values = True,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.

    src: https://github.com/dvlab-research/LongLoRA/blob/main/fine-tune.py
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    # TODO: save the input/output embedings as safetensors

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        if add_random_init_values:
            input_embeddings[-num_new_tokens:] += input_embeddings_avg
            output_embeddings[-num_new_tokens:] += output_embeddings_avg
        else:
            # This is the default, it seems odd to set all added values to exactly the same. 
            # However, I could see how adding random values to the mean could lead to instablity
            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

def author_special_tokens(author_names:list[str], pad_vocab_to:int=None):
    # https://github.com/huggingface/transformers/issues/27132
    # NOTE: no wonder I can't get AddedToken to work without special
    pad_usernames = []
    if pad_vocab_to is not None:
        pad_usernames = [f'placeholder_user_{i}' for i in range(pad_vocab_to-len(author_names))]
    
    return {'additional_special_tokens': author_names+pad_usernames}
    

def apply_special_tokens(tokenizer=None, custom_tokens=None, pad_vocab_to=None) -> None:
    '''Temporary No-Op'''
    if tokenizer is not None:
        raise NotImplementedError('This feature has been temporarily disabled.')
    num_custom_tokens = None
    if custom_tokens:
        custom_token_map = author_special_tokens(custom_tokens, pad_vocab_to=pad_vocab_to) if custom_tokens else None
        num_custom_tokens = tokenizer.add_special_tokens(custom_token_map)
    

    return num_custom_tokens


def get_tokenizer(model_id, padding_side=None):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    # " The DataCollatorForLanguageModeling always masks the pad_token in the labels and I set the pad_token = eos_token. "
    # https://discuss.huggingface.co/t/gpt2-finetuned-with-eos-token-will-never-yield-eos-token-during-generation/15437
    if padding_side is not None:
        tokenizer.padding_side = padding_side
    print('Using padding_side:', tokenizer.padding_side)
        
    return tokenizer

def bos_chat_template(tokenizer):
    '''Return a chat_template that begins with bos_token if tokenizer's does not already use it'''
    # Phi-3 medium requires BOS, but will not add as special token or in chat template.
    # it's unclear if this is a bug or intentional
    chat_template = tokenizer.chat_template
    if chat_template and ('bos_token' not in chat_template) and (tokenizer.bos_token is not None) and (tokenizer.bos_token not in chat_template):
        print(r'NOTE: bos_token not in chat template. {{ bos_token }} will be prepended to chat_template')
        chat_template = "{{ bos_token }}"+tokenizer.chat_template

    return chat_template

def configure_tokenizer(tokenizer, padding_side:str, custom_chat_template:str):
    if tokenizer.pad_token is None:
        # Llama 3 has no pad/unk, but has effectively has 2 eos. <|eot_id|> and <|end_of_text|>
        # The later is not used in the chat template
        tokenizer.pad_token=tokenizer.eos_token
    
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        msg = f'Warning: PAD = EOS: {tokenizer.eos_token}({tokenizer.eos_token_id})'
        if tokenizer.unk_token:
            tokenizer.pad_token_id = tokenizer.unk_token_id
            msg += ' Overriding with UNK token.'
        print(msg)

    if padding_side and padding_side != tokenizer.padding_side:
        print(f'NOTE: tokenizer.padding_side ({tokenizer.padding_side}) != config padding_side ({padding_side}). Setting padding_side={padding_side}.')
        tokenizer.padding_side = padding_side

    #if tokenizer.padding_side != 'left':
    #    print(f'Warning: padding_side({tokenizer.padding_side}) != left. This has inference implications. Proceed with caution:\nsee: https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side')

    if custom_chat_template:
        print('Using custom chat template')
        tokenizer.chat_template = custom_chat_template
    
    tokenizer.chat_template = bos_chat_template(tokenizer)
    print(f'Tokenizer using PAD = {tokenizer.pad_token}({tokenizer.pad_token_id})')
    return tokenizer

@contextmanager
def batchsafe_tokenizer(tokenizer):
    # TODO: In retrospect, this is not necessary. Could just set this on init. 
    # Any batch will *always* use this,
    # Any Non-batched won't use padding at all, so side/token is irrelevant anyway.
    # That said, still need to test this assertion before removing
    pad_side = tokenizer.padding_side
    pad_tokenid = tokenizer.pad_token_id
    
    tokenizer.padding_side  = 'left'
    # for non-chat_ml models, we set pad=unk, but batch needs pad=eos to work properly
    if tokenizer.pad_token_id == tokenizer.unk_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    try:
        yield tokenizer
    finally:
        tokenizer.padding_side = pad_side
        tokenizer.pad_token_id = pad_tokenid

def set_tokenizer_inference(tokenizer, uncomment_chat_template_bos:bool = True, force_bos_chat_template:bool = False):
    tokenizer.padding_side = 'left' # This is 100% necessary. Batched predictions will ALWAYS fail with right padded tokenizers.  
    
    # Hermes-2-Theta-Llama-3-8B breaks on batched unless pad=eos. Need batch for author probas
    # by default, pad = <|end_of_text|> (llama-3's eos token) but needs to be <eot_id>
    # there was a reason why I did this check, but I don't remember which model(s) it was necessary for
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == getattr(tokenizer,'unk_token_id', None):
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if uncomment_chat_template_bos and 'bos_token' in tokenizer.chat_template:
        # match commented out bos_token, remove comment tags e.g. {# {{- bos_token }} #} -> {{- bos_token }}
        # Needed for Llama3.1 variants where they are commented out during training to avoid double BOS, but need to add back for inference
        bos_re = re.compile(re.escape("{#") + "(.*" + re.escape("{{") + ".*bos_token.*" + re.escape("}}") + ".*)" + re.escape("#}"), re.I)
        tokenizer.chat_template = bos_re.sub(lambda m: m.group(1).strip(), tokenizer.chat_template)

    if force_bos_chat_template:
        tokenizer.chat_template = bos_chat_template(tokenizer)
        
    return tokenizer