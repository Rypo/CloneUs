import os
import typing
from contextlib import contextmanager
import torch
import transformers

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
    if chat_template and 'bos_token' not in chat_template and tokenizer.bos_token not in chat_template:
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

def set_tokenizer_inference(tokenizer, ensure_bos:bool = True,):
    tokenizer.padding_side = 'left'
    
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.unk_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if ensure_bos:
        tokenizer.chat_template = bos_chat_template(tokenizer)
        
    return tokenizer