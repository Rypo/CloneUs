import typing
import torch
import transformers
from transformers import GenerationConfig, StoppingCriteria
from transformers.generation.utils import GenerationMode



# THIS IS THE SINGLE SOURCE OF TRUTH NOW
GENOPTS = {
    'max_new_tokens': {'default': 256, 'type': int},
    'min_new_tokens': {'default': None, 'type': int}, # 'default': None,
    'temperature': {'default': 0.8, 'type': float}, # 'default': 1.0,
    'top_k': {'default': 50, 'type': int},
    'top_p': {'default': 1.0, 'type': float},
    
    'penalty_alpha': {'default': None, 'type': float},
    'low_memory': {'default': None, 'type': bool},
    
    'do_sample': {'default': True, 'type': bool}, # 'default': False,
    'repetition_penalty': {'default': 1.1, 'type': float}, # default: 1.0
    
    'typical_p': {'default': 1.0, 'type': float},
    'epsilon_cutoff': {'default': 0.0, 'type': float},
    'eta_cutoff': {'default': 0.0, 'type': float},
    
    
    'num_beams': {'default': 1, 'type': int},
    'num_beam_groups': {'default': 1, 'type': int},
    'diversity_penalty': {'default': 0.0, 'type': float},
    'length_penalty': {'default': 1.0, 'type': float},
    'early_stopping': {'default': False, 'type': bool},
    
    'no_repeat_ngram_size': {'default': 0, 'type': int},
    'renormalize_logits': {'default': True, 'type': bool}, # 'default': False,
    'exponential_decay_length_penalty': {'default': None, 'type': tuple[int, float]},
    
    'bad_words_ids': {'default': None, 'type': list[list[int]]},
    'force_words_ids': {'default': None, 'type': list[list[int]] | list[list[list[int]]]},
    'sequence_bias': {'default': None, 'type': dict[tuple[int], float]},
    
    'suppress_tokens': {'default': None, 'type': list[int]},
    'begin_suppress_tokens': {'default': None, 'type': list[int]},
    
    'guidance_scale': {'default': None, 'type': float}
}
GENOPT_DEFAULTS = {k: v['default'] for k,v in GENOPTS.items()}

def get_gencofs(pad_token_id, eos_token_id, include=('cs','bsms','dbsd','ms','gd'), max_new_tokens=256, **kwargs):

    # https://huggingface.co/docs/transformers/main_classes/text_generation

    #if pad_token_id is None:
    #    pad_token_id = eos_token_id
    if isinstance(include, str):
        include = (include,)
    shared_genargs = dict(
        max_new_tokens=max_new_tokens,
        renormalize_logits=True,
        repetition_penalty=1.1,
        eos_token_id=eos_token_id, #model.config.eos_token_id,
        pad_token_id=pad_token_id, #model.config.pad_token_id,
    )
    shared_genargs.update(kwargs)
    # https://github.com/oobabooga/text-generation-webui/blob/main/presets/Contrastive%20Search.yaml
    cs_genconf = GenerationConfig(do_sample=False, penalty_alpha=0.6, top_k=4, **shared_genargs) #   #contrastive -- Low memory cuts memory usage in ~half, makes shorter outputs with similar content, but slow
    bsms_genconf = GenerationConfig(do_sample=True, top_p=1, temperature=1, num_beams=4, early_stopping=True, **shared_genargs)
    dbsd_genconf = GenerationConfig(do_sample=False, num_beams=4, num_beam_groups=4, early_stopping=True, diversity_penalty=0.5, **shared_genargs)
    ms_genconf = GenerationConfig(do_sample=True, top_p=1, temperature=1, num_beams=1, **shared_genargs)
    gd_genconf = GenerationConfig(do_sample=False, num_beams=1, **shared_genargs)

    confmap = {k:v for k,v in zip(('cs','bsms','dbsd','ms','gd'),(cs_genconf,bsms_genconf,dbsd_genconf,ms_genconf,gd_genconf))}
    return [confmap[k] for k in include]

def get_config(tokenizer, alias:typing.Literal['cs','ms','gd'], **kwargs):
    print(f'eos: {tokenizer.eos_token!r} ({tokenizer.eos_token_id}), pad: {tokenizer.pad_token!r} ({tokenizer.pad_token_id}), padside: {tokenizer.padding_side}')

    shared = dict(
        max_new_tokens=256,
        renormalize_logits=True,
        repetition_penalty=1.1, # Setting this too high may prevent sequential same-author messages. 
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **kwargs,
    )
    #shared=dict(repetition_penalty=1.1, max_new_tokens=256, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, **kwargs)
    if alias=='cs':
        gen_config = GenerationConfig(penalty_alpha=0.6, top_k=4, low_memory=False, **shared)
    elif alias=='ms':
        gen_config = GenerationConfig(do_sample=True, top_p=1, temperature=1, top_k=50, **shared)
    elif alias=='gd':
        gen_config = GenerationConfig(do_sample=False, **shared)
    else:
        gen_config = GenerationConfig.from_dict(shared)
    
    return gen_config


def create_genconf(alias, pad_token_id, eos_token_id, **kwargs):
    #eos_token_id = kwargs.get('eos_token_id', pad_token_id)
    shared_args = dict(
        max_new_tokens=kwargs.get('max_new_tokens', GENOPTS['max_new_tokens']['default']),
        min_new_tokens=kwargs.get('min_new_tokens', None),
        renormalize_logits=kwargs.get('renormalize_logits', GENOPTS['renormalize_logits']['default']),
        repetition_penalty=kwargs.get('repetition_penalty', GENOPTS['repetition_penalty']['default']),
        eos_token_id=eos_token_id, 
        pad_token_id=pad_token_id, 
    )

    
    top_k = kwargs.get('top_k', GENOPTS['top_k']['default'])
    top_p = kwargs.get('top_p', GENOPTS['top_p']['default'])
    temperature = kwargs.get('temperature', GENOPTS['temperature']['default'])
    
    num_beams=kwargs.get('num_beams', 5)
    early_stopping = kwargs.get('early_stopping', False)


    alias_name = alias.lower()
    if alias_name in ['cs','contrastive_search']:
        top_k = kwargs.get('top_k', 4)
        penalty_alpha = kwargs.get('penalty_alpha',0.6)
        low_memory = kwargs.get('low_memory', False)

        genconfig = GenerationConfig(do_sample=False, penalty_alpha=penalty_alpha, top_k=top_k, low_memory=low_memory, **shared_args)
    
    elif alias_name in ['ms','multinomial_sampling']:
        genconfig = GenerationConfig(do_sample=True, top_p=top_p, temperature=temperature, top_k=top_k, **shared_args)
    
    elif alias_name in ['gd','greedy_decoding']:
        genconfig = GenerationConfig(do_sample=False, **shared_args)
    
    elif alias_name in ['bsd', 'beam_search_decoding']:
        genconfig = GenerationConfig(do_sample=False, num_beams=num_beams, early_stopping=early_stopping, **shared_args)
    
    elif alias_name in ['bsms','beam_search_multinomial_sampling']:
        # https://www.izzy.co/blogs/robo-boys.html#:~:text=model%20parameters%2C%20and%20settled%20on%20the%20following.
        temperature = kwargs.get('temperature',.75,)
        # top_p = kwargs.get('top_p', 0.85)
        top_p = kwargs.get('top_p', 1)
        top_k = kwargs.get('top_k', 80)
        num_beams=kwargs.get('num_beams',3,)
        
        genconfig = GenerationConfig(do_sample=True, top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams, early_stopping=early_stopping, **shared_args)
    
    elif alias_name in ['dbsd','diverse_beam_search_decoding']:
        num_beam_groups=kwargs.get('num_beam_groups', 5)
        diversity_penalty=kwargs.get('diversity_penalty', 1.0)
        genconfig = GenerationConfig(do_sample=False, num_beams=num_beams, num_beam_groups=num_beam_groups, early_stopping=early_stopping, diversity_penalty=diversity_penalty, **shared_args)

    else:
        raise ValueError('unknown alias "{alias}"')
    
    return genconfig

def get_generation_mode(generation_config:GenerationConfig, assistant_model:transformers.PreTrainedModel=None):
    """
    Returns the generation mode triggered by a [`GenerationConfig`] instance.
    """

    # https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/generation/utils.py#L939
    if generation_config.constraints is not None or generation_config.force_words_ids is not None:
        generation_mode = GenerationMode.CONSTRAINED_BEAM_SEARCH
    elif generation_config.num_beams == 1:
        if generation_config.do_sample is False:
            if (
                generation_config.top_k is not None
                and generation_config.top_k > 1
                and generation_config.penalty_alpha is not None
                and generation_config.penalty_alpha > 0
            ):
                generation_mode = GenerationMode.CONTRASTIVE_SEARCH
            else:
                generation_mode = GenerationMode.GREEDY_SEARCH
        else:
            generation_mode = GenerationMode.SAMPLE
    else:
        if generation_config.num_beam_groups > 1:
            generation_mode = GenerationMode.GROUP_BEAM_SEARCH
        elif generation_config.do_sample is True:
            generation_mode = GenerationMode.BEAM_SAMPLE
        else:
            generation_mode = GenerationMode.BEAM_SEARCH

    # Assisted generation may extend some generation modes
    if assistant_model is not None:
        if generation_mode in ("greedy_search", "sample"):
            generation_mode = GenerationMode.ASSISTED_GENERATION
        else:
            raise ValueError(
                "You've set `assistant_model`, which triggers assisted generate. Currently, assisted generate "
                "is only supported with Greedy Search and Sample."
            )
    return generation_mode


class NewLineTokensCriteria(StoppingCriteria):
    def __init__(self, stop_tokens):
        self.stop_tokens = stop_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        
        #last_tokens = input_ids[0,-2:]
        # this also works, but can't gaurentee output is identical in all circumstances. Seems like in the batch case, padding token would interfere 
        last_tokens = input_ids[:,-2:] 
        return torch.all(last_tokens==self.stop_tokens)