import typing
from pathlib import Path
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

GENMODE_ALIASES = {
    'ms': 'multinomial_sampling',
    'cs': 'contrastive_search',
    'gd': 'greedy_decoding',
    'bsd': 'beam_search_decoding',
    'bsms': 'beam_search_multinomial_sampling',
    'dbsd': 'diverse_beam_search_decoding'
 }

def load_gen_config(gen_config_path:str|Path, gen_config_name:str=None):
    gen_config_path = Path(gen_config_path)
    
    if gen_config_name is not None:
        if gen_config_path.suffix == '.json':
            gen_config_path = gen_config_path.with_name(gen_config_name)
        else:
            gen_config_path = gen_config_path/gen_config_name
    
    try:
        gen_config = GenerationConfig.from_pretrained(gen_config_path, config_file_name=gen_config_name, local_files_only=True)
        print(f'Found GenerationConfig: {gen_config_path}')
    except OSError as e:
        print('No existing GenerationConfig found, defaulting to GENOPTS (multinomial_sampling)')
        gen_config = GenerationConfig.from_dict(GENOPT_DEFAULTS.copy()) 

    return gen_config

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
    # https://huggingface.co/docs/transformers/main_classes/text_generation
    # https://github.com/oobabooga/text-generation-webui/blob/main/presets/Contrastive%20Search.yaml
     #contrastive -- Low memory cuts memory usage in ~half, makes shorter outputs with similar content, but slow
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

class WordListCriteria(StoppingCriteria):
    # https://discuss.huggingface.co/t/implimentation-of-stopping-criteria-list/20040/19
    # https://github.com/nestordemeure/stop_word/blob/main/stop_word_criteria.py
    def __init__(self, stop_token_ids: list[torch.Tensor]):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in self.stop_token_ids:
            if torch.all(input_ids[0, -stop_ids.shape[0]:] == stop_ids): 
                return True
        return False
    
    @classmethod
    def from_words(cls, word_list:list[str], tokenizer:transformers.PreTrainedTokenizer, device=0):
        # NOTE: The intentional space prefix on words. Different tokens may be used at sentence start vs first mid/end.
        # This is paricularily problematic when \n is part of a stop word
        # To normalize for this, add a space prefix and trim it off [0,1:]
        
        stop_token_ids = [tokenizer(' ' + x, return_tensors='pt', add_special_tokens=False)['input_ids'][0,1:].to(device) for x in word_list]
        return cls(stop_token_ids)