import typing
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
import transformers
from transformers import GenerationConfig, StoppingCriteria

# THIS IS THE SINGLE SOURCE OF TRUTH NOW
@dataclass
class GenOpts:
    max_new_tokens: int = 256
    min_new_tokens: int = None # 'default': None,
    temperature: float = 0.8 # 'default': 1.0,
    top_k: int = 50
    top_p: float = 1.0
    penalty_alpha: float = None
    low_memory: bool = None
    do_sample: bool = True # 'default': False,
    repetition_penalty: float = 1.1 # default: 1.0
    typical_p: float = 1.0
    epsilon_cutoff: float = 0.0
    eta_cutoff: float = 0.0
    num_beams: int = 1
    num_beam_groups: int = 1
    diversity_penalty: float = 0.0
    length_penalty: float = 1.0
    early_stopping: bool = False
    no_repeat_ngram_size: int = 0
    renormalize_logits: bool = True  # 'default': False,
    exponential_decay_length_penalty: tuple[int, float] = None
    bad_words_ids: list[list[int]] = None
    force_words_ids: list[list[int]] | list[list[list[int]]] = None
    sequence_bias: dict[tuple[int], float] = None
    suppress_tokens: list[int] = None
    begin_suppress_tokens: list[int] = None
    guidance_scale: float = None

GENMODE_ALIASES = {
    'ms': 'multinomial_sampling',
    'cs': 'contrastive_search',
    'gd': 'greedy_decoding',
    'bsd': 'beam_search_decoding',
    'bsms': 'beam_search_multinomial_sampling',
    'dbsd': 'diverse_beam_search_decoding'
 }

def load_gen_config(gen_config_path:str|Path, gen_config_name:str="generation_config.json"):
    gen_config_path = Path(gen_config_path)
    
   # GenerationConfig.from_pretrained expects dir and a str name
    if gen_config_path.suffix == '.json':
        if gen_config_name == "generation_config.json": # default used by from_pretraiend
            gen_config_name = gen_config_path.name
        gen_config_path = gen_config_path.parent
    
    try:
        gen_config = GenerationConfig.from_pretrained(gen_config_path, config_file_name=gen_config_name, local_files_only=True)
        print(f'Found GenerationConfig: {gen_config_path/gen_config_name}')
    except OSError as e:
        print('No existing GenerationConfig found, using GenOpts defaults (multinomial_sampling)')
        gen_config = GenerationConfig.from_dict(asdict(GenOpts())) 

    return gen_config


def create_genconf(alias:typing.Literal['ms','cs','gd','bsd','bsms','dbsd'], pad_token_id:int, eos_token_id:int|list[int], **kwargs):
    # https://huggingface.co/docs/transformers/main_classes/text_generation
    # https://github.com/oobabooga/text-generation-webui/blob/main/presets/Contrastive%20Search.yaml
     #contrastive -- Low memory cuts memory usage in ~half, makes shorter outputs with similar content, but slow
    default_gc = GenOpts()
    shared_args = dict(
        max_new_tokens=kwargs.get('max_new_tokens', default_gc.max_new_tokens),
        min_new_tokens=kwargs.get('min_new_tokens', None),
        renormalize_logits=kwargs.get('renormalize_logits', default_gc.renormalize_logits),
        repetition_penalty=kwargs.get('repetition_penalty', default_gc.repetition_penalty),
        eos_token_id=eos_token_id, #tokenizer.eos_token_id,
        pad_token_id=pad_token_id, # tokenizer.pad_token_id,
    )

    
    top_k = kwargs.get('top_k', default_gc.top_k)
    top_p = kwargs.get('top_p', default_gc.top_p)
    temperature = kwargs.get('temperature', default_gc.temperature)
    
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