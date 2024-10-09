import typing
import random
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field

import numpy as np
import torch
import transformers
from transformers import GenerationConfig, StoppingCriteria

from cloneus.plugins.sampler_hijack import hijack_samplers

hijack_samplers()

logger = logging.getLogger(__name__)
# THIS IS THE SINGLE SOURCE OF TRUTH NOW
@dataclass
class GenOpts:
    max_new_tokens: int = 256
    min_new_tokens: int = None # 'default': None,
    temperature: float = 0.8 # 'default': 1.0,
    top_k: int = 50
    top_p: float = 1.0
    min_p: float = None
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
    stop_strings: str| list[str] = None
    token_healing: bool = False
    cache_implementation: str =  None # "quantized"
    cache_config: transformers.CacheConfig|dict= None # transformers.QuantizedCacheConfig('HQQ', nbits=4, compute_dtype=torch.bfloat16, device='cuda')
    #dola_layers https://github.com/huggingface/transformers/blob/main/docs/source/en/generation_strategies.md#dola-decoding

    
    def to_dict(self):
       return asdict(self)

# src: https://github.com/oobabooga/text-generation-webui/blob/81f603d09fab9afad9fa54f123c57c187bc115df/extensions/openai/typing.py#L28
# src2: https://github.com/oobabooga/text-generation-webui/blob/81f603d09fab9afad9fa54f123c57c187bc115df/modules/presets.py#L13
# desc: https://github.com/oobabooga/text-generation-webui/wiki/03-%E2%80%90-Parameters-Tab#parameters-description
# updated params, descs: https://github.com/oobabooga/text-generation-webui/blob/dev/modules/ui_parameters.py
@dataclass
class GenOptsExtended(GenOpts):
    '''Transformers Generation Config with added oobabooga text-generation-webui config options'''

    #preset: str | None = field(default=None, metadata=dict(description="The name of a file under text-generation-webui/presets (without the .yaml extension). The sampling parameters that get overwritten by this option are the keys in the default_preset() function in modules/presets.py."))
    
    temperature_last: bool = False
    dynamic_temperature: bool = False
    dynatemp_low: float = 1
    dynatemp_high: float = 1
    dynatemp_exponent: float = 1

    smoothing_factor: float = 0
    smoothing_curve: float = 1

    frequency_penalty: float | None = 0
    presence_penalty: float | None = 0
    repetition_penalty_range: int = 0 #1024
    tfs: float = 1
    top_a: float = 0
    #negative_prompt: str = ''
    mirostat_mode: int = 0 # 0 or 2
    mirostat_tau: float = 5
    mirostat_eta: float = 0.1

    dry_multiplier: float = 0.0 # 0.8, ~ penalty weight, activator of DRY
    dry_base: float = 1.75 # pentalty--seqlength scaling factor 
    dry_allowed_length: int = 2 # max permitted repeats before pentalty applied
    dry_sequence_breakers: list[str] = None# field(default_factory=lambda: ['"\\n", ":", "\\"", "*"'])
    
    xtc_threshold: float = 0.1
    xtc_probability: float = 0
    # def to_a_dict(self):
    #     return asdict(self)
    #seed: int = -1
    #encoder_repetition_penalty: float = 1
    #truncation_length: int = 0
    #max_tokens_second: int = 0
    #prompt_lookup_num_tokens: int = 0
    #custom_token_bans: str = ""
    #sampler_priority: list[str] | str | None = field(default=None,  ['temperature', 'dynamic_temperature', 'quadratic_sampling', 'top_k', 'top_p', 'typical_p', 'epsilon_cutoff', 'eta_cutoff', 'tfs', 'top_a', 'min_p', 'mirostat']
    #                                                 metadata=dict(description='List of samplers where the first items will appear first in the stack. Example: ["top_k", "temperature", "top_p"].'))
    #auto_max_new_tokens: bool = False
    #ban_eos_token: bool = False
    #add_bos_token: bool = True
    #skip_special_tokens: bool = True
    #grammar_string: str = ""

# TODO: Randomizer - https://github.com/oobabooga/text-generation-webui/blob/81f603d09fab9afad9fa54f123c57c187bc115df/modules/presets.py#L82


def randomize_preset(gen_config:GenerationConfig):
    params_and_values = {
        'remove_tail_tokens': {
            'top_p': [0.5, 0.8, 0.9, 0.95, 0.99],
            'min_p': [0.5, 0.2, 0.1, 0.05, 0.01],
            'top_k': [3, 5, 10, 20, 30, 40],
            'typical_p': [0.2, 0.575, 0.95],
            'tfs': [0.5, 0.8, 0.9, 0.95, 0.99],
            'top_a': [0.5, 0.2, 0.1, 0.05, 0.01],
            'epsilon_cutoff': (1e-4*np.array([1, 3, 5, 7, 9])).round(6),
            'eta_cutoff': (1e-4*np.array([3, 6, 9, 12, 15, 18])).round(6),
        },
        'flatten_distribution': {
            'temperature': [0.1, 0.5, 0.7, 0.8, 1, 1.2, 1.5, ],#2.0, 5.0],
            'dynamic_temperature': [
                [0.1, 1],
                [0.1, 1.5],
                [0.1, 2],
                [0.1, 5],
                [0.5, 1],
                [0.5, 1.5],
                [0.5, 2],
                [0.5, 5],
                [0.8, 1],
                [0.8, 1.5],
                [0.8, 2],
                [0.8, 5],
                [1, 1.5],
                [1, 2],
                [1, 5]
            ],
            'smoothing_factor': [0.2, 0.3, 0.6, 1.2],
        },
        'repetition': {
            'repetition_penalty': [1, 1.05, 1.1, 1.15, 1.20, 1.25],
            'presence_penalty': [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0],
            'frequency_penalty': [0, 0.1, 0.2, 0.4, 0.6, 0.8, ],#1.0, 2.0],
        },
        'other': {
            'temperature_last': [True, False],
        }
    }

    generate_params = GenOptsExtended(temperature=1, repetition_penalty_range=1024, repetition_penalty=1,).to_dict()
    for cat in params_and_values:
        choices = list(params_and_values[cat].keys())
        #if shared.args.loader is not None:
        #    choices = [x for x in choices if loader_contains(x)]

        if len(choices) > 0:
            choice = random.choice(choices)
            value = random.choice(params_and_values[cat][choice])
            if choice == 'dynamic_temperature':
                generate_params['dynamic_temperature'] = True
                generate_params['dynatemp_low'] = value[0]
                generate_params['dynatemp_high'] = value[1]
            else:
                generate_params[choice] = value

    gen_config.update(**generate_params)
    #logger.info("GENERATED_PRESET=")
    #pprint.PrettyPrinter(indent=4, width=1, sort_dicts=False).pprint(remove_defaults(state))
    return gen_config# #*[generate_params[k] for k in presets_params()]



def load_gen_config(gen_config_path:str|Path=None, gen_config_name:str="generation_config.json"):
    if gen_config_path is None:
        logger.debug('using GenOptsExtended defaults (multinomial_sampling)')
        #gce = asdict(GenOptsExtended())
        return GenerationConfig.from_dict(GenOptsExtended().to_dict())
    gen_config_path = Path(gen_config_path)
    
   # GenerationConfig.from_pretrained expects dir and a str name
    if gen_config_path.suffix == '.json':
        if gen_config_name == "generation_config.json": # default used by from_pretraiend
            gen_config_name = gen_config_path.name
        gen_config_path = gen_config_path.parent
    
    try:
        gen_config = GenerationConfig.from_pretrained(gen_config_path, config_file_name=gen_config_name, local_files_only=True)
        logger.info(f'Found GenerationConfig: {gen_config_path/gen_config_name}')
    except OSError as e:
        # Try to fall back to default
        if gen_config_name != "generation_config.json":
            return load_gen_config(gen_config_path, gen_config_name="generation_config.json")
        #print(e)
        logger.info('No existing GenerationConfig found, using GenOpts defaults (multinomial_sampling)')
        #gen_config = GenerationConfig.from_dict(asdict(GenOpts())) 
        gen_config = GenerationConfig.from_dict(GenOptsExtended().to_dict())

    return gen_config


def preset_gen_config(preset:typing.Literal['ms','cs','gd','mp', 'miro','dyna','xtc', 'bsd','bsms','dbsd'], pad_token_id:int, eos_token_id:int|list[int], **kwargs):
    '''Returns a GenerationConfig with the minimum relevant argments for the preset.
    
    Any kwargs that are not used for the preset are ignored.
    '''
    # https://huggingface.co/docs/transformers/main_classes/text_generation
    # https://github.com/oobabooga/text-generation-webui/blob/main/presets/Contrastive%20Search.yaml
     #contrastive -- Low memory cuts memory usage in ~half, makes shorter outputs with similar content, but slow
    # default_gc = GenOpts()
    default_gc = GenOptsExtended()
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
    
    num_beams = kwargs.get('num_beams', 5)
    early_stopping = kwargs.get('early_stopping', False)

    mirostat_tau = kwargs.get('mirostat_tau', default_gc.mirostat_tau)
    mirostat_eta = kwargs.get('mirostat_eta', default_gc.mirostat_eta)
    
    dynatemp_low = kwargs.get('dynatemp_low', 0.6)
    dynatemp_high = kwargs.get('dynatemp_high', 1.2)
    dynatemp_exponent = kwargs.get('dynatemp_exponent', default_gc.dynatemp_exponent)

    preset_name = preset.lower()
    if preset_name in ['cs','contrastive_search']:
        top_k = kwargs.get('top_k', 4)
        penalty_alpha = kwargs.get('penalty_alpha',0.6)
        low_memory = kwargs.get('low_memory', False)

        genconfig = GenerationConfig(do_sample=False, penalty_alpha=penalty_alpha, top_k=top_k, low_memory=low_memory, **shared_args)
    
    elif preset_name in ['ms','multinomial_sampling']:
        genconfig = GenerationConfig(do_sample=True, top_p=top_p, temperature=temperature, top_k=top_k, **shared_args)
    
    elif preset_name in ['gd','greedy_decoding']:
        genconfig = GenerationConfig(do_sample=False, **shared_args)
    elif preset_name in ['mp', 'minp']:
        min_p = kwargs.get('min_p',0.1)# = 0.1
        temperature = kwargs.get('temperature', 1.5)
        
        genconfig = GenerationConfig(do_sample=True, temperature=temperature, min_p = min_p, **shared_args)
    elif preset_name in ['miro', 'mirostat']:
        # decent explaination of tau, eta: https://old.reddit.com/r/LocalLLaMA/comments/1furllv/which_llm_models_is_the_funniest_one_as_in_the/lq2povc/
        mirostat_mode = 2 # 0 or 2
        
        genconfig = GenerationConfig(do_sample=True, mirostat_mode=mirostat_mode, mirostat_tau=mirostat_tau, mirostat_eta=mirostat_eta, **shared_args)

    elif preset_name in ['dyna', 'dynamic_temperature']:
        dynamic_temperature = True
        genconfig = GenerationConfig(do_sample=True, dynamic_temperature=dynamic_temperature, dynatemp_low=dynatemp_low, dynatemp_high=dynatemp_high, dynatemp_exponent=dynatemp_exponent, **shared_args)
    
    elif preset_name in ['xtc', 'exclude_top_choice']:
        # https://github.com/oobabooga/text-generation-webui/pull/6335
        # https://www.reddit.com/r/LocalLLaMA/comments/1ev8n2s/exclude_top_choices_xtc_a_sampler_that_boosts/
        dry_multiplier = kwargs.get('dry_multiplier',  0.8)
        dry_sequence_breakers = kwargs.get('dry_sequence_breakers', default_gc.dry_sequence_breakers)
        xtc_threshold = kwargs.get('xtc_threshold',  0.1)
        xtc_probability = kwargs.get('xtc_probability',  0.5)
        temperature = kwargs.get('temperature',  1)
        min_p = kwargs.get('min_p',  0.1)
        
        repetition_penalty = kwargs.get('repetition_penalty', 1.0 )

        shared_args = {**shared_args, **{'repetition_penalty':repetition_penalty}}

        genconfig = GenerationConfig(do_sample=True, dry_multiplier=dry_multiplier, xtc_threshold=xtc_threshold, xtc_probability=xtc_probability, min_p=min_p, dry_sequence_breakers=dry_sequence_breakers, **shared_args)

    elif preset_name in ['bsd', 'beam_search_decoding']:
        genconfig = GenerationConfig(do_sample=False, num_beams=num_beams, early_stopping=early_stopping, **shared_args)
    
    elif preset_name in ['bsms','beam_search_multinomial_sampling']:
        # https://www.izzy.co/blogs/robo-boys.html#:~:text=model%20parameters%2C%20and%20settled%20on%20the%20following.
        temperature = kwargs.get('temperature',.75,)
        # top_p = kwargs.get('top_p', 0.85)
        top_p = kwargs.get('top_p', 1)
        top_k = kwargs.get('top_k', 80)
        num_beams=kwargs.get('num_beams',3,)
        
        genconfig = GenerationConfig(do_sample=True, top_k=top_k, top_p=top_p, temperature=temperature, num_beams=num_beams, early_stopping=early_stopping, **shared_args)
    
    elif preset_name in ['dbsd','diverse_beam_search_decoding']:
        num_beam_groups=kwargs.get('num_beam_groups', 5)
        diversity_penalty=kwargs.get('diversity_penalty', 1.0)
        genconfig = GenerationConfig(do_sample=False, num_beams=num_beams, num_beam_groups=num_beam_groups, early_stopping=early_stopping, diversity_penalty=diversity_penalty, **shared_args)

    else:
        raise ValueError(f'unknown preset "{preset}"')
    
    return genconfig