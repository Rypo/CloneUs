import typing
import discord
from discord import app_commands
from discord.ext import commands
from discord.ext.commands import Range

from discord.utils import MISSING 

from . import transformers as cmd_tfms

# Just use "words" in place of "tokens". It's a convenient lie.
class GenerationFlags(commands.FlagConverter):
    # alias: typing.Literal['contrastive_search','multinomial_sampling', 'greedy_decoding', 'beam_search_decoding','beam_search_multinomial_sampling', 'diverse_beam_search_decoding']
    preset: typing.Literal['ms','cs','gd','bsd','bsms','dbsd', 'mp'] = commands.flag(default=None, name='preset', description='Reset values to default preset state. ("ms","mp", and "cs" are best)')
    
    max_new_tokens: Range[int, 0, ]             = commands.flag(default=None, description='Maximum allowed number of words to generate (default: 256)')
    min_new_tokens: Range[int, 0, ]             = commands.flag(default=None, description='Minimum allowed number of words to generate (default: 0)')
    temperature: Range[float, 0, ]              = commands.flag(default=None, description='Controls randomness. > 1 = more random. < 1 = less random (default: 1.0)') # Modulation value for next token probabilities 
    top_k: Range[int, 0, ]                      = commands.flag(default=None, description='Fixed number of highest proba words to sample from (default: 50)')
    top_p: Range[float, 0, 1]                   = commands.flag(default=None, description='Dynamic number of highest proba words to sample from. Take while sum probas<top_p (default: 1.0)')
    min_p: Range[float, 0, 1]                   = commands.flag(default=None, description='Discard words with proba < (min_p) * max(proba words) (default: 0)') # 0
    
    do_sample: bool                             = commands.flag(default=None, description='Whether to use sampling or decoding. CS=False, MS=True (default: True)')
    penalty_alpha: Range[float, 0,]             = commands.flag(default=None, description='For Contrastive Search (CS). Balance model confidence and degeneration (default: 0.6)')
    
    repetition_penalty: Range[float, 1, 2]      = commands.flag(default=None, description='Values > 1.0 penalize word repetition (default: 1.1)')

    typical_p: Range[float, 0, 1]               = commands.flag(default=None, description='Local typicality threshold for generating words (default: 1.0)')
    epsilon_cutoff: Range[float, 0, 1]          = commands.flag(default=None, description='Probability cutoff for consideration during sampling (default: 0)')
    eta_cutoff: Range[float, 0, 1]              = commands.flag(default=None, description='Eta sampling threshold. Hybrid of locally typical, epsilon sampling (default: 0)')

    # NOTE: beam search is (almost) worthless. Need num_beams just for ban words/word weights. Otherwise blegh. 
    num_beams: int                              = commands.flag(default=None, description='Number of beams for beam search (default: 1)')
    # num_beam_groups: int      = commands.flag(default=None, description='Number of groups for beam search diversity.')
    # diversity_penalty: float  = commands.flag(default=None, description='Penalty for generating tokens same as other beam groups.')
    # length_penalty: float     = commands.flag(default=None, description='Exponential penalty to the length for beam-based generation.')
    # early_stopping: bool      = commands.flag(default=None, description='Controls stopping condition for beam-based methods.')
    low_memory: bool                            = commands.flag(default=None, description='For Contrastive Search (CS). Switch to sequential topk to reduce peak memory (default: False)')
    no_repeat_ngram_size: int                   = commands.flag(default=None, description='Size of ngrams that can only occur once (default: 0)')
    renormalize_logits: bool                    = commands.flag(default=None, description='Whether to renormalize logits after processing (default: True)')
    # NOTE: These values have special handlers
    guidance_scale: float                       = commands.flag(default=None, description='Guidance scale for classifier free guidance. 1=no guidance (default: 1)')
    #bad_words_ids: list[list[int]] = commands.flag(default=None,description='List of lists of token ids not allowed to be generated.')
    #force_words_ids: list[list[int]] | list[list[list[int]]] = commands.flag(default=None,description='List of token ids that must be generated.')
    #sequence_bias: dict[tuple[int], float] = commands.flag(default=None,description='Maps a sequence of tokens to selection bias term (+ inc, - dec).')
    
    # NOTE: These could be useful with handlers
    #exponential_decay_length_penalty: typing.Tuple[int, float] = commands.flag(default=None, description='tuple (start_idx, decay_factor) denoting decay start and penalty factor.')
    #constraints: list[Constraint] = commands.flag(description='Custom constraints for generation.')
    #forced_bos_token_id: int = commands.flag(description='Id of token to force as the first generated token.')
    #forced_eos_token_id: int | list[int] = commands.flag(description='Id(s) of token to force as the last generated token.')
    
    # NOTE: These are unlikely to be useful
    #max_length: int = commands.flag(default=20, description='Maximum length of generated tokens.')
    #min_length: int = commands.flag(default=0, description='Minimum length of the generated sequence.')
    #max_time: float = commands.flag(description='Maximum time allowed for generation in seconds.')
    #use_cache: bool = commands.flag(default=True, description='Whether to use past key/values attentions for decoding.')
    #encoder_repetition_penalty: float = commands.flag(default=1.0, description='Parameter for encoder repetition penalty.')
# Where to find some reasonable ranges:
# - https://github.com/oobabooga/text-generation-webui/blob/dev/modules/ui_parameters.py
class GenerationExtendedFlags(commands.FlagConverter):
    preset: typing.Literal['random','miro','dyna', 'xtc'] = commands.flag(default=None, description='Set values to default preset state')
    temperature_last: bool                  = commands.flag(default=None, description='Apply temp after filtering. This + high_tep + min_p = creative but coherent (default: False)') # False
    
    dynamic_temperature: bool               = commands.flag(default=None, description='Enable dynamic temperature - auto picks temp from [low, high] using cross-entropy (default: False)') # False
    dynatemp_low: Range[float, 0.01, ]      = commands.flag(default=None, description='Minumum dynamic temperature value (default: 1)') # 1
    dynatemp_high: Range[float, 0.01, ]     = commands.flag(default=None, description='Maximum dynamic temperature value (default: 1)') # 1
    dynatemp_exponent: Range[float, 0.01,]  = commands.flag(default=None, description='Dynamic temperature scale factor. How easy to reach high end temp, (default: 1)') # 1

    smoothing_factor: Range[float, 0, ]     = commands.flag(default=None, description='Enable Quadratic Sampling. 0<val<1 = even out probas. val>1 most likely more likely (default: 0)') # 0
    smoothing_curve: Range[float, 1, ]      = commands.flag(default=None, description='Quadratic Sampling dropoff curve factor (default: 1)') # 1
    
    top_a: Range[float, 0, 1]               = commands.flag(default=None, description='Discard words with proba < (top_a) * max(proba words)^2 (default: 0)') # 0
    tfs: Range[float, 0, 1]                 = commands.flag(default=None, description='Discard long tail proba words (lower than others). Closer to 0 = more discarded (default: 1)') # 1
    
    mirostat_mode: bool                     = commands.flag(default=None, description='Enable Mirostat - technique for sampling perplexity control using active learning (default: False)') # 0 or 2
    mirostat_tau: float                     = commands.flag(default=None, description='Target cross-entropy value. Paper says best=3, empirical says best=8 (default: 5)') # 5
    mirostat_eta: Range[float, 0, 1]        = commands.flag(default=None, description='Learning rate for updates during sampling (default: 0.1)') # 0.1    
    
    frequency_penalty: float | None         = commands.flag(default=None, description='Like repetition_penalty, but gets stronger each time a word is repeated (default: 0)') # 0
    presence_penalty: float | None          = commands.flag(default=None, description='Like repetition_penalty, but penalty is added rather than multiplied by score (default: 0)') # 0
    repetition_penalty_range: int           = commands.flag(default=None, description='Number of most recent tokens considered in repetition penalty. 0=full context window (default: 0)' ) # 0, -- 1024
    # https://github.com/oobabooga/text-generation-webui/pull/6335
    xtc_threshold: Range[float, 0, 1]       = commands.flag(default=None, description='Minimum probablity to mark tokens as a "top choice" (default: 0.1)') # If 2 or more tokens have probability above this threshold, consider removing all but the last one. 
    xtc_probability: Range[float, 0, 1]     = commands.flag(default=None, description='Probablity of excluding top choices. Enables when > 0. (default: 0)') # 0.5 
    # recommend: Min-P = 0.02, XTC threshold = 0.1, XTC probability = 0.5, DRY (multiplier 0.8),
    dry_multiplier: Range[float, 0, ]       = commands.flag(default=None, description='DRY penalty weight. Enables when > 0. (default: 0.0)') #0.8



class ModeFlags(commands.FlagConverter):
    streaming_mode: bool = commands.flag(default=None, name='streaming', description='Enable text streaming mode')
    tts_mode: bool = commands.flag(default=None, name='tts', description='Enable text to speech mode')
    #auto_author: bool = commands.flag(default=1, description='The number of days worth of messages to delete')
    #simulation_mode: bool = commands.flag(default=None, name='simulation', description='Enable full automated simulation mode')


class WordRuleFlags(commands.FlagConverter):
    banned_words: app_commands.Transform[str, cmd_tfms.WordListTransformer] = commands.flag(name='banned_words', default=None,) 
    #force_words: str = commands.flag(name='force_words', default=None, description='(non-functional) A comma separated list of forced words')
    weighted_words: app_commands.Transform[str, cmd_tfms.WordListTransformer]  = commands.flag(name='weighted_words', default=None,) 

class DrawFlags(commands.FlagConverter):
    #prompt: str
    n: Range[int, 1, ] = commands.flag(default=1, aliases=['n_images'])
    steps: int = commands.flag(default=None, aliases=['num_inference_steps'])
    negprompt: str = commands.flag(default=None, aliases=['neg_prompt','negative_prompt'], converter=cmd_tfms.dash_to_empty)
    guidance: float = commands.flag(default=None, aliases=['guide', 'guidance_scale'])
    detail: float = commands.flag(default=0.0, aliases=['details', 'detail_weight'])
    aspect: typing.Literal['square','portrait','landscape'] = commands.flag( default=None, aliases=['ar','orient', 'orientation']) # ['1:1', '13:19', 19:13]
    
    hdstrength: app_commands.Transform[float, cmd_tfms.PercentTransformer] = commands.flag(default=None,  aliases=['refine_strength']) 
    fast: bool = commands.flag(default=False, aliases=['lq'],) 
    seed: int = commands.flag(default=None, aliases=['random_seed'],) 


class RedrawFlags(commands.FlagConverter):
    # imgfile: discord.Attachment
    # prompt: str
    n: Range[int, 1, ] = commands.flag(default=1, aliases=['n_images'])
    imgfile: discord.Attachment = commands.flag(default=None)
    steps: int = commands.flag(default=None, aliases=['num_inference_steps'])
    strength: app_commands.Transform[float, cmd_tfms.PercentTransformer] = commands.flag(default=None,  aliases=['str']) 
    negprompt: str = commands.flag(default=None, aliases=['neg_prompt','negative_prompt'], converter=cmd_tfms.dash_to_empty)
    guidance: float = commands.flag(default=None, aliases=['guide', 'guidance_scale'])
    detail: float = commands.flag(default=0.0, aliases=['details', 'detail_weight'])
    aspect: typing.Literal['square','portrait','landscape'] = commands.flag( default=None, aliases=['ar','orient', 'orientation']) # ['1:1', '13:19', 19:13]
    
    hdstrength: app_commands.Transform[float, cmd_tfms.PercentTransformer] = commands.flag(default=None,  aliases=['refine_strength'])
    #dblend: app_commands.Transform[float, cmd_tfms.PercentTransformer] = commands.flag(default=None, aliases=['denoise_blend'],) 
    fast: bool = commands.flag(default=False, aliases=['lq'],) 
    seed: int = commands.flag(default=None, aliases=['random_seed'],) 

class UpsampleFlags(commands.FlagConverter):
    # imgfile: discord.Attachment
    # prompt: str
    hdstrength: app_commands.Transform[float, cmd_tfms.PercentTransformer] = commands.flag(default=0.3,  aliases=['refine_strength'])
    steps: int = commands.flag(default=None, aliases=['num_inference_steps'])
    negprompt: str = commands.flag(default=None, aliases=['neg_prompt','negative_prompt'], converter=cmd_tfms.dash_to_empty)
    guidance: float = commands.flag(default=None, aliases=['guide', 'guidance_scale'])
    detail: float = commands.flag(default=0.0, aliases=['details', 'detail_weight'])
    seed: int = commands.flag(default=None, aliases=['random_seed'],) 


class AnimateFlags(commands.FlagConverter):
    # prompt: str
    # imgurl: str = None
    nframes: int = commands.flag(default=11, aliases=['nf'])
    steps: int = commands.flag(default=None, aliases=['num_inference_steps'])
    strength_end: app_commands.Transform[float, cmd_tfms.PercentTransformer] = commands.flag(default=0.80,  aliases=['strmax']) 
    strength_start: app_commands.Transform[float, cmd_tfms.PercentTransformer] = commands.flag(default=0.30,  aliases=['strmin']) 
    negprompt: str = commands.flag(default=None, aliases=['neg_prompt','negative_prompt'], converter=cmd_tfms.dash_to_empty)
    guidance: float = commands.flag(default=None, aliases=['guide', 'guidance_scale'])
    detail: float = commands.flag(default=0.0, aliases=['details', 'detail_weight'])
    midframes:int = commands.flag(default=4)
    aspect: typing.Literal['square','portrait','landscape'] = commands.flag( default=None, aliases=['ar','orient', 'orientation']) # ['1:1', '13:19', 19:13]
    
    #dblend: app_commands.Transform[float, cmd_tfms.PercentTransformer] = commands.flag(default=None, aliases=['denoise_blend'],) 
    fast: bool = commands.flag(default=False, aliases=['lq'],) 
    seed: int = commands.flag(default=None, aliases=['random_seed'],) 


class ReanimateFlags(commands.FlagConverter):
    # imgurl: str
    # prompt: str
    steps: int = commands.flag(default=None, aliases=['num_inference_steps'])
    astrength: app_commands.Transform[float, cmd_tfms.PercentTransformer] = commands.flag(default=0.50,  aliases=['str'])
    imsize: typing.Literal['tiny','small','med','full'] = commands.flag(default='small', aliases=['imscale'])
    negprompt: str = commands.flag(default=None, aliases=['neg_prompt','negative_prompt'], converter=cmd_tfms.dash_to_empty)
    guidance: float = commands.flag(default=None, aliases=['guide', 'guidance_scale'])
    
    detail: float = commands.flag(default=0.0, aliases=['details', 'detail_weight'])
    #aspect: typing.Literal['square','portrait','landscape'] = commands.flag( default=None, aliases=['ar','orient', 'orientation']) # ['1:1', '13:19', 19:13]
    stage2:bool = commands.flag(default=False, aliases=['two_stage'])

    fast: bool = commands.flag(default=False, aliases=['lq'],) 
    aseed: int = commands.flag(default=None, aliases=['animation_seed'],) 