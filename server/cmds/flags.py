import typing
import discord
from discord import app_commands
from discord.ext import commands

from discord.utils import MISSING 

from . import transformers as cmd_tfms


class GenerationFlags(commands.FlagConverter):
    # alias: typing.Literal['contrastive_search','multinomial_sampling', 'greedy_decoding', 'beam_search_decoding','beam_search_multinomial_sampling', 'diverse_beam_search_decoding']
    alias: typing.Literal['ms','cs','gd','bsd','bsms','dbsd'] = commands.flag(default=None, name='alias', description='Reset values to default preset state. ("ms" and "cs" are best)')
    
    max_new_tokens: int       = commands.flag(default=None, aliases=['maxlen'], description='Maximum allowed number of tokens to generate.')
    min_new_tokens: int       = commands.flag(default=None, aliases=['minlen'], description='Minimum allowed number of tokens to generate.')
    temperature: float        = commands.flag(default=None, description='Modulation value for next token probabilities.')
    top_k: int                = commands.flag(default=None, description='Number of highest probability vocabulary tokens to keep for top-k-filtering.')
    top_p: float              = commands.flag(default=None, description='Probability threshold for top-p-filtering.')
    
    penalty_alpha: float      = commands.flag(default=None, description='Balance between model confidence and degeneration penalty.')
    low_memory: bool          = commands.flag(default=None, description='Switch to sequential topk for contrastive search to reduce peak memory.')

    do_sample: bool           = commands.flag(default=None, description='Whether to use sampling or decoding.')
    repetition_penalty: float = commands.flag(default=None, description='Values > 1.0 penalize word repetition.')

    typical_p: float          = commands.flag(default=None, description='Local typicality threshold for generating tokens.')
    epsilon_cutoff: float     = commands.flag(default=None, description='Conditional probability threshold for truncation sampling.')
    eta_cutoff: float         = commands.flag(default=None, description='Eta sampling threshold for hybrid sampling.')

    num_beams: int            = commands.flag(default=None, description='Number of beams for beam search.')
    num_beam_groups: int      = commands.flag(default=None, description='Number of groups for beam search diversity.')
    diversity_penalty: float  = commands.flag(default=None, description='Penalty for generating tokens same as other beam groups.')
    length_penalty: float     = commands.flag(default=None, description='Exponential penalty to the length for beam-based generation.')
    early_stopping: bool      = commands.flag(default=None, description='Controls stopping condition for beam-based methods.')

    no_repeat_ngram_size: int = commands.flag(default=None, description='Size of ngrams that can only occur once.')
    renormalize_logits: bool  = commands.flag(default=None, description='Whether to renormalize logits after processing.')
    # NOTE: These values have special handlers
    guidance_scale: float     = commands.flag(default=None, description='Guidance scale for classifier free guidance.')
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
    


class ModeFlags(commands.FlagConverter):
    streaming_mode: bool = commands.flag(default=None, name='streaming', description='Enable text streaming mode')
    tts_mode: bool = commands.flag(default=None, name='tts', description='Enable text to speech mode')
    #auto_author: bool = commands.flag(default=1, description='The number of days worth of messages to delete')
    #simulation_mode: bool = commands.flag(default=None, name='simulation', description='Enable full automated simulation mode')


class WordRuleFlags(commands.FlagConverter):
    banned_words: app_commands.Transform[str, cmd_tfms.WordListTransformer] = commands.flag(name='banned_words', default=None,) 
    #force_words: str = commands.flag(name='force_words', default=None, description='(non-functional) A comma separated list of forced words')
    weighted_words: app_commands.Transform[str, cmd_tfms.WordListTransformer]  = commands.flag(name='weighted_words', default=None,) 

class DrawFlags(commands.FlagConverter, delimiter=' ', prefix='--'):
    #prompt: str
    steps: int = commands.flag(default=None, aliases=['num_inference_steps'])
    no: str = commands.flag(default=None, aliases=['neg_prompt','negative_prompt'])
    guide: float = commands.flag(default=None, aliases=['guidance', 'guidance_scale'])
    aspect: typing.Literal['square','portrait','landscape'] = commands.flag( default=None, aliases=['ar','orient', 'orientation']) # ['1:1', '13:19', 19:13]
    
    hdsteps: int = commands.flag(default=0, aliases=['refine_steps'])
    hdstrength: app_commands.Transform[float, cmd_tfms.PercentTransformer] = commands.flag(default=None,  aliases=['refine_strength']) 
    dblend: app_commands.Transform[float, cmd_tfms.PercentTransformer] = commands.flag(default=None, aliases=['denoise_blend'],) 
    fast: bool = commands.flag(default=False, aliases=['lq'],) 


class RedrawFlags(commands.FlagConverter, delimiter=' ', prefix='--'):
    # imgfile: discord.Attachment
    # prompt: str
    steps: int = commands.flag(default=None, aliases=['num_inference_steps'])
    strength: app_commands.Transform[float, cmd_tfms.PercentTransformer] = commands.flag(default=None,  aliases=['str']) 
    no: str = commands.flag(default=None, aliases=['neg_prompt','negative_prompt'])
    guide: float = commands.flag(default=None, aliases=['guidance', 'guidance_scale'])
    aspect: typing.Literal['square','portrait','landscape'] = commands.flag( default=None, aliases=['ar','orient', 'orientation']) # ['1:1', '13:19', 19:13]
    hdsteps: int = commands.flag(default=0, aliases=['refine_steps'])
    hdstrength: app_commands.Transform[float, cmd_tfms.PercentTransformer] = commands.flag(default=None,  aliases=['refine_strength'])
    dblend: app_commands.Transform[float, cmd_tfms.PercentTransformer] = commands.flag(default=None, aliases=['denoise_blend'],) 
    fast: bool = commands.flag(default=False, aliases=['lq'],) 

# class RedrawFlags(commands.FlagConverter):
#     # image_url: str = commands.flag(
#     #     description="Image URL. Square = Best results. 1024x1024 = ideal (512x512 = Turbo ideal).")
#     # prompt: str = commands.flag(
#     #     description="A description of the image to be generated.")
#     steps: int = commands.flag(
#         default=None, description="Num of iterations to run. Increase = ⬆Quality, ⬆Run Time. Default=50 (Turbo: Default=4).")
#     #strength: commands.Range[float, 0, 100] = commands.flag(
#     strength: app_commands.Transform[float, cmd_tfms.PercentTransformer] = commands.flag(
#         default=80, description="How much to change input image. 0 = Change Nothing. 100=Change Entirely. Default=80")
    
#     neg_prompt: str = commands.flag(
#         default=None, description="Description of what you DON'T want. Usually comma sep list of words. Default=None (Turbo ignores).")
#     guidance: float = commands.flag(
#         default=10.0, description="Guidance scale. Increase = ⬆Prompt Adherence, ⬇Quality, ⬇Creativity. Default=10.0 (Turbo ignores).")
#     denoise_blend: app_commands.Transform[float, cmd_tfms.PercentTransformer]  = commands.flag( # commands.Range[float, 0, 100]
#         default=None, description="Percent of `steps` for Base stage before Refine stage. ⬇Quality, ⬇Run Time. Default=None (Turbo ignores).")
    
#     refine_strength: app_commands.Transform[float, cmd_tfms.PercentTransformer] = commands.flag(
#         default=30, description="Refinement stage alteration power. 0 = Alter Nothing. 100=Alter Everything. Default=30")