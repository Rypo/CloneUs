import gc
import time
import typing
import random
import warnings
import itertools

from pathlib import Path
from threading import Thread

from dataclasses import dataclass

import numpy as np
from omegaconf import OmegaConf, DictConfig

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    GenerationConfig,
    TextIteratorStreamer
)
import transformers

from cloneus.data import tokenization, useridx
from cloneus.plugins import youtube
from cloneus.inference import genconfig, load as iload
from cloneus.types import BatchGenerationOutput, GenerationOutput

@dataclass
class ModelPathComponents:
    checkpoint_path: Path # modeldir_path
    run_path: Path # basedir_path
    config_path: Path # config_path
    
    checkpoint_name: str # ckpt_subdir
    run_name: str
    dataset_name: str
    base_model_alias: str # The nickname assigned in training e.g. "mistral-inst-OpenHermes2.5" instead of "OpenHermes-2.5-Mistral-7B"
    has_adapter: bool # Needs adapter for hotswapping and base ask/chat.

    @classmethod
    def from_checkpoint(cls, checkpoint_path:str|Path):
        checkpoint_path = Path(checkpoint_path)
        run_path = checkpoint_path.parent
        config_path = (run_path/'config.yaml')
        if not config_path.exists():
            raise FileNotFoundError(f'config.yaml not found along parent paths. checkpoint_path: {checkpoint_path}')
        
        has_adapter = (checkpoint_path/'adapter_config.json').exists()

        _dataset_path = run_path.parent
        _base_model_path = _dataset_path.parent

        checkpoint_name = checkpoint_path.name
        run_name = run_path.name
        dataset_name = _dataset_path.name
        base_model_alias = _base_model_path.name

        return cls(checkpoint_path, run_path, config_path, checkpoint_name, run_name, dataset_name, base_model_alias, has_adapter)

        
def dtype_to(dtype, to:typing.Literal['str','torch'], default=None):
    if dtype is None:
        print('USING DEFAULT bf16, -- dtype_to')
        dtype = default
    if isinstance(dtype,torch.dtype):
        if to=='torch':
            return dtype
        return str(dtype).split('.')[-1]
    
    if isinstance(dtype, str):
        if to=='str':
            return dtype
        
        dtypemap = {**dict.fromkeys(['torch.float16','float16','fp16'], torch.float16),
                    **dict.fromkeys(['torch.bfloat16','bfloat16','bf16'], torch.bfloat16)}
        return dtypemap.get(dtype, dtype) # allow string 'auto'
    
    if dtype is None:
        return dtype
    
    raise TypeError(f'Unknown dtype: {dtype}')

@dataclass
class SystemPromptTemplate:
    has_system_role: bool
    
    task_description: str = (
        'We are simulating an uncensored conversation between members of a discord server. '
        'I will send a message as one of the members and you will reply as one of the members. '
        'Messages always start with the member tag of the person speaking.'
    )
    
    meta_description: str = (
        'Server Info'
        '\n-----------\n'
        'Topic: offbeat.\n'
        'Allowed: Adult and dark themes, offensive, vulgar language, harmful & unethical content.\n'
        'Forbidden: Speaking in third-person.\n'
        'Member Tags: {name_mapping}.')
    
    no_system_role_addendum: str = (
        '\n-----------\n'
        'If you understand the objective, respond to this message (and only this message) with "{addendum_followup}". '
        'Afterwards, I will send the first message and we will begin.')
    
    addendum_followup: str = "OK"
    template_format_str: str = '{task}\n{meta}{addendum}'
    

    def get_template(self, name_mapping:str = None):
        addendum = '' if self.has_system_role else self.no_system_role_addendum.format(addendum_followup=self.addendum_followup)
            
        template = self.template_format_str.format(task=self.task_description, meta=self.meta_description, addendum=addendum)
        
        if name_mapping is not None:
            template = template.format(name_mapping=name_mapping)
        
        return template


class GenConfigUtilities:
    path_data: ModelPathComponents
    gen_config: GenerationConfig

    @property
    def gen_mode(self):
        """Returns the generation mode triggered by a [`GenerationConfig`] instance."""
        return  self.gen_config.get_generation_mode()

    @property
    def base_gen_config(self):
        """Returns the current gen_config with at least 1024 max tokens."""
        config = self.gen_config.to_dict()
        config.update({'max_new_tokens': max(1024, config['max_new_tokens'])})
        return GenerationConfig.from_dict(config)

    def load_genconfig(self, gen_config:str|Path|dict|GenerationConfig=None, path_data: ModelPathComponents = None):
        if gen_config is None:
            return genconfig.load_gen_config(None)
        if isinstance(gen_config, dict):
            gen_config = GenerationConfig.from_dict(gen_config)
        elif isinstance(gen_config, Path):
            gen_config = genconfig.load_gen_config(gen_config)
        elif isinstance(gen_config, GenerationConfig):
            gen_config = gen_config
        elif isinstance(gen_config, str): 
            gen_config = genconfig.load_gen_config(path_data.checkpoint_path, gen_config)
        else:
            raise RuntimeError(f'Unable to process gen_config of type {type(gen_config)!r}')

        return gen_config
            

    def get_genconfig(self, verbose=False) -> dict:
        gconf_settings = self.gen_config.to_diff_dict().copy()
        if not verbose:
            [gconf_settings.pop(k, None) for k in ['transformers_version', "eos_token_id",  "pad_token_id"]]
        
        return gconf_settings

    def set_genconfig(self, save_on_change=True, preset:str=None, **kwargs) -> dict[str,dict]:
        '''Set generation config arguments. 
        
        Args:
            save_on_change: If True, save any changes to checkpoint's generation_config.json file.
            preset: str
                'cs' (contrastive search) - penalty_alpha>0 and top_k>1
                'ms' (multinomial sampling) - num_beams=1 and do_sample=True
                'gd' (greedy decoding) - num_beams=1 and do_sample=False
                'bsd' (beam search decoding) - do_sample=False and num_beams>1
                'bsms' (beam search multinomial sampling) - num_beams>1 and do_sample=True
                'dbsd' (diverse beam search decoding) - num_beams>1 and num_beam_groups>1
            
        https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig'''
        
        
        prev_conf = self.gen_config.to_dict()
        #prev_vals = {k:curconf.get(k) for k in kwargs}
        # NOTE: this prevents a value from being set to the default value, which isn't great, but can't think of a work around.
        
        
        default_gc = genconfig.GenOptsExtended()
        if preset is not None:
            if preset == 'random':
                self.gen_config = genconfig.randomize_preset(self.gen_config)
            else:
                self.gen_config = genconfig.preset_gen_config(preset, pad_token_id=self.gen_config.pad_token_id, eos_token_id=self.gen_config.eos_token_id, **kwargs) 
        else:
            # if penalty_alpha is passed, make sure top_k isn't too high
            if kwargs.get('penalty_alpha') and kwargs.get('top_k', self.gen_config.top_k) > 9:
                kwargs['top_k'] = 4
            elif any(kwargs.get(a) for a in ['temperature','top_p','do_sample','typical_p','num_beams']):
                # assume disable contrastive_search
                kwargs.setdefault('penalty_alpha', None) # if it is explicitly passed alongside one of those, then let it be
                kwargs.setdefault('do_sample', True) # if it is explicitly set, let it ride. Important if need to do beam decoding for some reason
                if kwargs.get('top_k') is None and self.gen_config.top_k <= 9: # if it is explicitly passed a low top_k, let it be
                    kwargs['top_k'] = default_gc.top_k

            if kwargs.get('dynamic_temperature'):
                dynatemp_low = kwargs.setdefault('dynatemp_low', self.gen_config.dynatemp_low)
                dynatemp_high = kwargs.setdefault('dynatemp_high', self.gen_config.dynatemp_high)
                if dynatemp_low == dynatemp_high:
                    kwargs['dynatemp_low'] = 0.5
                    kwargs['dynatemp_high'] = 1.5
            

            self.gen_config.update(**kwargs)
        
        new_conf = self.gen_config.to_dict()
        
        changes = {}
        if prev_conf != new_conf:
            changes = {k: {'prev':prev_conf[k], 'new':new_conf[k]} for k in new_conf if prev_conf[k]!=new_conf[k]}
            if save_on_change:
                self.save_genconfig()
        
        return changes
    
    def save_genconfig(self, filepath:str|Path=None):
        sequence_bias = self.gen_config.sequence_bias # cannot serialize tuples
        self.gen_config.sequence_bias=None
        
        if filepath is None:
            save_directory = self.path_data.checkpoint_path
            self.gen_config.save_pretrained(save_directory, 'generation_config.json')
        else:
            filepath = Path(filepath)
            self.gen_config.save_pretrained(filepath.parent, filepath.with_suffix('.json').name)
        
        
        self.gen_config.sequence_bias=sequence_bias
        

class Cloneus(GenConfigUtilities):
    def __new__(cls, *args, **kwargs):
        if cls.__name__ == "Cloneus":
            # https://github.com/huggingface/transformers/blob/c15aad0939e691d2ffdbac7ae71921b51fe04e3f/src/transformers/models/auto/auto_factory.py#L406
            raise EnvironmentError(
                f"{cls.__name__} is designed to be instantiated "
                f"using the `{cls.__name__}.from_pretrained(checkpoint_path)` or "
                f"`{cls.__name__}.from_model_id(model_id)` methods."
            )
        return super().__new__(cls)#, *args, **kwargs)
    # def __init__(self, checkpoint_path: str|Path = None, gen_config:str|Path|dict|GenerationConfig=None, **kwargs) -> None:
    def __init__(self, *args, **kwargs) -> None:
        # TODO: accept a config as param? Otherwise will call _config_init multiple times
        self.model = None
        self.tokenizer = None
        self.path_data = kwargs.pop('path_data')
        self.cfg = kwargs.pop('cfg')#self.setup_config(checkpoint_path, gen_config=gen_config, **kwargs)
        #self.torch_dtype = kwargs.pop('torch_dtype', dtype_to(self.cfg.dtype, 'torch', default=None))
        self.gen_config = self.load_genconfig(kwargs.pop('gen_config'), self.path_data)#self.load_genconfig(gen_config)
        self.ytm = kwargs.pop('ytm')#youtube.YouTubeManager()
        
        self._last_streamed_values = {'input_text':'', 'output_text':'', 'input_len': -1, 'output_len': -1}
        self._last_streamed_batch_values = {'input_text':'','author_prompts':[], 'output_texts':[], 'input_len': -1, 'output_lens': []}

    @property
    def torch_dtype(self):
        if self.model:
            return self.model.dtype
        return dtype_to(self.cfg.dtype, 'torch', default=None)  
    
    def config_path_data(self, checkpoint_path:str|Path, **kwargs):
        # https://huggingface.co/amazingvince/Not-WizardLM-2-7B
        path_data = ModelPathComponents.from_checkpoint(Path(checkpoint_path))
        
        config = OmegaConf.load(path_data.config_path)
        config.update(**{k:v for k,v in kwargs.items() if v is not None}) # filter to avoid overwriting with None
                   
        print(self.torch_dtype, config.attn_implementation) # Don't stop printing until fixed double call issue
        
        # Try to overide even if model wasen't trained with flash attn
        if config.attn_implementation is None:
            config.attn_implementation = 'flash_attention_2'
        
        if not config.postfix:
            config.postfix = ''

        return config,path_data
    
    def apply_stop_rules(self, tokenizer:transformers.PreTrainedTokenizer, gen_config:GenerationConfig, stop_criteria: list[transformers.StoppingCriteria]|None = None):
        '''Sets special cases for eos_tokens and stopping criteria'''
        tokenizer = tokenization.set_tokenizer_inference(tokenizer)

        #if gen_config.pad_token_id is None or gen_config.eos_token_id is None:
        gen_config.pad_token_id = tokenizer.pad_token_id
        gen_config.eos_token_id = tokenizer.eos_token_id
        
        if tokenizer.chat_template: 
            # TODO: This is going to become a mess if this trend continues. Need a automated way to do this.
            #  can't rely on base model generation config in case using the non-instruct version.
            
            extra_eos = {
                'Llama-3':'<|eot_id|>', 
                'Phi-3':'<|end|>',
            }
            for model_type, x_eos in extra_eos.items():
                # Check chat_template instead of tokens list in case using custom template override.
                if x_eos in tokenizer.chat_template:
                    new_eos = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(x_eos)]
                    if gen_config.eos_token_id != new_eos:
                        print(f'Using {model_type} format - {x_eos} added. eos_token_id: ({gen_config.eos_token_id})')     
                        gen_config.eos_token_id = new_eos

            # mistral models use [/INST] but it's NOT a single token, so can't be used in gc.eos_token_id.
            # In order to have model generate on both assistant AND user, need to stop on it.
            if '[/INST]' in tokenizer.chat_template:
                if stop_criteria is None:
                    stop_criteria = []
                stop_criteria += [genconfig.WordListCriteria.from_words(['[/INST]'], tokenizer)]

        return tokenizer, gen_config, stop_criteria


    @staticmethod
    def from_pretrained(checkpoint_path:str|Path, gen_config:GenerationConfig|Path = None, ytm:youtube.YouTubeManager = None, load:bool = True, **kwargs):
        try:
            path_data = ModelPathComponents.from_checkpoint(Path(checkpoint_path))
        except FileNotFoundError as e:
            from huggingface_hub import HfApi, utils as hub_utils
            try:
                _ = HfApi().model_info(checkpoint_path)
                warnings.warn('You passed a huggingface model_id. '
                              'Using a model without first finetuning is possible but not recommended. '
                              'To silence this warning use Cloneus.from_model_id() instead.')
                return CloneusUntuned(model_id=checkpoint_path, **kwargs)
            except hub_utils.RepositoryNotFoundError:
                raise e

        
        config = OmegaConf.load(path_data.config_path)
        
        if config.tag_placement == 'replace_role':
            ClonuesCls = CloneusRole
        elif config.tag_placement == 'content_prefix':
            ClonuesCls = CloneusUA
        elif config.tag_placement == 'tag_only':
            ClonuesCls = CloneusTag
        else:
            raise NotImplementedError('Unable to determine Cloneus Format Class')
        
        if gen_config is None:
            gen_config = 'generation_config.json'
        
        if ytm is None:
            ytm = youtube.YouTubeManager(enabled=True)
        clo = ClonuesCls(path_data=path_data, cfg=config, gen_config=gen_config, ytm=ytm, **kwargs)
        if load:
            return clo.load_model()
        return clo
    
    @staticmethod
    def from_model_id(model_id:str, author_names:list[str]=None, system_prompt:str|SystemPromptTemplate = None, load=True, **kwargs):
        cfg = CloneusUntuned.create_default_cfg(model_id, author_names=author_names, system_prompt=system_prompt, **kwargs)
        clo = CloneusUntuned(model_id, cfg=cfg)
        if load:
            return clo.load_model()
        return clo
        
    @iload.cleanup
    def load_model(self):
        self.unload_model(partial=True) # make sure all states are cleared first. If we want to preserve state, then should have called swap_model
        self.model, self.tokenizer = iload.load_any_inference(self.path_data.checkpoint_path, dtype=self.torch_dtype, attn_implementation=self.cfg.attn_implementation)
        self.tokenizer, self.gen_config, self.stop_criteria = self.apply_stop_rules(self.tokenizer, self.gen_config, stop_criteria=None)
        
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.model.config._name_or_path, trust_remote_code=True)
        self.base_dtype = AutoConfig.from_pretrained(self.model.config._name_or_path, trust_remote_code=True).torch_dtype
        if self.base_dtype not in [torch.float16, torch.bfloat16]: 
            self.base_dtype = torch.bfloat16 # Avoid ever running as float32
                
        self.base_has_system = tokenization.check_if_system(self.tokenizer) 
        self.base_needs_bos = self.base_tokenizer.chat_template and 'bos_token' not in self.base_tokenizer.chat_template

        print('Tk|Gc: (pad: {}|{}, eos: {}|{}), base_has_system: {}, has_prompt: {}, tag_placement: {}'.format(
            self.tokenizer.pad_token_id, self.gen_config.pad_token_id, 
            self.tokenizer.eos_token_id, self.gen_config.eos_token_id,
            self.base_has_system, (self.cfg.fprompt is not None), self.cfg.tag_placement,
        ))
        
        self.model.eval()

        return self
        
    @iload.cleanup
    def unload_model(self, partial:bool=True):
        '''Destroy model and tokenizers and try to free memory.
        
        Args:
            partial: If True, `cfg` and `path_data` are preserved for a subsequent call to `load_model()`. 
                Otherwise, the state is unrecoverable and `from_pretrained` must be called again. 
        '''
        self.model = None
        self.tokenizer = None
        self.base_tokenizer = None
        if not partial:
            self.cfg = None
            self.path_data = None
    
    def cast_dtype(self, dtype:str|torch.dtype):
        torch_dtype = dtype_to(dtype, to='torch')
        if self.model.dtype != torch_dtype:
            try:
                self.model.to(dtype=torch_dtype)
            except Exception as e:
                # AWQ will throw error if attempt to pass dtype in .to()
                print(e)
                if torch_dtype == torch.bfloat16:
                    self.model.bfloat16()
                elif torch_dtype == torch.float16:
                    self.model.half()

    def can_hotswap(self, new_cfg, new_path_data: ModelPathComponents):
        if any([self.model is None, self.cfg is None, self.path_data is None]):
            return False
        
        same_base_model = new_cfg.model_id == self.cfg.model_id
        lora_to_lora = new_path_data.has_adapter and self.path_data.has_adapter

        return same_base_model and lora_to_lora


    def _swap_adapter(self, new_cfg, new_path_data:ModelPathComponents, gen_config:str=None, dtype=None):
        adapter_name = (new_path_data.run_name + '-' + new_path_data.checkpoint_name).replace('.','')
        if not adapter_name in self.model.peft_config:
            self.model.load_adapter(new_path_data.checkpoint_path, adapter_name=adapter_name)
        
        self.model.set_adapter(adapter_name)
        # necessary -- https://huggingface.co/docs/peft/v0.10.0/en/package_reference/lora#peft.LoraModel.set_adapter
        for name,param in self.model.named_parameters():
            if hasattr(param,'requires_grad'):
                param.requires_grad_(False)
        
        if gen_config is not None:
            self.gen_config = self.load_genconfig(gen_config, new_path_data)

        if dtype is not None:
            self.cast_dtype(dtype=dtype)

        # For foundation models, changing the config changes how the tokenizer should behave.
        # So, to be safe, reapply stop rules if config changes
        if self.path_data.config_path != new_path_data.config_path:
            self.tokenizer, self.gen_config, self.stop_criteria = self.apply_stop_rules(self.tokenizer, self.gen_config, stop_criteria=self.stop_criteria)

        self.cfg, self.path_data = new_cfg, new_path_data
        
        return self


    @iload.cleanup
    def swap_model(self, checkpoint_path:(str|Path), gen_config:GenerationConfig|Path=None, dtype=None, attn_implementation: typing.Literal["eager", "sdpa", "flash_attention_2"]=None) -> None:
        # If the path is the same, assume that either want a dtype change or attn_impl change
        if Path(checkpoint_path) == self.path_data.checkpoint_path:
            if dtype:
                self.cast_dtype(dtype)
            if attn_implementation:
                return self.from_pretrained(checkpoint_path, gen_config=gen_config, dtype=dtype, attn_implementation=attn_implementation, ytm=self.ytm, load=True)#.load_model()
                
        cfg, path_data = self.config_path_data(checkpoint_path, dtype=dtype, attn_implementation=attn_implementation)
                
        if self.can_hotswap(cfg, path_data):
            return self._swap_adapter(cfg, path_data, gen_config=gen_config, dtype=dtype)
        
        # If it can't be hotswapped, nuke and pave
        self.unload_model(partial=False)
        
        return self.from_pretrained(checkpoint_path, gen_config=gen_config, dtype=dtype, attn_implementation=attn_implementation, ytm=self.ytm, load=True)#.load_model()
            
        
    def to_conversation_format(self, chat_history: list[tuple[str,str]]) -> list[dict[str,str]]:
        raise NotImplementedError('Cloneus is designed to be instantiated via Cloneus.from_pretrained')

    def to_text_input(self, chat_history: list[tuple[str,str]], author_seedtext: str|tuple[str,str]=None) -> str:
        '''Convert author message tuples into a text string, adding seed author text if provided, and applying youtube encoding if enabled'''
        chat_history = chat_history[:] # shallow copy to avoid mutation
        if author_seedtext is not None:
            text = self.to_seeded_text(chat_history, author_seedtext)
        else:
            text = self.tokenizer.apply_chat_template(self.to_conversation_format(chat_history), tokenize=False, add_generation_prompt=True)
                
        text = self.ytm.encode(text)
        
    
        return text
    
    def to_seeded_text(self, chat_history: list[tuple[str,str]], author_seedtext: str|tuple[str,str]):
        '''Converts to text by applying all normal formatting to chat_history + author but strips all formatting after seedtext. 
        
        This is done so that the model will continue generating from seedtext rather than stopping on a <|eot|> token or postfix.
        '''
        chat_history = chat_history[:] # shallow copy to avoid mutation
        if isinstance(author_seedtext, str):
            author_seedtext = (author_seedtext, '')
        
        author, seedtext = author_seedtext
        if seedtext is None:
            seedtext = ''
        
        seedtext_surrogate = '###_dummy_seedtext_###'
        chat_history += [(author, seedtext_surrogate)]
        text = self.tokenizer.apply_chat_template(self.to_conversation_format(chat_history), tokenize=False, add_generation_prompt=True)
        # split on dummy text to KEEP pre-content formatting but REMOVE post content formatting
        # this effectively removes the tag_sep dependency
        # which is important for models where the role has a special token before the content 
        # e.g. Llama-3: <|start_header_id|>(AUTHOR_TAG)<|end_header_id|>\n\n(CONTENT)<|eot_id|>
        # rather than requiring tag_sep = <|end_header_id|>, just don't use it all
        
        text = text.split(seedtext_surrogate)[0] + seedtext
        text = text.rstrip(' ') # IFF the formatted author_tag ends with a space ' ' should NOT trail with it
        #text += self.apply_content_prefix(author, seedtext, tag_sep).strip(' ') # IFF using ' ' as tag_sep, should NOT trail with it

        return text

    def cleanup_out_texts(self, out_texts: str|list[str]):
        '''Trim off added stop_criteria words, if any, from model output'''
        #out_texts = [ot.split(self.cfg.postfix)[0] for ot in output_texts] # old method
        if self.stop_criteria:
            for crit in filter(lambda c: isinstance(c, genconfig.WordListCriteria), self.stop_criteria):
                if isinstance(out_texts, str):
                    out_texts = crit.trim_stopwords(out_texts)
                else:
                    out_texts = [crit.trim_stopwords(text) for text in out_texts]
        return out_texts
    
    def encode_wordslist(self, wordslist:list[str]|list[tuple[str,float]]) -> (list[list[int]] | dict[tuple, float]):
        '''Use for GenerationConfig `bad_words_ids`, `force_words_ids`, or (if weights passed) with `sequence_bias`'''
        weights=None
        
        if isinstance(wordslist[0],tuple):
            wordslist,weights = list(zip(*wordslist))
        
        tokens = self.tokenizer(wordslist, add_special_tokens=False).input_ids # Do NOT add bos, since want exact token ids
        if weights is not None:
            assert len(wordslist) == len(weights), 'all words need a weight'
            return {tuple(t):w for t,w in zip(tokens,weights)}
        
        return tokens
    
    @torch.inference_mode()
    def _proba_next_token(self, chat_history, top_k=5):
        #author_surrogate = '###_dummy_author_###'
        #formatted_author_surrogate = useridx.format_author_tag(author_surrogate, self.cfg.author_tag, insert_raw=True)
        base_input = self.to_text_input(chat_history)#+[(author_surrogate, 'ignored')]).split(formatted_author_surrogate)[0]
        
        inputs = self.tokenizer(base_input, return_tensors="pt", add_special_tokens=False) # add_special_tokens=False
        outputs = self.model(**inputs.to(0))
        
        logprobs = torch.log_softmax(outputs.logits, dim=-1).detach_()
        
        topkn = logprobs[:,-1,:].topk(top_k)
        return topkn.values.exp().detach(), self.tokenizer.convert_ids_to_tokens(topkn.indices.squeeze(0))

    @torch.inference_mode()
    def _atag_shared_ntokens(self, formatted_atags:list[str]):
        '''Count the number of shared common author tag prefix tokens tokenized.
        e.g. for author tags ([USER:abc], USER:xyz]) might be 3 - { "[", "USER", ":" }
        '''
        tokd_fatags = [self.tokenizer.tokenize(fatag) for fatag in formatted_atags]
        #atag_lc_tokens = 0
        
        min_taglen = min(len(t) for t in tokd_fatags)  # longest possible common subtokens would be shortest tag
        lead_ntokens=0
        for i in range(min_taglen):
            # if all tokens at index i are the same, +1 and continue else done
            n_unique = len(set([t[i] for t in tokd_fatags]))
            if n_unique != 1:
                break
            
            lead_ntokens += 1
        # check trailing tokens
        trail_ntokens=0
        for i in range(1,min_taglen+1):
            # if all tokens at index i are the same, +1 and continue else done
            n_unique = len(set([t[-i] for t in tokd_fatags]))
            if n_unique != 1:
                break
            
            trail_ntokens += 1
        return lead_ntokens,trail_ntokens


    @torch.inference_mode()
    def author_probabilities(self, chat_history: list[tuple[str,str]], authors: list[str]=None) -> list[tuple[str,float]]:
        '''For each author, return the probability they will be the next to respond given the chat context.

        Args:
            chat_history: An ordered list of unformatted (author name, message) tuples representing the current chat conversation.
            authors: A list of author names to calculate probabilities for. Defaults to all authors in user index if None.

        Returns:
            A list of tuples containing an author and their probability to be the next to respond.
        '''
        # TODO: This causes a memory leak somehow. FIXME
        # Pros over old method:
        # - It's actually correct (?)
        # - If 2 authors have same first token, it can differentiate between them
        # - Don't need to worry about splitting name+author_tag in unexpected ways since whole tag is predicted
        # - It will always match one of the authors
        # - Doesn't depend on generation config settings

        # Cons:
        # - It will always match one of the authors. So unless a name is passed, it can't be predicted.
        # - Its batched, so padding weirdness affects probabilties (this could be changed, but it _needs_ to be as fast as possible)
        # - There is still potential for weird interaction between tag and chat template if preceding token isn't special (looking at you mistral..)
        # - Doesn't depend on generation config settings. Can't make the result more interesting by tweaking.

        # ref: https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/17
        if authors is None:
            authors = useridx.get_users('dname')

        n_author = len(authors)
        author_surrogate = '###_dummy_author_###'
        text_surrogate = '###_dummy_text_###'
        base_surrogate = self.to_text_input(chat_history, (author_surrogate, text_surrogate)) 

        # we DO want to split out the whole tag, since we are now getting whole tag probablities
        fmt_auth_surrogate = useridx.format_author_tag(author_surrogate, self.cfg.author_tag)
        #base_input = base_surrogate.split(fmt_auth_surrogate)[0] 

        fmt_atags = [useridx.format_author_tag(a, self.cfg.author_tag) for a in authors]
        tag_lens = torch.as_tensor(self.tokenizer(fmt_atags, add_special_tokens=False, return_length=True).length)
        # print(tag_lens)
        #candidate_batch = [base_input + a for a in fmt_atags]
        # NOTE: We need EVERYTHING around the author tag, if there is a newline after the tag, it may change the token
        # ex: Llama-3 -- If author tag is "Username (FName)" and tag sep = \n. Then "Username (FName)\n" will change rparen:  
        # [ ")", id: 8 ] becomes [ ")Ċ", id: 320 ]
        # probabilities will be off by ORDERS OF MAGNITITUDE if you use ")" instead of ")Ċ"
        candidate_batch = [base_surrogate.replace(fmt_auth_surrogate, ftag).split(text_surrogate)[0] for ftag in fmt_atags] # do **not** strip.
        b_inputs = self.tokenizer(candidate_batch, return_tensors="pt", padding=True, add_special_tokens=False)
        b_outputs = self.model(**b_inputs.to(0))

        # Remember - always proba of the *next* token, not current. So need to back step index by 1.
        b_logprobs = torch.log_softmax(b_outputs.logits, dim=-1).detach()[:, :-1, :].cpu() # (b, seq-1, V)
        b_input_ids = b_inputs.input_ids[:, 1:].cpu() # (b, seq-1)

        # skipping pre/post tag formatting tokens both gives a better estimate and fixes mistral spacing issues
        n_skip, n_trim = self._atag_shared_ntokens(fmt_atags)
        # -1 is needed or will be off by 1. Think it has to do with shifting indexing
        h_offset = max(n_skip-1, 0)
        # p_author = torch.zeros((n_author, tag_lens.max().item()))
        p_author = torch.full((n_author, tag_lens.max().item()), torch.nan) # torch.empty will throw off probas, zero might be okay 
        #nindices = []
        seq_len = b_input_ids.size(1)

        for i in range(n_author):
            
            taglen = tag_lens[i]
            start_idx = seq_len-taglen # idx of first author_tag token
            ustart_idx = start_idx+h_offset # idx of 1st non-shared author_tag token
            ustop_idx = seq_len-n_trim # idx of last non-shared author_tag token

            cur_name_logprobs = b_logprobs[i, ustart_idx:ustop_idx, :]
            cur_name_indices = b_input_ids[i, ustart_idx:ustop_idx, None]
            
            namelen = cur_name_logprobs.size(0)
            p_author[i, :namelen] = torch.gather(cur_name_logprobs, -1, cur_name_indices).squeeze()

            #nindices.append(cur_name_indices.squeeze())
        
        # for i in range(n_author):
        #     #cur_name_logprobs = b_logprobs[i, -tag_lens[i]-1:-1]
        #     cur_name_logprobs = b_logprobs[i, -tag_lens[i]:]
        #     cur_name_indices = b_input_ids[i, -tag_lens[i]:, None]
        #     p_author[i, :tag_lens[i]] = torch.gather(cur_name_logprobs, -1, cur_name_indices).squeeze()
        
        #return p_author,nindices,candidate_batch
        #return sorted(zip(authors, p_author.nansum(-1).exp().tolist()), key=lambda x: x[1], reverse=True)
        
        
        #a_tag_probas = p_author[:,n_skip:].nansum(-1).exp().tolist()
        a_tag_probas = p_author.nansum(-1).exp().tolist()
        author_probas = zip(authors, a_tag_probas)

        return sorted(author_probas, key=lambda x: x[1], reverse=True)
    

    @torch.inference_mode()
    def _batched_helper(self, msg_batch, true_batch_generate=False):
        t0 = time.perf_counter()

        if true_batch_generate:
            # BUG IMPORTANT: There is an issue with padding token. Whenever it is inserted, it makes those responses bad
            # I verified that ONLY when pad token is used, then it goes wrong.
            #with batchsafe_tokenizer(self.tokenizer) as tokenizer:
            inputs = self.tokenizer(msg_batch, return_length=True, return_tensors='pt', padding=True, add_special_tokens=False)
            
            input_len = inputs.pop('length')[0].item()
            outputs = self.model.generate(**inputs.to(0), generation_config=self.gen_config, stopping_criteria=self.stop_criteria).detach_()
            
            output_texts = self.tokenizer.batch_decode(outputs[:,input_len:], skip_special_tokens=True)
        else:
            # if not doing the whole batch, no point in incuring the 'penalty' for adding pad tokens to begining
            # better to just do tokenization on the fly
            output_texts  = []

            for inptext in msg_batch:
                inputs = self.tokenizer(inptext, return_tensors="pt", return_length=True, add_special_tokens=False)
                input_len = inputs.pop('length')[0].item()
                output = self.model.generate(**inputs.to(0), generation_config=self.gen_config, stopping_criteria=self.stop_criteria, negative_prompt_ids=None).detach_()
                output_texts.append(self.tokenizer.decode(output[0,input_len:], skip_special_tokens=True))

        print(('TRUE' if true_batch_generate else 'MOCK' ) + f' BATCH RUN TIME: {time.perf_counter()-t0:0.2f}s')
        #print('Raw Batch Outputs:\n', self.tokenizer.batch_decode(outputs, skip_special_tokens=False))
        
        # use last input_len for all. Shouldn't differ by more than a few tokens as long as author_tags are reasonable
        return output_texts,input_len
    

    def _get_batched_inputs(self, chat_history: list[tuple[str,str]], seed_authors: list[str], seed_text: str = None) -> tuple[str, list[str]]:
        # Need to be a little bit careful about how we combine previous context + author context
        # to_text_input(previous_context) + to_text_input((author,seed)) is not necessarily the same as to_text_input(previous_context+(author,seed))
        # because some chat_templates use index0 or otherwise for special behavior
        # simple but inefficent way is to_text_input(previous_context+(author,seed)) for each author. better way is split and replace
        
        input_context = self.to_text_input(chat_history, author_seedtext=('###_dummy_author_###','###_dummy_seedtext_###'))
        
        author_surrogate = useridx.format_author_tag('###_dummy_author_###', self.cfg.author_tag) # want to make sure we replace the whole, formatted tag
        seedtext_surrogate = '###_dummy_seedtext_###' # do NOT want to match surrounding markup, just the seed_text itself to be replaced

        input_context, fmt_dummy_seedtext = input_context.split(author_surrogate)
        authseed_template = author_surrogate+fmt_dummy_seedtext
        
        if seed_text is None: 
            seed_text = ''
        
        author_prompts = [authseed_template
                          .replace(author_surrogate, useridx.format_author_tag(u, self.cfg.author_tag))
                          .replace(seedtext_surrogate, seed_text)
                          .rstrip(' ') # IFF the formatted author_tag ends with a space ' ' should NOT trail with it 
                          for u in seed_authors]
        
        return input_context, author_prompts
    
    @torch.inference_mode()
    def batch_generate(self, chat_history: list[tuple[str,str]], seed_authors: list[str], seed_text: str = None, return_tuple: bool = False) -> list[str] | BatchGenerationOutput:
        """Generate responses for a batch of authors, each optionally starting with `seed_text` given the message context history.

        Args:
            chat_history: An ordered list of unformatted (author name, message) tuples representing the current chat conversation.
            seed_authors: The list of authors to generate responses for.
            seed_text: The seed text to start off all author's reponses with.
            return_tuple: If True, return BatchGenerationOutput tuple (input text, list formatted author+seed text, list of output texts, num input tokens, num output tokens for each output)

        Returns:
            BatchGenerationOutput if `return_tuple`, output text completions for `seed_authors` otherwise.
        """
        
        input_context,author_prompts = self._get_batched_inputs(chat_history, seed_authors, seed_text=seed_text)

        true_batched = (self.gen_mode == 'contrastive_search')  # Cuts time almost in half for CS. Worth the quality degradation.
        out_texts,input_len = self._batched_helper([input_context+ap for ap in author_prompts], true_batch_generate=true_batched)

        out_texts = self.cleanup_out_texts(out_texts)

        output_lens = self.tokenizer(out_texts, return_length=True, add_special_tokens=False,)['length']
        
        if return_tuple:
            return BatchGenerationOutput(input_context, author_prompts, out_texts, input_len, output_lens)
        return out_texts
   
    @torch.inference_mode()
    def generate(self, chat_history: list[tuple[str,str]], reply_author: str|tuple[str,str], return_tuple: bool = False) -> str | GenerationOutput:
        """Generate a response for `reply_author` given the message context history.

        Args:
            chat_history: An ordered list of unformatted (author name, message) tuples representing the current chat conversation.
            reply_author: The author name to respond as, or a tuple of (author name, seed text) to start the response with.
            return_tuple: If True, return GenerationOutput tuple (formatted input text, output text, num input tokens, num output tokens)

        Returns:
            GenerationOutput if `return_tuple`, output text string otherwise.
        """
        input_text = self.to_text_input(chat_history, reply_author)

        inputs = self.tokenizer(input_text, return_tensors="pt", return_length=True, add_special_tokens=False,)
        input_len = inputs.pop('length')[0].item()

        input_text = self.tokenizer.batch_decode(inputs.input_ids)[0]

        output = self.model.generate(**inputs.to(0), generation_config=self.gen_config, stopping_criteria=self.stop_criteria, negative_prompt_ids=None).detach_()
        out_tokens = output[0,input_len:]
        output_len = out_tokens.shape[0]
        # weird NOTE: if custom special tokens, decode skip_special_tokens **must**=FALSE. But encode add_special_tokens = (True | False), doesn't mater will be added regardless
        out_text = self.tokenizer.decode(out_tokens, skip_special_tokens=(not self.cfg.has_custom_tokens))
        out_text = self.cleanup_out_texts(out_text)
        
        if return_tuple:
            return GenerationOutput(input_text, out_text, input_len, output_len)
        return out_text
    
    
    @torch.inference_mode()
    def stream_generate(self, chat_history: list[tuple[str,str]], reply_author: str|tuple[str,str]):
        """Streamed version of `.generate()`

        Sets attribute `_last_streamed_values` with inputs and outputs for retrieval with `.last_streamed()`
        
        Args:
            chat_history: An ordered list of unformatted (author name, message) tuples representing the current chat conversation.
            reply_author: The author name to respond as, or a tuple of (author name, seed text) to start the response with.

        Yields:
           str: The next predicted token string.
        """
        
        # https://huggingface.co/docs/transformers/internal/generation_utils#transformers.TextStreamer
        self._last_streamed_values = {'input_text':'', 'output_text':'', 'input_len': -1, 'output_len': -1}
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=120.0, skip_special_tokens=(not self.cfg.has_custom_tokens))

        input_text = self.to_text_input(chat_history, reply_author)
        inputs = self.tokenizer(input_text, return_tensors="pt", return_length=True, add_special_tokens=False)#, max_length=1024, truncation=True)
        input_len = inputs.pop('length')[0].item()

        genkwargs = dict(inputs.to(0), generation_config=self.gen_config, streamer=streamer, stopping_criteria=self.stop_criteria, negative_prompt_ids=None)
        thread = Thread(target=self.model.generate, kwargs=genkwargs)
        thread.start()
        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            yield new_text
        
        generated_text = self.cleanup_out_texts(generated_text)
        output_len =  self.tokenizer(generated_text, return_length=True, add_special_tokens=False).length

        self._last_streamed_values.update({'input_text':input_text, 'output_text': generated_text, 'input_len': input_len, 'output_len': output_len})

    
    @torch.inference_mode()
    def stream_batch_generate(self, chat_history: list[tuple[str,str]], seed_authors: list[str], seed_text: str = None):
        """Streamed version of `.batch_generate()`

        Sets attribute `_last_streamed_batch_values` with inputs and outputs for retrieval with `.last_streamed(batch=True)`
        
        Args:
            chat_history: An ordered list of unformatted (author name, message) tuples representing the current chat conversation.
            seed_authors: The list of authors to generate responses for.
            seed_text: The seed text to start of each author's reponse with.

        Yields:
            tuple[int,str]: A tuple of the author index and the next predicted token string.
        """
        # https://huggingface.co/docs/transformers/internal/generation_utils#transformers.TextStreamer
        self._last_streamed_batch_values = {'input_text':'','author_prompts':[], 'output_texts':[], 'input_len': -1, 'output_lens': []}
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=120.0, skip_special_tokens=True)

       
        input_context,author_prompts = self._get_batched_inputs(chat_history, seed_authors, seed_text=seed_text)
        
        msg_batch = [input_context+ap for ap in author_prompts]

        for i, inptext in enumerate(msg_batch):
            inputs = self.tokenizer(inptext, return_tensors="pt", return_length=True, add_special_tokens=False)
            input_len = inputs.pop('length')[0].item()
            #genkwargs = dict(input_ids=inps.input_ids[[i]], attention_mask=inps.attention_mask[[i]], 
            genkwargs = dict(**inputs.to(0),
                             generation_config=self.gen_config, streamer=streamer, stopping_criteria=self.stop_criteria)
            thread = Thread(target=self.model.generate, kwargs=genkwargs)
            thread.start()
            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                yield i,new_text
            
            generated_text = self.cleanup_out_texts(generated_text)
            output_len = self.tokenizer(generated_text, return_length=True, add_special_tokens=False).length
            
            self._last_streamed_batch_values['output_texts'].append(generated_text)
            self._last_streamed_batch_values['output_lens'] += output_len
            
            
        self._last_streamed_batch_values.update(input_text=input_context, author_prompts=author_prompts, input_len=input_len)


    def last_streamed(self, batch: bool = False, return_tuple: bool = False) -> str | GenerationOutput | list[str] | BatchGenerationOutput:
        """Retrieve the last streamed generation outputs.

        Args:
            batch: If True, retrieve outputs from last streamed batch generation, otherwise last generation outputs.
            return_tuple: If True, return a GenerationOutput or BatchGenerationOutput tuple containing input(s), output(s), and token counts.

        Returns:
            generation output(s): The last streamed outputs.
            If `batch` is True returns BatchGenerationOutput if `return_tuple`, list of output text completions otherwise.
            If `batch` is False returns GenerationOutput tuple if `return_tuple`, output text string otherwise.
        """
        if batch:
            input_context, author_prompts, model_outputs, input_length, output_lengths = [
                self._last_streamed_batch_values.get(v) for v in ['input_text', 'author_prompts', 'output_texts', 'input_len', 'output_lens']]

            if return_tuple:
                return BatchGenerationOutput(input_context, author_prompts, model_outputs, input_length, output_lengths)
            return model_outputs

        input_text, model_output, input_length, output_length = [
            self._last_streamed_values.get(v) for v in ['input_text', 'output_text', 'input_len', 'output_len']]

        if return_tuple:
            return GenerationOutput(input_text, model_output, input_length, output_length)
        return model_output

    def base_to_conversation_format(self, chat_history: str|list[str], system_prompt:str=None):
        if isinstance(chat_history, str):
            chat_history = [chat_history]
        
        rolecycle = itertools.cycle(['user','assistant'])
        chat_content = []

        if system_prompt is not None and self.base_has_system:
            chat_content.append({"role": "system", "content": system_prompt})

        for message in chat_history:
            chat_content.append({"role": next(rolecycle), "content": message})

        input_text = self.base_tokenizer.apply_chat_template(chat_content, tokenize=False, add_generation_prompt=True)

        #print('chat_content:',chat_content)
        print(f'input_text: {input_text!r}',)
        
        return input_text

    @torch.inference_mode()
    def base_generate(self, messages:str|list[str], system_prompt:str=None, return_tuple: bool = False) -> str | GenerationOutput:
        """Generate a response using the underlying untuned base model.

        Note:
            This only works for models with an active peft adapter. 
        
        Args:
            messages: An ordered sequence of alternating user/assistant messages. 
            system_prompt: The system prompt message to guide the chat generation.
            return_tuple: If True, return GenerationOutput tuple (input text, output text, num input tokens, num output tokens)

        Returns:
            GenerationOutput if `return_tuple`, output text string otherwise.
        """
        # adapters = self.model.active_adapters()
        # self.model.disable_adapters()
        # TODO: Function for base/cloneus hybrid
        # - just user/assistant tags on a trained model. Surprisingly, sort of works to have AI style responsiveness but with custom vernacular

        input_text = self.base_to_conversation_format(messages, system_prompt=system_prompt)

        inputs = self.base_tokenizer(input_text, return_tensors="pt", return_length=True, add_special_tokens=self.base_needs_bos)
        input_len = inputs.pop('length')[0].item()
        
        with self.model.disable_adapter(), torch.cuda.amp.autocast(dtype=self.base_dtype):
            output = self.model.generate(**inputs.to(0), generation_config=self.base_gen_config, stopping_criteria=self.stop_criteria, negative_prompt_ids=None).detach_() # adapter_names=["__base__"]
        
        out_tokens = output[0,input_len:]
        output_len = out_tokens.shape[0]
        out_text = self.base_tokenizer.decode(out_tokens, skip_special_tokens=True)
        #self.model.set_adapter(adapters)
        if return_tuple:
            return GenerationOutput(input_text, out_text, input_len, output_len)
        return out_text
    

    @torch.inference_mode()
    def base_stream_generate(self, messages:str|list[str], system_prompt:str=None):
        """Streamed version of `.base_generate()`

        Sets attribute `_last_streamed_values` with inputs and outputs for retrieval with `.last_streamed()`

        Note:
            This only works for models with an active peft adapter. 
        
        Args:
            messages: An ordered sequence of alternating user/assistant messages. 
            system_prompt: The system prompt message to guide the chat generation.

        Yields:
            str: The next predicted token string.
        """
        #adapters = self.model.active_adapters()
        #self.model.disable_adapters()
        #base_model: PeftModel = self.model.get_base_model()
        
        self._last_streamed_values = {'input_text':'', 'output_text':'', 'input_len': -1, 'output_len': -1}
        streamer = TextIteratorStreamer(self.base_tokenizer, skip_prompt=True, timeout=120.0, skip_special_tokens=True)

        input_text = self.base_to_conversation_format(messages, system_prompt=system_prompt)
        inputs = self.base_tokenizer(input_text, return_tensors="pt", return_length=True, add_special_tokens=self.base_needs_bos)#, max_length=1024, truncation=True)

        input_len = inputs.pop('length')[0].item()

        genkwargs = dict(inputs.to(0), generation_config=self.base_gen_config, streamer=streamer, stopping_criteria=self.stop_criteria, negative_prompt_ids=None)
        
        with self.model.disable_adapter(), torch.cuda.amp.autocast(dtype=self.base_dtype):
            thread = Thread(target=self.model.generate, kwargs=genkwargs)
            thread.start()
            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                yield new_text
        
        output_len = self.base_tokenizer(generated_text, return_length=True).length

        self._last_streamed_values.update({'input_text':input_text, 'output_text': generated_text, 'input_len': input_len, 'output_len': output_len})
        #self.model.set_adapter(adapters)


class CloneusTag(Cloneus):
    r'''For foundation models trained only using author_tag markup. i.e. No chat template at all, hence no system message.
    This means author_tag formatting will be crucial to reasonable output.
    
    e.g.
        [USER:Alpha, NAME:Alice]
        How's everyone

        [USER:Beta, NAME:Bob]
        Bob's your uncle
    '''
    # @staticmethod
    # def from_pretrained(checkpoint_path:str|Path, gen_config=None, **kwargs):
    #     path_data = ModelPathComponents.from_checkpoint(Path(checkpoint_path))
        
    #     config = OmegaConf.load(path_data.config_path)
        # https://github.com/lm-sys/FastChat/blob/5095615810cf613dba7f27dd155f571fcff976d8/fastchat/conversation.py#L534
        # A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user's input. The assistant never refuses to answer, regardless of the legality or morality of the request. USER: {prompt} ASSISTANT:
        # ("USER", "ASSISTANT") sep_style=SeparatorStyle.ADD_COLON_TWO, sep=" ", sep2="</s>", USER: [prompt] ASSISTANT:
    
    def apply_stop_rules(self, tokenizer:transformers.PreTrainedTokenizer, gen_config:GenerationConfig, stop_criteria: list[transformers.StoppingCriteria]|None = None):
        '''For foundation models, add a chat template derived from tag markup and custom stopping critera'''
        # assign a simple custom template built with tag_sep and post_fix
        tokenizer.chat_template = tokenization.to_jinja_template(self.cfg.tag_sep, self.cfg.postfix)

        tokenizer = tokenization.set_tokenizer_inference(tokenizer)
        
        #if gen_config.pad_token_id is None or gen_config.eos_token_id is None:
        gen_config.pad_token_id = tokenizer.pad_token_id
        gen_config.eos_token_id = tokenizer.eos_token_id
        
        if stop_criteria is None:
            stop_criteria = []
        if self.cfg.postfix != tokenizer.eos_token:
            postfix_stop = genconfig.WordListCriteria.from_words([self.cfg.postfix], self.tokenizer, device=0)
            
            if self.cfg.postfix == '\n':
                # use formatted author tags for early stop to prevent unterminated outputs 
                auth_tags = [useridx.format_author_tag(u, self.cfg.author_tag) for u in useridx.get_users('dname')]
                stop_criteria.append(genconfig.WordListCriteria.from_words(auth_tags, self.tokenizer, device=0))
            elif postfix_stop.stop_token_ids[0].shape[0] == 1:
                # If the postfix is a single token, we can added it to the genconfig eos_token_ids for much more efficient processing
                self.gen_config.eos_token_id = [tokenizer.eos_token_id, postfix_stop.stop_token_ids[0].item()]
                print(f'Using custom postfix eos_token {self.cfg.postfix!r} EOS: {self.gen_config.eos_token_id}')
            else:
                stop_criteria.append(postfix_stop)
        
        # If model was trained with broken eos (i.e. no space between) it will fail to stop on eos
        # e.g. llama-2: "a </s>" -> input_ids=[263, 2]. But "a</s>" -> input_ids=[263, 829, 29879, 29958] (['</', 's', '>'])
        if tokenizer.eos_token in self.cfg.postfix:
            EOS = tokenizer.eos_token
            eos_id = tokenizer(f'{EOS}', add_special_tokens=False)['input_ids']
            nospace_eos_id = tokenizer(f'A{EOS}', add_special_tokens=False)['input_ids'][1:]
            
            if eos_id != nospace_eos_id:
                pretag = useridx.format_author_tag(user_display_name='###DUMMY', author_tag=self.cfg.author_tag, insert_raw=True).split('###DUMMY')[0]
                
                eos_pretag = f'{EOS}{pretag}'
                nospace_eos_pretag_id = tokenizer(f'A{eos_pretag}', add_special_tokens=False)['input_ids'][1:]
                
                broken_eos_stop = genconfig.WordListCriteria(
                    stop_token_ids = [torch.tensor(nospace_eos_id).to(0), torch.tensor(nospace_eos_pretag_id).to(0)], 
                    words = [EOS, eos_pretag]
                )
                
                print(f'Broken eos detected. Adding stop crit for segmented eos {tokenizer.batch_decode(nospace_eos_id)!r} + eos_pretag {tokenizer.batch_decode(nospace_eos_pretag_id)!r}')

                stop_criteria.append(broken_eos_stop)

        # print('GC EOS, TK EOS, POSTFIX:', gen_config.eos_token_id, tokenizer.eos_token, self.cfg.postfix)
        # print(['Words: {}, ids: {}'.format(s.words, s.stop_token_ids) for s in (stop_criteria if stop_criteria is not None else [])])
        if stop_criteria == []:
            stop_criteria = None
            
        return tokenizer, gen_config, stop_criteria
    

    def to_conversation_format(self, chat_history: list[tuple[str,str]]) -> list[dict[str,str]]:
        '''For tag only, markup free, format'''
        # why Foundation? : https://crfm.stanford.edu/2021/10/18/reflections.html#:~:text=are%20situated%20in.-,Naming,-The%20name%20%E2%80%9Cfoundation        
        # input_text = ''.join([self.apply_template(a,t, self.cfg.tag_sep, postfix=self.cfg.postfix) for a,t in author_messages])
        chat_content = []
        
        for author,message in chat_history:
            role_tag = useridx.format_author_tag(author, self.cfg.author_tag) # for TagFormat, tag_sep is baked in to chat_template via to_jinja_template
            chat_content.append({"role": role_tag, "content": message})
        
        return chat_content
    


    
class CloneusRole(Cloneus):
    r'''For ChatML models (and possibly other chat formats) trained with a custom chat template. i.e. does not alternate
    user/assistant, instead uses users,names in place of user/assistant. Requires a system message.
    
    e.g.::
        <|im_start|>system
         A conversation between Bob and a well meaning tree<|im_end|>
        <|im_start|>Beta (Bob)
         How's everyone<|im_end|>
        <|im_start|>Groot (Groot)
         I am Groot<|im_end|>
    '''
    def to_conversation_format(self, chat_history: list[tuple[str,str]]) -> list[dict[str,str]]:
        '''For chatml with custom roles as usernames'''
        chat_content = []

        #if self.use_sysprompt:
        chat_content.append({"role": "system", "content": self.cfg.fprompt})
        
        for author,message in chat_history:
            role_tag = useridx.format_author_tag(author, self.cfg.author_tag)
            chat_content.append({"role": role_tag, "content": message})
        
        return chat_content
    

class CloneusUA(Cloneus):
    r'''For QA/Chat models trained without a custom chat template, i.e. template must alternate
    user/assistant, possibly starting with system. This includes mistral where no explicit system/user/assistant
    is in the prompt, but apply_chat_template format requires u/a. Typically, this means author_tag will have formatting
    with an explicit USER:user, NAME:name type structure.
    
    e.g.::
        [INST] <USER:Alpha, NAME:Alice> hello [/INST]<USER:Beta, NAME:Bob> hi</s>
    
    e.g.::
        <|im_start|>user
         [USER:Beta, NAME:Bob] How's everyone<|im_end|>
        
        <|im_start|>assistant
         [USER:Gamma, NAME:Greg] I am Greg, thanks<|im_end|>
    '''
    def apply_content_prefix(self, author:str, text_content:str, tag_sep:str):
        atag=useridx.format_author_tag(author, self.cfg.author_tag)
        return f'{atag}{tag_sep}{text_content}' # postfix was never used in this function call, always was set to '' 


    def to_conversation_format(self, chat_history: list[tuple[str,str]]) -> list[dict[str,str]]:        
        chat_content = []
        rolecycle = itertools.cycle(['user','assistant'])
        
        if self.cfg.fprompt:
            if self.base_has_system:
                system_message = [{'role':'system', 'content': self.cfg.fprompt}]
            elif isinstance(self.cfg.prompt.append_msg, bool) and self.cfg.prompt.append_msg: # legacy
                content0 = self.apply_content_prefix(*chat_history[0], tag_sep=self.cfg.tag_sep)
                system_message = [{'role':next(rolecycle), 'content': self.cfg.fprompt + content0}]
                chat_history = chat_history[1:]
                
            elif isinstance(self.cfg.prompt.append_msg, str):
                system_message = [
                    {'role':'user', 'content': self.cfg.fprompt}, 
                    {'role':'assistant', 'content': self.cfg.prompt.append_msg}]
            else:
                system_message = [{'role': next(rolecycle), 'content': self.cfg.fprompt}]

            chat_content.extend(system_message)
            

        for auth,msg in chat_history:
            message = self.apply_content_prefix(auth, msg, tag_sep=self.cfg.tag_sep)
            chat_content.append({"role": next(rolecycle), "content": message})
        
        return chat_content
    




class CloneusUntuned(CloneusUA):
    r'''Use a untuned base model for user chat emulation. 
    
    Note: To have any success with this approach, a very detailed system prompt is required. 
    Regardless of prompt used, the quality cannot match that of a finetuned model.
    This is therefore not recommended in most instances.
    '''
    def __init__(self, model_id:str, gen_config:GenerationConfig|Path = None, cfg: DictConfig = None, **kwargs) -> None:
        
        self.model = None
        self.tokenizer = None
        self.path_data = None
        self.cfg = self.create_default_cfg(model_id, **kwargs) if cfg is None else cfg
        
        self.gen_config = self.load_genconfig(gen_config)
        self.ytm = None #kwargs.pop('ytm')#youtube.YouTubeManager()
        
        self._last_streamed_values = {'input_text':'', 'output_text':'', 'input_len': -1, 'output_len': -1}
        self._last_streamed_batch_values = {'input_text':'','author_prompts':[], 'output_texts':[], 'input_len': -1, 'output_lens': []}
    
    @staticmethod
    def create_default_cfg(model_id:str, author_names:list[str]=None, system_prompt:str|SystemPromptTemplate = None, author_tag:str=None, tag_sep:str=' ', attn_implementation:str='flash_attention_2', **kwargs):
        if author_names is not None:
            user_index = useridx.new_user_index(author_names)
        elif useridx.user_index_exists():
            user_index = useridx.get_users()
        else:
            raise ValueError('You must provide a list of `author_names`. No existing users.json to fallback on.')
            
        
        if author_names is None:
            author_names = useridx.get_users('dname', user_index=user_index)

        if author_tag is not None:
            assert r'{author}' in author_tag or r'{fname}' in author_tag, 'At least one of "{author}" or "{fname}" needs to be in author_tag template!'
        else:
            # Use first names if defined
            if all(useridx.get_users('fname', by='dname', user_index=user_index).get(author) for author in author_names):
                author_tag = '[USER:{author}, NAME:{fname}]:'
            else:
                author_tag = '[USER:{author}]:'
        
        name_mapping = ', '.join(useridx.format_author_tag(author, author_tag, user_index=user_index) for author in author_names)

        has_system = tokenization.check_if_system(AutoTokenizer.from_pretrained(model_id, trust_remote_code=True))

        if system_prompt is None:
            system_prompt = SystemPromptTemplate(has_system_role = has_system)
        
        if isinstance(system_prompt, SystemPromptTemplate):
            system_prompt = system_prompt.get_template()
        
        formatted_system_prompt = system_prompt.format(name_mapping=name_mapping)
        

        torch_dtype = AutoConfig.from_pretrained(model_id, trust_remote_code=True).torch_dtype
        
        # Don't want f32
        if torch_dtype in [torch.float16, torch.bfloat16]: 
            dtype = str(torch_dtype).split('.')[-1]
        else:
            dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'


        cfg = OmegaConf.create({
            'model_id': model_id,
            
            'author_tag': author_tag,
            'tag_sep': tag_sep,
            'postfix':'',

            'tag_placement': 'content_prefix',
            
            'prompt':{
                'append_msg': ("OK" if not has_system else None),
                'name_mapping': name_mapping,
                'template': system_prompt,
            },
            'fprompt': formatted_system_prompt,
            'dtype': dtype,
            'attn_implementation': attn_implementation,
            
            'has_custom_tokens':False
        })
        
        if kwargs:
            print('Unused kwargs added to cfg:',kwargs)
            cfg.update(kwargs)
    
        return cfg

    @iload.cleanup
    def load_model(self):
        self.model, self.tokenizer = iload.load_any_inference(self.cfg.model_id, quant_method='merged', dtype=self.torch_dtype, attn_implementation=self.cfg.attn_implementation)
        self.tokenizer, self.gen_config, self.stop_criteria = self.apply_stop_rules(self.tokenizer, self.gen_config, stop_criteria=None)
        
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.model.config._name_or_path, trust_remote_code=True)
        self.base_dtype = AutoConfig.from_pretrained(self.model.config._name_or_path, trust_remote_code=True).torch_dtype
        if self.base_dtype not in [torch.float16, torch.bfloat16]: 
            self.base_dtype = torch.bfloat16 # Avoid ever running as float32
                
        
        self.base_has_system = tokenization.check_if_system(self.tokenizer) #self.cfg.base_tune_type=='chat' or (self.cfg.base_tune_type=='instruct' and 'system' in str(self.tokenizer.chat_template)) 
        
        print('Tk|Gc: (pad: {}|{}, eos: {}|{}), base_has_system: {}, has_prompt: {}, tag_placement: {}'.format(
            self.tokenizer.pad_token_id, self.gen_config.pad_token_id, 
            self.tokenizer.eos_token_id, self.gen_config.eos_token_id,
            self.base_has_system, (self.cfg.fprompt is not None), self.cfg.tag_placement,
        ))
        
        self.model.eval()

        return self
    
    # def to_conversation_format(self, author_messages: list[tuple[str,str]]) -> list[dict[str,str]]:        
    #     rolecycle = itertools.cycle(['user','assistant'])

    #     chat_content = []

    #     if self.base_has_system:
    #         chat_content.append({"role": "system", "content": self.cfg.fprompt})
    #     else:
    #         chat_content.append({"role": "user", "content": self.cfg.fprompt})
            
    #         if self.cfg.prompt.append_msg:
    #             chat_content.append({"role": "assistant", "content": "OK"})
        
    #     for auth,msg in author_messages:
    #         message = self.apply_content_prefix(auth, msg, tag_sep=self.cfg.tag_sep)
    #         chat_content.append({"role": next(rolecycle), "content": message})
        
    #     return chat_content
    

    def to_text_input(self, chat_history: list[tuple[str,str]], author_seedtext: tuple[str,str]=None):
        chat_history = chat_history[:]
        if author_seedtext is not None:
            text = self.to_seeded_text(chat_history, author_seedtext)
        else:
            text = self.tokenizer.apply_chat_template(self.to_conversation_format(chat_history), tokenize=False, add_generation_prompt=True)
        
        # Untuned models will not understand the encoding, so leave youtube as links
        #text = self.ytm.encode(text)
    
        return text