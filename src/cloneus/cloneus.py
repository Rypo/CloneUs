import gc
import time
import typing
import random
import itertools

from pathlib import Path
from threading import Thread

from dataclasses import dataclass

import numpy as np
from omegaconf import OmegaConf

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    GenerationConfig,
    TextIteratorStreamer
)
import transformers

from cloneus.data import roles, tokenization
from cloneus.plugins import youtube
from cloneus.inference import genconfig, load


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

    def load_genconfig(self, gen_config:str|Path|dict|GenerationConfig, path_data: ModelPathComponents):
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
    # def __init__(self, *args, **kwargs):
    #     # https://github.com/huggingface/transformers/blob/c15aad0939e691d2ffdbac7ae71921b51fe04e3f/src/transformers/models/auto/auto_factory.py#L406
    #     raise EnvironmentError(
    #         f"{self.__class__.__name__} is designed to be instantiated "
    #         f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or "
    #         f"`{self.__class__.__name__}.from_config(config)` methods."
    #     )
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
    def from_pretrained(checkpoint_path:str|Path, gen_config=None, ytm:youtube.YouTubeManager = None, **kwargs):
        # TODO: allow model_id for limited init with base_generate functionality
        # Maybe allow attempt to use without training, but need a LOT of assumptions to work without a cfg
        path_data = ModelPathComponents.from_checkpoint(Path(checkpoint_path))
        
        config = OmegaConf.load(path_data.config_path)
        
        if config.custom_chat_template: #config.base_tune_type == 'chat' or :
            ClonuesCls = CloneusRole
        elif config.base_tune_type in ['instruct','chat']:
            ClonuesCls = CloneusUA
        elif config.base_tune_type == 'foundation':
            ClonuesCls = CloneusTag
        else:
            raise NotImplementedError('Unable to determine Cloneus Format Class')
        
        if gen_config is None:
            gen_config = 'generation_config.json'
        
        if ytm is None:
            ytm = youtube.YouTubeManager(enabled=True)
        return ClonuesCls(path_data=path_data, cfg=config, gen_config=gen_config, ytm=ytm, **kwargs)
        
    @load.cleanup
    def load_model(self):
        self.model, self.tokenizer = load.load_any_inference(self.path_data.checkpoint_path, dtype=self.torch_dtype, attn_implementation=self.cfg.attn_implementation)
        self.tokenizer, self.gen_config, self.stop_criteria = self.apply_stop_rules(self.tokenizer, self.gen_config, stop_criteria=None)
        
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.model.config._name_or_path, trust_remote_code=True)
        self.base_dtype = AutoConfig.from_pretrained(self.model.config._name_or_path, trust_remote_code=True).torch_dtype
        if self.base_dtype not in [torch.float16, torch.bfloat16]: 
            self.base_dtype = torch.bfloat16 # Avoid ever running as float32
                
        # when adding custom tokens (e.g. <|im_end|>) use_default_system_prompt will be false, so check the tune_type
        self.base_has_system = self.cfg.base_tune_type=='chat' or (self.cfg.base_tune_type=='instruct' and 'system' in str(self.tokenizer.chat_template)) 
        
        print('Tk|Gc: (pad: {}|{}, eos: {}|{}) base_tune_type: {}, base_has_system: {} (use: {}), custom_roles: {}'.format(
            self.tokenizer.pad_token_id, self.gen_config.pad_token_id, 
            self.tokenizer.eos_token_id, self.gen_config.eos_token_id,
            self.cfg.base_tune_type, self.base_has_system,
            (self.cfg.fprompt and not self.cfg.prompt.append_msg), (self.cfg.custom_chat_template is not None)
        ))
        
        self.model.eval()

        return self
        
    @load.cleanup
    def unload_model(self):
        self.model = None
        self.tokenizer = None
        self.base_tokenizer = None
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
        # So, to be safe just reload the tokenizer if config changes
        if self.path_data.config_path != new_path_data.config_path:
            self.tokenizer, self.gen_config, self.stop_criteria = self.apply_stop_rules(self.tokenizer, self.gen_config, stop_criteria=self.stop_criteria)
            #self.tokenizer = self.setup_tokenizer(new_path_data.checkpoint_path)

        self.cfg, self.path_data = new_cfg, new_path_data
        
        return self


    @load.cleanup
    def swap_model(self, checkpoint_path:(str|Path), gen_config:GenerationConfig|Path=None, dtype=None, attn_implementation: typing.Literal["eager", "sdpa", "flash_attention_2"]=None) -> None:
        # If the path is the same, assume that either want a dtype change or attn_impl change
        if Path(checkpoint_path) == self.path_data.checkpoint_path:
            if dtype:
                self.cast_dtype(dtype)
            if attn_implementation:
                return self.from_pretrained(checkpoint_path, gen_config=gen_config, dtype=dtype, attn_implementation=attn_implementation, ytm=self.ytm).load_model()
                
        cfg, path_data = self.config_path_data(checkpoint_path, dtype=dtype, attn_implementation=attn_implementation)
                
        if self.can_hotswap(cfg, path_data):
            return self._swap_adapter(cfg, path_data, gen_config=gen_config, dtype=dtype)
        
        #if self.cfg.quant_method != cfg.quant_method:
        self.unload_model()
        
        return self.from_pretrained(checkpoint_path, gen_config=gen_config, dtype=dtype, attn_implementation=attn_implementation, ytm=self.ytm).load_model()
            
        

    def apply_template(self, author:str, text_content:str, tag_sep:str):
        atag=roles.format_author_tag(author, self.cfg.author_tag)
        return f'{atag}{tag_sep}{text_content}' # postfix was never used in this function call, always was set to '' 

    def to_conversation_format(self, author_messages: list[tuple[str,str]]) -> list[dict[str,str]]:
        raise NotImplementedError('Cloneus is designed to be instantiated via Cloneus.from_pretrained')

    def to_text_input(self, author_messages: list[tuple[str,str]], prompt_author_seedtext: tuple[str,str]=None):
        text = self.tokenizer.apply_chat_template(self.to_conversation_format(author_messages), tokenize=False, add_generation_prompt=True)
        text = self.ytm.encode(text)
        text = self.append_author_seedtext(text, prompt_author_seedtext, tag_sep=self.cfg.tag_sep)
    
        return text
    
    def append_author_seedtext(self, text: str, prompt_author_seedtext: tuple[str,str], tag_sep:str=None):
        if prompt_author_seedtext is not None:
            if tag_sep is None:
                tag_sep = self.cfg.tag_sep
            author, seedtext = prompt_author_seedtext
            if seedtext is None:
                seedtext = ''
            text += self.apply_template(author, seedtext, tag_sep).strip(' ') # IFF using ' ' as tag_sep, should NOT trail with it

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
        
        tokens = self.tokenizer(wordslist, add_special_tokens=False).input_ids
        if weights is not None:
            assert len(wordslist) == len(weights), 'all words need a weight'
            return {tuple(t):w for t,w in zip(tokens,weights)}
        
        return tokens


    @torch.inference_mode()
    def next_author_probs(self, author_messages: list[tuple[str,str]], top_k_next_tokens: int, author_list: list[str]=None):
        '''Returns a list of (authtok, proba) pairs. Note this is only the FIRST part of an author name unless author_list is provided.
        If author_list is given, try to map one of the author names in the list to the token segment
        '''
        # another possible method
        # - https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationMixin.compute_transition_scores
        # In theory, could allow seed text by comparing prob of each author saying seed_text given context. But that's a lot more work. 
        # Might be an easier way with BeamScorer
        # fill tag to split on a known fixed value. Allows author_tag to change without breaking.
        # needs to be lower case for compat with lauthor. TODO: fix
        FAKE_AUTH_MSG = ('###dummy_author_name', '###dummy_message_content')
        input_text = self.to_text_input(author_messages+[FAKE_AUTH_MSG], prompt_author_seedtext=None)
        
        # By splitting on dummy author, all prior formatting will be applied exactly, take this and discard the rest
        input_text = input_text.split(FAKE_AUTH_MSG[0], maxsplit=1)[0]
        #print(input_text)
        inputs = self.tokenizer(input_text, return_tensors="pt", add_special_tokens=False)

        author_gen_config = GenerationConfig(
            #force_words_ids=[],#[[wtok[:3]] for wtok in self.encode_wordslist(userutils.author_display_names)], 
            max_new_tokens=3, #min_new_tokens=1, 
            #do_sample=True, top_k=len(roles.author_display_names),temperature=1, #remove_invalid_values=True,
            renormalize_logits=True,
            num_beams=1,
            eos_token_id=self.gen_config.eos_token_id, 
            pad_token_id=self.gen_config.pad_token_id
        )
        
        output = self.model.generate(**inputs.to(0), generation_config=author_gen_config, stopping_criteria=self.stop_criteria, 
                                     output_scores=True, return_dict_in_generate=True)
        
        # Sorta redundant with top_k in generate, but handles sorting, masking, slicing without effort. 
        # using output.scores[0][0] avoid issues with num_beams > 1
        # when forcing words, the other beam dims are not useful anyway
        logits,token_ids = output.scores[0][0].topk(top_k_next_tokens, dim=0)
        authtok_prob_pairs = list(zip(
            self.tokenizer.batch_decode(token_ids),
            logits.softmax(0).tolist()
        ))
        
        if author_list is not None:
            if 'lauthor' in self.cfg.author_tag:
                author_list = [a.lower() for a in author_list] # make sure are lower cased if using lauthor
            for i, (aseg, prob) in enumerate(authtok_prob_pairs):
                try:
                    authtok_prob_pairs[i] = (next(filter(lambda a: a.startswith(aseg.strip()),author_list)), prob)
                except StopIteration:
                    print(f'WARNING: could not match "{aseg}" author in `author_list`, using ("{aseg}", {prob}) in return')
                    #authtok_prob_pairs.append((aseg, prob))

        return authtok_prob_pairs


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
    

    @torch.inference_mode()
    def batch_generate(self, author_messages: list[tuple[str,str]], prompt_authors: list[str], prompt_seedtext: str) -> tuple[str, list[str], list[str], int, list[int]]:
        input_context = self.to_text_input(author_messages, prompt_author_seedtext=None)
        if prompt_seedtext is None: prompt_seedtext = ''
        # IFF using ' ' as tag_sep, should NOT trail with it
        author_prompts = [self.apply_template(author, prompt_seedtext, self.cfg.tag_sep).strip(' ') for author in prompt_authors]

        true_batched = (self.gen_mode == 'contrastive_search')  # Cuts time almost in half for CS. Worth the quality degradation.
        out_texts,input_len = self._batched_helper([input_context+ap for ap in author_prompts], true_batch_generate=true_batched)
        
        out_texts = self.cleanup_out_texts(out_texts)
        
        output_lens = self.tokenizer(out_texts, return_length=True, add_special_tokens=False,)['length']

        return input_context, author_prompts, out_texts, input_len, output_lens
   
    @torch.inference_mode()
    def generate(self, author_messages: list[tuple[str,str]], prompt_author_seedtext: tuple[str,str]) -> tuple[str, str, int, int]:
        input_text = self.to_text_input(author_messages, prompt_author_seedtext)
        
        inputs = self.tokenizer(input_text, return_tensors="pt", return_length=True, add_special_tokens=False,)
        input_len = inputs.pop('length')[0].item()

        input_text = self.tokenizer.batch_decode(inputs.input_ids)[0]
        
        output = self.model.generate(**inputs.to(0), generation_config=self.gen_config, stopping_criteria=self.stop_criteria, negative_prompt_ids=None).detach_()
        out_tokens = output[0,input_len:]
        output_len = out_tokens.shape[0]
        # weird NOTE: if custom special tokens, decode skip_special_tokens **must**=FALSE. But encode add_special_tokens = (True | False), doesn't mater will be added regardless
        out_text = self.tokenizer.decode(out_tokens, skip_special_tokens=(not self.cfg.has_custom_tokens))
        out_text = self.cleanup_out_texts(out_text)

        return input_text, out_text, input_len, output_len
    
    
    @torch.inference_mode()
    def stream_generate(self, author_messages: list[tuple[str,str]], prompt_author_seedtext: tuple[str,str]):
        # https://huggingface.co/docs/transformers/internal/generation_utils#transformers.TextStreamer
        self._last_streamed_values = {'input_text':'', 'output_text':'', 'input_len': -1, 'output_len': -1}
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=120.0, skip_special_tokens=(not self.cfg.has_custom_tokens))

        input_text = self.to_text_input(author_messages, prompt_author_seedtext)
        inputs = self.tokenizer(input_text, return_tensors="pt", return_length=True)#, max_length=1024, truncation=True)
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
    def stream_batch_generate(self, author_messages: list[tuple[str,str]], prompt_authors: list[str], prompt_seedtext: str):
        # https://huggingface.co/docs/transformers/internal/generation_utils#transformers.TextStreamer
        self._last_streamed_batch_values = {'input_text':'','author_prompts':[], 'output_texts':[], 'input_len': -1, 'output_lens': []}
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=120.0, skip_special_tokens=True)

       
        input_context = self.to_text_input(author_messages, prompt_author_seedtext=None)
        if prompt_seedtext is None: prompt_seedtext = ''
        # IFF using ' ' as tag_sep, should NOT trail with it
        author_prompts = [self.apply_template(author, prompt_seedtext, self.cfg.tag_sep).strip(' ') for author in prompt_authors]
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
            
            #print(output_len)
        self._last_streamed_batch_values.update(input_text=input_context, author_prompts=author_prompts, input_len=input_len)
        #self._last_streamed_values.update({'input_text':trunc_context, 'output_texts': generated_text, 'input_len': input_len, 'output_len': output_len})

    def get_last_streamed(self, batch=False):
        if batch:
            base_input_text, author_prompts, model_outputs, input_length, output_lengths = [
                self._last_streamed_batch_values.get(v) for v in ['input_text', 'author_prompts', 'output_texts', 'input_len', 'output_lens']]
            return base_input_text, author_prompts, model_outputs, input_length, output_lengths
        
        input_text, model_output, input_length, output_length = [
            self._last_streamed_values.get(v) for v in ['input_text', 'output_text', 'input_len', 'output_len']]
        return input_text, model_output, input_length, output_length


    def base_instr_to_text(self, chat_history: list[str], sys_prompt:str=None):
        rolecycle = itertools.cycle(['user','assistant'])
        chat_content = []

        if sys_prompt is not None and self.base_has_system:
            chat_content.append({"role": "system", "content": sys_prompt})

        for message in chat_history:
            chat_content.append({"role": next(rolecycle), "content": message})

        input_text = self.base_tokenizer.apply_chat_template(chat_content, tokenize=False, add_generation_prompt=True)

        #print('chat_content:',chat_content)
        print(f'input_text: {input_text!r}',)
        
        return input_text

    @torch.inference_mode()
    def base_generate(self, chat_text:str|list[str], sys_prompt:str=None) -> tuple[str, str, int, int]:
        # adapters = self.model.active_adapters()#()
        # self.model.disable_adapters()
        # TODO: Function for base/cloneus hybrid
        # - just user/assistant tags on a trained model. Surprisingly, sort of works to have AI style responsiveness but with custom vernacular
        chat_history = [chat_text] if isinstance(chat_text, str) else chat_text
        input_text = self.base_instr_to_text(chat_history, sys_prompt=sys_prompt)
        
        inputs = self.base_tokenizer(input_text, return_tensors="pt", return_length=True, add_special_tokens=False)
        input_len = inputs.pop('length')[0].item()
        #self.model.to(dtype = self.base_dtype)
        with self.model.disable_adapter(), torch.cuda.amp.autocast(dtype=self.base_dtype):
            output = self.model.generate(**inputs.to(0), generation_config=self.base_gen_config, stopping_criteria=self.stop_criteria, negative_prompt_ids=None).detach_() # adapter_names=["__base__"]
        #self.model.to(dtype = self.torch_dtype)
        out_tokens = output[0,input_len:]
        output_len = out_tokens.shape[0]
        out_text = self.base_tokenizer.decode(out_tokens, skip_special_tokens=True)
        
        #self.model.set_adapter(adapters)
        
        return input_text, out_text, input_len, output_len
    

    @torch.inference_mode()
    def base_stream_generate(self, chat_text:str|list[str], sys_prompt:str=None):
        #adapters = self.model.active_adapters()
        #self.model.disable_adapters()
        #base_model: PeftModel = self.model.get_base_model()
        
        self._last_streamed_values = {'input_text':'', 'output_text':'', 'input_len': -1, 'output_len': -1}
        streamer = TextIteratorStreamer(self.base_tokenizer, skip_prompt=True, timeout=120.0, skip_special_tokens=True)

        chat_history = [chat_text] if isinstance(chat_text, str) else chat_text
        input_text = self.base_instr_to_text(chat_history, sys_prompt=sys_prompt)
        inputs = self.base_tokenizer(input_text, return_tensors="pt", return_length=True, add_special_tokens=False)#, max_length=1024, truncation=True)

        input_len = inputs.pop('length')[0].item()

        genkwargs = dict(inputs.to(0), generation_config=self.base_gen_config, streamer=streamer, stopping_criteria=self.stop_criteria, negative_prompt_ids=None)
        
        #self.model.to(dtype = self.base_dtype)
        
        with self.model.disable_adapter(), torch.cuda.amp.autocast(dtype=self.base_dtype):
            thread = Thread(target=self.model.generate, kwargs=genkwargs)
            thread.start()
            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                yield new_text
        
        #self.model.to(dtype = self.torch_dtype)
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
        tokenizer.chat_template = roles.to_jinja_template(self.cfg.tag_sep, self.cfg.postfix)

        tokenizer = tokenization.set_tokenizer_inference(tokenizer)
        
        #if gen_config.pad_token_id is None or gen_config.eos_token_id is None:
        gen_config.pad_token_id = tokenizer.pad_token_id
        gen_config.eos_token_id = tokenizer.eos_token_id
        
        
        if self.cfg.postfix != tokenizer.eos_token:
            postfix_stop = genconfig.WordListCriteria.from_words([self.cfg.postfix], self.tokenizer, device=0)
            # If the postfix is a single token, we can added it to the genconfig eos_token_ids for much more efficient processing

            if self.cfg.postfix == '\n':
                # use formatted author tags for early stop to prevent unterminated outputs 
                auth_tags = [roles.format_author_tag(u, self.cfg.author_tag) for u in roles.author_display_names]
                stop_criteria = [genconfig.WordListCriteria.from_words(auth_tags, self.tokenizer, device=0)]
            elif postfix_stop.stop_token_ids[0].shape[0] == 1:
                stop_criteria = None
                self.gen_config.eos_token_id = [tokenizer.eos_token_id, postfix_stop.stop_token_ids[0].item()]
                print(f'Using custom postfix eos_token {self.cfg.postfix!r} EOS: {self.gen_config.eos_token_id}')
            else:
                stop_criteria = [postfix_stop]
        
        # If model was trained with broken eos (i.e. no space between) it will fail to stop on eos
        # e.g. llama-2: "a </s>" -> input_ids=[263, 2]. But "a</s>" -> input_ids=[263, 829, 29879, 29958] (['</', 's', '>'])
        if tokenizer.eos_token in self.cfg.postfix:
            eos_tok_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False)['input_ids']
            ns_eos_tok_id = tokenizer(f'A{tokenizer.eos_token}', add_special_tokens=False)['input_ids'][1:]
            if eos_tok_id != ns_eos_tok_id:
                pre_tag = roles.format_author_tag(user_display_name='###DUMMY', author_tag=self.cfg.author_tag, insert_raw=True).split('###DUMMY')[0]
                
                eos_pre_tag = f'{tokenizer.eos_token}{pre_tag}'
                ns_eos_pre_tag_id = tokenizer(f'A{eos_pre_tag}', add_special_tokens=False)['input_ids'][1:]
                
                broken_eos_stop = genconfig.WordListCriteria(
                    [torch.tensor(ns_eos_tok_id).to(0), torch.tensor(ns_eos_pre_tag_id).to(0)], 
                    [tokenizer.eos_token, eos_pre_tag]
                )
                
                print(f'Broken eos detected. Adding stop crit for segmented eos {tokenizer.batch_decode(ns_eos_tok_id)!r} + eos_pre_tag {tokenizer.batch_decode(ns_eos_pre_tag_id)!r}')

                if stop_criteria is None:
                    stop_criteria = []
                stop_criteria += [broken_eos_stop]

        # print('GC EOS, TK EOS, POSTFIX:', gen_config.eos_token_id, tokenizer.eos_token, self.cfg.postfix)
        # print(['Words: {}, ids: {}'.format(s.words, s.stop_token_ids) for s in (stop_criteria if stop_criteria is not None else [])])
        if stop_criteria == []:
            stop_criteria = None
            
        return tokenizer, gen_config, stop_criteria
    

    def to_conversation_format(self, author_messages: list[tuple[str,str]]) -> list[dict[str,str]]:
        '''For tag only, markup free, format'''
        # why Foundation? : https://crfm.stanford.edu/2021/10/18/reflections.html#:~:text=are%20situated%20in.-,Naming,-The%20name%20%E2%80%9Cfoundation        
        # input_text = ''.join([self.apply_template(a,t, self.cfg.tag_sep, postfix=self.cfg.postfix) for a,t in author_messages])
        chat_content = []
        
        for author,message in author_messages:
            role_tag = roles.format_author_tag(author, self.cfg.author_tag) # for TagFormat, tag_sep is baked in to chat_template via to_jinja_template
            chat_content.append({"role": role_tag, "content": message})
        
        return chat_content
    


    
class CloneusRole(Cloneus):
    r'''For ChatML models (and possibly other chat formats) trained with a custom chat template. i.e. does not alternate
    user/assistant, instead uses users,names in place of user/assistant. Requires a system message.
    
    e.g.
        <|im_start|>system
        A conversation between Bob and a well meaning tree<|im_end|>
        <|im_start|>Beta (Bob)
        How's everyone<|im_end|>
        <|im_start|>Groot (Groot)
        I am Groot<|im_end|>
    '''
    def to_conversation_format(self, author_messages: list[tuple[str,str]]) -> list[dict[str,str]]:
        '''For chatml with custom roles as usernames'''
        chat_content = []

        #if self.use_sysprompt:
        chat_content.append({"role": "system", "content": self.cfg.fprompt})
        
        for author,message in author_messages:
            role_tag = roles.format_author_tag(author, self.cfg.author_tag)
            chat_content.append({"role": role_tag, "content": message})
        
        return chat_content
    

class CloneusUA(Cloneus):
    r'''For QA/Chat models trained without a custom chat template, i.e. template must alternate
    user/assistant, possibly starting with system. This includes mistral where no explicit system/user/assistant
    is in the prompt, but apply_chat_template format requires u/a. Typically, this means author_tag will have formatting
    with an explicit USER:user, NAME:name type structure.
    
    e.g.
        [INST] <USER:Alpha, NAME:Alice> hello [/INST]<USER:Beta, NAME:Bob> hi</s>
    
    e.g.
        <|im_start|>user
        [USER:Beta, NAME:Bob] How's everyone<|im_end|>
        <|im_start|>assistant
        [USER:Gamma, NAME:Greg] I am Greg, thanks<|im_end|>
    '''
    
    def to_conversation_format(self, author_messages: list[tuple[str,str]]) -> list[dict[str,str]]:        
        rolecycle = itertools.cycle(['user','assistant'])

        chat_content = []

        if self.base_has_system:
            chat_content.append({"role": "system", "content": self.cfg.fprompt})
        else:
            pcontent = self.cfg.fprompt
            if self.cfg.prompt.append_msg:
                auth, msg = author_messages[0]
                pcontent += self.apply_template(auth, msg, tag_sep=self.cfg.tag_sep)
                author_messages = author_messages[1:]
            
            chat_content.append({"role": next(rolecycle), "content": pcontent})
        
        for auth,msg in author_messages:
            message = self.apply_template(auth, msg, tag_sep=self.cfg.tag_sep)
            chat_content.append({"role": next(rolecycle), "content": message})
        
        return chat_content
    
# TODO: 
# fix configs

# base_generation config

# cloneus from_base_model
# cloneus from_checkpoint
# from_pretrained = from_checkpoint | from_base_model

class CloneusBasePrediction(Cloneus):
    r'''For base model prediction. 
    
    Attempt use a untuned, base model for user chat emulation. 

    Note: To have any success with this approach, a very detailed system prompt is required. 
    Regardless of prompt used, the quality cannot match that of a finetuned model.
    This is therefore not recommended for standalone use in most instances.
    '''
    def to_conversation_format(self, author_messages: list[tuple[str,str]]) -> list[dict[str,str]]:
        '''For base model prediction'''
        rolecycle = itertools.cycle(['user','assistant'])
        chat_content = []

        #if self.use_sysprompt:
        chat_content.append({"role": "system", "content": self.cfg.fprompt})
        
        for auth,msg in author_messages:
            message = self.apply_template(auth, msg, tag_sep=self.cfg.tag_sep)
            chat_content.append({"role": next(rolecycle), "content": message})
        
        return chat_content
    

    def to_text_input(self, author_messages: list[tuple[str,str]], prompt_author_seedtext: tuple[str,str]=None):
        text = self.tokenizer.apply_chat_template(self.to_conversation_format(author_messages), tokenize=False, add_generation_prompt=True)
        # Untuned models will not understand the encoding, so leave youtube as links
        #text = self.ytm.encode(text)
        text = self.append_author_seedtext(text, prompt_author_seedtext, tag_sep=self.cfg.tag_sep)
    
        return text