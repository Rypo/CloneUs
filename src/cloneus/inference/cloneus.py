import time
import typing
import random
import itertools

from pathlib import Path
from threading import Thread
from collections import namedtuple
from contextlib import contextmanager
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import datasets
from omegaconf import OmegaConf, DictConfig

import torch
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    TextIteratorStreamer
)
import transformers
from transformers.generation.utils import GenerationMode

from unsloth import FastLanguageModel

from peft import PeftModel, LoraConfig, get_peft_model, AutoPeftModelForCausalLM, PeftConfig

from safetensors.torch import load_model as load_model_safetensors, save_model as save_model_safetensors
from cloneus.data import roles
from cloneus.plugins import youtube

from cloneus.core import paths as cpaths, types

from cloneus.inference import genconfig
from cloneus.inference import load


ModelFiles = namedtuple('ModelFiles',['basedir_path','ckpt_subdir','config_path', 'modeldir_path'])

def modeldir_components(model_dir, checkpoint_dir=None):
    model_path = Path(model_dir)
    base_path = model_path
    while not (base_path/'config.yaml').exists():
        base_path = base_path.parent
        if base_path.parent==base_path:
            raise FileNotFoundError(f'config.yaml not found along parent paths. model_path: {model_path}')


    if checkpoint_dir is None:
        checkpoint_dir = str(model_path.relative_to(base_path))
        if checkpoint_dir in ['config.yaml', '.']:
            checkpoint_dir=None
    
    if checkpoint_dir is None:
        raise FileNotFoundError('No checkpoint dir found')
    
    return ModelFiles(basedir_path=base_path, ckpt_subdir=checkpoint_dir, config_path=base_path/'config.yaml', modeldir_path=base_path/checkpoint_dir)
    
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


@contextmanager
def batchsafe_tokenizer(tokenizer):
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


class Cloneus:
    def __init__(self, model_dir: str|Path = None, ckpt_subdir: str=None, gconfig_fname=None, **kwargs) -> None:
        self.config = self._config_init(model_dir, ckpt_subdir, gconfig_fname=gconfig_fname, **kwargs)
        self.ckpt_subdir = ckpt_subdir
        self.model = None
        self.tokenizer = None
        self.ytm = youtube.YouTubeManager()
        #self.gen_config, self._author_gen_config = self.load_genconf(gconfig_fname)
                
        self._last_streamed_values = {'input_text':'', 'output_text':'', 'input_len': -1, 'output_len': -1}
        self._last_streamed_batch_values = {'input_text':'','author_prompts':[], 'output_texts':[], 'input_len': -1, 'output_lens': []}

        self.filler_message = (random.choice(roles.author_display_names), 'ye')
        #self.gen_config=None
        

    def _config_init(self, model_dir, ckpt_subdir, gconfig_fname=None, **kwargs):

        # if we don't filter out None from kwargs, config is over written with None
        kwargs = {k:v for k,v in kwargs.items() if v is not None}
        mdir_comps = modeldir_components(model_dir, ckpt_subdir)
        self.mdir_comps = mdir_comps
        self.basemodelnick=self.mdir_comps.basedir_path.resolve().relative_to(cpaths.RUNS_DIR).parts[0]
        #config = self.load_config(config_file, ckpt_subdir, **kwargs)
        config = OmegaConf.load(mdir_comps.config_path)
        config.update(model_dir = mdir_comps.modeldir_path, **kwargs)
        
        self.model_dir = config.model_dir
        self.ctx_len = config.ctx_len
        
        self.tag_sep = config.tag_sep
        self.postfix = config.postfix
        self.author_tag = config.author_tag
        # weird NOTE: if custom special tokens, decode skip_special_tokens **must**=FALSE. But encode add_special_tokens = (True | False), doesn't mater will be added regardless
        self.has_custom_tokens = config.has_custom_tokens 
        
        self.dtype = dtype_to(config.get('dtype'), 'torch', default=None)
        self.attn_implementation = config.get('attn_implementation', None)
        
        self.prompt = config.get('prompt') if config.get('prompt') is not None else {}
        self.fprompt = config.get('fprompt')
        self._prompt_append_msg = self.prompt.get('append_msg')

        self._is_instruct_model = config.get('instruct_model')
        self._is_chat_model = config.get('custom_chat_template') is not None

        #self.guidance_phrases=dict.fromkeys(userutils.author_display_names)
        print(self.dtype, self.attn_implementation)
        if gconfig_fname is not None or not hasattr(self,'gen_config'):
            self.gen_config, self._author_gen_config = self.load_genconf(gconfig_fname)

        return config


    @load.cleanup
    def switch_model(self, model_dir:(str|Path), ckpt_subdir:str=None, dtype=None, attn_implementation: typing.Literal["eager", "sdpa", "flash_attention_2"]=None, gconfig_fname:str=None) -> None:
        cur_basemodelnick=self.basemodelnick
        self.config = self._config_init(model_dir, ckpt_subdir, dtype=dtype, attn_implementation=attn_implementation, gconfig_fname=gconfig_fname)
        new_basemodelnick=self.basemodelnick
        
        if cur_basemodelnick==new_basemodelnick and self.model is not None:
            #print('Swap Adapter. dtype, attn_implementation will be ignored')
            # NOTE: This makes dtype and attn_implementation be completely ignored
            checkpoint = self.mdir_comps.ckpt_subdir
            adapter_name = (self.mdir_comps.basedir_path.name+'-'+checkpoint).replace('.','')
            if not adapter_name in self.model.peft_config:
                self.model.load_adapter(self.mdir_comps.basedir_path/checkpoint, adapter_name=adapter_name)
            self.model.set_adapter(adapter_name)
        else:
            self.load_model()

    @load.cleanup
    def load_model(self):
        
        #if self.model is None or dtype != self.dtype or attn_implementation != self.attn_implementation:
        self.model, self.tokenizer = load.load_any_inference(self.model_dir, dtype=self.dtype, attn_implementation=self.attn_implementation)
        #self.model, self.tokenizer = minfer.load_unmerged_lowrsc(self.model_dir, dtype=None, attn_implementation='sdpa')
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.model.config._name_or_path)
        self.base_dtype = transformers.AutoConfig.from_pretrained(self.model.config._name_or_path).torch_dtype
        if self.base_dtype not in [torch.float16, torch.bfloat16]: 
            self.base_dtype = torch.bfloat16 # Avoid ever running as float32

        self._is_instruct_model = self.tokenizer.chat_template is not None
        # when adding custom tokens (e.g. <|im_end|>) use_default_system_prompt will be false, so check the tune_type
        self.has_sysprompt = self.tokenizer.use_default_system_prompt or self.config.get('tune_type') == 'chatml' 
        self._use_sysprompt = self.has_sysprompt and not self.config.prompt.append_msg
        
        print(self.tokenizer.pad_token_id, self.tokenizer.eos_token_id, 
              f'Instruct: {self._is_instruct_model}, has_system: {self.has_sysprompt} (use: {self._use_sysprompt}), chat: {self._is_chat_model}')

        self.stop_criteria = None if (self._is_instruct_model or '</s>' in self.postfix) else [
            genconfig.NewLineTokensCriteria(self.tokenizer('\n\n', add_special_tokens=False, return_tensors='pt')['input_ids'][0,1:].to(0))]
        
        self.model.eval()

        return self
        
    @load.cleanup
    def unload_model(self):
        self.model = None
        self.tokenizer = None


    @property
    def gen_alias(self):
        """Returns the generation mode triggered by a [`GenerationConfig`] instance."""
        return genconfig.get_generation_mode(self.gen_config, None)
        


    def load_genconf(self, gconfig_fname=None):
        if gconfig_fname is None:
            gconfig_fname = 'generation_config.json'

        gconf_path = Path(self.model_dir)/gconfig_fname
        if not gconf_path.exists():
            print(f'No GenConfig found at: {gconf_path}')
        
        try:
            self.gen_config = GenerationConfig.from_pretrained(self.model_dir, gconfig_fname, local_files_only=True)
            print(f'Found GenerationConfig: {Path(self.model_dir)/gconfig_fname}')
        except OSError as e:
            print('No existing GenerationConfig found, defaulting to GENOPTS (multinomial_sampling)')
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            gcdefault = genconfig.GENOPT_DEFAULTS.copy()
            gcdefault.update(pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            self.gen_config = GenerationConfig.from_dict(gcdefault) 
            
        self._author_gen_config = GenerationConfig(
            #force_words_ids=[],#[[wtok[:3]] for wtok in self.encode_wordslist(userutils.author_display_names)], 
            max_new_tokens=3, #min_new_tokens=1, 
            #do_sample=True, #remove_invalid_values=True,
            #top_k=len(userutils.author_display_names),
            #temperature=1,
            renormalize_logits=True,
            num_beams=1,
            eos_token_id=self.gen_config.eos_token_id, 
            pad_token_id=self.gen_config.pad_token_id
        )
        
        return self.gen_config, self._author_gen_config
            
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
    
    
    def neg_inpids(self, author: str):
        authvibe = self.guidance_phrases.get(author)
        if authvibe is None or self.tokenizer is None:
            return None
        
        return self.tokenizer(authvibe, return_tensors="pt")["input_ids"].to(0)    
    
    def get_genconf(self, verbose=False) -> dict:
        gconf_settings = self.gen_config.to_diff_dict().copy()# if self.model else {}
        if not verbose:
            [gconf_settings.pop(k, None) for k in ['transformers_version', "eos_token_id",  "pad_token_id"]]
        
        return gconf_settings

    def set_genconf(self, **kwargs) -> dict[str,dict]:
        '''Set generation config arguments. Special arg "alias" to use presets.
        Args:
            alias: str
            'cs' (contrastive search) - penalty_alpha>0 and top_k>1
            'ms' (multinomial sampling) - num_beams=1 and do_sample=True
            'gd' (greedy decoding) - num_beams=1 and do_sample=False
            'bsd' (beam search decoding) - do_sample=False and num_beams>1
            'bsms' (beam search multinomial sampling) - num_beams>1 and do_sample=True
            'dbsd' (diverse beam search decoding) - num_beams>1 and num_beam_groups>1
            
        https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig'''
        
        alias = kwargs.pop('alias',None)
        print(kwargs)
        prev_conf = self.gen_config.to_dict()
        #prev_vals = {k:curconf.get(k) for k in kwargs}
        # NOTE: this prevents a value from being set to the default value, which isn't great, but can't think of a work around.
        #filt_kwargs = {k:v for k,v in kwargs.items() if v != GENOPT_DEFAULTS[k] and v != prev_conf[k]}

        if alias is not None:
            abrv_gmodes = dict(zip(['ms','cs','gd','bsd','bsms','dbsd'],
                ['multinomial_sampling', 'contrastive_search', 'greedy_decoding', 'beam_search_decoding','beam_search_multinomial_sampling', 'diverse_beam_search_decoding']))
            self.gen_config = genconfig.create_genconf(abrv_gmodes.get(alias, alias), pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id, **kwargs) 
        else:
            # if penalty_alpha is passed, make sure top_k isn't too high
            if kwargs.get('penalty_alpha') and kwargs.get('top_k', self.gen_config.top_k) > 9:
                kwargs['top_k'] = 4
            elif any(kwargs.get(a) for a in ['temperature','top_p','do_sample','typical_p','num_beams']):
                # assume disable contrastive_search
                kwargs.setdefault('penalty_alpha', None) # if it is explicitly passed alongside one of those, then let it be
                kwargs.setdefault('do_sample', True) # if it is explicitly set, let it ride. Important if need to do beam decoding for some reason
                if kwargs.get('top_k') is None and self.gen_config.top_k <= 9: # if it is explicitly passed a low top_k, let it be
                    kwargs['top_k'] = genconfig.GENOPT_DEFAULTS['top_k']
            #self.gen_config.update(**kwargs)
            self.gen_config.update(**kwargs)
        
        new_conf = self.gen_config.to_dict()

        #new_vals = {k:newconf.get(k) for k in kwargs}
        
        changes = {}
        if prev_conf != new_conf:
            changes = {k: {'prev':prev_conf[k], 'new':new_conf[k]} for k in new_conf if prev_conf[k]!=new_conf[k]}
            sequence_bias = self.gen_config.sequence_bias # cannot serialize tuples
            self.gen_config.sequence_bias=None
            self.gen_config.save_pretrained(self.model_dir, 'generation_config.json')
            self.gen_config.sequence_bias=sequence_bias
        
        return changes



    def discrete_front_truncate(self, input_text: str, new_tokbuf: int = 256):
        '''Truncate full samples split on postfix from left side of text to max_len'''
        # split keep ends so they are included in token lengths. NOTE: without the "if t", will have a double postfix. No clue how that bug slipped by.
        split_text = [t+self.postfix for t in input_text.split(self.postfix) if t] #
        #split_text = input_text.split(self.postfix)
        lengths = self.tokenizer(split_text, return_length=True).length
        # argmax returns first index of True, so we need to reverse the cumsum and then reverse the argmax
        first_idx = (np.cumsum(lengths[::-1]) <= self.ctx_len-new_tokbuf)[::-1].argmax()
        
        trunc_text = ''.join(split_text[first_idx:])
        #trunc_text = self.postfix.join(split_text[first_idx:])

        return trunc_text


    def to_text_input(self, author_messages: list[tuple[str,str]], prompt_author_seedtext: tuple[str,str]=None):
        if self._is_instruct_model:
            if self._is_chat_model:
                return self.chat_to_text(author_messages, prompt_author_seedtext)
            
            return self.instruct_to_text(author_messages, prompt_author_seedtext)
        return self.foundation_to_text(author_messages, prompt_author_seedtext)


    def apply_template(self, author, text_content, tag_sep, postfix):
        atag=roles.format_author_tag(author, self.author_tag)
        #atag=self.author_tag.format(author=author, lauthor=author.lower(), fname=crew.author_to_fname[author])
        return f'{atag}{tag_sep}{text_content}{postfix}'
    

    def to_foundation_format(self, author_messages: list[tuple[str,str]]):
        # why Foundation?
        # https://crfm.stanford.edu/2021/10/18/reflections.html#:~:text=are%20situated%20in.-,Naming,-The%20name%20%E2%80%9Cfoundation
        
        input_text = ''.join([self.apply_template(a,t, self.tag_sep, postfix=self.postfix) for a,t in author_messages])
        return input_text


    def to_instruct_format(self, author_messages: list[tuple[str,str]]):
        if not author_messages:
            author_messages = [self.filler_message]
        
        elif len(author_messages) % 2 == 0:
            author_messages=[self.filler_message,*author_messages]
            #author_messages = author_messages[1:]
        
        rolecycle = itertools.cycle(['user','assistant'])

        if self.has_sysprompt:
            chat_content = [{"role": "system", "content": self.fprompt}]
        else:
            pcontent = self.fprompt
            if self._prompt_append_msg:
                # TODO: Assess if it's okay to pop(0) even though it will always be the filler message if even
                pcontent += self.apply_template(*author_messages.pop(0), tag_sep=self.tag_sep, postfix='')
            
            chat_content = [{"role": next(rolecycle), "content": pcontent}]
        
        for auth,msg in author_messages:
            message = self.apply_template(auth, msg, tag_sep=self.tag_sep, postfix='')
            chat_content.append({"role": next(rolecycle), "content": message})
        
        return chat_content
       
    def to_chat_format(self, author_messages: list[tuple[str,str]]):
        '''For chatml with custom roles as usernames'''
        # if not author_messages:
        #     author_messages = [self.filler_message]
        
        chat_content = []

        if self.has_sysprompt:
            chat_content.append({"role": "system", "content": self.fprompt})
        
        for author,message in author_messages:
            role_tag = roles.format_author_tag(author, self.author_tag)
            chat_content.append({"role": role_tag, "content": message})
        
        return chat_content

    def append_author_seedtext(self, text: str, prompt_author_seedtext: tuple[str,str], tag_sep:str=None):
        if tag_sep is None:
            tag_sep = self.tag_sep

        if prompt_author_seedtext is not None:
            author, seedtext = prompt_author_seedtext
            if seedtext is None:
                seedtext = ''
            text += self.apply_template(author, seedtext, tag_sep, postfix='').strip(' ') # IFF using ' ' as tag_sep, should NOT trail with it

        return text

    def foundation_to_text(self, author_messages: list[tuple[str,str]], prompt_author_seedtext: tuple[str,str]=None):
        text = self.to_foundation_format(author_messages)
        text = self.ytm.encode(text)

        if (text_len:=self.tokenizer(text, return_length=True).length[0]) >= self.ctx_len:
            # technically this can still go over the limit undetected after adding author tag/ seed text, but going to let it slide for now unless it becomes an issue.
            print(f'MAX CONTENT LENGTH EXCEEDED. Front Truncating. {text_len} ≥ {self.ctx_len}')
            text = self.discrete_front_truncate(text, self.gen_config.max_new_tokens)
        
        text = self.append_author_seedtext(text, prompt_author_seedtext)
        
        if f'{self.postfix}{self.postfix}' in text:
            print('ERROR: Double postfix found in input text')
            print(repr(text))

        
        return text

    ################################################################
    #
    # TODO: THIS DOES NOT TRIM IF CONTEXT GROWS TOO LARGE. WATCH OUT
    #
    ################################################################
    def instruct_to_text(self, author_messages: list[tuple[str,str]], prompt_author_seedtext: tuple[str,str]=None):
        # add_generation_prompt will = '' if doesnt have, like mistral. So, always okay to add?
        text = self.tokenizer.apply_chat_template(self.to_instruct_format(author_messages), tokenize=False, add_generation_prompt=True)
        text = self.ytm.encode(text)
        text = self.append_author_seedtext(text, prompt_author_seedtext)
    
        return text
    
    def chat_to_text(self, author_messages: list[tuple[str,str]], prompt_author_seedtext: tuple[str,str]=None):
        # add_generation_prompt will = '' if doesnt have, like mistral. So, always okay to add?
        text = self.tokenizer.apply_chat_template(self.to_chat_format(author_messages), tokenize=False, add_generation_prompt=True)
        text = self.ytm.encode(text)
        text = self.append_author_seedtext(text, prompt_author_seedtext, tag_sep='\n') # ChatML always has \n after role
    
        return text
    
    def get_top_tokprobs(self, out_score, k=5) -> list[tuple[str, float]]:
        probs = torch.nn.functional.softmax(out_score.squeeze(),dim=0)
        top_probs,top_toks = torch.topk(probs, k)
        tokstrs = self.tokenizer.batch_decode(top_toks)
        
        return list(zip(tokstrs,top_probs.tolist()))
    
    @torch.inference_mode()
    def next_author_probs(self, author_messages: list[tuple[str,str]], top_k_next_tokens: int = 5, author_list: list[str]=None):
        '''Returns a list of (authtok, proba) pairs. Note this is only the FIRST part of an author name unless author_list is provided.
        If author_list is given, try to map one of the author names in the list to the token segment
        '''
        # another possible method
        # - https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationMixin.compute_transition_scores
        # In theory, could allow seed text by comparing prob of each author saying seed_text given context. But that's a lot more work. 
        # Might be an easier way with BeamScorer
        trunc_input_text = self.to_text_input(author_messages, prompt_author_seedtext=None)

        # leave this as max_new_tokens because assume setting to 1 could include context that changes the author
        #trunc_input_text = self.discrete_front_truncate(input_text, self.gen_config.max_new_tokens) 
        # fill tag to split on a known fixed value. Allows author_tag to change without breaking.
        trunc_input_text += self.author_tag.format(author='#DUMMY', lauthor='#DUMMY', fname='#NAME').split('#DUMMY',1)[0]
        inputs = self.tokenizer(trunc_input_text, return_tensors="pt")

        
        output = self.model.generate(**inputs.to(0), generation_config=self._author_gen_config, stopping_criteria=self.stop_criteria, 
                                     output_scores=True, return_dict_in_generate=True)
        
        # Sorta redundant with top_k in generate, but handles sorting, masking, slicing without effort. 
        # using output.scores[0][0] avoid issues with num_beams > 1
        # when forcing words, the other beam dims are not useful anyway
        logits,token_ids = output.scores[0][0].topk(top_k_next_tokens, dim=0)
        authtok_prob_pairs = list(zip(
            self.tokenizer.batch_decode(token_ids),
            logits.softmax(0).tolist()
        ))
        #authtok_prob_pairs = self.get_top_tokprobs(output.scores[0], k=top_k_next_tokens)
        if author_list is not None:
            for i, (aseg, prob) in enumerate(authtok_prob_pairs):
                try:
                    authtok_prob_pairs[i] = (next(filter(lambda a: a.startswith(aseg),author_list)), prob)
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
            with batchsafe_tokenizer(self.tokenizer) as tokenizer:
                inputs = tokenizer(msg_batch, return_length=True, return_tensors='pt', padding=True)
            
            input_len = inputs.pop('length')[0].item()
            outputs = self.model.generate(**inputs.to(0), generation_config=self.gen_config, stopping_criteria=self.stop_criteria).detach()
            
            output_texts = self.tokenizer.batch_decode(outputs[:,input_len:], skip_special_tokens=True)
        else:
            # if not doing the whole batch, no point in incuring the 'penalty' for adding pad tokens to begining
            # better to just do tokenization on the fly
            output_texts  = []

            for inptext in msg_batch:
                inputs = self.tokenizer(inptext, return_tensors="pt", return_length=True)
                input_len = inputs.pop('length')[0].item()
                output = self.model.generate(**inputs.to(0), generation_config=self.gen_config, stopping_criteria=self.stop_criteria, negative_prompt_ids=None).detach()
                output_texts.append(self.tokenizer.decode(output[0,input_len:], skip_special_tokens=True))
            
        print(f'TOTAL BATCHED RUN TIME: {time.perf_counter()-t0:0.2f}s')
        #print('Raw Batch Outputs:\n', self.tokenizer.batch_decode(outputs, skip_special_tokens=False))
        
        # use last input_len for all. Shouldn't differ by more than a few tokens as long as author_tags are reasonable
        return output_texts,input_len
    

    @torch.inference_mode()
    def batch_generate(self, author_messages: list[tuple[str,str]], prompt_authors: list[str], prompt_seedtext: str) -> tuple[str, list[str], list[str], int, list[int]]:
        trunc_context = self.to_text_input(author_messages, prompt_author_seedtext=None)
        if prompt_seedtext is None: prompt_seedtext = ''
        # IFF using ' ' as tag_sep, should NOT trail with it
        author_prompts = [self.apply_template(author, prompt_seedtext, self.tag_sep, postfix='').strip(' ') for author in prompt_authors]

        true_batched = (self.gen_alias == 'contrastive_search')  # Cuts time almost in half for CS. Worth the quality degradation.
        output_texts,input_len = self._batched_helper([trunc_context+ap for ap in author_prompts], true_batch_generate=true_batched)
        
        out_texts = [ot.split(self.postfix)[0] for ot in output_texts]
        output_lens = self.tokenizer(out_texts, add_special_tokens=False, return_length=True)['length']

        return trunc_context, author_prompts, out_texts, input_len, output_lens
   
    @torch.inference_mode()
    def generate(self, author_messages: list[tuple[str,str]], prompt_author_seedtext: tuple[str,str]) -> tuple[str, str, int, int]:
        trunc_input_text = self.to_text_input(author_messages, prompt_author_seedtext)
        
        inputs = self.tokenizer(trunc_input_text, return_tensors="pt", return_length=True)
        input_len = inputs.pop('length')[0].item()

        trunc_input_text = self.tokenizer.batch_decode(inputs.input_ids)[0]

        # neg_inp = self.neg_inpids(prompt_author_seedtext[0])
        
        output = self.model.generate(**inputs.to(0), generation_config=self.gen_config, stopping_criteria=self.stop_criteria, negative_prompt_ids=None).detach()
        out_tokens = output[0,input_len:]
        output_len = out_tokens.shape[0]
        out_text = self.tokenizer.decode(out_tokens, skip_special_tokens=(not self.has_custom_tokens))

        return trunc_input_text, out_text, input_len, output_len
    
    
    
    @torch.inference_mode()
    def stream_generate(self, author_messages: list[tuple[str,str]], prompt_author_seedtext: tuple[str,str]):
        # https://huggingface.co/docs/transformers/internal/generation_utils#transformers.TextStreamer
        self._last_streamed_values = {'input_text':'', 'output_text':'', 'input_len': -1, 'output_len': -1}
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=120.0, skip_special_tokens=(not self.has_custom_tokens))

        trunc_input_text = self.to_text_input(author_messages, prompt_author_seedtext)
        inputs = self.tokenizer(trunc_input_text, return_tensors="pt", return_length=True)#, max_length=1024, truncation=True)
        input_len = inputs.pop('length')[0].item()

        neg_inp=None#self.neg_inpids(prompt_author_seedtext[0])
        genkwargs = dict(inputs.to(0), generation_config=self.gen_config, streamer=streamer, stopping_criteria=self.stop_criteria, negative_prompt_ids=neg_inp)
        thread = Thread(target=self.model.generate, kwargs=genkwargs)
        thread.start()
        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            yield new_text
        
        output_len =  self.tokenizer(generated_text, return_length=True).length

        self._last_streamed_values.update({'input_text':trunc_input_text, 'output_text': generated_text, 'input_len': input_len, 'output_len': output_len})

    

    @torch.inference_mode()
    def stream_batch_generate(self, author_messages, prompt_authors, prompt_seedtext):
        # https://huggingface.co/docs/transformers/internal/generation_utils#transformers.TextStreamer
        self._last_streamed_batch_values = {'input_text':'','author_prompts':[], 'output_texts':[], 'input_len': -1, 'output_lens': []}
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=120.0, skip_special_tokens=True)

       
        trunc_context = self.to_text_input(author_messages, prompt_author_seedtext=None)
        if prompt_seedtext is None: prompt_seedtext = ''
        # IFF using ' ' as tag_sep, should NOT trail with it
        author_prompts = [self.apply_template(author, prompt_seedtext, self.tag_sep, postfix='').strip(' ') for author in prompt_authors]
        #inputs, author_prompts = self._batch_process(trunc_context, prompt_authors, prompt_seedtext)
        #inputs, trunc_context, author_prompts = self._batch_process(author_messages, prompt_authors, prompt_seedtext)
        msg_batch = [trunc_context+ap for ap in author_prompts]
        #input_len = inputs.pop('length')[0].item()
        

        #inps=inputs.to(0)

        for i, inptext in enumerate(msg_batch):
            inputs = self.tokenizer(inptext, return_tensors="pt", return_length=True)
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
            
            output_len = self.tokenizer(generated_text, return_length=True).length
            
            self._last_streamed_batch_values['output_texts'].append(generated_text)
            self._last_streamed_batch_values['output_lens'] += output_len
            
            #print(output_len)
        self._last_streamed_batch_values.update(input_text=trunc_context, author_prompts=author_prompts, input_len=input_len)
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

        if sys_prompt is not None and self.has_sysprompt:
            chat_content.append({"role": "system", "content": sys_prompt})

        for message in chat_history:
            chat_content.append({"role": next(rolecycle), "content": message})

        trunc_input_text = self.base_tokenizer.apply_chat_template(chat_content, tokenize=False, add_generation_prompt=True)

        print('chat_content:',chat_content)
        print(f'trunc_input_text: {trunc_input_text!r}',)
        
        return trunc_input_text

    @torch.inference_mode()
    def base_generate(self, input_text:str|list[str], sys_prompt:str=None) -> tuple[str, str, int, int]:
        # adapters = self.model.active_adapters()#()
        # self.model.disable_adapters()
        # TODO: Function for base/cloneus hybrid
        # - just user/assistant tags on a trained model. Surprisingly, sort of works to have AI style responsiveness but with custom vernacular
        chat_history = [input_text] if isinstance(input_text, str) else input_text
        trunc_input_text = self.base_instr_to_text(chat_history, sys_prompt=sys_prompt)
        
        inputs = self.base_tokenizer(trunc_input_text, return_tensors="pt", return_length=True)
        input_len = inputs.pop('length')[0].item()
        self.model.to(dtype = self.base_dtype)
        with self.model.disable_adapter():
            output = self.model.generate(**inputs.to(0), generation_config=self.gen_config, stopping_criteria=self.stop_criteria, negative_prompt_ids=None).detach_() # adapter_names=["__base__"]
        self.model.to(dtype = self.dtype)
        #output = base_model.generate(**inputs.to(0), generation_config=self.gen_config, stopping_criteria=self.stop_criteria, negative_prompt_ids=None).detach_() # adapter_names=["__base__"]
        out_tokens = output[0,input_len:]
        output_len = out_tokens.shape[0]
        out_text = self.base_tokenizer.decode(out_tokens, skip_special_tokens=True)
        
        #self.model.set_adapter(adapters)
        
        return trunc_input_text, out_text, input_len, output_len
    


    @torch.inference_mode()
    def base_stream_generate(self, input_text:str|list[str], sys_prompt:str=None):
        #adapters = self.model.active_adapters()
        #self.model.disable_adapters()
        #base_model: PeftModel = self.model.get_base_model()
        
        self._last_streamed_values = {'input_text':'', 'output_text':'', 'input_len': -1, 'output_len': -1}
        streamer = TextIteratorStreamer(self.base_tokenizer, skip_prompt=True, timeout=120.0, skip_special_tokens=True)

        chat_history = [input_text] if isinstance(input_text, str) else input_text
        trunc_input_text = self.base_instr_to_text(chat_history, sys_prompt=sys_prompt)
        inputs = self.base_tokenizer(trunc_input_text, return_tensors="pt", return_length=True)#, max_length=1024, truncation=True)

        input_len = inputs.pop('length')[0].item()

        genkwargs = dict(inputs.to(0), generation_config=self.gen_config, streamer=streamer, stopping_criteria=self.stop_criteria, negative_prompt_ids=None)
        
        self.model.to(dtype = self.base_dtype)
        print(self.model.dtype)
        with self.model.disable_adapter():
            thread = Thread(target=self.model.generate, kwargs=genkwargs)
            thread.start()
            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                yield new_text
        self.model.to(dtype = self.dtype)
        output_len = self.base_tokenizer(generated_text, return_length=True).length

        self._last_streamed_values.update({'input_text':trunc_input_text, 'output_text': generated_text, 'input_len': input_len, 'output_len': output_len})
        #self.model.set_adapter(adapters)
