import copy
import typing
import itertools
import functools
from dataclasses import dataclass


import more_itertools
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import datasets
from transformers import PreTrainedTokenizerFast

from ..plugins import youtube as youtube
#from ..core import paths as rpaths
from . import roles, etl
from .tokenization import check_if_system



def convo_token_count(convo:list[dict], tokenizer):
    if isinstance(convo, dict):
        convo = [convo]
    # This method adds BOS (e.g '<s>' (1)) to beginning
    # n_tokens =  tokenizer(tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False), return_length=True)['length'][0]
    # This method does not
    n_tokens = tokenizer.apply_chat_template(convo, tokenize=True, add_generation_prompt=False, return_dict=True, tokenizer_kwargs={'return_length':True})['length'][0]
    return n_tokens


def assign_roles(messages:list[str], roles:list[str]=None):
    rolecycle = itertools.cycle(['user','assistant']) if roles is None else iter(roles)

    chat_content = []
    for msg in filter(None,messages):
        chat_content.append({"role": next(rolecycle), "content": msg})

    return chat_content


def add_sys_msg(chat_convo:list[dict], system_msg:str|dict[str,str], has_system:bool, append_first_msg:bool):
    if isinstance(system_msg, str):
        system_msg = {'role':'system', 'content':system_msg}

    if has_system: 
        if chat_convo[0]['role'] != 'system':
            assert system_msg not in chat_convo, 'system found in wrong position!'
            chat_convo = [system_msg]+chat_convo
        return chat_convo

    # Below here, roles are overwritten. Custom roles should never reach this point in code 
    # because if roles are customizable then it follows that a role can = "system".
    
    system_msg['role'] = 'user'
    rolecycle = itertools.cycle(['user','assistant'])

    if system_msg['content'] not in chat_convo[0]['content']:
        if append_first_msg:
            system_msg['content'] += chat_convo[0]['content']
            chat_convo[0] = system_msg
        else:
            # if not appending, first message will be user with system prompt, assistant as first chat message
            chat_convo = [system_msg] + chat_convo
        
        for c in chat_convo:
            c['role'] = next(rolecycle)


    return chat_convo

def map_to_inputs(dset:datasets.Dataset, tokenizer:PreTrainedTokenizerFast, max_length:int, truncation:bool, text_field='text'):
    '''Maps Dataset text field to those for model input (input_ids, special_tokens_mask, length)
    
    Importantly, it does NOT add special tokens. This is primarily a concern after calling apply_chat_template(tokenize=False) which,
    depending on the template, may or may not insert a bos_token. This will cause double bos_token on all entries. 
    '''
    dset = dset.map(lambda s: tokenizer(s[text_field], add_special_tokens=False, return_special_tokens_mask=True, max_length=max_length, return_length=True, truncation=truncation), batched=True)
    return dset

@dataclass
class TimeGapPartitioner:
    '''Helper class for recursive partitioning of chat conversations based on time between consecutive messages.'''
    content: list[dict]
    time_gaps: list[float]
    min_session_length: int
    tokenizer: PreTrainedTokenizerFast
    topk_longest_gaps: int = 10

    @property
    def candidate_inds(self):
        tg_inds = np.argsort(self.time_gaps)
        # avoid too small if sessions after splits. Want at least min_session_length items on either side of split
        lb = self.min_session_length
        ub = tg_inds.size-self.min_session_length
        # remember: indexing.. don't get off-by-one'd. Should be <= and <
        candidate_inds = tg_inds[(lb <= tg_inds) & (tg_inds < ub)]
        # Filter to top k longest gaps, otherwise no better than just picking midpoint.
        return candidate_inds[-self.topk_longest_gaps:]
    
    @property
    def data(self):
        return self.content#[self.content,self.time_gaps] #self

    def count_tokens(self, content=None):
        if content is None:
            content = self.content
        return convo_token_count(content, self.tokenizer)

    def cost_fn(self, i:int, system_msg: str, has_system:bool, append_first_msg:bool):
        '''Absolute difference between token counts on each side of the split'''
        # need deep copy because add_sys_msg mutates dicts, which we don't want during exploration phase
        r_content =  add_sys_msg(copy.deepcopy(self.content[i:]), system_msg, has_system, append_first_msg)
        
        return abs(self.count_tokens(self.content[:i]) - self.count_tokens(r_content))
    

    def make_splits(self, system_msg: str, has_system:bool, append_first_msg:bool):
        i = min(self.candidate_inds, key=lambda i: self.cost_fn(i, system_msg, has_system, append_first_msg))

        lt,rt = self.time_gaps[:i], self.time_gaps[i:]
        r_content = add_sys_msg(self.content[i:], system_msg, has_system, append_first_msg)
        # if new sys_msg is added to convo, add pad to keep indices aligned        
        rt = [0.0]*(len(r_content)-len(rt)) + rt
        
        return [TimeGapPartitioner(self.content[:i], lt, self.min_session_length, self.tokenizer, self.topk_longest_gaps), 
                TimeGapPartitioner(r_content, rt, self.min_session_length, self.tokenizer, self.topk_longest_gaps)]    


    def find_best_splits(self, system_msg:str, max_len:int, has_system, append_first_msg):
        if self.count_tokens() < max_len:
            return self#.data

        ctg_left,ctg_right = self.make_splits(system_msg, has_system, append_first_msg)
        
        return [ctg_left.find_best_splits(system_msg, max_len, has_system, append_first_msg), ctg_right.find_best_splits(system_msg, max_len, has_system, append_first_msg)]

        # return [ctg_left.data if ctg_left.count_tokens() < max_len else ctg_left.find_best_splits(system_msg, max_len, has_system, append_first_msg),
        #     ctg_right.data if ctg_right.count_tokens() < max_len else ctg_right.find_best_splits(system_msg, max_len, has_system, append_first_msg)]
        
    def time_gap_partition(self, cfg, has_system):
        if self.count_tokens() < cfg.chunk_size:
            return [self.content]
        # TODO: verify order is preserved
        splits = self.find_best_splits(system_msg = cfg.fprompt, max_len=cfg.chunk_size,  has_system=has_system, append_first_msg=cfg.prompt.append_msg)
        return [s.content for s in more_itertools.collapse(splits)]
    

def format_ua_tgap(df_all:pd.DataFrame, cfg, has_system):
    df_all['gtime_gap'] = df_all.groupby(['split','chat_session'])['Date'].diff().dt.total_seconds().fillna(0)

    df_rolechat = df_all[['formatted_text','split','chat_session','gtime_gap']].copy()
    
    convo = df_rolechat.groupby(['split','chat_session'])[['formatted_text','gtime_gap']].agg(list).drop_duplicates('formatted_text')

    convo['content'] = convo.apply(lambda r: assign_roles(r.formatted_text, roles=None), axis=1) # convo['formatted_text'].apply(assign_roles)
    convo['content'] = convo['content'].apply(add_sys_msg, system_msg = cfg.fprompt, has_system=has_system, append_first_msg = cfg.prompt.append_msg)
    
    convo = convo[['content','gtime_gap']].drop_duplicates('content')

    return convo

def format_role_tgap(df_all:pd.DataFrame, cfg, has_system):
    df_all['gtime_gap'] = df_all.groupby(['split','chat_session'])['Date'].diff().dt.total_seconds().fillna(0)

    df_rolechat = df_all[['formatted_author_tag','text','split','chat_session', 'gtime_gap']].copy()

    convo = df_rolechat.groupby(['split','chat_session'])[['formatted_author_tag','text','gtime_gap']].agg(list).drop_duplicates('text')

    convo['content'] = convo.apply(lambda r: assign_roles(r.text, roles=r.formatted_author_tag), axis=1)
    convo['content'] = convo['content'].apply(add_sys_msg, system_msg = cfg.fprompt, has_system=has_system, append_first_msg = cfg.prompt.append_msg)
    
    convo = convo[['content','gtime_gap']].drop_duplicates('content')

    return convo

def ua_tags_dataset(chat_csv, tokenizer, cfg, text_only=False):
    topk_longest_gaps = 10
    has_system = check_if_system(tokenizer)
    
    msl = 3 if cfg.prompt.append_msg else 2
    cfg.dataset.min_session_length = msl if cfg.dataset.min_session_length is None else max(cfg.dataset.min_session_length, msl)

    df_all = etl.format_chat_groups(etl.preprocess_df(chat_csv), author_tag=cfg.author_tag, tag_sep=cfg.tag_sep, postfix=None, 
                                    hours_between_sessions=cfg.dataset.hours_between_sessions, min_session_length=cfg.dataset.min_session_length, eval_frac=cfg.dataset.get('eval_frac',0.01))
    

    convo = format_ua_tgap(df_all, cfg, has_system=has_system)
    convo['n_tokens'] = tokenizer(tokenizer.apply_chat_template(convo['content'].to_list(), tokenize=False), return_length=True, add_special_tokens=False)['length']
    print(f'Splitting over length samples. n = {(convo.n_tokens > cfg.chunk_size).sum()}')
    
    conversations = convo.apply(lambda r: [r.content] if r.n_tokens < cfg.chunk_size else 
                                TimeGapPartitioner(r.content, r.gtime_gap, cfg.dataset.min_session_length, tokenizer, topk_longest_gaps).time_gap_partition(cfg, has_system), axis=1
                                ).explode().droplevel(1).drop_duplicates()

    dset = datasets.DatasetDict({
        'train': datasets.Dataset.from_pandas(conversations['train'].to_frame(name='text').reset_index(drop=True).copy(), split='train', preserve_index=False),
        'validation': datasets.Dataset.from_pandas(conversations['eval'].to_frame(name='text').reset_index(drop=True).copy(), split='validation', preserve_index=False),
        # 'train': datasets.Dataset.from_dict(conversations['train'].to_frame(name='text').to_dict(orient='list')),
        # 'validation': datasets.Dataset.from_dict(conversations['eval'].to_frame(name='text').to_dict(orient='list')),
    })

    dset = dset.map(lambda x: {"text": tokenizer.apply_chat_template(x["text"], tokenize=False, add_generation_prompt=False)}, batched=True)
    
    if not text_only:
        dset = map_to_inputs(dset, tokenizer, max_length=cfg.chunk_size, truncation=cfg.dataset.allow_truncation)
    
    return dset

def author_roletags_dataset(chat_csv, tokenizer, cfg, text_only=False):
    topk_longest_gaps = 10
    has_system = check_if_system(tokenizer)

    df_all = etl.format_chat_groups(etl.preprocess_df(chat_csv), author_tag=cfg.author_tag, tag_sep=None, postfix=None, 
                                    hours_between_sessions=cfg.dataset.hours_between_sessions, min_session_length=cfg.dataset.min_session_length, eval_frac=cfg.dataset.get('eval_frac',0.01))

    convo = format_role_tgap(df_all, cfg, has_system=has_system)
    convo['n_tokens'] = tokenizer(tokenizer.apply_chat_template(convo['content'].to_list(), tokenize=False), return_length=True, add_special_tokens=False)['length']
    
    print(f'Splitting over length samples. n = {(convo.n_tokens > cfg.chunk_size).sum()}')
    
    conversations = convo.apply(lambda r: [r.content] if r.n_tokens < cfg.chunk_size else 
                                TimeGapPartitioner(r.content, r.gtime_gap, cfg.dataset.min_session_length, tokenizer, topk_longest_gaps).time_gap_partition(cfg, has_system), 
                                axis=1).explode().droplevel(1).drop_duplicates()

    dset = datasets.DatasetDict({
        'train': datasets.Dataset.from_pandas(conversations['train'].to_frame(name='text').reset_index(drop=True).copy(), split='train', preserve_index=False),
        'validation': datasets.Dataset.from_pandas(conversations['eval'].to_frame(name='text').reset_index(drop=True).copy(), split='validation', preserve_index=False),
    })

    dset = dset.map(lambda x: {'text':tokenizer.apply_chat_template(x["text"], tokenize=False, add_generation_prompt=False)}, batched=True)
    
    if not text_only:
        dset = map_to_inputs(dset, tokenizer, max_length=cfg.chunk_size, truncation=cfg.dataset.allow_truncation)
    
    return dset

def tag_only_dataset(chat_csv, tokenizer, cfg, text_only=False):
    tag_chat_template = roles.to_jinja_template(cfg.tag_sep, cfg.postfix)
    if tokenizer.chat_template is not None:
        print('WARNING: existing chat_template will be overwritten with one created from tag_sep + postfix. '
              'This dataset type only recommended for foundation model tuning.')
    tokenizer.chat_template = tag_chat_template
    return author_roletags_dataset(chat_csv, tokenizer, cfg, text_only)


def jsonl_dataset(train_jsonl, eval_jsonl, tokenizer, cfg, text_only=False):
    dset = datasets.load_dataset("json", data_files={"train": train_jsonl, "validation": eval_jsonl})
    if not text_only:
        dset = map_to_inputs(dset, tokenizer, max_length=cfg.chunk_size, truncation=cfg.dataset.allow_truncation)
    
    return dset


def ungrouped_dataset(chat_csv, tokenizer, cfg, text_only=False):
    '''Dataset of chat messages without any grouping by time or chunk size. 
    
    SFTTrainer with packing=True must be used since order is lost through normal batching
    '''
    if not cfg.use_sft_trainer:
        raise ValueError('for dataset "ungrouped_eos" SFTTrainer must be used to preserve message order')
    

    df_all = etl.format_chat_groups(etl.preprocess_df(chat_csv), author_tag=cfg.author_tag,  tag_sep=cfg.tag_sep, postfix=cfg.postfix, 
                                    hours_between_sessions=None, min_session_length=cfg.dataset.min_session_length, eval_frac=cfg.dataset.eval_frac)
    
    df_all = df_all[['split','formatted_text']].rename(columns={'formatted_text':'text'})
    
    ds_ungrouped = datasets.DatasetDict({
        'train': datasets.Dataset.from_pandas(df_all[df_all.split=='train'], split='train', preserve_index=False),
        'validation': datasets.Dataset.from_pandas(df_all[df_all.split=='eval'], split='validation', preserve_index=False)
    })
    if not text_only:
        ds_ungrouped = map_to_inputs(ds_ungrouped, tokenizer, max_length=cfg.chunk_size, truncation=cfg.dataset.allow_truncation, text_field='text')
    
    return ds_ungrouped



def chunk_max_length(formatted_texts:list[str], tokenizer, max_length:int):
    '''Creates a list of sequentially concatenated texts where each contains at most `max_length` tokens.'''
    def _accum(acc, t):
        text_accum,tok_accum = acc[-1]
        if (new_len := tok_accum + t[1]) < max_length:
            acc[-1] = (text_accum+ t[0], new_len)
        else:
            acc.append(t)
        return acc
    
    token_counts = tokenizer(formatted_texts, return_length=True, add_special_tokens=False)['length']
    texts_toks = functools.reduce(lambda acc,txt_tok: _accum(acc, txt_tok), zip(formatted_texts,token_counts), [('',0)])
    
    return [text for text,_ in texts_toks]


def max_tokens_dataset(chat_csv, tokenizer, cfg, text_only=False):
    '''Dataset groupings of sequential texts concatenated up to a maximum of `cfg.chunk_size` total tokens
    
    Chat sessions grouping are maintained to avoid non-sequential chat concatenation.
    '''
    df_all = etl.format_chat_groups(etl.preprocess_df(chat_csv), author_tag=cfg.author_tag,  tag_sep=cfg.tag_sep, postfix=cfg.postfix, 
                                    hours_between_sessions=cfg.dataset.hours_between_sessions, min_session_length=cfg.dataset.min_session_length, eval_frac=cfg.dataset.get('eval_frac',0.01))
    
    df_all['session_hr'] = df_all['chat_session'].astype(str).str[0]
    df_chunked_sessions = df_all.groupby(['split', 'session_hr'])['formatted_text'].agg(list).apply(chunk_max_length, tokenizer=tokenizer, max_length=cfg.chunk_size)
        
    dset = datasets.DatasetDict({
        'train': datasets.Dataset.from_dict({'text': [chunk for session_chunks in df_chunked_sessions['train'].to_list() for chunk in session_chunks]}),
        'validation': datasets.Dataset.from_dict({'text': [chunk for session_chunks in df_chunked_sessions['eval'].to_list() for chunk in session_chunks]}),
    })
    # https://huggingface.co/learn/nlp-course/chapter5/3
    if not text_only:
        dset = map_to_inputs(dset, tokenizer, max_length=cfg.chunk_size, truncation=cfg.dataset.allow_truncation)

    return dset