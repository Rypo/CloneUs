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
from . import etl
from .tokenization import check_if_system, to_jinja_template



def convo_token_count(convo:list[dict], tokenizer: PreTrainedTokenizerFast):
    if isinstance(convo, dict):
        convo = [convo]
    # This method adds BOS (e.g '<s>' (1)) to beginning
    #n_tokens =  tokenizer(tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False), return_length=True)['length']
    n_tokens =  tokenizer(tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False), add_special_tokens=False, return_length=True)['length']

    # This method does not
    #n_tokens = tokenizer.apply_chat_template(convo, tokenize=True, add_generation_prompt=False, return_dict=True, tokenizer_kwargs={'return_length':True})['length']
    if not isinstance(n_tokens, int):
        n_tokens=n_tokens[0]
    return n_tokens


def assign_roles(messages:list[str], role_labels:list[str]=None):
    rolecycle = itertools.cycle(['user','assistant']) if role_labels is None else iter(role_labels)

    chat_content = []
    for msg in filter(None,messages):
        chat_content.append({"role": next(rolecycle), "content": msg})

    return chat_content

def add_sys_msg(chat_convo:list[dict], system_msg:str|dict[str,str]|list[dict[str,str]], tag_placement:typing.Literal['tag_only', 'content_prefix', 'replace_role'],):
    if isinstance(system_msg, str):
        system_msg = {'role':'system', 'content':system_msg}

    # TODO: Should tag_only accept a system message? Foundation models don't really do that, but I guess it's not gonna break anything
    if tag_placement != 'content_prefix':
        chat_convo = [system_msg]+chat_convo
        return chat_convo

    # Below here, roles are overwritten. Custom roles should never reach this point in code 
    # because if roles are customizable then it follows that a role can = "system".
    
    
    rolecycle = itertools.cycle(['user','assistant'])
    if isinstance(system_msg, list):
        assert len(system_msg) == 2,'SYN ACK format must be 2 messages exactly'
        assert system_msg[0]['role'] == 'user', 'SYN ACK format must be role0=user'
        assert system_msg[1]['role'] == 'assistant', 'SYN ACK format must be role1=assistant'
        # [{"role": "user", "content": fprompt}), {"role": "assistant", "content": "OK"}]
    else:
        assert system_msg['role'] != 'assistant', 'System message cannot start with assistant'
        system_msg = [{'role': next(rolecycle), 'content': system_msg['content']}]
    

    chat_convo = system_msg + chat_convo

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

    convo['content'] = convo.apply(lambda r: assign_roles(r.formatted_text, role_labels=None), axis=1) # convo['formatted_text'].apply(assign_roles)
    convo['content'] = convo['content'].apply(add_sys_msg, system_msg = cfg.fprompt, has_system=has_system, append_first_msg = cfg.prompt.append_msg)
    
    convo = convo[['content','gtime_gap']].drop_duplicates('content')

    return convo

def format_role_tgap(df_all:pd.DataFrame, cfg, has_system):
    df_all['gtime_gap'] = df_all.groupby(['split','chat_session'])['Date'].diff().dt.total_seconds().fillna(0)

    df_rolechat = df_all[['formatted_author_tag','text','split','chat_session', 'gtime_gap']].copy()

    convo = df_rolechat.groupby(['split','chat_session'])[['formatted_author_tag','text','gtime_gap']].agg(list).drop_duplicates('text')

    convo['content'] = convo.apply(lambda r: assign_roles(r.text, role_labels=r.formatted_author_tag), axis=1)
    convo['content'] = convo['content'].apply(add_sys_msg, system_msg = cfg.fprompt, has_system=has_system, append_first_msg = cfg.prompt.append_msg)
    
    convo = convo[['content','gtime_gap']].drop_duplicates('content')

    return convo

def ua_tags_dataset(chat_csv, tokenizer, cfg, text_only=False):
    topk_longest_gaps = 10
    has_system = check_if_system(tokenizer)
    
    msl = 3 if cfg.prompt.append_msg else 2
    cfg.dataset.min_session_length = msl if cfg.dataset.min_session_length is None else max(cfg.dataset.min_session_length, msl)

    df_all = etl.format_chat_groups(etl.process_csv(chat_csv), author_tag=cfg.author_tag, tag_sep=cfg.tag_sep, postfix=None, 
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

    df_all = etl.format_chat_groups(etl.process_csv(chat_csv), author_tag=cfg.author_tag, tag_sep=None, postfix=None, 
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
    tag_chat_template = to_jinja_template(cfg.tag_sep, cfg.postfix)
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
    

    df_all = etl.format_chat_groups(etl.process_csv(chat_csv), author_tag=cfg.author_tag,  tag_sep=cfg.tag_sep, postfix=cfg.postfix, 
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
    df_all = etl.format_chat_groups(etl.process_csv(chat_csv), author_tag=cfg.author_tag,  tag_sep=cfg.tag_sep, postfix=cfg.postfix, 
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


def fill_cfg_from_data(formatted_author_tag_col:pd.Series, cfg):
    fprompt = cfg.fprompt
    name_mapping = cfg.prompt.name_mapping
    
    if name_mapping is None:
        name_mapping = ', '.join(formatted_author_tag_col.unique())
        cfg.prompt.name_mapping = name_mapping
    
    if fprompt is None:
        fprompt = cfg.prompt.template.format(name_mapping=name_mapping, task=cfg.prompt.task)
        cfg.fprompt = fprompt
    
    return cfg

def apply_conv_format(df_all:pd.DataFrame, cfg, has_system:bool=None):
    '''WILL SET cfg.fprompt and cfg.name_mapping IF THEY ARE NONE'''
    fprompt = cfg.fprompt
    name_mapping = cfg.prompt.name_mapping
    
    if name_mapping is None:
        name_mapping = ', '.join(df_all.formatted_author_tag.unique())
        cfg.prompt.name_mapping = name_mapping
    
    if fprompt is None:
        fprompt = cfg.prompt.template.format(name_mapping=name_mapping, task=cfg.prompt.task)
        cfg.fprompt = fprompt
    
    group_keys = ['split']
    if 'chat_session' in df_all:
        group_keys += ['chat_session']
    
    sr_convo = df_all.groupby(group_keys)[['formatted_author_tag','text']].agg(list).apply(
        lambda r: to_conversation_format(r.formatted_author_tag, r.text, 
                                         tag_placement=cfg.tag_placement, fprompt=cfg.fprompt, 
                                         has_system=has_system, append_msg=cfg.prompt.append_msg), axis=1)
    
    #sr_convo.apply(tokenizer.apply_chat_template, tokenize = False,).apply(lambda x: tokenizer(x, add_special_tokens=False, return_length=True)['length'])
    
    return sr_convo

def to_conversation_format(formatted_author_tags:list[str], raw_texts:list[str], tag_placement:typing.Literal['tag_only', 'content_prefix', 'replace_role'],  fprompt:str=None, has_system: bool=None, append_msg:bool=None) -> list[dict[str,str]]:
    # NOTE: formatted_author_tags INCLUDE TAG SEP (if not None)
    # https://github.com/benfred/py-spy
    atag_tcontent = zip(formatted_author_tags, raw_texts)
    chat_content = []
    
    if fprompt is None:
        if tag_placement == 'content_prefix':
            for fauth_tag,text in atag_tcontent:
                content = fauth_tag + text
                chat_content.append({"role": next(rolecycle), "content": content})
        else:
            for role_tag,content in atag_tcontent:
                # for TagFormat, tag_sep is baked in to chat_template via to_jinja_template
                chat_content.append({"role": role_tag, "content": content})

        return chat_content
    
    
    if tag_placement in ['replace_role','tag_only']:
        # '''For tag only, markup free, format''' # '''For chatml with custom roles as usernames'''
        
        if tag_placement == 'replace_role':
            chat_content.append({"role": "system", "content": fprompt})
        
        for role_tag,content in atag_tcontent:
            #role_tag = useridx.format_author_tag(author, self.cfg.author_tag) # for TagFormat, tag_sep is baked in to chat_template via to_jinja_template
            chat_content.append({"role": role_tag, "content": content})
        
        return chat_content
    
    elif tag_placement == 'content_prefix':
        rolecycle = itertools.cycle(['user','assistant'])
            
        if fprompt is not None:
            assert append_msg is not None, 'content_prefix requires append_msg set True/False'
            assert has_system is not None, 'content_prefix requires has_system set True/False'
            
            if has_system:
                chat_content.append({"role": "system", "content": fprompt})
            elif append_msg=='legacy':
                tag0, text0 = next(atag_tcontent)
                content0 = tag0 + text0 
                # make sure first iter starts on assistant now
                chat_content.append({"role": next(rolecycle), "content": fprompt+content0})
            elif append_msg:
                chat_content.append({"role": "user", "content": fprompt})
                chat_content.append({"role": "assistant", "content": "OK"})
            
        for fauth_tag,text in atag_tcontent:
            content = fauth_tag + text
            chat_content.append({"role": next(rolecycle), "content": content})

        return chat_content

    


def batched_token_count(convo:list[dict[str,str]], tokenizer:PreTrainedTokenizerFast, ) -> np.ndarray[int]:
    '''Returns an array of approximate token lens for all items in a list. Each will be treated as it's own batch if not already batched.
    
    [{"role":...,"content":... }, {"role":...,"content":... }, ] will become [[{"role":...,"content":... }], [{"role":...,"content":... }], ]
    [[{"role":...,"content":... }], [{"role":...,"content":... }], ] will be left as is
    '''
    
    batched_convo = [[c] if isinstance(c, dict) else c for c in convo]
    return tokenizer(tokenizer.apply_chat_template(batched_convo, tokenize=False, add_generation_prompt=False), add_special_tokens=False, return_length=True, return_tensors='np')['length']

def split_overflow(sys_convo_batches, tokenizer, system_msg, tag_placement, max_length):
    convo_lens = batched_token_count(sys_convo_batches, tokenizer)
    overlen_indices = np.flatnonzero(convo_lens > max_length)
    
    while overlen_indices.size > 0:
        for i in overlen_indices:
            b_over = sys_convo_batches.pop(i)
            h = len(b_over)//2
            h_left = b_over[:h]
            h_right = add_sys_msg(b_over[h:], system_msg, tag_placement=tag_placement)
            # right first, then left since it inserts *before* index. Alt would be left, right@i+1
            sys_convo_batches.insert(i, h_right)
            sys_convo_batches.insert(i, h_left)
    
        convo_lens = batched_token_count(sys_convo_batches, tokenizer)
        overlen_indices = np.flatnonzero(convo_lens > max_length)
        if overlen_indices.size == 1:
            assert len(sys_convo_batches[overlen_indices[0]]) > 1, 'At least one message+system > max_length tokens. Truncation required.'
        
        print(overlen_indices, convo_lens[overlen_indices])
    
    return sys_convo_batches

def consecutive_max_batch(convo, all_msg_lens, token_limit:int):
    #all_msg_lens = batched_token_count(convo, tokenizer)
    batches = []
    batch = []
    total = 0
    
    for i,v in enumerate(all_msg_lens):
        if total + v > token_limit:
            batches.append(batch)
            batch = [convo[i]]
            total = v
        else:
            batch.append(convo[i])
            total+=v
    
    return batches

def convo_batch_max_tokens(convo:list[dict[str,str]], tokenizer:PreTrainedTokenizerFast, system_msg:str | dict[str, str] | list[dict[str, str]], tag_placement:typing.Literal['tag_only', 'content_prefix', 'replace_role'], max_length:int):
    syslen = batched_token_count([system_msg],tokenizer)[0]
    all_msg_lens = batched_token_count(convo, tokenizer)
    
    token_limit = (max_length-syslen)
    assert all_msg_lens.max() <= token_limit,'At least one message+system > max_length tokens. Truncation required.'

    convo_batches = consecutive_max_batch(convo, all_msg_lens, token_limit)

    sys_convo_batches = [add_sys_msg(c, system_msg, tag_placement=tag_placement) for c in convo_batches]
    sys_convo_batches = split_overflow(sys_convo_batches, tokenizer, system_msg, tag_placement, max_length=max_length) # not token_limit, since system has been added
    

    return sys_convo_batches


def dateless_dataset(chat_csv, tokenizer, cfg, text_only=False):
    '''Dataset groupings of sequential texts concatenated up to a maximum of `cfg.chunk_size` total tokens
    
    Chat sessions are NOT assigned, and no Date or timestamps fields are used or created.
    '''
    df_proc = etl.process_csv(chat_csv, youtube_encode_fetch=True, filter_prefixes=('!', '/'), merge_window=None)
    df_all = etl.format_text_tags(df_proc, author_tag=cfg.author_tag,  tag_sep=cfg.tag_sep, postfix=cfg.postfix,  eval_frac=cfg.dataset.get('eval_frac',0.01))
    
    df_tall = df_all.query('split=="train"')
    df_eall = df_all.query('split=="eval"')
    # get base tokens before any system is added
    convo_train = to_conversation_format(df_tall['formatted_author_tag'], df_tall['text'], tag_placement=cfg.tag_placement)
    convo_eval = to_conversation_format(df_eall['formatted_author_tag'], df_eall['text'], tag_placement=cfg.tag_placement)
    
    system_message = cfg.fprompt
    if cfg.tag_placement == 'content_prefix' and cfg.prompt.append_msg:
        if cfg.prompt.append_msg != 'legacy':
            system_message = [{'role':'user', 'content': cfg.fprompt}, 
                              {'role':'assistant', 'content':cfg.prompt.append_msg}]


    dset = datasets.DatasetDict({
        'train': datasets.Dataset.from_dict({'text': convo_batch_max_tokens(convo_train, tokenizer, system_message, cfg.tag_placement, max_length=cfg.chunk_size)}, split='train'),
        'validation': datasets.Dataset.from_dict({'text': convo_batch_max_tokens(convo_eval, tokenizer, system_message, cfg.tag_placement, max_length=cfg.chunk_size)}, split='validation'),
    })
    # https://huggingface.co/learn/nlp-course/chapter5/3
    if not text_only:
        dset = map_to_inputs(dset, tokenizer, max_length=cfg.chunk_size, truncation=cfg.dataset.allow_truncation)

    return dset