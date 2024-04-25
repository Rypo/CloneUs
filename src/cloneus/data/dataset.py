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


def check_if_system(tokenizer):
    from jinja2.exceptions import TemplateError
    try:
        _=tokenizer.apply_chat_template([{'role':'system', 'content':'abc'}])
        has_system = True
    except TemplateError:
        has_system = False

    return has_system

def index_chunks(len_arr, max_chunk_tokens=256):
    idxsets = []
    valsets = []
    
    cur_set = []
    cur_sum = 0
    
    for i,vlen in enumerate(len_arr):
        if cur_sum+vlen <= max_chunk_tokens:
            cur_sum+=vlen
            cur_set.append(i)
        else:
            idxsets.append(cur_set)
            valsets.append(cur_sum)
            cur_sum = vlen
            cur_set = [i]

    # 0th index to last index +1 (so it's inclusive)
    idxbnds = [[ix[0],ix[-1]+1] for ix in idxsets]

    return idxbnds


def token_partion(df_chat, maxlen):
    samples=[]
    cursample=('',0)
    for row in df_chat[['text','toklen']].itertuples(index=False):
        ctext,clen = cursample
        if clen+row.toklen <= maxlen:
            ctext+=row.text
            clen+=row.toklen
            cursample=(ctext,clen)
        else:
            samples.append(cursample)
            cursample=(row.text,row.toklen)

    return datasets.Dataset.from_list([{'text': sample} for sample,toklen in samples])

def token_chunk_chat(df_chat, idx_bounds):
    chunked = [df_chat['text'][lb:ub] for lb, ub in tqdm(idx_bounds)]
    mchunk = [{'text': ''.join(chunk)} for chunk in chunked]
    ds_chunk = datasets.Dataset.from_list(mchunk)
    return ds_chunk

def dataset_token_chunks(df_train, df_val, tokenizer, chunk_size=512):
    #if df_train['text'].str.startswith('[USER').any() and msg_sep == '\n':
    #    print("Warning: [USER] tokens found in training data and inter-message seperator is 1xNEWLINE. This will cause issues with the tokenizer.")
    # The majority of the extemely small number (~100) of examples that are longer than 256 tokens are shitposts

    train_lens = tokenizer(df_train['text'].to_list(), return_length=True, add_special_tokens=True)['length']
    val_lens = tokenizer(df_val['text'].to_list(), return_length=True, add_special_tokens=True)['length']

    ds_chunkblocks = datasets.DatasetDict({
        'train':token_chunk_chat(df_train, index_chunks(train_lens, chunk_size)),
        'validation': token_chunk_chat(df_val, index_chunks(val_lens, chunk_size)),
    })

    # return_special_tokens_mask for DataCollatorForLanguageModeling
    ds_chunkblocks = ds_chunkblocks.map(lambda s: tokenizer(s['text'], return_special_tokens_mask=True, max_length=chunk_size, truncation=True), batched=True)
    #ds_chunkblocks = ds_chunkblocks.map(lambda s: tokenizer(s['text'], return_special_tokens_mask=True, return_length=True), batched=True)

    return ds_chunkblocks




def convo_token_count(convo:list[dict], tokenizer):
    if isinstance(convo, dict):
        convo = [convo]
    # This method adds BOS (e.g '<s>' (1)) to beginning
    n_tokens =  tokenizer(tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False), return_length=True)['length'][0]
    # This method does not
    # n_tokens = tokenizer.apply_chat_template(convo.iloc[0], tokenize=True, add_generation_prompt=False, return_dict=True, tokenizer_kwargs={'return_length':True})['length'][0]
    return n_tokens

def token_cnt_role(convo:list[dict], tokenizer, sys_role_msg:dict, count_special=True):
    convo = list(convo)
    if convo[0]!=sys_role_msg:
        # include added system msg in partition length
        convo = [sys_role_msg]+convo
    return convo_token_count(convo, tokenizer)
    #text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
    #return tokenizer(text, return_length=True, add_special_tokens=count_special)['length'][0]

def sanity_check(tokenizer, orig_convo:list[dict], convo_splits:list[list[dict]], sys_role_msg:dict,):
    sys_ntokens = token_cnt_role([sys_role_msg], tokenizer, sys_role_msg, count_special=True)
    orig_ntokens = token_cnt_role(orig_convo, tokenizer, sys_role_msg, count_special=True)
    
    n_partitions = len(convo_splits)
    new_ntokens = sum([token_cnt_role(s, tokenizer, sys_role_msg, count_special=True) for s in convo_splits])
    n_added_systokens = (n_partitions-1)*sys_ntokens # -1 for the original system message before splitting

    return new_ntokens - n_added_systokens == orig_ntokens

def split_convo(over_len_convo: list[dict[str,str]], max_len:int, tokenizer, count_special=True, verify=True)-> list[list[dict[str,str]]]:
    sys_role_msg = over_len_convo[0]
    # if under max length, return as its own partion of size 1. The list wrap ensures it's not exploded into dicts
    if token_cnt_role(over_len_convo, tokenizer, sys_role_msg, count_special)<=max_len:
        return [over_len_convo]
        
    assert sys_role_msg['role']=='system', 'First message not a system role!'
    cbatches = []
    for cb in more_itertools.constrained_batches(over_len_convo, max_size=max_len, get_len=lambda t: token_cnt_role([t], tokenizer, sys_role_msg, count_special)):
        cbatches.append(list(cb))
       
    # append system to each if not already
    cbatches = [cb if cb[0] == sys_role_msg else [sys_role_msg]+cb for cb in cbatches] 
    if verify:
        assert sanity_check(tokenizer, over_len_convo, cbatches, sys_role_msg), 'Verification Failed! Some tokens cannot be accounted for.'
    
    return cbatches

def group_role_replace(df_all:pd.DataFrame, tokenizer:PreTrainedTokenizerFast, max_len:int, role_tag:str, fprompt:str) -> pd.Series:
    # NOTE: last implementation was wrong. Some chat templates have special behavior for index0.
    # So, was getting special behavior twice, for formatted_system + first user message
    
    system_entry = {"role": "system", "content": fprompt}
    #formatted_system = tokenizer.apply_chat_template([{"role": "system", "content": fprompt}], tokenize=False)

    df_all['role'] = df_all['user'].apply(roles.format_author_tag, author_tag=role_tag)
    
    df_rolechat = df_all[['role','text','split','chat_session']].rename(columns={'text':'content'}).copy()
    
    # insert system message as first item in chat convos groups.
    conversations = df_rolechat.groupby(['split','chat_session'])[['role','content']].apply(lambda df: [system_entry] + df.to_dict(orient='records'))

    # Drop duplicates created if list of hours_between_sessions 
    conversations.drop_duplicates(inplace=True)

    # split any groups that have total tokens exceeding max_len into sub groups, drop chat_session index as it's no longer needed
    conversations = conversations.apply(split_convo, max_len=max_len, tokenizer=tokenizer, count_special=True, verify=True).explode().droplevel(1)

    return conversations

def _author_role_dataset(chat_csv:str, tokenizer:PreTrainedTokenizerFast, cfg):
    role_tag = cfg.author_tag
    fprompt = cfg.fprompt
    max_len = cfg.chunk_size
    prompt_length = tokenizer(tokenizer.apply_chat_template([{"role": "system", "content": fprompt}], tokenize=False), return_length=True).length[0]
    #max_len-=prompt_length
    eval_frac = cfg.dataset.eval_frac

    if cfg.custom_chat_template:
        assert tokenizer.chat_template == cfg.custom_chat_template, 'Custom chat template not assigned to tokenizer!'

    # https://old.reddit.com/r/LocalLLaMA/comments/1aiz6zu/roleplaying_system_prompts/
    
    df_all = etl.format_chat_groups(etl.preprocess_df(chat_csv), tag_sep='<not_used>', postfix='<not_used>', author_tag=role_tag, 
                                    hours_between_sessions=cfg.dataset.hours_between_sessions, min_session_length=cfg.dataset.min_session_length, eval_frac=eval_frac)
    
    
    ds_chat = group_role_replace(df_all, tokenizer, max_len=max_len, role_tag=role_tag, fprompt=fprompt)

    ds_chat = ds_chat.apply(tokenizer.apply_chat_template, tokenize=False)

    ds_timechunk_chatml = datasets.DatasetDict({
        'train': datasets.Dataset.from_pandas(ds_chat['train'].to_frame(name='text').reset_index(drop=True).copy(), split='train', preserve_index=False),
        'validation': datasets.Dataset.from_pandas(ds_chat['eval'].to_frame(name='text').reset_index(drop=True).copy(), split='validation', preserve_index=False),
    })

    ds_timechunk_chatml = ds_timechunk_chatml.map(lambda s: tokenizer(s['text'], return_special_tokens_mask=True, return_length=True, truncation=False), batched=True)
    return ds_timechunk_chatml

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






@dataclass
class ContentTimeGaps:
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

    def cost_fn(self, i, system_msg: str, has_system:bool, append_first_msg:bool):
        # closest to balanced token_count between left,right side of split  
        r_content =  add_sys_msg(copy.deepcopy(self.content[i:]), system_msg, has_system, append_first_msg)
        # need deep copy because add_sys_msg mutates dicts 
        return abs(self.count_tokens(self.content[:i]) - self.count_tokens(r_content))
    

    def make_splits(self, system_msg: str, has_system:bool, append_first_msg:bool):
        i = min(self.candidate_inds, key=lambda i: self.cost_fn(i, system_msg, has_system, append_first_msg))

        lc,rc = self.content[:i], self.content[i:]
        lt,rt = self.time_gaps[:i], self.time_gaps[i:]
        
        rc = add_sys_msg(rc, system_msg, has_system, append_first_msg)
        # if new sys_msg is added to convo, add pad to keep indices aligned
        roffset = (len(rc)-len(rt))
        
        rt = [0.0]*roffset + rt
        
        return [ContentTimeGaps(lc,lt, self.min_session_length, self.tokenizer, self.topk_longest_gaps), 
                ContentTimeGaps(rc,rt, self.min_session_length, self.tokenizer, self.topk_longest_gaps)]    


    def find_best_splits(self, system_msg:str, max_len:int, has_system, append_first_msg):
        if self.count_tokens() < max_len:
            return self#.data

        ctg_left,ctg_right = self.make_splits(system_msg, has_system, append_first_msg)
        
        return [ctg_left.find_best_splits(system_msg, max_len, has_system, append_first_msg), ctg_right.find_best_splits(system_msg, max_len, has_system, append_first_msg)]

        # return [ctg_left.data if ctg_left.count_tokens() < max_len else ctg_left.find_best_splits(system_msg, max_len, has_system, append_first_msg),
        #     ctg_right.data if ctg_right.count_tokens() < max_len else ctg_right.find_best_splits(system_msg, max_len, has_system, append_first_msg)]
        
        # return [ctg_left if ctg_left.count_tokens() < max_len else ctg_left.find_best_splits(system_msg, max_len, has_system, append_first_msg),
        #         ctg_right if ctg_right.count_tokens() < max_len else ctg_right.find_best_splits(system_msg, max_len, has_system, append_first_msg)]

    def time_gap_partition(self, cfg, has_system):
        #if convo_token_count(self.content, tokenizer) < cfg.chunk_size:
        if self.count_tokens() < cfg.chunk_size:
            return [self.content]
        
        splits = self.find_best_splits(system_msg = cfg.fprompt, max_len=cfg.chunk_size,  has_system=has_system, append_first_msg=cfg.prompt.append_msg)
        return [s.content for s in more_itertools.collapse(splits)]
    
# def time_gap_partition(content, gtime_gap, cfg, has_system, tokenizer):
#     if convo_token_count(content, tokenizer) < cfg.chunk_size:
#         return [[content],[gtime_gap]]
    
#     ctg = ContentTimeGaps(content, gtime_gap, cfg.dataset.min_session_length, tokenizer)
    
#     splits = ctg.find_best_splits(system_msg = cfg.fprompt, max_len=cfg.chunk_size,  has_system=has_system, append_first_msg=cfg.prompt.append_msg)
    
#     conts,times = [],[]
#     # TODO: verify order is preserved
#     for ctgs in more_itertools.collapse(splits):
#         conts.append(ctgs.content)
#         times.append(ctgs.time_gaps)
#     return [conts,times] #pd.Series([conts,times], index=['content','gtime_gaps'])

def format_ua_tgap(df_all:pd.DataFrame, cfg, has_system):
    df_all['formatted_text'] = df_all['formatted_text'].str.removesuffix('<|not_used|>')
    df_all['gtime_gap'] = df_all.groupby(['split','chat_session'])['Date'].diff().dt.total_seconds().fillna(0)

    df_rolechat = df_all[['formatted_text','split','chat_session','gtime_gap']].rename(columns={'formatted_text':'text'}).copy()
    
    convo = df_rolechat.groupby(['split','chat_session'])[['text','gtime_gap']].agg(list).drop_duplicates('text')

    convo['content'] = convo['text'].apply(assign_roles)
    convo['content'] = convo['content'].apply(add_sys_msg, system_msg = cfg.fprompt, has_system=has_system, append_first_msg = cfg.prompt.append_msg)
    
    convo.drop_duplicates('content',inplace=True)

    return convo

def format_role_tgap(df_all:pd.DataFrame, cfg, has_system):
    df_all['role'] = df_all['user'].apply(roles.format_author_tag, author_tag=cfg.author_tag, )
    df_all['gtime_gap'] = df_all.groupby(['split','chat_session'])['Date'].diff().dt.total_seconds().fillna(0)

    df_rolechat = df_all[['role','text','split','chat_session', 'gtime_gap']].copy()

    convo = df_rolechat.groupby(['split','chat_session'])[['role','text','gtime_gap']].agg(list).drop_duplicates('text')

    convo['content'] = convo.apply(lambda r: assign_roles(r.text, r.role), axis=1)
    convo['content'] = convo['content'].apply(add_sys_msg, system_msg = cfg.fprompt, has_system=has_system, append_first_msg = cfg.prompt.append_msg)
    
    convo = convo[['content','gtime_gap']].drop_duplicates('content')

    return convo

def ua_tags_dataset(chat_csv, tokenizer, cfg):
    topk_longest_gaps = 10
    has_system = check_if_system(tokenizer)
    
    if cfg.dataset.min_session_length is None:
        cfg.dataset.min_session_length = 2
    if not has_system:
        msl = 3 if cfg.prompt.append_msg else 2
        cfg.dataset.min_session_length = max(msl, cfg.dataset.min_session_length)

    df_all = etl.format_chat_groups(etl.preprocess_df(chat_csv), tag_sep=cfg.tag_sep, postfix='<|not_used|>', author_tag=cfg.author_tag, 
                                    hours_between_sessions=cfg.dataset.hours_between_sessions, min_session_length=cfg.dataset.min_session_length, eval_frac=cfg.dataset.get('eval_frac',0.01))
    
    convo = format_ua_tgap(df_all, cfg, has_system=has_system)
    convo['n_tok'] = tokenizer(tokenizer.apply_chat_template(convo['content'].to_list(), tokenize=False), return_length=True)['length']
    
    conversations = convo.apply(lambda r: ContentTimeGaps(r.content, r.gtime_gap, cfg.dataset.min_session_length, tokenizer, topk_longest_gaps)
                                .time_gap_partition(cfg, has_system) if r.n_tok>cfg.chunk_size else [r.content], axis=1).explode().droplevel(1).drop_duplicates()

    dset = datasets.DatasetDict({
        'train': datasets.Dataset.from_dict(conversations['train'].to_frame(name='text').to_dict(orient='list')),
        'validation': datasets.Dataset.from_dict(conversations['eval'].to_frame(name='text').to_dict(orient='list')),
    })

    dset = dset.map(lambda x: {"text": tokenizer.apply_chat_template(x["text"], tokenize=False, add_generation_prompt=False)}, batched=True)

    #dataset = dataset.map(lambda s: tokenizer(s['text'], return_special_tokens_mask=True, max_length=maxlen,  return_length=True, truncation=truncation), batched=True)
    return dset

def author_roletags_dataset(chat_csv, tokenizer, cfg):
    topk_longest_gaps = 10
    has_system = check_if_system(tokenizer)
    
    df_all = etl.format_chat_groups(etl.preprocess_df(chat_csv), tag_sep='<not_used>', postfix='<not_used>', author_tag=cfg.author_tag, 
                                    hours_between_sessions=cfg.dataset.hours_between_sessions, min_session_length=cfg.dataset.min_session_length, eval_frac=cfg.dataset.get('eval_frac',0.01))

    convo = format_role_tgap(df_all, cfg, has_system=has_system)
    convo['n_tok'] = tokenizer(tokenizer.apply_chat_template(convo['content'].to_list(), tokenize=False),return_length=True)['length']
    print(f'Splitting overlen samples. n = {(convo.n_tok>cfg.chunk_size).sum()}')
    conversations = convo.apply(lambda r: ContentTimeGaps(r.content, r.gtime_gap, cfg.dataset.min_session_length, tokenizer, topk_longest_gaps)
                                .time_gap_partition(cfg, has_system) if r.n_tok>cfg.chunk_size else [r.content], axis=1).explode().droplevel(1).drop_duplicates()

    dset = datasets.DatasetDict({
        'train': datasets.Dataset.from_dict(conversations['train'].to_frame(name='chat').to_dict(orient='list')),
        'validation': datasets.Dataset.from_dict(conversations['eval'].to_frame(name='chat').to_dict(orient='list')),
    })

    dset = dset.map(lambda x: {"text": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)}, batched=True)

    dset = dset.map(lambda s: tokenizer(s['text'], return_special_tokens_mask=True, max_length=cfg.chunk_size, return_length=True, truncation=False), batched=True)
    return dset

def dataset_tc_files(tokenizer, maxlen, train_jsonl, eval_jsonl):
    ds_timechunk = datasets.load_dataset("json", data_files={"train": train_jsonl, "validation": eval_jsonl})
    ds_timechunk = ds_timechunk.map(lambda s: tokenizer(s['text'], return_special_tokens_mask=True, max_length=maxlen,  return_length=True, truncation=False), batched=True)
    
    return ds_timechunk



def dataset_ungrouped(chat_csv, tokenizer, cfg, text_only=False):
    if not cfg.use_sft_trainer:
        raise ValueError('for dataset "ungrouped_eos" SFTTrainer must be used')
    # hours_between_sessions: ignored
    maxlen= cfg.chunk_size
    tag_sep=cfg.tag_sep
    postfix=cfg.postfix
    author_tag=cfg.author_tag
    min_session_length=cfg.dataset.min_session_length
    eval_frac = cfg.dataset.eval_frac
    df_all = etl.format_chat_groups(etl.preprocess_df(chat_csv), tag_sep, postfix=postfix, author_tag=author_tag, 
                                    hours_between_sessions=4, min_session_length=min_session_length, eval_frac=eval_frac).drop(columns=['chat_session']) 
    #df_proc = etl.preprocess_df(chat_csv)
    #df_all = etl.create_dftext(df_proc, tag_sep=tag_sep, postfix=postfix, author_tag=author_tag, min_session_length=min_session_length).drop(columns=['chat_session']) 
    #df_all = create_dftext(chat_csv, tag_sep=tag_sep, postfix=postfix, author_tag=author_tag, hours_between_sessions=4 ).drop(columns=['chat_session']) 
    
    ds_ungrouped = datasets.DatasetDict({
        'train': datasets.Dataset.from_pandas(df_all[df_all.split=='train'], split='train', preserve_index=False),
        'validation': datasets.Dataset.from_pandas(df_all[df_all.split=='eval'], split='validation', preserve_index=False)
    })
    if not text_only:
        ds_ungrouped = ds_ungrouped.map(lambda s: tokenizer(s['text'], return_special_tokens_mask=True, max_length=maxlen,  return_length=True, truncation=False), batched=True)
    
    return ds_ungrouped

def dataset_df_all(chat_csv, tokenizer, maxlen=1024, tag_sep=' ', postfix='\n\n'):
    df_all = etl.format_chat_groups(etl.preprocess_df(chat_csv), tag_sep=tag_sep, postfix=postfix, author_tag='[USER:{author}]', hours_between_sessions=4, min_session_length=1, eval_frac=0.005)
    # df_all = etl.create_dftext(chat_csv, tag_sep=tag_sep, postfix=postfix)
    
    df_train = df_all[df_all.split=='train'].drop(columns=['pre_bot','split', 'user_sequence']).reset_index(drop=True).copy()
    df_eval = df_all[df_all.split=='eval'].drop(columns=['pre_bot','split', 'user_sequence']).reset_index(drop=True).copy()
    
    ds_raw_msgs = datasets.DatasetDict({
        'train': datasets.Dataset.from_pandas(df_train, split='train', preserve_index=False),
        'validation': datasets.Dataset.from_pandas(df_eval, split='validation', preserve_index=False)
    })
    ds_raw_msgs = ds_raw_msgs.map(lambda s: tokenizer(s['text'], return_special_tokens_mask=True, max_length=maxlen, truncation=True), batched=True)
    
    return ds_raw_msgs

def dataset_all_chunks(chat_csv, tokenizer, cfg):
    # HEY: look at this: https://huggingface.co/learn/nlp-course/chapter5/3
    maxlen= cfg.chunk_size
    tag_sep=cfg.tag_sep
    postfix=cfg.postfix
    author_tag=cfg.author_tag
    min_session_length=cfg.dataset.min_session_length
    eval_frac = cfg.dataset.eval_frac
    
    df_all = etl.format_chat_groups(etl.preprocess_df(chat_csv),  tag_sep=tag_sep, postfix=postfix, author_tag=author_tag,  min_session_length=min_session_length, eval_frac=eval_frac).drop(columns=['chat_session']) 
    #df_all = create_dftext(chat_csv, tag_sep=tag_sep, postfix=postfix)
    
    df_train = df_all[df_all.split=='train'].drop(columns=['split', 'text']).reset_index(drop=True).copy()
    df_eval = df_all[df_all.split=='eval'].drop(columns=['split', 'text']).reset_index(drop=True).copy()

    df_train['toklen'] = tokenizer(df_train['formatted_text'].to_list(), return_length=True, add_special_tokens=True)['length']
    df_eval['toklen'] = tokenizer(df_eval['formatted_text'].to_list(), return_length=True, add_special_tokens=True)['length']
    
    ds_chunks = datasets.DatasetDict({
        'train': token_partion(df_train.rename(columns={'formatted_text':'text'}), maxlen),
        'validation': token_partion(df_eval.rename(columns={'formatted_text':'text'}), maxlen),
    })

    # return_special_tokens_mask for DataCollatorForLanguageModeling
    ds_chunks = ds_chunks.map(lambda s: tokenizer(s['text'], return_special_tokens_mask=True, max_length=maxlen, truncation=True), batched=True)
    # ds_raw_msgs = datasets.DatasetDict({
    #     'train': datasets.Dataset.from_dict(tokenizer(cattrain, max_length=maxlen, return_overflowing_tokens=True, return_special_tokens_mask=True, return_length=True)),
    #     'validation': datasets.Dataset.from_dict(tokenizer(cateval, max_length=maxlen, return_overflowing_tokens=True, return_special_tokens_mask=True, return_length=True))
    # })
    # ds_raw_msgs = ds_raw_msgs.map(lambda s: tokenizer(s['text'], return_special_tokens_mask=True, max_length=maxlen, truncation=True), batched=True)
    
    return ds_chunks


# def resplit_overflow(df_data, tokenizer, maxlen, postfix='\n\n'):
#     '''Takes a dataframe split with a text column and splits those that exceed maxlen into multiple new rows using postfix'''
#     df_data['toklen'] = tokenizer(df_data['formatted_text'].to_list(), return_length=True, add_special_tokens=True)['length']
#     df_overlen = df_data[df_data['toklen'] > maxlen]
    
#     def resplit_overlen(overlen_text):
#         # NOTE: without the "if t", will have a double postfix. No clue how that bug slipped by.
#         split_text_list = [t+postfix for t in overlen_text.split(postfix) if t]
#         return [''.join(cb) for cb in more_itertools.constrained_batches(split_text_list, maxlen, get_len=lambda t: tokenizer(t, return_length=True, add_special_tokens=True)['length'][0])]
    
#     df_etext = df_overlen['formatted_text'].apply(resplit_overlen).explode().to_frame()
#     df_etext = df_etext.reindex(columns=df_overlen.columns).fillna(df_overlen)
    
#     df_etext['toklen'] = df_etext['formatted_text'].apply(lambda t: tokenizer(t, return_length=True)['length'][0])
#     df_etext['chat_session'] = df_etext['chat_session'].astype(int)

#     return pd.concat([df_data.drop(index=df_overlen.index), df_etext]).sort_values(['chat_session','toklen'], ascending=[True,False])


# def dataset_timechunk(chat_csv, tokenizer, cfg, text_only=False):
#     maxlen= cfg.chunk_size
#     tag_sep=cfg.tag_sep
#     postfix=cfg.postfix
#     author_tag=cfg.author_tag
#     hours_between_sessions=cfg.dataset.hours_between_sessions
#     min_session_length=cfg.dataset.min_session_length
#     eval_frac = cfg.dataset.eval_frac
    
#     df_all = etl.format_chat_groups(etl.preprocess_df(chat_csv), tag_sep, postfix, author_tag, hours_between_sessions, min_session_length, eval_frac=eval_frac)
#     df_concat = df_all.groupby(['split','chat_session'])['formatted_text'].agg(''.join).drop_duplicates()
    
#     # TODO: maybe split on longest chat pause instead of arbitrary length.
#     df_train = resplit_overflow(df_concat['train'].reset_index(), tokenizer, maxlen=maxlen, postfix=postfix)
#     df_eval = resplit_overflow(df_concat['eval'].reset_index(), tokenizer, maxlen=maxlen, postfix=postfix)

#     ds_timechunk = datasets.DatasetDict({
#         'train': datasets.Dataset.from_pandas(df_train.drop(columns=['chat_session','toklen']).rename(columns={'formatted_text':'text'}), split='train', preserve_index=False),
#         'validation': datasets.Dataset.from_pandas(df_eval.drop(columns=['chat_session','toklen']).rename(columns={'formatted_text':'text'}), split='validation', preserve_index=False)
#     })
#     if not text_only:
#         ds_timechunk = ds_timechunk.map(lambda s: tokenizer(s['text'], return_special_tokens_mask=True, max_length=maxlen,  return_length=True, truncation=False), batched=True)
#     # ds_timechunk = ds_timechunk.map(lambda s: tokenizer(s['text'], return_special_tokens_mask=True, max_length=maxlen, return_overflowing_tokens=True, return_length=True, truncation=True), 
#     #                             batched=True, remove_columns=ds_timechunk['train'].column_names)
    
#     return ds_timechunk

# def to_instruct_format(dataset, postfix, prompt, append_first_to_prompt=False, has_system=False):
#     messages = dataset['text'].split(postfix)
#     rolecycle = itertools.cycle(['user','assistant'])

#     if has_system:
#         chat_content = [{"role": "system", "content": prompt}]
        
#     else:
#         pcontent = prompt
#         if append_first_to_prompt:
#             pcontent+=messages.pop(0)
#         chat_content = [{"role": next(rolecycle), "content": pcontent}]

#     for msg in filter(None,messages):
#         chat_content.append({"role": next(rolecycle), "content": msg})
    
#     return chat_content

# def instruct_dataset_timechunks(chat_csv, tokenizer, cfg, has_system=None, ):
#     # TODO: make dataset where [INST] AuthorName [/INST] *response*. Might also be able to include seed text.
#     # First instruction would be roughly the same.
#     # but all subsequent will just be some form of username, first name
#     # WHY???
#     # Then can use the DataCollatorForCompletionOnlyLM and not worry about prompt effecting loss 
    
#     maxlen= cfg.chunk_size
#     postfix=cfg.postfix
#     append_first_to_prompt=cfg.prompt.append_msg

#     truncation=cfg.dataset.allow_truncation
#     fprompt = cfg.fprompt

#     prompt_length = tokenizer(fprompt, return_length=True).length[0]

#     if has_system is None:
#         has_system = check_system(tokenizer)
    
    
#     dataset = dataset_timechunk(chat_csv, tokenizer, cfg, text_only=True)
#     dataset = dataset.map(lambda s: {'text': tokenizer.apply_chat_template(to_instruct_format(s, postfix, fprompt, append_first_to_prompt, has_system), tokenize=False)})
#     dataset = dataset.map(lambda s: tokenizer(s['text'], return_special_tokens_mask=True, max_length=maxlen,  return_length=True, truncation=truncation), batched=True)
#     return dataset
