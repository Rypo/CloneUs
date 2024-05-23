import copy
import typing
import itertools
import functools
from dataclasses import dataclass


import more_itertools
import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange
import datasets
from transformers import PreTrainedTokenizerFast

from ..plugins import youtube as youtube
#from ..core import paths as rpaths
from . import etl
from .tokenization import check_if_system, to_jinja_template

def map_to_inputs(dset:datasets.Dataset, tokenizer:PreTrainedTokenizerFast, max_length:int, truncation:bool, text_field='text'):
    '''Maps Dataset text field to those for model input (input_ids, special_tokens_mask, length)
    
    Importantly, it does NOT add special tokens. This is primarily a concern after calling apply_chat_template(tokenize=False) which,
    depending on the template, may or may not insert a bos_token. This will cause double bos_token on all entries. 
    '''
    dset = dset.map(lambda s: tokenizer(s[text_field], add_special_tokens=False, return_special_tokens_mask=True, max_length=max_length, return_length=True, truncation=truncation), batched=True)
    return dset

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


def batched_token_count(convo:list[list[dict]]|list[dict[str,str]], tokenizer:PreTrainedTokenizerFast, ) -> np.ndarray[int]:
    '''Returns an array of approximate token lens for all items in a list. Each will be treated as it's own batch if not already batched.
    
    [{"role":...,"content":... }, {"role":...,"content":... }, ] will become [[{"role":...,"content":... }], [{"role":...,"content":... }], ]
    [[{"role":...,"content":... }], [{"role":...,"content":... }], ] will be left as is
    '''
    
    batched_convo = [[c] if isinstance(c, dict) else c for c in convo]
    return tokenizer(tokenizer.apply_chat_template(batched_convo, tokenize=False, add_generation_prompt=False), add_special_tokens=False, return_length=True, return_tensors='np')['length']


def add_sys_msg(chat_convo:list[dict], system_msg:dict[str,str]|list[dict[str,str]], tag_placement:typing.Literal['tag_only', 'content_prefix', 'replace_role'],):
    if isinstance(system_msg, str):
        raise TypeError('No more loosey goosey. dict or list[dict, dict] only')
        #system_msg = {'role':'system', 'content':system_msg}

    # TODO: Should tag_only accept a system message? Foundation models don't really do that, but I guess it's not gonna break anything
    if tag_placement in [ 'replace_role', 'tag_only',]:
        assert system_msg['role'] == 'system', 'Only tag_placement="content_prefix" can have system message with role != system'
        chat_convo = [system_msg]+chat_convo
        return chat_convo

    # Below here, roles are overwritten. Custom roles should never reach this point in code 
    # because if roles are customizable then it follows that a role can = "system".
    

    # because splits can occur anywhere, it may more may not be assistant. So need to reassign roles to prevent issue.
    if chat_convo[0]['role'] != 'user':
        rolecycle = itertools.cycle(['user','assistant'])
        for c in chat_convo:
            c['role'] = next(rolecycle)
    # content_prefix
    # prompt.append_msg: str != legacy
    if isinstance(system_msg, list): 
         # [{"role": "user", "content": fprompt}), {"role": "assistant", "content": "OK"}]
        assert len(system_msg) == 2,'SYN ACK format must be 2 messages exactly'
        assert system_msg[0]['role'] == 'user', 'SYN ACK format must be role0=user'
        assert system_msg[1]['role'] == 'assistant', 'SYN ACK format must be role1=assistant'
        chat_convo = system_msg + chat_convo
    
    elif system_msg['role'] == 'system':  # has_system = True
        #system_msg = [system_msg]
        chat_convo = [system_msg] + chat_convo
    
    else:  # append_msg = 'legacy'
        #rolecycle = itertools.cycle(['assistant', 'user']) # NOTE: Reversed because need to take
        assert system_msg['role'] == 'user', 'System message role must be user for systemless content prefix'
        msg0 = chat_convo[0]
        chat_convo = chat_convo[1:]
        # make sure first iter starts on assistant now
        system_msg = [{'role': msg0['role'], 'content': system_msg['content'] + msg0['content']}]
        
        chat_convo = system_msg + chat_convo
        #for c in chat_convo:
        #    c['role'] = next(rolecycle)

    #chat_convo = system_msg + chat_convo
        

    return chat_convo


def to_conversation_format(formatted_author_tags:list[str], raw_texts:list[str], tag_placement:typing.Literal['tag_only', 'content_prefix', 'replace_role'],  system_msg:dict[str,str]|list[dict[str,str]]|None=None) -> list[dict[str,str]]: # has_system: bool=None, append_msg:bool=None
    # NOTE: formatted_author_tags INCLUDE TAG SEP (if not None)
    # https://github.com/benfred/py-spy
    atag_tcontent = zip(formatted_author_tags, raw_texts)
    chat_content = []
    
    if system_msg is None:
        if tag_placement == 'content_prefix':
            rolecycle = itertools.cycle(['user','assistant'])

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
        #if tag_placement == 'replace_role':
        #chat_content.append({"role": "system", "content": system_msg})
        
        for role_tag,content in atag_tcontent:
            chat_content.append({"role": role_tag, "content": content})
        

        return add_sys_msg(chat_content, system_msg=system_msg, tag_placement=tag_placement)
    
    elif tag_placement == 'content_prefix':
        rolecycle = itertools.cycle(['user','assistant'])
            
        # assert append_msg is not None, 'content_prefix requires append_msg set True/False'
        # assert has_system is not None, 'content_prefix requires has_system set True/False'
        
        # if has_system:
        #     chat_content.append({"role": "system", "content": system_msg})
        # elif append_msg=='legacy':
        #     tag0, text0 = next(atag_tcontent)
        #     content0 = tag0 + text0 
        #     # make sure first iter starts on assistant now
        #     chat_content.append({"role": next(rolecycle), "content": system_msg+content0})
        # elif append_msg:
        #     chat_content.append({"role": "user", "content": system_msg})
        #     chat_content.append({"role": "assistant", "content": "OK"})
            
        for fauth_tag,text in atag_tcontent:
            content = fauth_tag + text
            chat_content.append({"role": next(rolecycle), "content": content})

        return add_sys_msg(chat_content, system_msg=system_msg, tag_placement=tag_placement)
    
    else:
        raise ValueError('unknown tag_placement value: '+tag_placement)



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

def prepare_system_msg(cfg, tokenizer):
    if not cfg.fprompt and cfg.prompt.template:
        raise RuntimeError('Prompt not formatted. Call `fill_cfg_from_data(formatted_author_tag, cfg)` before proceeding.')
    
    system_message = {'role':'system', 'content': cfg.fprompt}

    has_system = True
    append_msg = None
    
    if cfg.tag_placement == 'tag_only':
        tag_chat_template = to_jinja_template(cfg.tag_sep, cfg.postfix)
        if tokenizer.chat_template is not None and tokenizer.chat_template != tag_chat_template:
            print('WARNING: existing chat_template will be overwritten with one created from tag_sep + postfix. '
                'This dataset type only recommended for foundation model tuning.')
        tokenizer.chat_template = tag_chat_template
    
    elif cfg.tag_placement == 'replace_role':
        system_message = {'role':'system', 'content': cfg.fprompt}
    
    elif cfg.tag_placement == 'content_prefix':
        has_system = check_if_system(tokenizer)
        append_msg = cfg.prompt.append_msg
        
        if has_system:
            system_message = {'role':'system', 'content': cfg.fprompt}
        
        elif cfg.prompt.append_msg == 'legacy':
            system_message = {'role':'user', 'content': cfg.fprompt}
            # legacy handling done by add_sys_message since requires first chat message
        elif cfg.prompt.append_msg:
            system_message = [
                {'role':'user', 'content': cfg.fprompt}, 
                {'role':'assistant', 'content': cfg.prompt.append_msg}]


    return system_message, has_system, append_msg




def chat_lull_idx(time_gaps:list[float], tail_strip:float=0.05, top_one:bool=True):
    tg_asort = np.argsort(time_gaps)
    if tg_asort.size < 1/tail_strip:
        cand_inds = tg_asort
    else:
        ntrim = int(tail_strip*tg_asort.size)
        cand_inds = tg_asort[(ntrim<tg_asort) & (tg_asort<tg_asort.size-ntrim)]
    if top_one:
        return cand_inds[-1]
    
    return cand_inds[::-1]
    
def exact_time_split_overlength(batched_sys_convos:list[list[dict]], batched_time_gaps:list[list[float]], tokenizer, system_msg, tag_placement, max_length):
    '''Slower version of time_split. Always used the longest time gap for splitting. 
    
    Runs multiple iterations until complete. May split one batch in to many more than 2 sub-batches.'''
    # intermission, latency, inactivity, idle time, lull, elapsed, lapse, timeout, interval, interlude, downtime.. why naming so hard
    convo_lens = batched_token_count(batched_sys_convos, tokenizer)
    overlen_indices = np.flatnonzero(convo_lens > max_length)
    if overlen_indices.size > 0:
        print(f'Splitting {overlen_indices.size} conversations with tokens > {max_length} using max intermission time.')
        assert all([len(batched_sys_convos[i]) > 1 for i in overlen_indices]), 'At least one message+system > max_length tokens. Truncation required.'

    while overlen_indices.size > 0:
        for offset,i in enumerate(overlen_indices):
            i+=offset
            c_over = batched_sys_convos.pop(i)
            t_over = batched_time_gaps.pop(i)
            h = chat_lull_idx(t_over, tail_strip=0.05)
            c_left = c_over[:h]
            c_right = add_sys_msg(c_over[h:], system_msg, tag_placement=tag_placement)
            
            t_left = t_over[:h]
            t_right = [0.0]+t_over[h:] # prepend 0.0 for added system message
            
            # right first, then left since it inserts *before* index. Alt would be left, right@i+1
            batched_sys_convos.insert(i, c_right)
            batched_time_gaps.insert(i, t_right)
            
            batched_sys_convos.insert(i, c_left)
            batched_time_gaps.insert(i, t_left)
    
        convo_lens = batched_token_count(batched_sys_convos, tokenizer)
        overlen_indices = np.flatnonzero(convo_lens > max_length)
        if overlen_indices.size == 1:
            assert len(batched_sys_convos[overlen_indices[0]]) > 1, 'At least one message+system > max_length tokens. Truncation required.'
        
        print('lens:',convo_lens[overlen_indices]) # 'idx:',overlen_indices, 
    
    return batched_sys_convos

# def time_split_overlength(batched_sys_convos:list[list[dict]], batched_time_gaps:list[list[float]], tokenizer, system_msg, tag_placement, max_length):
#     '''Split on the longest time gap that successfully partitions both sides to be under max_length. 
    
#     Each over length batch is split into exactly 2 sub batches.'''
#     # intermission, latency, inactivity, idle time, lull, elapsed, lapse, timeout, interval, interlude, downtime.. why naming so hard
#     convo_lens = batched_token_count(batched_sys_convos, tokenizer)
#     overlen_indices = np.flatnonzero(convo_lens > max_length)
#     new_batched_sys_convos = []
#     if overlen_indices.size > 0:
#         print(f'Splitting {overlen_indices.size} conversations with tokens > {max_length} using max intermission time.')
#         assert all([len(batched_sys_convos[i]) > 1 for i in overlen_indices]), 'At least one message+system > max_length tokens. Truncation required.'
    
        
#         for offset,i in enumerate(overlen_indices):
#             orig_length = convo_lens[i]
#             i += offset
#             c_over = batched_sys_convos.pop(i) #overlen_sys_convos[i]
#             t_over = batched_time_gaps.pop(i)#overlen_time_gaps[i]
#             cand_splits = chat_lull_idx(t_over, tail_strip=0.02, top_one=False)
#             for h in cand_splits:
#                 c_left = c_over[:h]
#                 c_right = add_sys_msg(c_over[h:], system_msg, tag_placement=tag_placement)
#                 n_sys = len(c_right)-len(c_over[h:]) # might change anywhere from 0-2 new messages depending on append_msg/tag_placement
#                 tok_split = batched_token_count([c_left, c_right], tokenizer) 
#                 if (tok_split <= max_length).all():
#                     t_left = t_over[:h]
#                     t_right = ([0.0]*n_sys) + t_over[h:] # prepend 0.0 for added system message
#                     print(orig_length, '->', tok_split, 'sum', tok_split.sum(), 'n_batch', len(batched_sys_convos), 'n_sys', n_sys)
#                     # right first, then left since it inserts *before* index. Alt would be left, right@i+1
#                     batched_sys_convos.insert(i, c_right)
#                     batched_time_gaps.insert(i, t_right)
                    
#                     batched_sys_convos.insert(i, c_left)
#                     batched_time_gaps.insert(i, t_left)
#                     print('nbatch after insert', len(batched_sys_convos))
#                     break
        
#         convo_lens = batched_token_count(batched_sys_convos, tokenizer)
#         overlen_indices = np.flatnonzero(convo_lens > max_length)
        
#         if overlen_indices.size > 0:
#             print('remaining over length:', convo_lens[overlen_indices], 'idx:',overlen_indices)
#             return time_split_overlength(batched_sys_convos, batched_time_gaps, tokenizer, system_msg, tag_placement, max_length)

    
#     return batched_sys_convos

def time_split_overlength(batched_sys_convos:list[list[dict]], batched_time_gaps:list[list[float]], tokenizer, system_msg, tag_placement, max_length):
    '''Split on the longest time gap that successfully partitions both sides to be under max_length. 
    
    Each over length batch is split into exactly 2 sub batches.'''
    # intermission, latency, inactivity, idle time, lull, elapsed, lapse, timeout, interval, interlude, downtime.. why naming so hard
    original_n_batches = len(batched_sys_convos)

    convo_lens = batched_token_count(batched_sys_convos, tokenizer)
    overlen_indices = np.flatnonzero(convo_lens > max_length)
    new_batched_sys_convos = []
    new_batched_time_gaps = []

    if overlen_indices.size > 0:
        print(f'Splitting {overlen_indices.size} conversations with tokens > {max_length} using max intermission time.')
        assert all([len(batched_sys_convos[i]) > 1 for i in overlen_indices]), 'At least one message+system > max_length tokens. Truncation required.'
    
    for i in range(original_n_batches):
        if i not in overlen_indices:
            new_batched_sys_convos.append(batched_sys_convos[i])
            new_batched_time_gaps.append(batched_time_gaps[i])
            continue

        orig_length = convo_lens[i]
        
        c_over = batched_sys_convos[i]
        t_over = batched_time_gaps[i]
        cand_splits = chat_lull_idx(t_over, tail_strip=0.02, top_one=False)
        
        for h in cand_splits:
            c_left = c_over[:h]
            c_right = add_sys_msg(copy.deepcopy(c_over[h:]), system_msg, tag_placement=tag_placement)
            
            n_sys = len(c_right)-len(c_over[h:]) # might change anywhere from 0-2 new messages depending on append_msg/tag_placement
            #print('n_sys', n_sys,)
            tok_split = batched_token_count([c_left, c_right], tokenizer) 
            
            if (tok_split <= max_length).all():
                t_left = t_over[:h]
                t_right = ([0.0]*n_sys) + t_over[h:] # prepend 0.0 for added system message
                
                new_batched_sys_convos.append(c_left)
                new_batched_sys_convos.append(c_right)

                new_batched_time_gaps.append(t_left)
                new_batched_time_gaps.append(t_right)

                print(orig_length, '->', tok_split, 'sum:', tok_split.sum(),  f'n_batches: {len(new_batched_sys_convos)} / {original_n_batches}')

                break
    
    convo_lens = batched_token_count(new_batched_sys_convos, tokenizer)
    overlen_indices = np.flatnonzero(convo_lens > max_length)
    
    if overlen_indices.size > 0:
        print('remaining over length:', convo_lens[overlen_indices], 'idx:',overlen_indices)
        return time_split_overlength(new_batched_sys_convos, new_batched_time_gaps, tokenizer, system_msg, tag_placement, max_length)

    
    return new_batched_sys_convos

def chat_sessions_dataset(chat_csv, tokenizer, cfg, text_only=False):
    df_proc= etl.process_csv(chat_csv, youtube_encode_fetch=True, filter_prefixes=('!', '/'), merge_window=7.0)
    df_all = etl.format_text_tags(df_proc, author_tag=cfg.author_tag, tag_sep=cfg.tag_sep, postfix=cfg.postfix, eval_frac=cfg.dataset.get('eval_frac',0.01))
    df_all = etl.label_chat_sessions(df_all, hours_between_sessions=cfg.dataset.hours_between_sessions, min_session_length=cfg.dataset.min_session_length)
    
    cfg = fill_cfg_from_data(df_all['formatted_author_tag'], cfg) # fill fprompt, name_mappings
    system_message, has_system, append_msg = prepare_system_msg(cfg, tokenizer)
    # intrasession_time_gap
    df_all['intrn_time_gap'] = df_all.groupby(['split','chat_session'])['Date'].diff().dt.total_seconds().fillna(0) 
    
    df_rolechat = df_all[['formatted_author_tag','text','split','chat_session', 'intrn_time_gap']].copy()
    df_convo = df_rolechat.groupby(['split','chat_session'])[['formatted_author_tag','text','intrn_time_gap']].agg(list).drop_duplicates('text')


    df_convo['conversation'] = df_convo.apply(lambda r: to_conversation_format(r.formatted_author_tag, r.text, cfg.tag_placement, system_message), axis=1)
    # prepend 0.0 for inserted system message
    df_convo['intrn_time_gap'] = df_convo['intrn_time_gap'].apply(lambda x: [0.0]+x)

    eval_convo = df_convo.loc['eval']
    train_convo = df_convo.loc['train']

    eval_batches = time_split_overlength(eval_convo['conversation'].tolist(), eval_convo['intrn_time_gap'].tolist(), tokenizer, system_msg=system_message, tag_placement=cfg.tag_placement, max_length=cfg.chunk_size)
    train_batches = time_split_overlength(train_convo['conversation'].tolist(), train_convo['intrn_time_gap'].tolist(), tokenizer, system_msg=system_message, tag_placement=cfg.tag_placement, max_length=cfg.chunk_size)
    
    dset = datasets.DatasetDict({
        'train': datasets.Dataset.from_dict({'text': train_batches}, split='train'),
        'validation': datasets.Dataset.from_dict({'text': eval_batches}, split='validation'),
    })
    dset = dset.map(lambda x: {"text": tokenizer.apply_chat_template(x["text"], tokenize=False, add_generation_prompt=False)})
    
    if not text_only:
        dset = map_to_inputs(dset, tokenizer, max_length=cfg.chunk_size, truncation=cfg.dataset.allow_truncation)

    return dset


def bisect_overlength(batched_sys_convos, tokenizer, system_msg, tag_placement, max_length):
    convo_lens = batched_token_count(batched_sys_convos, tokenizer)
    overlen_indices = np.flatnonzero(convo_lens > max_length)
    if overlen_indices.size > 0:
        print(f'Splitting {overlen_indices.size} conversations with tokens > {max_length} in half.')
        assert all([len(batched_sys_convos[i]) > 1 for i in overlen_indices]), 'At least one message+system > max_length tokens. Truncation required.'
    
    while overlen_indices.size > 0:
        for offset,i in enumerate(overlen_indices):
            i+=offset
            b_over = batched_sys_convos.pop(i)
            h = len(b_over)//2
            h_left = b_over[:h]
            h_right = add_sys_msg(b_over[h:], system_msg, tag_placement=tag_placement)
            # right first, then left since it inserts *before* index. Alt would be left, right@i+1
            batched_sys_convos.insert(i, h_right)
            batched_sys_convos.insert(i, h_left)
    
        convo_lens = batched_token_count(batched_sys_convos, tokenizer)
        overlen_indices = np.flatnonzero(convo_lens > max_length)
        
        print('idx:',overlen_indices, 'lens:',convo_lens[overlen_indices])
    
    return batched_sys_convos

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

def convo_batch_max_tokens(convo:list[dict[str,str]], tokenizer:PreTrainedTokenizerFast, system_msg: dict[str, str] | list[dict[str, str]], tag_placement:typing.Literal['tag_only', 'content_prefix', 'replace_role'], max_length:int):
    syslen = batched_token_count([system_msg], tokenizer)[0] #TODO: this will break for append_msg contpfx
    all_msg_lens = batched_token_count(convo, tokenizer)
    
    token_limit = (max_length-syslen)
    assert all_msg_lens.max() <= token_limit,'At least one message+system > max_length tokens. Truncation required.'
    # TODO: allow overlength for later truncation
    convo_batches = consecutive_max_batch(convo, all_msg_lens, token_limit)

    sys_convo_batches = [add_sys_msg(c, system_msg, tag_placement=tag_placement) for c in convo_batches]
    sys_convo_batches = bisect_overlength(sys_convo_batches, tokenizer, system_msg, tag_placement, max_length=max_length) # not token_limit, since system has been added
    

    return sys_convo_batches


def max_tokens_dataset(chat_csv, tokenizer, cfg, text_only=False):
    '''Dataset groupings of sequential texts concatenated up to a maximum of `cfg.chunk_size` total tokens
    
    Chat sessions are not assigned, and the only use of Date or timestamp if present is consecutive message merge.
    '''
    df_proc = etl.process_csv(chat_csv, youtube_encode_fetch=True, filter_prefixes=('!', '/'), merge_window=7.0)
    df_all = etl.format_text_tags(df_proc, author_tag=cfg.author_tag,  tag_sep=cfg.tag_sep, postfix=cfg.postfix,  eval_frac=cfg.dataset.get('eval_frac',0.01))
    
    cfg = fill_cfg_from_data(df_all['formatted_author_tag'], cfg) # fill fprompt, name_mappings
    system_message, has_system, append_msg = prepare_system_msg(cfg, tokenizer) # may update tokenizer

    df_tall = df_all.query('split=="train"')
    df_eall = df_all.query('split=="eval"')
    # get base tokens before any system is added
    convo_train = to_conversation_format(df_tall['formatted_author_tag'], df_tall['text'], tag_placement=cfg.tag_placement)
    convo_eval = to_conversation_format(df_eall['formatted_author_tag'], df_eall['text'], tag_placement=cfg.tag_placement)
    
    dset = datasets.DatasetDict({
        'train': datasets.Dataset.from_dict({'text': convo_batch_max_tokens(convo_train, tokenizer, system_message, cfg.tag_placement, max_length=cfg.chunk_size)}, split='train'),
        'validation': datasets.Dataset.from_dict({'text': convo_batch_max_tokens(convo_eval, tokenizer, system_message, cfg.tag_placement, max_length=cfg.chunk_size)}, split='validation'),
    })
    # https://huggingface.co/learn/nlp-course/chapter5/3
    if not text_only:
        dset = map_to_inputs(dset, tokenizer, max_length=cfg.chunk_size, truncation=cfg.dataset.allow_truncation)

    return dset



def jsonl_dataset(train_jsonl, eval_jsonl, tokenizer, cfg, text_only=False):
    dset = datasets.load_dataset("json", data_files={"train": train_jsonl, "validation": eval_jsonl})
    if not text_only:
        dset = map_to_inputs(dset, tokenizer, max_length=cfg.chunk_size, truncation=cfg.dataset.allow_truncation)
    
    return dset


def ungrouped_dataset(chat_csv, tokenizer, cfg, text_only=False):
    '''Dataset of chat messages without any grouping by time or chunk size. 

    The dataset is constructed from the raw dataframe. No further processing or parsing is preformed.
    The dataset will be split into train and validation and have fields:
      ['Date', 'time_gap', 'formatted_author_tag','text'] 
    that will need to be handled before feeding to a model.
    
    SFTTrainer with packing=True must be used since order is lost through normal batching
    '''
    if not cfg.use_sft_trainer:
        raise ValueError('for dataset "ungrouped_eos" SFTTrainer must be used to preserve message order')
    
    df_proc= etl.process_csv(chat_csv, youtube_encode_fetch=True, filter_prefixes=('!', '/'), merge_window=7.0)
    df_all = etl.format_text_tags(df_proc, author_tag=cfg.author_tag, tag_sep=cfg.tag_sep, postfix=cfg.postfix, eval_frac=cfg.dataset.get('eval_frac',0.01))
    
    cfg = fill_cfg_from_data(df_all['formatted_author_tag'], cfg) # fill fprompt, name_mappings
    system_message, has_system, append_msg = prepare_system_msg(cfg, tokenizer) # may update tokenizer
    
    df_all = df_all[['split', 'Date', 'time_gap', 'formatted_author_tag','text']].set_index('split')
    
    ds_ungrouped = datasets.DatasetDict({
        'train': datasets.Dataset.from_pandas(df_all.loc['train'], split='train', preserve_index=False),
        'validation': datasets.Dataset.from_pandas(df_all.loc['eval'], split='validation', preserve_index=False)
    })
    if not text_only:
        ds_ungrouped = map_to_inputs(ds_ungrouped, tokenizer, max_length=cfg.chunk_size, truncation=cfg.dataset.allow_truncation, text_field='text')
    
    return ds_ungrouped