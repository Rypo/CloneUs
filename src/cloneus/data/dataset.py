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


def batched_token_count(convo:list[list[dict]]|list[dict[str,str]], tokenizer:PreTrainedTokenizerFast) -> np.ndarray[int]:
    '''Returns an array of approximate token lens for all items in a list. Each will be treated as it's own batch if not already batched.
    
    [{"role":...,"content":... }, {"role":...,"content":... }, ] will become [[{"role":...,"content":... }], [{"role":...,"content":... }], ]
    [[{"role":...,"content":... }], [{"role":...,"content":... }], ] will be left as is
    '''
    # This will throw off sum lengths by 1*n_messages when applying batching to a single conversation
    # But, on the otherhand, this is exactly what would happen if {{bos_token}} was in the chat_template... so it's consistent I guess? 
    batched_convo = convo[:]
    bos_discount = 0
    if isinstance(convo[0], dict):
        # NOTE: this will break if mistral format. Because we are treating each message like it's own conversation, the roles will be wrong. 
        # batched_convo = [[c] if isinstance(c, dict) else c for c in convo]
        batched_convo = [[c] for c in convo]
        bos_discount = 1
        #print('treating each item as own convo')
    return tokenizer(tokenizer.apply_chat_template(batched_convo, tokenize=False, add_generation_prompt=False), add_special_tokens=False, return_length=True, return_tensors='np')['length'] - bos_discount


def add_sys_msg(chat_convo:list[dict], system_msg:list[dict[str,str]], tag_placement:typing.Literal['tag_only', 'content_prefix', 'replace_role'],):
    if isinstance(system_msg, str):
        raise TypeError('No more loosey goosey. dict or list[dict, dict] only')
        #system_msg = {'role':'system', 'content':system_msg}

    # TODO: Should tag_only accept a system message? Foundation models don't really do that, but I guess it's not gonna break anything
    if tag_placement in [ 'replace_role', 'tag_only',]:
        assert len(system_msg) == 1 and system_msg[0]['role'] == 'system', 'Only tag_placement="content_prefix" can have system message with role != system'
        chat_convo = system_msg+chat_convo
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
    if len(system_msg) == 2: 
         # [{"role": "user", "content": fprompt}), {"role": "assistant", "content": "OK"}]
        # assert len(system_msg) == 2,'SYN ACK format must be 2 messages exactly'
        assert system_msg[0]['role'] == 'user', 'SYN ACK format must be role0=user'
        assert system_msg[1]['role'] == 'assistant', 'SYN ACK format must be role1=assistant'
        chat_convo = system_msg + chat_convo
    
    elif system_msg[0]['role'] == 'system':  # has_system = True
        #system_msg = [system_msg]
        chat_convo = system_msg + chat_convo
    
    else:  # append_msg = 'legacy'
        #rolecycle = itertools.cycle(['assistant', 'user']) # NOTE: Reversed because need to take
        assert system_msg[0]['role'] == 'user', 'System message role must be user for systemless content prefix'
        msg0 = chat_convo[0]
        chat_convo = chat_convo[1:]
        # make sure first iter starts on assistant now
        system_msg = [{'role': msg0['role'], 'content': system_msg[0]['content'] + msg0['content']}]
        
        chat_convo = system_msg + chat_convo
        

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
        name_mapping = ', '.join(formatted_author_tag_col.str.strip().unique())
        cfg.prompt.name_mapping = name_mapping
    
    if fprompt is None:
        fprompt = cfg.prompt.template.format(name_mapping=name_mapping, task=cfg.prompt.task)
        cfg.fprompt = fprompt
    
    return cfg

def prepare_system_msg(cfg, tokenizer):
    if not cfg.fprompt and cfg.prompt.template:
        raise RuntimeError('Prompt not formatted. Call `fill_cfg_from_data(formatted_author_tag, cfg)` before proceeding.')
    
    system_message = [{'role':'system', 'content': cfg.fprompt}]

    has_system = True
    append_msg = None
    
    if cfg.tag_placement == 'tag_only':
        tag_chat_template = to_jinja_template(cfg.tag_sep, cfg.postfix)
        if tokenizer.chat_template is not None and tokenizer.chat_template != tag_chat_template:
            print('WARNING: existing chat_template will be overwritten with one created from tag_sep + postfix. '
                'This dataset type only recommended for foundation model tuning.')
        tokenizer.chat_template = tag_chat_template
    
    elif cfg.tag_placement == 'replace_role':
        system_message = [{'role':'system', 'content': cfg.fprompt}]
    
    elif cfg.tag_placement == 'content_prefix':
        has_system = check_if_system(tokenizer)
        append_msg = cfg.prompt.append_msg
        
        if has_system:
            system_message = [{'role':'system', 'content': cfg.fprompt}]
        
        elif cfg.prompt.append_msg == 'legacy':
            system_message = [{'role':'user', 'content': cfg.fprompt}]
            # legacy handling done by add_sys_message since requires first chat message
        elif cfg.prompt.append_msg:
            system_message = [
                {'role':'user', 'content': cfg.fprompt}, 
                {'role':'assistant', 'content': cfg.prompt.append_msg}]


    return system_message, has_system, append_msg




def chat_lull_idx(time_gaps:list[float], tail_strip:float=0.05, top_one:bool=True):
    tg_asort = np.argsort(time_gaps)
    if not tail_strip or tg_asort.size < 1/tail_strip:
        cand_inds = tg_asort
    else:
        ntrim = int(tail_strip*tg_asort.size)
        cand_inds = tg_asort[(ntrim<tg_asort) & (tg_asort<tg_asort.size-ntrim)]
    if top_one:
        return cand_inds[-1:]
    
    return cand_inds[::-1]
    
def greedy_time_split_overlength(prepared_conversations:list[list[dict]], convo_time_gaps:list[list[float]], tokenizer, system_msg, tag_placement, max_length):
    '''Slower version of time_split. Always used the longest time gap for splitting. 
    
    Runs multiple iterations until complete. May split one batch in to many small sub batches depending on how sparse the conversation is.'''
    # intermission, latency, inactivity, idle time, lull, elapsed, lapse, timeout, interval, interlude, downtime.. why naming so hard
    original_n_convos = len(prepared_conversations)

    convo_lens = batched_token_count(prepared_conversations, tokenizer)
    overlen_indices = np.flatnonzero(convo_lens > max_length)
    new_conversations = []
    new_time_gaps = []

    if overlen_indices.size > 0:
        print(f'Splitting {overlen_indices.size} conversations with tokens > {max_length} using max intermission time.')
        assert all([len(prepared_conversations[i]) > 1 for i in overlen_indices]), 'At least one message+system > max_length tokens. Truncation required.'

    for i in range(original_n_convos):
        if i not in overlen_indices:
            new_conversations.append(prepared_conversations[i])
            new_time_gaps.append(convo_time_gaps[i])
            continue

        orig_length = convo_lens[i]
        excess_tokens = (orig_length-max_length)
        
        c_over = prepared_conversations[i]
        t_over = convo_time_gaps[i]
        
        h = chat_lull_idx(t_over, tail_strip=0.05, top_one=True)[0]
        
        c_left = c_over[:h]
        c_right = add_sys_msg(copy.deepcopy(c_over[h:]), system_msg, tag_placement=tag_placement)
        
        t_left = t_over[:h]
        t_right= t_over[h:] 
        # might change anywhere from 0-2 new messages depending on append_msg/tag_placement
        zpad = [0.0]*(len(c_right) - len(t_right))
        t_right = zpad + t_right # prepend 0.0s for added system message
        
        #tok_left, tok_right = batched_token_count([c_left, c_right], tokenizer) 
        
        new_conversations.extend([c_left, c_right])
        new_time_gaps.extend([t_left, t_right])
        
    
    convo_lens = batched_token_count(new_conversations, tokenizer)
    overlen_indices = np.flatnonzero(convo_lens > max_length)
    
    if overlen_indices.size > 0:
        print('remaining over length:', convo_lens[overlen_indices], 'idx:',overlen_indices)
        return greedy_time_split_overlength(new_conversations, new_time_gaps, tokenizer, system_msg, tag_placement, max_length)
        
    return new_conversations


def top_time_split_indices(cuml_tokens:np.ndarray[int], time_gaps:list[float], excess_tokens:int, max_length:int, ):
    # if convo_length_start is None:
    #     convo_length_start = cuml_tokens[-1]
    # excess_tokens = (convo_length_start-max_length)

    valid_mask = (cuml_tokens > excess_tokens) & (cuml_tokens < max_length) # create a mask s.t. True if it partitions into 2 sides where both < max_tokens
    time_gap_order = np.argsort(time_gaps) # sort index from smallest to largest time gap
    valid_mask_sorted = valid_mask[time_gap_order] # get mask in argsorted order by time gaps
    splitable_indices = time_gap_order[valid_mask_sorted] # filter indices down to only those that can partion both sides < max_length while also preserving sort order
    cand_splits = splitable_indices[::-1] # reverse order so longest gaps are first in array

    # depending on system message length, this should reduce the search down to a few iterations

    return cand_splits

def time_split_overlength(prepared_conversations:list[list[dict]], convo_time_gaps:list[list[float]], tokenizer, system_msg, tag_placement, max_length):
    '''Split on the longest time gap that successfully partitions both sides to be under max_length. 
    
    Each over length batch is split into exactly 2 sub batches if possible, 
    otherwise if it exceeds 2*`max_length` it will be split again via a recursive call.
    
    Args:
        prepared_conversations: list of conversation with system message included for each
    '''
    # intermission, latency, inactivity, idle time, lull, elapsed, lapse, timeout, interval, interlude, downtime.. why naming so hard
    original_n_convos = len(prepared_conversations)

    convo_lens = batched_token_count(prepared_conversations, tokenizer)
    overlen_indices = np.flatnonzero(convo_lens > max_length)
    new_conversations = []
    new_time_gaps = []

    if overlen_indices.size > 0:
        print(f'Splitting {overlen_indices.size} conversations with tokens > {max_length} using max intermission time.')
        assert all([len(prepared_conversations[i]) > 1 for i in overlen_indices]), 'At least one message+system > max_length tokens. Truncation required.'
    
   # pbar := tqdm(range(original_n_batches), leave=None)
    for i in (pbar := tqdm(range(original_n_convos), leave=False)):
        if i not in overlen_indices:
            new_conversations.append(prepared_conversations[i])
            new_time_gaps.append(convo_time_gaps[i])
            continue

        orig_length = convo_lens[i]
        
        c_over = prepared_conversations[i]
        t_over = convo_time_gaps[i]
        
        excess_tokens = (orig_length-max_length)
        top_only = excess_tokens > max_length # if it's going to require multiple runs anyway, no sense in iterating over everything
        try:
            assert not top_only, 'Multi-iter needed'
            # need try-catch for jinja exceptions since will be treating each role+content as it's own convo
            cuml_tokens = batched_token_count(c_over, tokenizer).cumsum()
            cand_splits = top_time_split_indices(cuml_tokens, t_over, excess_tokens, max_length)
            # c_over_toksum = cuml_tokens[-1]
            #print(f'{orig_length=} | {c_over_toksum=}')
        except Exception as e:
            cand_splits = chat_lull_idx(t_over, tail_strip=0.05, top_one=top_only)

        for h in cand_splits:
            c_left = c_over[:h]
            c_right = add_sys_msg(copy.deepcopy(c_over[h:]), system_msg, tag_placement=tag_placement)
            
            tok_split = batched_token_count([c_left, c_right], tokenizer) 
            
            if (tok_split <= max_length).all() or cand_splits.size==1:
                t_left = t_over[:h]
                t_right= t_over[h:] 
                # might change anywhere from 0-2 new messages depending on append_msg/tag_placement
                zpad = [0.0]*(len(c_right) - len(t_right))
                t_right = zpad + t_right # prepend 0.0s for added system message, only matters if need to do another iteration
                
                new_conversations.append(c_left)
                new_conversations.append(c_right)

                new_time_gaps.append(t_left)
                new_time_gaps.append(t_right)

                pbar.set_description(f'{orig_length} -> {tok_split} (Δ={tok_split.sum()-orig_length})')
                pbar.set_postfix({'n_convo': len(new_conversations)})
                #print(orig_length, '->', tok_split, 'sum:', tok_split.sum(),  f'n_batch: {len(new_batched_sys_convos)} / {original_n_batches}')

                break
    
    pbar.close()
    convo_lens = batched_token_count(new_conversations, tokenizer)
    overlen_indices = np.flatnonzero(convo_lens > max_length)
    
    if overlen_indices.size > 0:
        print('remaining over length:', convo_lens[overlen_indices], 'idx:',overlen_indices)
        return time_split_overlength(new_conversations, new_time_gaps, tokenizer, system_msg, tag_placement, max_length)

    
    return new_conversations

def dedupe_conversations(conversations:list[list[dict]]):
    unique_convos = []
    for convo in conversations:
        if convo not in unique_convos:
            unique_convos.append(convo)
    return unique_convos

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
    # prepend from 0 to 2 0.0 for inserted system message(s)
    df_convo['intrn_time_gap'] = (df_convo['conversation'].str.len() - df_convo['intrn_time_gap'].str.len()).apply(lambda zpad: [0.0]*zpad) + df_convo['intrn_time_gap']

    # eval_convo = df_convo.loc['eval']
    # train_convo = df_convo.loc['train']

    eval_convos = time_split_overlength(df_convo.loc['eval','conversation'].tolist(), df_convo.loc['eval','intrn_time_gap'].tolist(), 
                                        tokenizer, system_msg=system_message, tag_placement=cfg.tag_placement, max_length=cfg.chunk_size)
    train_convos = time_split_overlength(df_convo.loc['train','conversation'].tolist(), df_convo.loc['train','intrn_time_gap'].tolist(), 
                                         tokenizer, system_msg=system_message, tag_placement=cfg.tag_placement, max_length=cfg.chunk_size)
        
    
    dset = datasets.DatasetDict({
        'train': datasets.Dataset.from_dict({'text': dedupe_conversations(train_convos)}, split='train'),
        'validation': datasets.Dataset.from_dict({'text': dedupe_conversations(eval_convos)}, split='validation'),
    })
    dset = dset.map(lambda x: {"text": tokenizer.apply_chat_template(x["text"], tokenize=False, add_generation_prompt=False)})
    
    if not text_only:
        dset = map_to_inputs(dset, tokenizer, max_length=cfg.chunk_size, truncation=cfg.dataset.allow_truncation)

    return dset


def bisect_overlength(prepared_conversations, tokenizer, system_msg, tag_placement, max_length):
    original_n_convos = len(prepared_conversations)

    convo_lens = batched_token_count(prepared_conversations, tokenizer)
    overlen_indices = np.flatnonzero(convo_lens > max_length)
    new_conversations = []
    
    if overlen_indices.size > 0:
        print(f'Splitting {overlen_indices.size} conversations with tokens > {max_length} in half.')
        assert all([len(prepared_conversations[i]) > 1 for i in overlen_indices]), 'At least one message+system > max_length tokens. Truncation required.'
        if overlen_indices.size == 1:
            i = overlen_indices[0]
            print(len(prepared_conversations[i]))
            print(batched_token_count(prepared_conversations[i], tokenizer))

    
    for i in (pbar := tqdm(range(original_n_convos), leave=False)):
        if i not in overlen_indices:
            new_conversations.append(prepared_conversations[i])
            continue
    
        orig_length = convo_lens[i]
        c_over = prepared_conversations[i]    
        #excess_tokens = (orig_length-max_length)
        #top_only = excess_tokens > max_length # if it's going to require multiple runs anyway, no sense in iterating over everything
        try:
            cuml_tokens = batched_token_count(c_over, tokenizer).cumsum()
            #orig_length = cuml_tokens[-1]
            #cand_inds = np.abs(cuml_tokens-(max_length//2)).argsort() # .argmin()
            h = np.abs(cuml_tokens-(max_length//2)).argmin()
            # nearest to balanced tokens on both sides of split
            # for h in cand_inds:
            #     if h >= 3 or cuml_tokens.size-h >= 3: 
            #         break # need at least 3 to have a mid idx to split
           
        except Exception as e:
            #print(e)
            h = len(c_over)//2 # fall back is just cut the convo in half
        
        c_left = c_over[:h]
        c_right = add_sys_msg(copy.deepcopy(c_over[h:]), system_msg, tag_placement=tag_placement)
        new_conversations.extend([c_left, c_right])
        
        tok_split = batched_token_count([c_left, c_right], tokenizer) 

        pbar.set_description(f'{orig_length} -> {tok_split} (Δ={tok_split.sum()-orig_length})')
        pbar.set_postfix({'n_convo': len(new_conversations)})
        #print(orig_length, '->', tok_split, 'sum:', tok_split.sum(),)

    pbar.close()
    convo_lens = batched_token_count(new_conversations, tokenizer)
    overlen_indices = np.flatnonzero(convo_lens > max_length)
    
    if overlen_indices.size > 0:
        print('remaining over length:', convo_lens[overlen_indices], 'idx:',overlen_indices)
        return bisect_overlength(new_conversations, tokenizer, system_msg, tag_placement, max_length)
   
    return new_conversations

def consecutive_max_tokens(unified_conversation:list[dict[str,str]], all_msg_lens:np.ndarray[int], token_limit:int) -> list[list[dict[str,str]]]:
    conversations = []

    convo = []
    convo_tokenlen = 0
    for message, msg_tokenlen in zip(unified_conversation, all_msg_lens):
    
        if convo_tokenlen + msg_tokenlen >= token_limit:
            conversations.append(convo)
            convo = [message]
            convo_tokenlen = msg_tokenlen
        else:
            convo.append(message)
            convo_tokenlen += msg_tokenlen
    
    return conversations

def convo_batch_max_tokens(unified_conversation:list[dict[str,str]], tokenizer:PreTrainedTokenizerFast, system_msg: dict[str, str] | list[dict[str, str]], tag_placement:typing.Literal['tag_only', 'content_prefix', 'replace_role'], max_length:int):
    
    # TODO: this will break for append_msg content_prefix
    syslen = batched_token_count(system_msg, tokenizer).sum() # sum in case is SYN ACK
    all_msg_lens = batched_token_count(unified_conversation, tokenizer)
    
    token_limit = (max_length-syslen)
    # TODO: allow overlength for later truncation
    assert all_msg_lens.max() <= token_limit,'At least one message+system > max_length tokens. Truncation required.'
    
    # use token_limit instead of max_length because we haven't added system messages yet
    max_token_conversations = consecutive_max_tokens(unified_conversation, all_msg_lens, token_limit)

    prepared_conversations = [add_sys_msg(c, system_msg, tag_placement=tag_placement) for c in max_token_conversations]
    # Now use max_length, not token_limit, since system has been added
    prepared_conversations = bisect_overlength(prepared_conversations, tokenizer, system_msg, tag_placement, max_length=max_length) 
    
    return prepared_conversations


def max_tokens_dataset(chat_csv, tokenizer, cfg, text_only=False):
    '''Dataset groupings of sequential texts concatenated up to a maximum of `cfg.chunk_size` total tokens
    
    Chat sessions are not assigned, and the only use of Date or timestamp if present is consecutive message merge.
    '''
    df_proc = etl.process_csv(chat_csv, youtube_encode_fetch=True, filter_prefixes=('!', '/'), merge_window=7.0)
    df_all = etl.format_text_tags(df_proc, author_tag=cfg.author_tag,  tag_sep=cfg.tag_sep, postfix=cfg.postfix,  eval_frac=cfg.dataset.get('eval_frac',0.01))
    
    cfg = fill_cfg_from_data(df_all['formatted_author_tag'], cfg) # fill fprompt, name_mappings
    system_message, has_system, append_msg = prepare_system_msg(cfg, tokenizer) # may update tokenizer

    # get base tokens before any system is added
    # Do not add system message since it is a flat list of messages as a single mega conversation
    sr_flat_convo = df_all.groupby('split')[['formatted_author_tag', 'text']].agg(list).apply(lambda r: to_conversation_format(r.formatted_author_tag, r.text, 'replace_role'), axis=1)
    
    train_convos = convo_batch_max_tokens(sr_flat_convo['train'], tokenizer, system_message, cfg.tag_placement, max_length=cfg.chunk_size)
    eval_convos = convo_batch_max_tokens(sr_flat_convo['eval'], tokenizer, system_message, cfg.tag_placement, max_length=cfg.chunk_size)
    
    dset = datasets.DatasetDict({
        'train': datasets.Dataset.from_dict({'text': train_convos}, split='train'),
        'validation': datasets.Dataset.from_dict({'text': eval_convos}, split='validation'),
    })
    
    dset = dset.map(lambda x: {"text": tokenizer.apply_chat_template(x["text"], tokenize=False, add_generation_prompt=False)})
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
    #dset = dset.map(lambda x: {"text": tokenizer.apply_chat_template(x["text"], tokenize=False, add_generation_prompt=False)})
    if not text_only:
        ds_ungrouped = map_to_inputs(ds_ungrouped, tokenizer, max_length=cfg.chunk_size, truncation=cfg.dataset.allow_truncation, text_field='text')
    
    return ds_ungrouped