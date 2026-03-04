import re
import copy
import typing
import dateutil
import itertools
import functools
from dataclasses import dataclass
import json

import more_itertools
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from omegaconf import OmegaConf,DictConfig
import torch
import datasets
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizerBase

from ..plugins import youtube

from . import etl, useridx
from .tokenization import check_if_system, to_jinja_template

def get_roletag_regex_masks(masked_role_tags: str | list[str] = 'system', role_tag_template: str = r'<|im_start|>{role_tag}'):
    '''Helper function to build properly escaped regex mask bounds for use in mask_message_roles'''
    if not isinstance(masked_role_tags, list):
        masked_role_tags = [masked_role_tags]
    
    if len(masked_role_tags) > 1:
        re_esc_roles = [re.escape(role_tag_template.format(role_tag = tag)) for tag in masked_role_tags]
        re_mask_start = '(?:{exclude})'.format(exclude = "|".join(re_esc_roles))
    else:
        re_mask_start = re.escape(role_tag_template.format(role_tag = masked_role_tags[0]))
    
    re_mask_end = re.escape(role_tag_template.format(role_tag = ''))
    return re_mask_start, re_mask_end

def mask_message_roles(dset:datasets.Dataset, tokenizer:PreTrainedTokenizerBase, re_mask_start:str = r'<\|im_start\|>system', re_mask_end:str = r'<\|im_start\|>', num_proc=16):
    '''This function is only necessary because Unsloth requires raw text inputs. 
    
    🤗 SFTTrainer could do this better/simpler by adding {% generation %} fences in chat_template and feeding message logs `with assistant_only_loss=True`'''
    # Match mask_start + anything as long as it is followed by mask_end or endline (`|\Z`). 
    # match endline in case last role in sequence.
    re_mask = re.compile(rf'{re_mask_start}.+?(?={re_mask_end}|\Z)', flags = re.DOTALL + re.MULTILINE) # re.IGNORECASE
    
    output_keys = ['labels', 'input_ids']

    if (is_packed := 'seq_lengths' in dset.column_names):
        output_keys += ['seq_lengths', 'completion_mask']
    else:
        output_keys += ['attention_mask']
        # "For padding-free, we should NOT create attention_mask as it causes FlashAttention to ignore position_ids and compute wrong cu_seq_lens from the all-1s mask"
        # - https://github.com/huggingface/trl/blob/17acd61f28140243a458d10656a2194af788ff38/trl/trainer/sft_trainer.py#L171C1-L172C57
    
    def _mask_message_roles(example:dict[str, list[int]]):
        seq_lengths = example.get('seq_lengths', [len(example['input_ids'])])

        offset = 0
        
        adjusted = {
            'labels': [],
            'input_ids': [],
            'attention_mask': [],
            'seq_lengths': [],
            'completion_mask': [],
        }

        for seqlen in seq_lengths:
            inp_seq = example['input_ids'][offset:(offset+seqlen)]
            text_sample = tokenizer.decode(inp_seq, skip_special_tokens=False)
            offset += seqlen
            # this is more consistent in tokenization than just using `len(example['input_ids'])`
            # e.g. LLama3.1 failed because tokenization != forward(backward())
            # ['rel', '????????', '??', '?\n', 'AND'] 
            # ['rel', '?????', '????', '?', '?\n']
            orig_length = tokenizer(text=text_sample, add_special_tokens=False, return_length=True)['length'][0]
            adjusted['seq_lengths'].append(orig_length)
            
            inds = []
            for m in re_mask.finditer(text_sample):
                inds.extend(m.span())
        
            # pairwise index with alternating 0,1 labels. Add len for last segment.
            char_idx_labels = [(s, e, i%2) for i,(s,e) in enumerate(zip(inds, inds[1:]+[len(text_sample)]))]
            
            start_idx = char_idx_labels[0][0]
            char_idx_labels = [(0, start_idx, 1)] + char_idx_labels # add back anything that comes before first match
            
            labels = []
            for s,e,label in char_idx_labels:
                if s!=e:
                    inps = tokenizer(text=text_sample[s:e], add_special_tokens=False, return_length=True, return_attention_mask=True, return_tensors='pt')
                    token_len = inps['length'][0]
                    
                    input_ids = torch.as_tensor(inps['input_ids']).squeeze().tolist() # an expensive flatten
                    attention_mask = torch.as_tensor(inps['attention_mask']).squeeze().tolist()

                    labels.extend([-100]*token_len if label==0 else input_ids)
                    
                    adjusted['input_ids'].extend(input_ids)
                    adjusted['attention_mask'].extend(attention_mask)
                    adjusted['completion_mask'].extend([label]*token_len)

            if len(labels) != orig_length:
                raise RuntimeError(f'Parser failure. Length mismatch: {len(labels)} != {orig_length}')
            
            adjusted['labels'].extend(labels)
        
        result = {key: adjusted[key] for key in output_keys}
        # result = {'labels': adjusted['labels'], 'input_ids': adjusted['input_ids']}
        
        # if 'seq_lengths' in example:
        #     result.update({'seq_lengths': adjusted['seq_lengths'], 'completion_mask': adjusted['completion_mask']})
        # else:
        #     # "For padding-free, we should NOT create attention_mask as it causes FlashAttention to ignore position_ids and compute wrong cu_seq_lens from the all-1s mask"
        #     # - https://github.com/huggingface/trl/blob/17acd61f28140243a458d10656a2194af788ff38/trl/trainer/sft_trainer.py#L171C1-L172C57
        #     result.update({'attention_mask': adjusted['attention_mask']})
        return result

    # samp = dset.take(10).map(_mask_message_roles, num_proc=1)
    # samp.column_names
    # non_numer_col = [t for t in ['text','messages'] if t in trainer.train_dataset.column_names]
    return dset.map(_mask_message_roles, num_proc=num_proc)

    # trainer.train_dataset = trainer.train_dataset.map(_mask_message_roles, num_proc=num_proc)#.remove_columns(non_numer_col)
    # # if 'seq_lengths' not in trainer.train_dataset.column_names:
    # #     trainer.train_dataset = trl.pack_dataset(trainer.train_dataset, trainer.args.max_seq_length)
    
    # if trainer.eval_dataset:
    #     trainer.eval_dataset = trainer.eval_dataset.map(_mask_message_roles, num_proc=num_proc)#.remove_columns(non_numer_col)

    # return trainer

def apply_role_mask_tokens(dset:datasets.Dataset, tokenizer:PreTrainedTokenizerBase, cfg:DictConfig, num_proc = 32, *, no_format_roles: tuple[str] = ('system', 'user', 'assistant')):
    default_mask_roles = {
        'content_prefix': ['system'],
        'replace_role': ['system'],
        'content_prefix_ot': ['system', 'user']
    }
    
    role_tag_template = cfg.get('mask_roletag_template')
    masked_authors = cfg.get('masked_authors')
    # Here is where we could set masked_role_tags to ['system','user'] 
    # or even ['system', 'BOT (BotName)'] if we don't drop bot messages TODO: try
    # or for max inefficency, ['system', 'User1 (firstName1)', 'User3 (firstName3)', ...]
    if role_tag_template:
        if not masked_authors:
            masked_authors = default_mask_roles[cfg.tag_placement]
        
        masked_role_tags = [author if author in no_format_roles else useridx.format_author_tag(author, cfg.author_tag) for author in masked_authors]

        mask_start, mask_end = get_roletag_regex_masks(masked_role_tags=masked_role_tags, role_tag_template=role_tag_template)
        dset = mask_message_roles(dset, tokenizer, mask_start, mask_end, num_proc = num_proc)
    return dset

def map_to_inputs(dset:datasets.Dataset, tokenizer:PreTrainedTokenizerBase, max_length:int, truncation:bool, text_field='text'):
    '''Maps Dataset text field to those for model input (input_ids, special_tokens_mask, length)
    
    Importantly, it DOES add special tokens. This is primarily a concern after calling apply_chat_template(tokenize=False) which,
    depending on the template, may or may not insert a bos_token. This will cause double bos_token on all entries unless 
    apply_chat_template(tokenize=False, tokenizer_kwargs={'add_special_tokens': False}) is used. 
    '''
    dset = dset.map(lambda s: tokenizer(text = s[text_field], add_special_tokens=True, return_special_tokens_mask=True, max_length=max_length, return_length=True, truncation=truncation), batched=True)
    return dset


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
    return tokenizer(text = tokenizer.apply_chat_template(batched_convo, tokenize=False, add_generation_prompt=False), add_special_tokens=False, return_length=True, return_tensors='np')['length']- bos_discount


def add_sys_msg(chat_convo:list[dict], system_msg:list[dict[str,str]], tag_placement:typing.Literal['tag_only', 'content_prefix', 'replace_role', 'content_prefix_ot',],):
    if isinstance(system_msg, str):
        raise TypeError('No more loosey goosey. dict or list[dict, dict] only')
        #system_msg = {'role':'system', 'content':system_msg}

    # TODO: Should tag_only accept a system message? Foundation models don't really do that, but I guess it's not gonna break anything
    if tag_placement != 'content_prefix':
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


def to_conversation_format(formatted_author_tags:list[str], raw_texts:list[str], tag_placement:typing.Literal['tag_only', 'content_prefix', 'replace_role' 'content_prefix_ot'], system_msg:dict[str,str]|list[dict[str,str]]|None=None) -> list[dict[str,str]]:
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
    
    
    if tag_placement == 'content_prefix':
        rolecycle = itertools.cycle(['user','assistant'])
                        
        for fauth_tag,text in atag_tcontent:
            content = fauth_tag + text
            chat_content.append({"role": next(rolecycle), "content": content})

        return add_sys_msg(chat_content, system_msg=system_msg, tag_placement=tag_placement)
    else:
        # '''For tag only, markup free, format''' # '''For chatml with custom roles as usernames'''
        for role_tag,content in atag_tcontent:
            chat_content.append({"role": role_tag, "content": content})
        
        return add_sys_msg(chat_content, system_msg=system_msg, tag_placement=tag_placement)



def fill_cfg_from_data(formatted_author_tag_col:pd.Series, cfg:DictConfig):
    fprompt = cfg.fprompt
    name_mapping = cfg.prompt.name_mapping
    name_map_json = [{'username': k, 'firstName': v} for k,v in useridx.get_users('fname',by='dname').items()]
    if name_mapping is None:
        name_mapping = ', '.join(formatted_author_tag_col.str.strip().unique())
        cfg.prompt.name_mapping = name_mapping
    
    if fprompt is None:
        fprompt = cfg.prompt.template.format(name_mapping=name_mapping, append_msg=cfg.prompt.append_msg, name_mapping_json=json.dumps(name_map_json, indent=1))
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
    
    elif cfg.tag_placement == 'content_prefix':
        has_system = check_if_system(tokenizer)
        append_msg = cfg.prompt.append_msg
        
        if has_system:
            system_message = [{'role':'system', 'content': cfg.fprompt}]
        
        elif isinstance(append_msg, bool) and append_msg:
            system_message = [{'role':'user', 'content': cfg.fprompt}]
            # legacy handling done by add_sys_message since requires first chat message
        elif isinstance(append_msg, str):
            system_message = [
                {'role':'user', 'content': cfg.fprompt}, 
                {'role':'assistant', 'content': append_msg}
            ]

    else:
        system_message = [{'role':'system', 'content': cfg.fprompt}]

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

def time_split_overlength(prepared_conversations:list[list[dict]], convo_time_gaps:list[list[float]], tokenizer:PreTrainedTokenizerBase, system_msg:str, tag_placement:str, max_length:int):
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

def dedupe_conversations(conversations:list[list[dict]]) -> list[list[dict]]:
    unique_convos = []
    for convo in conversations:
        if convo not in unique_convos:
            unique_convos.append(convo)
    return unique_convos


def prepare_dataset_dataframe(chat_csv: str, cfg: DictConfig, **cfg_dot_kwargs):
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"{k}={v}" for k,v in cfg_dot_kwargs.items()]))

    df_chat, user_index = etl.process_csv(chat_csv, youtube_encode_fetch=True, filter_prefixes=('!', '/'), merge_window=7.0, drop_cloneus_bot=True)
    df_chat = etl.merge_user_sequences(df_chat)
    
    df_chat['formatted_author_tag'] = df_chat['user'].apply(useridx.format_author_tag, author_tag=cfg.author_tag, user_index=user_index) 
    
    if cfg.tag_sep and cfg.tag_placement != 'content_prefix_ot':
        df_chat['formatted_author_tag'] += cfg.tag_sep # append sep to end of tag
    
    df_chat = etl.assign_split(df_chat, cfg.dataset.get('eval_frac', 0.005))
    
    if 'chunkh' in cfg.dataset.name:
        df_chat = etl.label_chat_sessions(df_chat, hours_between_sessions=cfg.dataset.hours_between_sessions, min_session_length=cfg.dataset.min_session_length)
        # intrasession_time_gap
        df_chat['intrn_time_gap'] = df_chat.groupby(['split','chat_session'])['Date'].diff().dt.total_seconds().fillna(0) 
    
    df_chat['local_date'] = df_chat['Date'].dt.tz_convert(dateutil.tz.gettz())
    df_chat['date_string'] = df_chat['local_date'].dt.strftime("%d %b %Y") # Llama3.1
    
    return df_chat


def chat_sessions_dataset(df_chat: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, cfg:DictConfig, dataset_format: typing.Literal['text','tokens','messages'] = 'tokens'):
    system_message, has_system, append_msg = prepare_system_msg(cfg, tokenizer)
    
    df_convo = df_chat.groupby(['split','chat_session'])[['formatted_author_tag','text','intrn_time_gap']].agg(list).drop_duplicates('text')

    df_convo['conversation'] = df_convo.apply(lambda r: to_conversation_format(r.formatted_author_tag, r.text, cfg.tag_placement, system_message), axis=1)
    # prepend either 0,1,or 2 "0.0" values to ensure alignment with inserted system message(s)
    df_convo['intrn_time_gap'] = (df_convo['conversation'].str.len() - df_convo['intrn_time_gap'].str.len()).apply(lambda zpad: [0.0]*zpad) + df_convo['intrn_time_gap']

    eval_convos = time_split_overlength(df_convo.loc['eval','conversation'].tolist(), df_convo.loc['eval','intrn_time_gap'].tolist(), 
                                        tokenizer, system_msg=system_message, tag_placement=cfg.tag_placement, max_length=cfg.chunk_size)
    train_convos = time_split_overlength(df_convo.loc['train','conversation'].tolist(), df_convo.loc['train','intrn_time_gap'].tolist(), 
                                         tokenizer, system_msg=system_message, tag_placement=cfg.tag_placement, max_length=cfg.chunk_size)
        
    
    dset = datasets.DatasetDict({
        'train': datasets.Dataset.from_dict({'messages': dedupe_conversations(train_convos)}, split='train'),
        'validation': datasets.Dataset.from_dict({'messages': dedupe_conversations(eval_convos)}, split='validation'),
    })
    
    if dataset_format=='messages':
        return dset
    
    # TODO: pass date_string in as kwarg to apply_chat_template. Use the earliest date in chat sequence if multiple
    tokenizer_kwargs={'add_special_tokens': False} # If returning as text, do NOT add spec tokens. When the Trainer tokenizes, it WILL add special tokens,
    dset = dset.map(lambda x: {"text": tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=False, tokenizer_kwargs=tokenizer_kwargs, batched=True)}).remove_columns(['messages'])
    
    if dataset_format == 'tokens':
        dset = dset.map(lambda s: tokenizer(text = s['text'], max_length=cfg.chunk_size, truncation=cfg.dataset.allow_truncation, 
                                            add_special_tokens=True, return_special_tokens_mask=True, return_length=True), batched=True).remove_columns(['text'])
        #dset = map_to_inputs(dset, tokenizer, max_length=cfg.chunk_size, truncation=cfg.dataset.allow_truncation).remove_columns(['text'])
    
        if cfg.get('mask_roletag_template'):
            dset = apply_role_mask_tokens(dset, tokenizer, cfg, num_proc=32)

    return dset

def to_chat_triplets(conversation:list[dict[str,str]], system_message:str, ctx_template:str='{role} {content}', ctx_sep:str = '\n') -> list[list[dict[str,str]]]:
    '''Converts conversation format messages into expanding (sys, ctx, resp) triplets where ctx grows by 1 message until exhausted'''
    chat_sequence = [c for c in conversation if c['role'] != 'system']

    messages = []
    for i in range(len(chat_sequence)-1, 0, -1):
        messages.append([
            {'role':'system', 'content': system_message},
            {'role':'user', 'content': ctx_sep.join(ctx_template.format_map(c) for c in chat_sequence[:i])},
            {'role':'assistant','content': ctx_template.format_map(chat_sequence[i])},
        ])
    return messages[::-1]


def subsession_completions_dataset(df_chat: pd.DataFrame, tokenizer: PreTrainedTokenizerBase, cfg:DictConfig, ctx_template:str='{role}{tag_sep}{content}', ctx_sep:str = '\n', dataset_format: typing.Literal['text','tokens','messages'] = 'text'):
    ctx_template = ctx_template.format(role='{role}', tag_sep=cfg.tag_sep, content='{content}')

    chat_session_dset = chat_sessions_dataset(df_chat, tokenizer, cfg, dataset_format='messages')
    dset = chat_session_dset.map(lambda ex: {'conversations': to_chat_triplets(ex['messages'], cfg.fprompt, ctx_template, ctx_sep)},)
    
    dset = datasets.DatasetDict({
        'train': datasets.Dataset.from_dict({'messages': [y for x in dset['train']['conversations'] for y in x]}, split='train'),
        'validation': datasets.Dataset.from_dict({'messages': [y for x in dset['validation']['conversations'] for y in x]}, split='validation'),
    })
    
    if dataset_format == 'messages':
        return dset

    tokenizer_kwargs={'add_special_tokens': False} # If returning as text, do NOT add spec tokens. When the Trainer tokenizes, it WILL add special tokens,
    dset = dset.map(lambda ex: {'text': tokenizer.apply_chat_template(ex['messages'], tokenize=False, add_generation_prompt=False, tokenizer_kwargs=tokenizer_kwargs)}, batched=True).remove_columns(['messages'])
    
    if dataset_format=='tokens':
        dset = dset.map(lambda s: tokenizer(text = s['text'], max_length=cfg.chunk_size, truncation=cfg.dataset.allow_truncation, 
                                            add_special_tokens=True, return_special_tokens_mask=True, return_length=True), batched=True).remove_columns(['text'])
        #dset = map_to_inputs(dset, tokenizer, max_length=cfg.chunk_size, truncation=cfg.dataset.allow_truncation).remove_columns(['text'])
    
        if cfg.get('mask_roletag_template'):
            dset = apply_role_mask_tokens(dset, tokenizer, cfg, num_proc=32)
    
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

def convo_batch_max_tokens(unified_conversation:list[dict[str,str]], tokenizer:PreTrainedTokenizerBase, system_msg: dict[str, str] | list[dict[str, str]], tag_placement:typing.Literal['tag_only', 'content_prefix', 'replace_role','content_prefix_ot',], max_length:int):
    
    # TODO: this will break for append_msg content_prefix
    syslen = batched_token_count([system_msg], tokenizer).sum() # sum in case is SYN ACK
    if tag_placement == 'content_prefix':
        # since treating each msg as its own convo, need to reassign all roles as user so mistral format doesn't complain. Changes lengths slighty, but it's an approximation anyway
        all_msg_lens = batched_token_count([{'role':'user', 'content': u['content']} for u in unified_conversation], tokenizer)
    else:
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


def max_tokens_dataset(df_chat: pd.DataFrame, tokenizer:PreTrainedTokenizerBase, cfg:DictConfig, dataset_format: typing.Literal['text','tokens','messages'] = 'tokens'):
    '''Dataset groupings of sequential texts concatenated up to a maximum of `cfg.chunk_size` total tokens
    
    Chat sessions are not assigned, and the only use of Date or timestamp if present is consecutive message merge.
    '''
    system_message, has_system, append_msg = prepare_system_msg(cfg, tokenizer) # may update tokenizer

    # get base tokens before any system is added
    # Do not add system message since it is a flat list of messages as a single mega conversation
    sr_flat_convo = df_chat.groupby('split')[['formatted_author_tag', 'text']].agg(list).apply(lambda r: to_conversation_format(r.formatted_author_tag, r.text, cfg.tag_placement), axis=1)
    print('Grouping messages into conversations of maximal length..')
    train_convos = convo_batch_max_tokens(sr_flat_convo['train'], tokenizer, system_message, cfg.tag_placement, max_length=cfg.chunk_size)
    eval_convos = convo_batch_max_tokens(sr_flat_convo['eval'], tokenizer, system_message, cfg.tag_placement, max_length=cfg.chunk_size)
    
    dset = datasets.DatasetDict({
        'train': datasets.Dataset.from_dict({'messages': train_convos}, split='train'),
        'validation': datasets.Dataset.from_dict({'messages': eval_convos}, split='validation'),
    })
    
    if dataset_format == 'messages':
        return dset

    tokenizer_kwargs={'add_special_tokens': False} # If returning as text, do NOT add spec tokens. When the Trainer tokenizes, it WILL add special tokens,
    dset = dset.map(lambda x: {"text": tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=False, tokenizer_kwargs=tokenizer_kwargs)}).remove_columns(['messages'])
    
    # https://huggingface.co/learn/nlp-course/chapter5/3
    if dataset_format=='tokens':
        dset = dset.map(lambda s: tokenizer(text = s['text'], max_length=cfg.chunk_size, truncation=cfg.dataset.allow_truncation, 
                                            add_special_tokens=True, return_special_tokens_mask=True, return_length=True), batched=True).remove_columns(['text'])    
        
        if cfg.get('mask_roletag_template'):
            dset = apply_role_mask_tokens(dset, tokenizer, cfg, num_proc=32)

    return dset



def jsonl_dataset(train_jsonl:str, eval_jsonl:str, tokenizer:PreTrainedTokenizerBase, cfg:DictConfig, dataset_format: typing.Literal['tokens','raw'] = 'tokens'):
    dset = datasets.load_dataset("json", data_files={"train": train_jsonl, "validation": eval_jsonl})
    
    if dataset_format=='tokens':
        dset = dset.map(lambda s: tokenizer(text = s['text'], max_length=cfg.chunk_size, truncation=cfg.dataset.allow_truncation, 
                                            add_special_tokens=True, return_special_tokens_mask=True, return_length=True), batched=True).remove_columns(['text'])
    
        if cfg.get('mask_roletag_template'):
            dset = apply_role_mask_tokens(dset, tokenizer, cfg, num_proc=32)
    
    return dset


def ungrouped_dataset(df_chat: pd.DataFrame, tokenizer:PreTrainedTokenizerBase, cfg:DictConfig, dataset_format: typing.Literal['tokens','raw'] = 'tokens'):
    '''Dataset of chat messages without any grouping by time or chunk size. 

    The dataset is constructed from the raw dataframe. No further processing or parsing is preformed.
    The dataset will be split into train and validation and have fields:
      ['Date', 'time_gap', 'formatted_author_tag','text'] 
    that will need to be handled before feeding to a model.
    
    SFTTrainer with packing=True must be used since order is lost through normal batching
    '''
    if not cfg.use_sft_trainer:
        raise ValueError('for dataset "ungrouped_eos" SFTTrainer must be used to preserve message order')
    
    # No grouping means system message cannot be added anywhere.
    system_message, has_system, append_msg = prepare_system_msg(cfg, tokenizer) # may update tokenizer
    
    df_all = df_chat[['split', 'Date', 'time_gap', 'formatted_author_tag','text']]
    
    # Concatenate the author tag and text, adding tag_sep if need be
    if cfg.tag_sep and not df_all['formatted_author_tag'].str.endswith(cfg.tag_sep).all():
        df_all['formatted_author_tag'] = df_all['formatted_author_tag'].str.removesuffix(cfg.tag_sep) + cfg.tag_sep
    
    df_all['text'] = df_all['formatted_author_tag'] + df_all['text']
    df_all = df_all.set_index('split')
    
    dset = datasets.DatasetDict({
        'train': datasets.Dataset.from_pandas(df_all.loc['train'], split='train', preserve_index=False),
        'validation': datasets.Dataset.from_pandas(df_all.loc['eval'], split='validation', preserve_index=False)
    })
    
    if dataset_format == 'raw':
        return dset
    
    #dset = dset.map(lambda x: {"text": tokenizer.apply_chat_template(x["text"], tokenize=False, add_generation_prompt=False)})
    if dataset_format=='tokens':
        dset = dset.map(lambda s: tokenizer(text = s['text'], max_length=cfg.chunk_size, truncation=cfg.dataset.allow_truncation, 
                                            add_special_tokens=True, return_special_tokens_mask=True, return_length=True), batched=True).remove_columns(['text'])
    
        if cfg.get('mask_roletag_template'):
            dset = apply_role_mask_tokens(dset, tokenizer, cfg, num_proc=32)
    
    return dset