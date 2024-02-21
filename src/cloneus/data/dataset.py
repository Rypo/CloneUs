import os
import json

import random
import datetime
import itertools
from typing import Dict

import more_itertools
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import datasets

from ..plugins import youtube as youtube
#from ..core import paths as rpaths
from . import roles, etl




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


def resplit_overflow(df_data, tokenizer, maxlen, postfix='\n\n'):
    '''Takes a dataframe split with a text column and splits those that exceed maxlen into multiple new rows using postfix'''
    df_data['toklen'] = tokenizer(df_data['formatted_text'].to_list(), return_length=True, add_special_tokens=True)['length']
    df_overlen = df_data[df_data['toklen'] > maxlen]
    
    def resplit_overlen(overlen_text):
        # NOTE: without the "if t", will have a double postfix. No clue how that bug slipped by.
        split_text_list = [t+postfix for t in overlen_text.split(postfix) if t]
        return [''.join(cb) for cb in more_itertools.constrained_batches(split_text_list, maxlen, get_len=lambda t: tokenizer(t, return_length=True, add_special_tokens=True)['length'][0])]
    
    df_etext = df_overlen['formatted_text'].apply(resplit_overlen).explode().to_frame()
    df_etext = df_etext.reindex(columns=df_overlen.columns).fillna(df_overlen)
    
    df_etext['toklen'] = df_etext['formatted_text'].apply(lambda t: tokenizer(t, return_length=True)['length'][0])
    df_etext['chat_session'] = df_etext['chat_session'].astype(int)

    return pd.concat([df_data.drop(index=df_overlen.index), df_etext]).sort_values(['chat_session','toklen'], ascending=[True,False])





def dataset_timechunk(chat_csv, tokenizer, cfg, text_only=False):
    maxlen= cfg.chunk_size
    tag_sep=cfg.tag_sep
    postfix=cfg.postfix
    author_tag=cfg.author_tag
    hours_between_sessions=cfg.dataset.hours_between_sessions
    min_session_length=cfg.dataset.min_session_length
    
    df_all = etl.format_chat_groups(etl.preprocess_df(chat_csv),  tag_sep, postfix, author_tag, hours_between_sessions, min_session_length, eval_frac=0.005)
        
    df_train = df_all[df_all.split=='train'].groupby('chat_session',as_index=False)['formatted_text'].agg(''.join)
    df_eval = df_all[df_all.split=='eval'].groupby('chat_session',as_index=False)['formatted_text'].agg(''.join)
    
    # TODO: maybe split on longest chat pause instead of arbitrary length.
    df_train = resplit_overflow(df_train, tokenizer, maxlen=maxlen, postfix=postfix)
    df_eval = resplit_overflow(df_eval, tokenizer, maxlen=maxlen, postfix=postfix)

    ds_timechunk = datasets.DatasetDict({
        'train': datasets.Dataset.from_pandas(df_train.drop(columns=['chat_session','toklen']).rename(columns={'formatted_text':'text'}), split='train', preserve_index=False),
        'validation': datasets.Dataset.from_pandas(df_eval.drop(columns=['chat_session','toklen']).rename(columns={'formatted_text':'text'}), split='validation', preserve_index=False)
    })
    if not text_only:
        ds_timechunk = ds_timechunk.map(lambda s: tokenizer(s['text'], return_special_tokens_mask=True, max_length=maxlen,  return_length=True, truncation=False), batched=True)
    # ds_timechunk = ds_timechunk.map(lambda s: tokenizer(s['text'], return_special_tokens_mask=True, max_length=maxlen, return_overflowing_tokens=True, return_length=True, truncation=True), 
    #                             batched=True, remove_columns=ds_timechunk['train'].column_names)
    
    return ds_timechunk



def split_over(fchatformat:list[str], max_len:int, tokenizer, formatted_system: str, special_tokens_counted=True):
    r'''Takes a list of messages with ALL formatting <|im_start|>, [INST], </s>\n ... etc. 
    and splits into into multiple lists if the total token length exceeds `max_len`

    Note: special_tokens_counted is kind of a hack. It will add a <s> to EVERY message, therefore adding +1 to EVERY message in the chunk
    This will deflate the max length significantly, but I can't figure out why it's going ~15 tokens over length without out (4111 vs 4096) 
    '''
    return [''.join((formatted_system,*cb) if cb[0] != formatted_system else cb )
     for cb in more_itertools.constrained_batches(fchatformat, max_len, get_len=lambda t: tokenizer(t, return_length=True, add_special_tokens=special_tokens_counted)['length'][0])]


def group_chatml_format(df_all:pd.DataFrame, tokenizer, max_len:int, role_tag:str, fprompt:str, custom_chat_template:str=None) -> pd.Series:
    # NOTE: This can only work with chat_ml format
    if custom_chat_template is None:
        # Same as chat_ml but removed "assistant\n" from add_generation_prompt
        custom_chat_template="{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>' }}{% endif %}"
    
    formatted_system = tokenizer.apply_chat_template([{"role": "system", "content": fprompt}], tokenize=False)

    df_all['role'] = df_all['user'].apply(roles.format_author_tag, author_tag=role_tag)
    
    df_rolechat = df_all[['role','text','split','chat_session']].rename(columns={'text':'content'}).copy()

    df_rolechat['chatfmt'] = df_rolechat[['role','content']].apply(lambda x: tokenizer.apply_chat_template([dict(x)], custom_chat_template, tokenize=False), axis=1)
    # insert formatted system message as first item in formatted message list groups  
    ds_session_syschats = df_rolechat.groupby(['split','chat_session'])['chatfmt'].agg(lambda x: [formatted_system,*list(x)])
    # Explode out overlength lists, drop chat_session index as it's no longer needed
    ds_syschats = ds_session_syschats.apply(split_over, max_len=max_len, tokenizer=tokenizer, formatted_system=formatted_system).explode().droplevel(1)

    return ds_syschats


def author_role_dataset(chat_csv, tokenizer, cfg, custom_chat_template=None):
    role_tag = cfg.author_tag
    fprompt = cfg.fprompt
    max_len = cfg.chunk_size

    prompt_length = tokenizer(fprompt, return_length=True).length[0]

    max_len-=prompt_length
    

    # https://old.reddit.com/r/LocalLLaMA/comments/1aiz6zu/roleplaying_system_prompts/
    
    df_all = etl.format_chat_groups(etl.preprocess_df(chat_csv), cfg.tag_sep, postfix='not_used', author_tag=role_tag, 
                                    hours_between_sessions=cfg.dataset.hours_between_sessions, min_session_length=cfg.dataset.min_session_length, eval_frac=0.005)
    
    # use max_len-2 in split threshold to account for <s> </s> tokens in final tokenization
    ds_chatml = group_chatml_format(df_all, tokenizer, max_len=max_len, role_tag=role_tag, fprompt=fprompt, custom_chat_template=custom_chat_template)

    ds_timechunk_chatml = datasets.DatasetDict({
        'train': datasets.Dataset.from_pandas(ds_chatml['train'].to_frame(name='text'), split='train', preserve_index=False),
        'validation': datasets.Dataset.from_pandas(ds_chatml['eval'].to_frame(name='text'), split='validation', preserve_index=False),
    })

    ds_timechunk_chatml = ds_timechunk_chatml.map(lambda s: tokenizer(s['text'], return_special_tokens_mask=True, return_length=True, truncation=False), batched=True)
    return ds_timechunk_chatml


def to_instruct_format(dataset, postfix, prompt, append_first_to_prompt=False, has_system=False):
    messages = dataset['text'].split(postfix)
    rolecycle = itertools.cycle(['user','assistant'])

    if has_system:
        chat_content = [{"role": "system", "content": prompt}]
        
    else:
        pcontent = prompt
        if append_first_to_prompt:
            pcontent+=messages.pop(0)
        chat_content = [{"role": next(rolecycle), "content": pcontent}]

    for msg in filter(None,messages):
        chat_content.append({"role": next(rolecycle), "content": msg})
    
    return chat_content

def instruct_dataset_timechunks(chat_csv, tokenizer, cfg, has_system=None, ):
    # TODO: make dataset where [INST] AuthorName [/INST] *response*. Might also be able to include seed text.
    # First instruction would be roughly the same.
    # but all subsequent will just be some form of username, first name
    # WHY???
    # Then can use the DataCollatorForCompletionOnlyLM and not worry about prompt effecting loss 

    maxlen= cfg.chunk_size
    postfix=cfg.postfix
    append_first_to_prompt=cfg.prompt.append_msg

    truncation=cfg.dataset.allow_truncation
    fprompt = cfg.fprompt

    prompt_length = tokenizer(fprompt, return_length=True).length[0]

    if has_system is None:
        has_system = tokenizer.use_default_system_prompt
    
    
    dataset = dataset_timechunk(chat_csv, tokenizer, cfg, text_only=True)
    dataset = dataset.map(lambda s: {'text': tokenizer.apply_chat_template(to_instruct_format(s, postfix, fprompt, append_first_to_prompt, has_system), tokenize=False)})
    dataset = dataset.map(lambda s: tokenizer(s['text'], return_special_tokens_mask=True, max_length=maxlen,  return_length=True, truncation=truncation), batched=True)
    return dataset


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
    df_all = etl.format_chat_groups(etl.preprocess_df(chat_csv), tag_sep, postfix=postfix, author_tag=author_tag, 
                                    hours_between_sessions=4, min_session_length=min_session_length, eval_frac=0.005).drop(columns=['chat_session']) 
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
    
    
    df_all = etl.format_chat_groups(etl.preprocess_df(chat_csv),  tag_sep=tag_sep, postfix=postfix, author_tag=author_tag,  min_session_length=min_session_length, eval_frac=0.005).drop(columns=['chat_session']) 
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
