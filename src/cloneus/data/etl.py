import json
import urllib
import typing
import itertools

import numpy as np
import pandas as pd

from cloneus.data import roles
from cloneus.core import paths as cpaths
from cloneus.plugins import youtube

def cleanup_url(url):
    if not isinstance(url,str):
        return url
    
    urls = url.split(',') if ',' in url else [url]
    cleaned_urls = []
    for u in urls:
        pr=urllib.parse.urlparse(u)
        new_pr = urllib.parse.ParseResult(pr.scheme, pr.netloc, pr.path, '', '', '')
        cleaned_urls.append(urllib.parse.urlunparse(new_pr))
    
    return ','.join(cleaned_urls)


def ids_to_displaynames() -> tuple[dict,int]:
    with open(cpaths.ROOT_DIR/'config/users.json') as f:
        users = json.load(f)

    cloneus_bot_id = int(users['BOT']['id'])
    bot_id_display = {cloneus_bot_id: users['BOT']['displayName']}
    user_id_display = {int(usr['id']): usr['displayName'] for usr in users['USERS']}

    id2display = {**user_id_display, **bot_id_display}
    
    
    return id2display, cloneus_bot_id


def _targeted_chat_filter(df_chat):
     # Drop very specific instance in 2018 when Char RNN trained on chat was dumped in chat (= 8 rows)
    mask_user_colon = df_chat['Content'].str.contains('|'.join(u + ':' for u in roles.author_display_names), case=False)
    mask_23ju18 = df_chat['Date'].dt.date == pd.to_datetime('2018-07-23').date()
    df_chat = df_chat.drop(index=df_chat[mask_user_colon & mask_23ju18].index)
    return df_chat

def preprocess_df(chat_csv: str, cmd_prefixes=('!','/')):
    df_chat = pd.read_csv(chat_csv)
    #RE_ATTACHMENTS = re.compile(r'https://cdn.discordapp.com/attachments/\d+/\d+/([^\s,]+)')
    #df_gsc['Attachments'] = df_gsc.Attachments.apply(cleanup_url).str.replace(RE_ATTACHMENTS, r'attachments/\g<1>',regex=True)
    #df_gsc['Content'] = df_gsc.Content.str.replace(RE_ATTACHMENTS, 'attachments/\g<1>', regex=True)
    #df_gsc['Content'] = df_gsc['Content'].fillna(df_gsc['Attachments'])
    # format="%Y-%m-%dT%H:%M:%S.%f%z")#, format='%m/%d/%Y %I:%M %p')
    df_chat['Date'] = df_chat['Date'].pipe(pd.to_datetime, utc=True, format='ISO8601')

    # strip out all discord image urls
    df_chat['Content'] = df_chat['Content'].str.replace('(https://cdn.discordapp.com\S+)','', regex=True).replace('',None) 
    df_chat = df_chat.dropna(subset='Content').reset_index(drop=True)

    #df_chat = _targeted_chat_filter(df_chat)
    
    id2display, cloneus_bot_id = ids_to_displaynames()
    df_chat['user'] = df_chat['AuthorId'].map(id2display)
    
    # Drop any messages from users not in the users.json (too few messages or from a different bot) 
    df_chat = df_chat.dropna(subset='user').reset_index(drop=True)
    
    is_command_call = df_chat['Content'].str.startswith(cmd_prefixes)
    is_cloneus_bot = df_chat['AuthorId'] == cloneus_bot_id

    # use the first message by bot to denote potentially unsafe training data
    df_chat['pre_bot'] = df_chat.index < df_chat[is_cloneus_bot].index[0] if is_cloneus_bot.any() else True
    # drop Cloneus messages and Command calls
    df_chat = df_chat[~is_command_call & ~is_cloneus_bot]

    # replace 2+ newlines with 1 newline, since we may use \n\n to separate messages
    df_chat['text'] = df_chat['Content'].str.replace(r'\n{2,}','\n', regex=True)
   
    try:
        # Want this to be AFTER filtering to avoid YouTube API calls on excluded content
        ytm = youtube.YouTubeManager(allow_fetch=True)
        # Transform all youtube URLs into custom metadata tag for better LLM comprehension
        df_chat.loc[:, 'text'] = df_chat['text'].apply(ytm.encode)
    except KeyError as e:
        print('.env missing: "YOUTUBE_API_KEY". YouTube video links will not encoded.')
    # this will drop ~< 500 samples with invalid YouTube links as the only source of text. Verified that these are indeed not valid links
    df_chat = df_chat[df_chat['text'] != ''].reset_index(drop=True)


    # TODO: Consider what to do about extremely long chat streaks (e.g. 367 consectutive messages by a user on 2023-01-19)
    # TODO: Consider implications of assigning sequence/sessions after filtering vs before filtering
    df_chat['time_gap'] = df_chat['Date'].diff().fillna(pd.Timedelta(0)).dt.total_seconds()
    # If message is from a different user or time gap is greater than 7 minutes, then it's a new sequence
    # - https://www.reddit.com/r/discordapp/comments/9u2tdz/whats_the_new_cutoff_time_for_separating_messages/
    df_chat['user_sequence'] = (((df_chat['user'] != df_chat['user'].shift(1)) | (df_chat['time_gap']>(7*60))).cumsum())
    #df_chat['chat_session'] = (df_chat['time_gap'] >= (hours_between_sessions*60*60)).cumsum() # 4hr time gap between sessions

    return df_chat

def expand_sessions(chat_session: pd.Series, min_session_length=1):
    '''Forward Merge session groups that have less than `min_session_msgs` items'''
    # NOTE: this could inf loop, if last entry is it's own single session
    # Could be avoided if -= 1, but +=1 aligns better with the assumption we take that 4hr break = new topic session
    # but if we have __4hr_break__ [TEXT] __4hr_break__ then seems more likely that text should be mapped forward instead of backward
    n_small_session = min_session_length-1 # skip loop and return unless > 1
    sess_sizes = []
    while n_small_session > 0:
        under_len = chat_session.value_counts() < min_session_length
        small_sessions = under_len[under_len].index
        
        chat_session = chat_session.where(~chat_session.isin(small_sessions), other = chat_session + 1)
        
        n_small_session = small_sessions.shape[0]
        sess_sizes.append(n_small_session)
    if sess_sizes:    
        print('sessions < min_session_length:',' -> '.join(map(str,sess_sizes)))
    return chat_session


def expand_and_split(df_chats:pd.DataFrame, hours_between_sessions:int=4, min_session_length:int=1):
    te_sessions = (df_chats['time_gap'] >= (hours_between_sessions*60*60)).cumsum()
    #train_chat_session = expand_sessions((df_chats.loc[df_chats.split=='train', 'time_gap'] >= (hours_between_sessions*60*60)).cumsum(), min_session_length=min_session_length)
    #eval_chat_session += (train_chat_session.max()+1) 
    train_chat_session = expand_sessions(te_sessions[df_chats.split=='train'], min_session_length=min_session_length)
    eval_chat_session = expand_sessions(te_sessions[df_chats.split=='eval'], min_session_length=min_session_length) + 1 # need +1 to force break 1st group
    
    assert train_chat_session.max() < eval_chat_session.min(), 'Possible data leak. Train chat_session overlaps eval chat_session'
    chat_session = pd.concat([train_chat_session, eval_chat_session])
    
    return chat_session


def assign_split(df_chat,  eval_frac: (float|typing.Literal['after_bot']) = 0.005):
    if eval_frac=='after_bot':
        df_chat['split'] = df_chat['pre_bot'].apply(lambda x: 'train' if x else 'eval')
    else:
        # Use the last eval_frac chat groups for eval
        n_eval = int(df_chat.shape[0]*eval_frac)
        eval_start_date = df_chat.iloc[-n_eval].Date
        
        df_chat['split'] = (df_chat.Date < eval_start_date).apply(lambda x: 'train' if x else 'eval')
        print(f'Using messages after {eval_start_date} for eval set. n_messages = {n_eval}')
    
    return df_chat

def format_chat_groups(df_proc: pd.DataFrame, tag_sep:str, postfix:str, author_tag:str, hours_between_sessions:(int|list[int]) = 4, min_session_length:int=1, eval_frac: (float|typing.Literal['after_bot']) = 0.005):
    """Prepares data for use in hf dataset. Creates a formatted text col, merges user messages, and assigns train, eval split.

    Groups chat data by user_sequence and join by a new line. Creates formatted_text column where each takes the form: 
        `author_tag` `tag_sep` <TEXT> `postfix`.
    Assigns a split column for training and evaluation based on the `eval_frac` parameter.
    If min_session_length>1, any groups containing fewer than `min_session_length` messages will be forcably regrouped until.

    Args:
        df_proc: DataFrame containing preprocessed chat data.
        tag_sep: Separator string to use between the author tag and the text.
        postfix: String to append at the end of each formatted text. 
        author_tag: The format string for the author tag. e.g. '[USER:{author}]'.
        hours_between_sessions: Hours of silence before starting a new group. If a list is given, will copy the data group for each (default: 4).
        min_session_length: The minimum number of messages required for a session to be considered valid (default: 1).
        eval_frac (float|'after_bot'): If float, the fraction of chat groups to use for evaluation. If 'after_bot', use all messages after the bot's first message (default: 0.005).

    Returns:
        pd.DataFrame: A DataFrame with the formatted text and session splits for training and evaluation.

    """
    df_chats = df_proc.groupby('user_sequence', as_index=False)[
        ['user', 'Date', 'time_gap', 'text','pre_bot']].agg(
        {'user':'first', 'Date':'last', 'time_gap':'first', 'text':list, 'pre_bot':'first'})
    # Join consecutive author messages with new line
    df_chats['text'] = df_chats['text'].str.join('\n')

    df_chats['tag_prefix'] = df_chats['user'].apply(roles.format_author_tag, author_tag=author_tag) + tag_sep

    # BUG until 2023-11-29 was df_proc.text instead of df_all
    df_chats['formatted_text'] = df_chats['tag_prefix'] + df_chats['text'] + postfix 
    
    df_chats = assign_split(df_chats, eval_frac)
    if isinstance(hours_between_sessions, (int,float)):
        df_chats['chat_session'] = expand_and_split(df_chats, hours_between_sessions, min_session_length)
        return df_chats
    
    chatgrps = {h: expand_and_split(df_chats, h, min_session_length) for h in hours_between_sessions}
    
    for h1,h2 in itertools.combinations(hours_between_sessions, 2):
        print(f'({h1}h, {h2}h) - duplicate grouped items:',(chatgrps[h1]==chatgrps[h2]).sum())
        # Not the number of identical groups, number of items assigned same group at that index
        # TODO: Duplicate groups would be more useful to know
        # TODO: Need to know if dupe groups in eval at least. With so few items, could give a wildly misleading eval loss.
    
    # collisions possible if > 1 bill groups, but I mean...
    return pd.concat([df_chats.assign(chat_session=chatgrps[h] + int(h*1e9)) for h in hours_between_sessions]) 





