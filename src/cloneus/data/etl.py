import json
import urllib


import numpy as np
import pandas as pd

from . import roles
from ..core import paths as cpaths
from ..plugins import youtube as youtube

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

def preprocess_df(chat_csv: str, video_datafile='video_data.jsonl', hours_between_sessions: int = 4, cmd_prefixes=('!','/')):
    df_chat = pd.read_csv(chat_csv)
    #RE_ATTACHMENTS = re.compile(r'https://cdn.discordapp.com/attachments/\d+/\d+/([^\s,]+)')
    #df_gsc['Attachments'] = df_gsc.Attachments.apply(cleanup_url).str.replace(RE_ATTACHMENTS, r'attachments/\g<1>',regex=True)
    #df_gsc['Content'] = df_gsc.Content.str.replace(RE_ATTACHMENTS, 'attachments/\g<1>', regex=True)
    #df_gsc['Content'] = df_gsc['Content'].fillna(df_gsc['Attachments'])

    # BUG: Not converting to dt object? 
    df_chat['Date'] = df_chat['Date'].pipe(pd.to_datetime, infer_datetime_format=True)# format="%Y-%m-%dT%H:%M:%S.%f%z")#, format='%m/%d/%Y %I:%M %p')

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
    # df_chat['pre_ai'] = df_chat.index < df_chat[is_cloneus_bot].index[0]
    df_chat['pre_ai'] = df_chat.index < df_chat.index[-256]
    # TODO: parameter for eval split
    
    # drop Cloneus messages and Command calls
    df_chat = df_chat[~is_command_call & ~is_cloneus_bot]


    # replace 2+ newlines with 1 newline, since we may use \n\n to separate messages
    df_chat['text'] = df_chat['Content'].str.replace(r'\n{2,}','\n', regex=True)
   
    try:
        # Want this to be AFTER filtering to avoid YouTube API calls on excluded content
        ytm = youtube.YouTubeManager(video_datafile)
        # Transform all youtube URLs into custom metadata tag for better LLM comprehension
        df_chat.loc[:, 'text'] = df_chat['text'].apply(ytm.encode, allow_fetch=False)
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
    df_chat['chat_session'] = (df_chat['time_gap'] >= (hours_between_sessions*60*60)).cumsum() # 4hr time gap between sessions
    

    return df_chat

def expand_sessions(chat_session: pd.Series, min_session_length=1):
    '''Forward Merge session groups that have less than `min_session_msgs` items'''
    # NOTE: this could inf loop, if last entry is it's own single session
    # Could be avoided if -= 1, but +=1 aligns better with the assumption we take that 4hr break = new topic session
    # but if we have __4hr_break__ [TEXT] __4hr_break__ then seems more likely that text should be mapped forward instead of backward
    n_small_session = min_session_length-1 # skip loop and return unless > 1
    while n_small_session > 0:
        under_len = chat_session.value_counts() < min_session_length
        small_sessions = under_len[under_len].index
        
        chat_session = chat_session.where(~chat_session.isin(small_sessions), other = chat_session + 1)
        
        n_small_session = small_sessions.shape[0]
        print(n_small_session)
    
    return chat_session



def create_dftext(df_proc: pd.DataFrame, tag_sep='\n', postfix='\n\n', author_tag='[USER:{author}]', min_session_length=1):
    
    df_all = df_proc.groupby('user_sequence', as_index=False)[
        ['user', 'Date', 'chat_session', 'time_gap', 'text','pre_ai']].agg(
        {'user':'first', 'Date':'last', 'chat_session':'first', 'time_gap':'first', 'text':list, 'pre_ai':'first'})
    
    # Join consecutive author messages with new line
    df_all['text'] = df_all['text'].str.join('\n')

    df_all['tag_prefix'] = df_all['user'].apply(roles.format_author_tag, author_tag=author_tag) + tag_sep

    # BUG until 2023-11-29 was df_proc.text instead of df_all
    df_all['formatted_text'] = df_all['tag_prefix'] + df_all['text'] + postfix 
    
    df_all['split'] = df_all['pre_ai'].apply(lambda x: 'train' if x else 'eval')
    # df_all.loc[df_all.pre_ai, 'split'] = 'train'
    # df_all.loc[(~df_all.pre_ai), 'split'] = 'eval'

    df_all['chat_session'] = expand_sessions(df_all['chat_session'], min_session_length=min_session_length)

    return df_all



def create_df_all(chat_csv, tag_sep, postfix, author_tag, hours_between_sessions:(int|list[int]) = 4, min_session_length=1):
    if isinstance(hours_between_sessions, (int,float)):
        df_proc = preprocess_df(chat_csv, hours_between_sessions=hours_between_sessions)
        df_all = create_dftext(df_proc, tag_sep=tag_sep, postfix=postfix, author_tag=author_tag,  min_session_length=min_session_length)
    else:
        dfs = []
        for hrsess in hours_between_sessions:
            df_proc = preprocess_df(chat_csv, hours_between_sessions=hrsess)
            dfhr = create_dftext(df_proc, tag_sep=tag_sep, postfix=postfix, author_tag=author_tag, min_session_length=min_session_length)
            dfhr.chat_session+=int(hrsess*1e6)
            dfs.append(dfhr)
        df_all = pd.concat(dfs)
    
    return df_all