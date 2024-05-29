import sys
import json
import urllib
import typing
import warnings
import itertools

import numpy as np
import pandas as pd

from cloneus.data import useridx
from cloneus.plugins import youtube

def ask_fill_fnames(display_names: list[str]):
    n_missing = len(display_names)
    inp=''
    while (inp+'?')[0] not in ['y','n','q']:
        inp=input(f'Missing {n_missing} users\' firstName. Do you want to add these for the model to learn? (y/n/q) ').lower()
    
    if inp.startswith('q'):
        print('Aborted.')
        sys.exit(0)
    
    dname_to_fname = {}
    if inp.startswith('y'):
        for i,dn in enumerate(display_names,1): 
            dname_to_fname[dn] = input(f'[{i}/{n_missing}] First Name ({dn!r}):')
        print("Done. Be sure to update `author_tag` your training_config.yaml with the 'fname' fmtstring to utilize first names: e.g. '{author} ({fname})'")
    
    return dname_to_fname

def user_index_from_chat(df_chat: pd.DataFrame, msg_threshold:float|int = 0.005, excluded_users:list[str] = None, bot_usernames: list[str] = None, interactive:bool=False) -> list[dict]:
    '''Create user index from chat with assumptions. 
    Column format: ['AuthorID', 'Author', ...]. 
    Optional columns: ['username', 'displayName', 'firstName', 'authorInitial', 'isBot']
    '''
    df_users = df_chat.copy()
    # NOTE: AuthorID needs to be uint64, do NOT cast int, will overflow
    
    # one must be in included
    if 'Author' in df_users:
        if 'username' not in df_users:
            df_users.rename(columns={'Author':'username'}, inplace=True)
    elif 'username' not in df_users:
        raise KeyError('At least one of "Author" or "username" must be specified in columns')

    
    if excluded_users:
        df_users = df_users[~df_users['username'].isin(excluded_users)]
    
    # Exclude users with too few messages
    msg_counts = df_users['AuthorID'].value_counts(normalize=isinstance(msg_threshold, float))
    df_users = df_users[df_users['AuthorID'].isin(msg_counts[msg_counts >= msg_threshold].index)]

    for col in ['displayName', 'firstName', 'authorInitial', 'isBot']:
        df_users[col] = df_users.get(col, default=None)
    
    df_users.rename(columns={'AuthorID':'id'}, inplace=True)
    df_users = df_users[['id', 'firstName', 'authorInitial', 'username', 'displayName', 'isBot']].drop_duplicates().reset_index(drop=True)

     # Try to fill missing values with current user index, if any
    if useridx.user_index_exists():
        df_user_index = pd.DataFrame(useridx.get_users(include_bot=True))
        df_users = df_users.set_index('id').fillna(df_user_index.set_index('id')).reset_index()
    

    df_users['displayName'] = df_users['displayName'].fillna(df_users['username'])
    
    df_users['isBot'] = df_users['isBot'].where(df_users['isBot'].notna(), 
        False if bot_usernames is None else df_users['username'].isin(bot_usernames))
   
    if interactive:
        dnames_missing_fnames = df_users['displayName'].where(df_users['firstName'].isna()).dropna().to_list()
        if dnames_missing_fnames:
            dname_to_fname = ask_fill_fnames(dnames_missing_fnames)
            df_users['firstName'] = df_users['firstName'].where(df_users['firstName'].notna(), df_users['displayName'].map(dname_to_fname)).fillna('').str.title()
    #elif df_users['firstName'].isna().any():
    #   print("(Recommended) Update config/users.json with each user's first name")

    user_index = df_users.to_dict('records')
    
    if df_users['authorInitial'].isna().any():
        priority_order = ['fname', 'dname', 'uname']
        pcols = ['firstName', 'displayName','username']
        for i,pcol in enumerate(pcols):
            if df_users[pcol].isna().any():
                priority_order.pop(i)

        useridx.update_initials(user_index, priority_order=tuple(priority_order))

    clobot = useridx.get_cloneus_user()
    if clobot not in user_index:
        user_index = [clobot, *user_index]
    else:
        # Only purpose is to shuffle bot to front of list. Wasteful? probably.
        user_index = useridx.get_users(user_index=user_index, include_bot=True)

    return user_index


def get_make_userindex(df_chat, msg_threshold=0.005, excluded_users:list[str] = None, bot_usernames: list[str] = None, username_column:str = 'Author'):
    ''' Get or create user index. Always includes cloneus record first entry.'''
    # TODO: Maybe WildCard user for if data has long tail distrib? -- all users < threshold = WildCardUser

    if useridx.user_index_exists():
        all_names = set([*useridx.get_users('uname'), *useridx.get_users('fname'), *useridx.get_users('dname') ])

        hits = df_chat[username_column].isin(all_names).sum()
        samples = df_chat.shape[0]
        
        print(f'({hits}/{samples}) messages have usernames matching existing user index')
        
        if hits/samples < 0.5:
            print(r'Usernames do not match existing users.json. Creating ephemeral user index...')
            user_data_map = user_index_from_chat(df_chat, msg_threshold=msg_threshold, excluded_users=excluded_users, bot_usernames=bot_usernames)
        else:
            user_data_map = useridx.get_users(include_bot=True)

    else:
        user_data_map = user_index_from_chat(df_chat, msg_threshold=msg_threshold, excluded_users=excluded_users, bot_usernames=bot_usernames)
        useridx.write_user_index(user_data_map)


    return user_data_map


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

def _targeted_chat_filter(df_chat, enable=False):
    '''Drop very specific instance in 2018 when Char RNN trained on chat was dumped in chat.
    Nothing will be dropped unless these conditions apply, but set enable=False to bypass.
    '''
    if enable:
        mask_user_colon = df_chat['text'].str.contains('|'.join(u + ':' for u in useridx.get_users('dname')), case=False)
        mask_23ju18 = df_chat['Date'].dt.date == pd.to_datetime('2018-07-23').date()
        df_chat = df_chat.drop(index=df_chat[mask_user_colon & mask_23ju18].index)
    return df_chat

def discord_text_filters(text_col: pd.Series, cmd_prefixes:tuple[str, ...]=('!','/'),):
    '''Null out: command calls, discord image urls'''
    #RE_ATTACHMENTS = re.compile(r'https://cdn.discordapp.com/attachments/\d+/\d+/([^\s,]+)')
    #df_gsc['Attachments'] = df_gsc.Attachments.apply(cleanup_url).str.replace(RE_ATTACHMENTS, r'attachments/\g<1>',regex=True)
    #df_gsc['Content'] = df_gsc.Content.str.replace(RE_ATTACHMENTS, 'attachments/\g<1>', regex=True)
    
    # Drop Command calls
    text_col = text_col.where(~text_col.str.startswith(cmd_prefixes, na=True))
    # strip out all discord image urls
    text_col = text_col.str.replace('(https://cdn.discordapp.com\S+)','', regex=True).replace('',None) 
    
    return text_col

def to_common_format(df_chat:pd.DataFrame):
    '''Converts generic csv columns to standard discord-style format
    
    ['username', 'text', 'timestamp'] -> ['AuthorID', 'Author', 'Date', 'Content']
    '''
    df_chat = df_chat.rename(columns=str.lower) # normalize col names
    if 'timestamp' in df_chat:
        df_chat.rename(columns={'timestamp':'Date'}, inplace=True)
        df_chat['Date'] = df_chat['Date'].pipe(pd.to_datetime, utc=True, format='mixed')#'ISO8601')
    else:
        print()
        warnings.warn('No timestamp column. Chats will chunk by max tokens only. Expect data quality loss.')
        print()
        df_chat['Date'] = pd.NaT#pd.date_range(end='1/1/2024', periods=df_chat.shape[0], unit="s", freq="5min")
    
    df_chat.rename(columns={'username':'Author', 'text':'Content'}, inplace=True)
    
    if 'AuthorID' not in df_chat:
        # Assign an arbitrary id
        df_chat['AuthorID'] = df_chat['Author'].apply(useridx.fake_author_id)

    return df_chat[['AuthorID', 'Author', 'Date', 'Content']]


def apply_youtube_encoding(text_col:pd.Series, youtube_encode_fetch: bool|tuple[bool,bool]=(True,True)):
    if isinstance(youtube_encode_fetch, bool):
        youtube_encode_fetch = (youtube_encode_fetch,youtube_encode_fetch)
    
    yt_encode, yt_fetch = youtube_encode_fetch
    ytm = youtube.YouTubeManager(allow_fetch=yt_fetch, enabled=yt_encode)
    
    if yt_fetch and ytm.enabled:
        # do fetch in a batches of 50 for best api quota usage efficency
        video_ids = text_col.str.extractall(youtube.RE_YOUTUBE_URL_ID).video_id.to_list()
        _ = ytm.get_videos(video_ids=video_ids, query='', allow_fetch=yt_fetch)
    
    # Transform all youtube URLs into custom metadata tag for better LLM comprehension
    return text_col.apply(ytm.encode)

def has_time_column(df_or_csv:str|pd.DataFrame):
    if not isinstance(df_or_csv, pd.DataFrame):
        df_or_csv = pd.read_csv(df_or_csv, nrows=0)
    
    column_names = df_or_csv.columns.str.upper()

    return column_names.isin(['DATE','TIMESTAMP']).any()

def data_source_format(df_chat: pd.DataFrame, ):
    csv_columns = set(df_chat.columns.str.upper())

    required_cols_discord = ["AuthorID","Author", "Content", "Date"]
    required_cols_other = ["username", "text"]

    cols_discord_upset = set(map(str.upper, required_cols_discord))
    cols_other_upset = set(map(str.upper, required_cols_other))

    if (cols_discord_upset & csv_columns) == cols_discord_upset:
        return 'discord'
    
    if (cols_other_upset & csv_columns) == cols_other_upset:
        return 'other'
    
    raise RuntimeError('Missing required columns. Columns should have at least: '
                       f'{required_cols_discord!r} for Discord Chat or '
                       f'{required_cols_other!r} for other chat exports.')


def process_csv(chat_csv:str, youtube_encode_fetch:bool|tuple[bool,bool]=True, filter_prefixes:tuple[str, ...] = ('!','/'), merge_window:float = 7.0):
    df_chat = pd.read_csv(chat_csv)
    data_source = data_source_format(df_chat)

    if data_source == 'discord':
        print('Using format: Discord')
        return process_discord_chat(df_chat, cmd_prefixes=filter_prefixes, youtube_encode_fetch=youtube_encode_fetch)
    
    if data_source == 'other':
        print('Using format: Other')
        return process_other_chat(df_chat, youtube_encode_fetch, merge_window=merge_window)
    
    
def process_other_chat(df_chat: pd.DataFrame, youtube_encode_fetch:bool|tuple[bool,bool]=True, merge_window:float = 7.0):
    '''Required columns: ["username", "text"]. _Strongly_ recommended columns: ["timestamp"].
    
    timestamp should be column of datetimes with at least minute resolution and a consistent formatting (e.g. YYYY-mm-dd HH:MM:SS).

    If timestamp is not provided, all messages will be assumed to have been sent at an exact 5 minute interval ending 2024-01-01.
    '''
    df_chat = to_common_format(df_chat)
    df_chat = _process_chat(df_chat, 'other', youtube_encode_fetch=youtube_encode_fetch, merge_window=merge_window)

    return df_chat

def process_discord_chat(df_chat: pd.DataFrame, cmd_prefixes:tuple[str, ...]=('!','/'), youtube_encode_fetch: bool|tuple[bool,bool]=True):
    '''Required columns: ["AuthorID", "Author", "Date", "Content" ]. Unused columns: ["Reactions", "Attachments"]'''
    df_chat = df_chat.rename(columns=str.title).rename(columns={'Authorid':'AuthorID'}) # normalize col names
    # format="%Y-%m-%dT%H:%M:%S.%f%z")#, format='%m/%d/%Y %I:%M %p')
    df_chat['Date'] = df_chat['Date'].pipe(pd.to_datetime, utc=True, format='mixed') #  ISO8601
    df_chat = _process_chat(df_chat, 'discord', youtube_encode_fetch=youtube_encode_fetch, merge_window=7.0, cmd_prefixes=cmd_prefixes)

    return df_chat

def _process_chat(df_chat:pd.DataFrame, data_source:typing.Literal['discord','other'], youtube_encode_fetch:bool|tuple[bool,bool]=True, merge_window:float = 7.0, cmd_prefixes:tuple[str, ...]=None):
    user_index = get_make_userindex(df_chat)
    bot_data = user_index[0]
    
    nrec_init = df_chat.shape[0]
    
    # format="%Y-%m-%dT%H:%M:%S.%f%z")#, format='%m/%d/%Y %I:%M %p')
    #df_chat['Date'] = df_chat['Date'].pipe(pd.to_datetime, utc=True, format='mixed') #  ISO8601

    # replace 2+ newlines with 1 newline, since we may use \n\n to separate messages
    df_chat['text'] = df_chat['Content'].str.replace(r'\n{2,}','\n', regex=True)
    df_chat = df_chat.dropna(subset='text').reset_index(drop=True)

    if data_source == 'discord':
        # Feel free to set enable=False unless you also happen to have spammed your server with a CharRNN July 23, 2018
        df_chat = _targeted_chat_filter(df_chat, enable=True).reset_index(drop=True)
        df_chat.loc[:,'text'] = discord_text_filters(df_chat['text'], cmd_prefixes=cmd_prefixes)
        df_chat = df_chat.dropna(subset='text').reset_index(drop=True) # Drop nulled texts
    
    # use displayName for user column
    id_to_dname = useridx.get_users('dname', by='id', user_index=user_index, include_bot=True)
    df_chat['user'] = df_chat['AuthorID'].map(id_to_dname)
    #return df_chat, user_index
    is_cloneus_bot = df_chat['AuthorID'] == bot_data['id']

    df_chat['pre_bot'] = True
    if is_cloneus_bot.any():
        # use the first message by bot to denote potentially unsafe training data
        df_chat['pre_bot'] = df_chat.index < df_chat[is_cloneus_bot].index[0]
        df_chat = df_chat[~is_cloneus_bot] # drop Cloneus messages
        
    # Drop any messages from users not in the users.json (too few messages, in exclusion list) 
    nrec_before_user = df_chat.shape[0]
    df_chat = df_chat.dropna(subset='user').reset_index(drop=True)
    nrec_drop_users = df_chat.shape[0]

    
    # Want this to be AFTER filtering to avoid YouTube API calls on omitted content
    df_chat.loc[:,'text'] = apply_youtube_encoding(df_chat['text'], youtube_encode_fetch)
    # this will drop ~< 500 samples with invalid YouTube links as the only source of text. Verified that these are indeed not valid links
    df_chat = df_chat[df_chat['text'] != ''].reset_index(drop=True)
    
    if df_chat['Date'].isna().any():
        df_chat['time_gap'] = None
        df_chat['user_sequence'] =(df_chat['user'] != df_chat['user'].shift(1)).cumsum()
    else:
        # TODO: Consider what to do about extremely long chat streaks (e.g. 367 consectutive messages by a user on 2023-01-19)
        # TODO: Consider implications of assigning sequence/sessions after filtering vs before filtering
        df_chat['time_gap'] = df_chat['Date'].diff().fillna(pd.Timedelta(0)).dt.total_seconds()
        # If message is from a different user or time gap is greater than 7 minutes, then it's a new sequence
        # - https://www.reddit.com/r/discordapp/comments/9u2tdz/whats_the_new_cutoff_time_for_separating_messages/
        df_chat['user_sequence'] = (((df_chat['user'] != df_chat['user'].shift(1)) | (df_chat['time_gap']>(merge_window*60))).cumsum())


    nrec_final = df_chat.shape[0]
    print(
        f'Init Record Count: {nrec_init}\n'
        #f'Drop Discord links: {nrec_init-nrec_drop_discapp} records\n'
        #f'Drop Targeted: {nrec_drop_discapp-nrec_drop_targeted} records\n'
        #f'Drop Cmd/Bot Messages: {nrec_drop_targeted-nrec_drop_cmdbot} records\n'
        f'Dropped {nrec_before_user-nrec_drop_users} messages with missing/removed users\n'
        #f'Drop Bad YT Links: {nrec_drop_users-nrec_drop_bad_ytlink} records\n'
        f'Final Record Count: {nrec_final} (drop total: {nrec_init-nrec_final})'
        )
    
    return df_chat



def expand_sessions(chat_session: pd.Series, min_session_length:int=1):
    '''Forward Merge session groups that have less than `min_session_msgs` items'''
    if min_session_length<=1:
        # skip loop and return unless > 1
        return chat_session

    # just need a number large enough to not exit early during prev_n_smsess comparison
    n_small_session = chat_session.shape[0]+1
    sess_sizes = []
    prev_n_smsess = n_small_session
    sess_max = chat_session.max()
    sess_min = chat_session.min()
    while n_small_session > 0:
        under_len = chat_session.value_counts() < min_session_length
        small_sessions = under_len[under_len].index
        n_small_session = small_sessions.shape[0]
        # Avoid inf loop: 
        # sess-= 1 if didn't reduce the number of small sessions, 
        # +=1 is default case since it aligns better with the assumption we take that 4hr break = new topic session
        # but if we have __4hr_break__ [TEXT] __4hr_break__ then seems more likely that text should be mapped forward instead of backward
        shift = -1 if prev_n_smsess == n_small_session else 1        
        chat_session = chat_session.where(~chat_session.isin(small_sessions), other = (chat_session + shift).clip(sess_min, sess_max))
        
        prev_n_smsess = n_small_session
        sess_sizes.append(n_small_session)
        
    if sess_sizes and sess_sizes[0]>0:    
        print('sessions < min_session_length:',' -> '.join(map(str,sess_sizes)))
    return chat_session


def delineate_sessions(df_chats:pd.DataFrame, hours_between_sessions:int=4, min_session_length:int=1) -> pd.Series:
    #te_sessions = (df_chats['time_gap'] >= (hours_between_sessions*60*60)).cumsum()
    train_chat_session = expand_sessions((df_chats.loc[df_chats.split=='train', 'time_gap'] >= (hours_between_sessions*60*60)).cumsum(), min_session_length=min_session_length)
    eval_chat_session = expand_sessions((df_chats.loc[df_chats.split=='eval', 'time_gap'] >= (hours_between_sessions*60*60)).cumsum(), min_session_length=min_session_length)
    # max+1 to avoid collisions since 0 will be min of eval
    eval_chat_session += train_chat_session.max()+1
    #train_chat_session = expand_sessions(te_sessions[df_chats.split=='train'].copy(), min_session_length=min_session_length)
    #eval_chat_session = expand_sessions(te_sessions[df_chats.split=='eval'].copy(), min_session_length=min_session_length) + 1 # need +1 to force break 1st group
    
    #assert train_chat_session.max() < eval_chat_session.min(), 'Possible data leak. Train chat_session overlaps eval chat_session'
    chat_session = pd.concat([train_chat_session, eval_chat_session])
    
    return chat_session


def assign_split(df_chat:pd.DataFrame, eval_frac: (float|typing.Literal['after_bot']) = 0.005):
    if eval_frac=='after_bot':
        df_chat['split'] = df_chat['pre_bot'].apply(lambda x: 'train' if x else 'eval')
    else:
        # Use the last eval_frac chat groups for eval
        n_eval = int(df_chat.shape[0]*eval_frac)
        if df_chat['Date'].isna().any():
            df_chat['split'] = 'train'
            df_chat.iloc[-n_eval:, -1] = 'eval'
            print(f'Using last {n_eval} messages for evaluation set')
        else:
            eval_start_date = df_chat['Date'].iloc[-n_eval]
            df_chat.loc[:, 'split'] = (df_chat['Date'] < eval_start_date).apply(lambda x: 'train' if x else 'eval')
            n_date_eval = (df_chat['split']=='eval').sum()
            assert n_date_eval == n_eval, f'Bad split assignment! Eval set size missmatch: {n_date_eval} != {n_eval}'
            print(f'Using messages after {eval_start_date} for eval set. n_messages = {n_date_eval}')
    
    return df_chat

def format_text_tags(df_proc:pd.DataFrame, author_tag:str, tag_sep:str=None, postfix:str=None, eval_frac: (float|typing.Literal['after_bot']) = 0.005):
    '''Creates a formatted text col, merges user messages, and assigns train, eval split.

    Groups chat data by user_sequence and join by a new line. Creates formatted_text column where each takes the form: 
        `author_tag` `tag_sep` <TEXT> `postfix`.
    Assigns a split column for training and evaluation based on the `eval_frac` parameter.

    Args:
        df_proc: DataFrame containing preprocessed chat data.
        author_tag: The format string for the author tag. e.g. '[USER:{author}]'.
        tag_sep: Separator string to use between the author tag and the text.
        postfix: String to append at the end of each formatted text. 
        eval_frac (float|'after_bot'): If float, the fraction of chat groups to use for evaluation. If 'after_bot', use all messages after the bot's first message (default: 0.005).
    '''
    user_index = get_make_userindex(df_proc)
    df_chats = df_proc.groupby('user_sequence', as_index=False)[
        ['user', 'Date', 'time_gap', 'text','pre_bot']].agg(
        {'user':'first', 'Date':'last', 'time_gap':'first', 'text':list, 'pre_bot':'first'}).copy()
    
    # Join consecutive author messages with new line
    df_chats['text'] = df_chats['text'].str.join('\n')
    
    df_chats['formatted_author_tag'] = df_chats['user'].apply(useridx.format_author_tag, author_tag=author_tag, user_index=user_index) + ('' if tag_sep is None else tag_sep)

    # BUG until 2023-11-29 was df_proc.text instead of df_all
    df_chats['formatted_text'] = df_chats['formatted_author_tag'] + df_chats['text'] + ('' if postfix is None else postfix)  
    
    df_chats = assign_split(df_chats, eval_frac)
    return df_chats

def label_chat_sessions(df_proc: pd.DataFrame, hours_between_sessions:(int|list[int]|None) = 4, min_session_length:int=1):
    """Prepares data for use in hf dataset. 
    If min_session_length>1, any groups containing fewer than `min_session_length` messages will be forcably regrouped until.

    Args:
        df_proc: DataFrame containing preprocessed chat data.
        hours_between_sessions: Hours of silence before starting a new group. If a list is given, will copy the data group for each (default: 4).
        min_session_length: The minimum number of messages required for a session to be considered valid (default: 1).

    Returns:
        pd.DataFrame: A DataFrame with the formatted text and session splits for training and evaluation.
    """
    # TODO: Argument to allow for data subset training 
    # df_proc = df_proc[(df_proc.Date > '2020') & (df_proc.Date < '2022')].copy()
    #df_chats = format_text_tags(df_proc, author_tag, tag_sep=tag_sep, postfix=postfix, eval_frac=eval_frac)
    df_chats = df_proc.copy()
    if hours_between_sessions is None:
        return df_chats
    
    if isinstance(hours_between_sessions, (int,float)):
        df_chats['chat_session'] = delineate_sessions(df_chats, hours_between_sessions, min_session_length) + int(hours_between_sessions*1e9)
        return df_chats
    
    chatgrps = {h: delineate_sessions(df_chats, h, min_session_length) for h in hours_between_sessions}
    
    for h1,h2 in itertools.combinations(hours_between_sessions, 2):
        print(f'({h1}h, {h2}h) - duplicate grouped items:',(chatgrps[h1]==chatgrps[h2]).sum())
        # Not the number of identical groups, number of items assigned same group at that index
        # TODO: Duplicate groups would be more useful to know
    
    # collisions possible if > 1 bill groups, but I mean...
    return pd.concat([df_chats.assign(chat_session=chatgrps[h] + int(h*1e9)).copy() for h in hours_between_sessions]) 





