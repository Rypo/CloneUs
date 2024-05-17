import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cloneus.data import useridx, etl
ROOT_DIR = Path(__file__).parent.parent
USERS_FILEPATH = useridx.USERS_FILEPATH # USERS_FILEPATH = ROOT_DIR/'config/users.json'

# Keep the read json in build.py
# move csv processing to etl


# If build.py
# 	Ask for first names
# else
# 	Assume no first names, but announce via print 

# Json Pros:
# - Tells you the Guild ID + Guild Channel
# - Uses nickname as displayName rather than username
# - Automatically differentiates bot user from human users
# - Better Date/timestamp resolution

# Json Cons:
# - Data export with DiscordChatExporter takes longer
# - Requires an additional processing step. (Only run once, takes ~15 seconds) 

# Put in readme:
# 	quick start
# 		edit train_config
# 		train -d path_to_data.csv
# 		use this method if you 
# 			1. don't know first names
# 			2. have no interesting using discord server
# 			3. youtube api
# 	better start
# 		run build.py your_data.csv / you data json
#   Zero start (not recommended)
#   - show the Cloneus.from_model_id()
	
# 	section for non-discord
# 	section for discord
# 		from json
# 		from csv
# 		mention pros of json


def move_dotenv(root_dir:Path):
    if (root_dir/'.env_example').exists() and not (root_dir/'.env').exists():
        (root_dir/'.env_example').rename(root_dir/'.env')

def ask_overwrite(filepath:str|Path) -> bool:
    can_write = True
    if Path(filepath).exists():
        inp=''
        while (inp+'?')[0] not in ['y','n','q']:
            inp=input(f'Overwrite File: {filepath}? (y/n/q) ').lower()
        
        if inp.startswith('q'):
            print('Aborted.')
            sys.exit(0)

        can_write = inp.startswith('y')
    return can_write

def discord_json_to_csv(df_chat: pd.DataFrame, out_csvfile: str|Path):
    df_chatcsv = df_chat.copy().rename(columns={'username':'Author'})
    print(df_chatcsv.columns)
    df_chatcsv = df_chatcsv[['AuthorID','Author','Date','Content','Attachments','Reactions']]
    df_chatcsv.loc[:,['Attachments','Reactions']] = df_chatcsv[['Attachments','Reactions']].map(lambda v: np.nan if isinstance(v,list) and not v else v)
    df_chatcsv.to_csv(out_csvfile, index=False)
    print('Saved converted discord csv file:', out_csvfile)
    return out_csvfile

def read_discord_json(chat_jsonfile:str|Path) -> pd.DataFrame:
    with open(chat_jsonfile) as f:
        chat_json = json.load(f)
    
    print('~~~~~ Add these to your .env file ~~~~~')
    print('GUILDS_ID: ', chat_json['guild']['id'])
    print('CHANNEL_ID:', chat_json['channel']['id'])
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    
    df_chat = pd.json_normalize(chat_json['messages'])[
        ['timestamp', 'content', 'attachments', 'reactions', 'author.id', 'author.name', 'author.nickname', 'author.isBot']
    ]
    df_chat.rename(columns={'timestamp':'Date', 'content':'Content', 'attachments':'Attachments', 'reactions':'Reactions',  
                            'author.id':'AuthorID', 'author.name':'username', 'author.nickname':'displayName', 'author.isBot':'isBot'}, inplace=True)
    return df_chat

def read_chat_csv(chat_csvfile:str|Path):
    df_chat = pd.read_csv(chat_csvfile)
    data_source = etl.data_source_format(df_chat)

    if data_source=='discord':
        df_chat = df_chat.rename(columns=str.title).rename(columns={'Authorid':'AuthorID'}) # normalize col names
        return df_chat
    
    if data_source=='other':
        df_chat = df_chat.rename(columns=str.lower) # normalize col names
        df_chat['AuthorID'] = df_chat['username'].apply(lambda a: abs(hash(a)))
        return df_chat


def build_and_save(chat_filepath:str|Path, msg_threshold:float|int = 0.005, excluded_users:list[str] = None, bot_usernames: list[str] = None, exclude_bots:bool=True):
    chat_filepath = Path(chat_filepath)
    if chat_filepath.suffix == '.json':
        df_chat = read_discord_json(chat_filepath)
        csv_outpath = discord_json_to_csv(df_chat, chat_filepath.with_suffix('.csv'))
    else:
        df_chat = read_chat_csv(chat_filepath)
        csv_outpath = chat_filepath

    if bot_usernames and exclude_bots:
        if excluded_users is None:
            excluded_users = []
        
        excluded_users = list(set(excluded_users+bot_usernames))
    
    if msg_threshold >= 1:
        msg_threshold = int(msg_threshold)
    
    user_index = etl.user_index_from_chat(df_chat, msg_threshold, excluded_users, bot_usernames, interactive=True)
    
    if (can_write := ask_overwrite(USERS_FILEPATH)):
        useridx.write_user_index(user_index, overwrite=can_write)
    
    return csv_outpath


def get_args():
    parser = argparse.ArgumentParser(description='Build user index (config/users.json) and if necessary, convert format and save')
    parser.add_argument('chat_file', type=str,
                        help='path/to/chat.csv, or discordExport.json. If non-discord source: columns must include ["username", "text"]')
    
    parser.add_argument('-t','--threshold', default=0.005, type=float, required=False,
                        help='The minimum fraction of total messages that must belong to a user included in users.json. If >1 minimum messages count by user to be in users.json.')
    
    parser.add_argument('-e','--exclude', nargs='*',
                        help='List of usernames to exclude from user.json regardless of message count.')
    
    parser.add_argument('-b','--bots', nargs='*',
                        help='List of non-Cloneus bot usernames to optionally exclude from users.json. Can be determined automatically if data is a discord Json export.')
    
    parser.add_argument('--include-bots', action='store_true', 
                        help='Whether to include any non-Cloneus bots in users.json. Has no effect if not discord Json export or --bots flag is used.')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    move_dotenv(ROOT_DIR)
    csv_outpath = build_and_save(args.chat_file, msg_threshold=args.threshold, excluded_users=args.exclude, bot_usernames=args.bots, exclude_bots=(not args.include_bots))
    print(f'DONE. Update train_config.yaml with your csv path and start cloning or call `python scripts/train.py -d {csv_outpath}`')


