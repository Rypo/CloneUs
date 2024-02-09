import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import cloneus.core.paths as cpaths


USERS_FILEPATH = cpaths.ROOT_DIR/'config/users.json'

def get_args():
    parser = argparse.ArgumentParser(description='Setup the initial files needed to run the rest of the program.')
    parser.add_argument('chat_file', type=str,
                        help='Json file exported from DiscordChatExporter.')
    
    parser.add_argument('-t','--threshold', default=100, type=int, required=False,
                        help='Minimum number of user message count to include them in the users file.')
    
    parser.add_argument('--include-bots', action='store_true', 
                        help='Whether to include any existing server bots as users in the users file.')
    return parser.parse_args()


def read_chatjson(chat_jsonfile:str|Path) -> pd.DataFrame:
    with open(chat_jsonfile) as f:
        chat_json = json.load(f)
    
    print('~~~~~ Add these to your .env file ~~~~~')
    print('GUILDS_ID: ', chat_json['guild']['id'])
    print('CHANNEL_ID:', chat_json['channel']['id'])
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    
    df_msgs = pd.json_normalize(chat_json['messages'])
    return df_msgs


def create_usersfile(df_msgs: pd.DataFrame, msg_threshold:int = 100, exclude_bots:bool = True): 
    df_users = df_msgs[['author.id', 'author.name', 'author.nickname', 'author.isBot']].rename(columns = lambda c: c.split('.')[-1])
    
    # Exclude any existing bots from the users file
    if exclude_bots:
        df_users = df_users[~df_users.isBot]

    # Exclude users with too few messages
    msg_counts = df_users.id.value_counts()
    df_users = df_users[df_users.id.isin(msg_counts[msg_counts >= msg_threshold].index)]

    df_users = df_users.rename(columns={'name':'username', 'nickname':'displayName'}).drop_duplicates().reset_index(drop=True).copy()
    #users['id'] = users['id'].astype(int)
    df_users.insert(1, 'firstName', None)
    df_users.insert(2, 'authorInitial', None)

    users = {
        'BOT': {
            'id': '000000000000000000',   # replace in config/users.json (required)
            # 'firstName': None,          # Not used.
            # 'authorInitial': None,      # Not used.
            'username': 'cloneus',        # replace in config/users.json (required)
            'displayName': 'Cloneus',     # replace in config/users.json (optional)
            'isBot': True},
        'USERS': df_users.to_dict('records')
    }
    
    with open(USERS_FILEPATH, 'w') as f:
        json.dump(users, f, indent=2)
        print('Created users file at:', str(USERS_FILEPATH))
        print('*'*100)
        print("(required) DON'T FORGET: You NEED to manually update the config/users.json file with your bot's info.")
        print("(recommended) Update user.jsons with each user's first name and a unique initial")
        print('*'*100)
    
def to_chatcsv(df_msgs: pd.DataFrame, out_csvfile: str|Path):
    df_chatcsv = df_msgs[['author.id','author.name','timestamp','content','attachments','reactions']].copy()
    df_chatcsv.rename(columns=dict(zip(df_chatcsv.columns, ['AuthorId','Author','Date','Content','Attachments','Reactions'])), inplace=True)
    df_chatcsv.loc[:,['Attachments','Reactions']] = df_chatcsv[['Attachments','Reactions']].applymap(lambda v: np.nan if isinstance(v,list) and not v else v)
    df_chatcsv.to_csv(out_csvfile, index=False)
    print('Created Message csv file:', out_csvfile)        


def check_overwrite(test_path:str|Path) -> bool:
    can_write = True
    if Path(test_path).exists():
        inp=''
        while inp not in ['y','n','yes','no','q','quit']:
            inp=input(f'File Exists! {test_path}. Overwrite? (y/n/q) ').lower()
        
        if inp.startswith('q'):
            sys.exit(0)

        can_write = inp.startswith('y')
    return can_write

if __name__ == '__main__':
    args = get_args()

    df_msgs = read_chatjson(args.chat_file)

    if check_overwrite(USERS_FILEPATH):
        create_usersfile(df_msgs, msg_threshold=args.threshold, exclude_bots=(not args.include_bots))

    out_csvpath = Path(args.chat_file).with_suffix('.csv')
    
    
    if check_overwrite(out_csvpath):
        to_chatcsv(df_msgs, out_csvpath)


