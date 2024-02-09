import json
import pandas as pd

import cloneus.core.paths as cpaths

USERS_FILEPATH = cpaths.ROOT_DIR/'config/users.json'


def _load_author_helpers():
    try:
        with open(USERS_FILEPATH,'r') as f:
            userdata = json.load(f)
    except FileNotFoundError as e:
        print('users.json Not Found! Call scripts/build.py before running.')
        raise e

    author_display_names = []
    author_to_fname = {} # May have nulls if no firstName
    initial_to_author = {} # may have duplicate drops if no authorInitial
    
    for u in userdata['USERS']:
        author_display_names.append(u['displayName'])
        author_to_fname[u['displayName']] = u['firstName']
        initial = u['authorInitial'] if u['authorInitial'] else u['username'][0]
        initial_to_author[initial] = u['displayName']

    return author_display_names, author_to_fname, initial_to_author

# Consts
author_display_names, author_to_fname, initial_to_author = _load_author_helpers()


def format_author_tag(user_display_name:str, author_tag:str):
    return author_tag.format(author=user_display_name, lauthor=user_display_name.lower(), fname=author_to_fname.get(user_display_name,user_display_name))


def get_name_map():
    names = [(n,u) for u,n in author_to_fname.items()]
    name_mapping = ', '.join([f'{n} ({u})' for n,u in names[:-1]]) + f', and {names[-1][0]} ({names[-1][1]})'
    return name_mapping