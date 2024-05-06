import re
import json
from collections import defaultdict

import cloneus.core.paths as cpaths

USERS_FILEPATH = cpaths.ROOT_DIR/'config/users.json'

def _read_userdata():
    try:
        with open(USERS_FILEPATH,'r') as f:
            USER_DATA: dict = json.load(f)
            return USER_DATA
    except FileNotFoundError as e:
        print('users.json Not Found! Call scripts/build.py before running.')
        raise e

def _load_author_helpers():
    USER_DATA = _read_userdata()
    author_display_names = []
    author_to_fname = {} # May have nulls if no firstName
    initial_to_author = {} # may have duplicate drops if no authorInitial
    author_to_id = {}
    
    for u in USER_DATA['USERS']:
        author = u['displayName']
        author_display_names.append(author)
        author_to_fname[author] = u['firstName']
        author_to_id[author] = u['id']

        if u['authorInitial']:
            initial_to_author[u['authorInitial'].lower()] = author
    
    # If initials are not provided, assign using first names if available
    # Otherwise, use display names for initials
    # TODO: allow partial initial assignment. 

    if not initial_to_author:
        fnames = list(author_to_fname.values())
        if all(fnames):
            initial_to_author = default_initials(fnames)
        else:
            initial_to_author = default_initials(author_display_names)

    return USER_DATA, author_display_names, author_to_fname, author_to_id, initial_to_author

# Consts
USER_DATA, author_display_names, author_to_fname, author_to_id, initial_to_author = _load_author_helpers()
BOT_NAME = USER_DATA['BOT']['displayName']


def format_author_tag(user_display_name:str, author_tag:str, *, insert_raw:bool=False):
    if insert_raw:
        return author_tag.format(author=user_display_name, lauthor=user_display_name, fname=user_display_name)
    
    return author_tag.format(author=user_display_name, lauthor=user_display_name.lower(), fname=author_to_fname.get(user_display_name,user_display_name))


def to_jinja_template(tag_sep:str, postfix:str):
    # role will be the pre-formated author tag
    template=(
        "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
        "{{ bos_token }}"
        "{% for message in messages %}"
        "{{ message['role'] + '__TAG_SEP__' + message['content'] + '__POSTFIX__' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '' }}"
        "{% endif %}"
    )
    template = template.replace('__TAG_SEP__',tag_sep).replace('__POSTFIX__', postfix)
    return template

def check_author_initials():
    if initial_to_author:
        assert len(set([i.lower() for i in initial_to_author])) == len(author_display_names), 'Each user must be assigned a unique, case-insensitive initial or char+num(s).'

def parse_initials(initials_seq:str, return_dispname=False):
    '''aBC|A,b,c|a B c -> [a,b,c]. a2b1c1d11 -> [a2,b1,c1,d11]'''
    initials = re.findall(r'([a-z]\d*)[, ]?', initials_seq, re.I)
    if not return_dispname:
        return [i.lower() for i in initials]
    return [initial_to_author[i] for i in initials]
    

def default_initials(names:list[str]):
    snames = sorted(names, key=str.lower)
    initials = [name[0] for name in snames]
    if len(set(initials)) == len(snames):
        return dict(zip(initials, snames))
    
    dd = defaultdict(lambda: 1)
    uinitials = []

    for c in initials:
        uinitials.append(f'{c}{dd[c]}')
        dd[c]+=1
    
    return dict(zip(uinitials, snames))


def get_name_map():
    names = [(n,u) for u,n in author_to_fname.items()]
    name_mapping = ', '.join([f'{n} ({u})' for n,u in names[:-1]]) + f', and {names[-1][0]} ({names[-1][1]})'
    return name_mapping