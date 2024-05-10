import re
import json
import typing
import warnings
import functools
from collections import defaultdict

from unidecode import unidecode

import cloneus.core.paths as cpaths

USERS_FILEPATH = cpaths.ROOT_DIR/'config/users.json'

USER_DATA = None

def _read_userdata():
    try:
        with open(USERS_FILEPATH,'r') as f:
            USER_DATA: dict = json.load(f)
            return USER_DATA
    except FileNotFoundError as e:
        warnings.warn('Missing config/users.json. You must run `scripts/build.py` or otherwise create users.json to proceed.')
        return placeholder_userdata(False)

def placeholder_userdata(populate=False):
    user_data = {
        "BOT": {"id": 0, "username": "cloneus", "displayName": "Cloneus", "isBot": True},
        "USERS": []
    }
    if populate:
        import string
        for i in range(1,8):
            c = string.ascii_lowercase[i-1]
            c_up = c.upper()

            placeholder_user = {
                "id": i,
                "firstName": f"{c_up}FirstName",
                "authorInitial": c,
                "username": f"placeholder_username_{i}",
                "displayName": "PlaceHolder_displayName{i}",
                "isBot": False
            }

            user_data["USERS"].append(placeholder_user)
    return user_data

    

# if k and v: return dict mapping all user's k -> v
# if k and no v: return dict of dicts indexed by k where values are each user's whole data dict (including k)
# if v and no k: return list of values, all user's v in no particular order
# if neither k nor v: return list of dicts of all user's data dicts
@functools.cache
def get_users(get: typing.Literal['dname','initial','fname','uname','bot','id']|None = None, 
              by: typing.Literal['dname','initial','fname','uname','id']|None = None, 
              *, include_bot:bool=False
              ) -> dict[str,] | dict[str, dict[str,]] | list | list[dict[str,]]:
    '''Reindex and filter user data.

    aliases: dname -> displayName; initial -> authorInitial; fname -> firstName; uname -> username; bot -> isBot

    Args:
        get: The value to select from each user's data. If None, return unfiltered user data
        by: The key to index user data by. If None, return a list instead of a dict
        include_bot: If True, include Cloneus bot as first entry in the returned data

    Returns:
        dict: { by : userdata[get] } if both `by` and `get`
        dict[dict]: { by : userdata } if `by` and not `get`
        list: userdata[get] if `get` and not `by`
        list[dict]: userdata if neither `get` nor `by`

    Raises:
        ValueError: if `by` has missing or duplicate values
    '''
    
    global USER_DATA
    
    if USER_DATA is None:
        USER_DATA = _read_userdata()
    
    user_data = USER_DATA["USERS"]
    
    if include_bot:
        bot_data = {"firstName":'BottyMcBotface', "authorInitial":'b', **USER_DATA["BOT"]}
        user_data = [bot_data, *user_data]
    
    alias = {'dname':'displayName', 'initial':'authorInitial', 'fname':'firstName', 'uname':'username', 'bot':'isBot', 'id':'id'}
    
    # I know I'm going to try pluralize it, just save myself the headache
    if by:
        by = by.rstrip('s')
    if get:
        get = get.rstrip('s')

    if by:
        # this could go later, but it *feels* better earlier
        _key = alias.get(by, by)
        _vals = [u[_key] for u in user_data]
        if not all(_vals):
            raise ValueError(f'Cannot index by {by!r}. Not all users assigned a value for {by!r}.')
        if len(set(_vals)) != len(_vals):
            raise ValueError(f'Cannot index by {by!r}. Values for {by!r} are not unique across users.')


    if by is None and get is None:
        return user_data
    
    if get and not by:
        val = alias.get(get, get)
        return [u[val] for u in user_data]

    
    key = alias.get(by, by)

    if by and not get:
        return {u[key]: u for u in user_data}
    
    # idx_by and select
    val = alias.get(get, get)
    return {u[key]: u[val] for u in user_data}



# def _load_author_helpers():
#     USER_DATA = _read_userdata()
#     author_display_names = []
#     author_to_fname = {} # May have nulls if no firstName
#     initial_to_author = {} # may have duplicate drops if no authorInitial
#     author_to_id = {}
    
#     for u in USER_DATA['USERS']:
#         author = u['displayName']
#         author_display_names.append(author)
#         author_to_fname[author] = u['firstName']
#         author_to_id[author] = u['id']

#         if u['authorInitial']:
#             initial_to_author[u['authorInitial'].lower()] = author
    
#     # If initials are not provided, assign using first names if available
#     # Otherwise, use display names for initials
#     # TODO: allow partial initial assignment. 

#     if not initial_to_author:
#         fnames = list(author_to_fname.values())
#         if all(fnames):
#             initial_to_author = default_initials(fnames)
#         else:
#             initial_to_author = default_initials(author_display_names)

#     return USER_DATA, author_display_names, author_to_fname, author_to_id, initial_to_author

# Consts
# author_display_names : get_users('dname')
# author_to_fname: get_users('fname', 'dname')
# author_to_id: get_users('id', 'dname')
# initial_to_author: get_users('dname', 'initial')
# BOT_NAME: get_users('dname', include_bot = True)[0]
# try:
#     USER_DATA, author_display_names, author_to_fname, author_to_id, initial_to_author = _load_author_helpers()
#     BOT_NAME = USER_DATA['BOT']['displayName']
# except FileNotFoundError as e:
#     warnings.warn('Missing config/users.json. Until you run scripts/build.py or otherwise create user.json, the following constants will be unavailable:'
#                   '`USER_DATA`, `author_display_names`, `author_to_fname`, `author_to_id`, `initial_to_author`, `BOT_NAME`')



def format_author_tag(user_display_name:str, author_tag:str, *, insert_raw:bool=False):
    if insert_raw:
        return author_tag.format(author=user_display_name, lauthor=user_display_name, fname=user_display_name)
    
    return author_tag.format(author=user_display_name, lauthor=user_display_name.lower(), fname=get_users('fname', 'dname').get(user_display_name,user_display_name))


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
    i_to_dname = get_users('dname', by='initial')
    if i_to_dname:
        assert len(set([i.lower() for i in i_to_dname])) == len(i_to_dname), 'Each user must be assigned a unique, case-insensitive initial or char+num(s).'

def parse_initials(initials_seq:str, return_dispname=False):
    '''aBC|A,b,c|a B c -> [a,b,c]. a2b1c1d11 -> [a2,b1,c1,d11]'''
    initials = re.findall(r'([a-z]\d*)[, ]?', initials_seq, re.I)
    if not return_dispname:
        return [i.lower() for i in initials]
    i_to_dname = get_users('dname', 'initial')
    return [i_to_dname[i] for i in initials]
    

def assign_initials(names:list[str]):
    snames = sorted(set(names), key=str.lower)
    # take the first alpha letter from the decoded name, if none, default is "x"
    initials = [next(filter(str.isalpha, unidecode(name)), 'x').lower() for name in snames]
    
    # if they are all unique, no digits required.
    if len(set(initials)) == len(snames):
        return dict(zip(initials, snames))
    
    dd = defaultdict(lambda: 1)
    uinitials = []

    for c in initials:
        uinitials.append(f'{c}{dd[c]}')
        dd[c]+=1
    
    return dict(zip(uinitials, snames))

# If initials are not provided, assign using first names if available
# Otherwise, use display names for initials
# TODO: allow partial initial assignment. 
def default_initials():
    '''Get initials for each user
    priorty: 
        1: unique initial first names
        2: unique initial display names
        3: first names
        4: display names

    Give priority to first names, if set.'''
    fnames = get_users('fname')
    dnames = get_users('dname')
    
    i_to_fname = {}
    i_to_dname = {}

    if all(fnames):
        i_to_fname = assign_initials(fnames)
        if ''.join(i_to_fname.keys()).isalpha():
            return i_to_fname
        
    if all(dnames):
        i_to_dname = assign_initials(dnames)
        if ''.join(i_to_dname.key()).isalpha():
            return i_to_dname
        
    if i_to_fname:
        return i_to_fname
    
    return i_to_dname
