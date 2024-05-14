import re
import json
import typing
import warnings
import functools
from collections import defaultdict

from unidecode import unidecode

import cloneus.core.paths as cpaths

USERS_FILEPATH = cpaths.ROOT_DIR/'config/users.json'

USER_INDEX = None

def _read_user_index():
    try:
        with open(USERS_FILEPATH,'r') as f:
            USER_INDEX: dict = json.load(f)
            return USER_INDEX
    except FileNotFoundError as e:
        raise FileNotFoundError('Missing config/users.json. You must run `scripts/build.py` or otherwise create users.json to proceed.')

def write_user_index(user_index:dict, overwrite:bool=False):
    user_initials = get_users('initial', user_index=user_index)
    if not all(user_initials):
        ini_to_name,namekey = create_default_initials(
            get_users('fname', user_index=user_index), 
            get_users('dname', user_index=user_index), 
            get_users('uname', user_index=user_index)
        )
        
        ud = get_users(by=namekey, user_index=user_index)
        
        for ini,name in ini_to_name.items():
            # if not ud[name]['authorInitial']:
            ud[name]['authorInitial'] = ini

    
    if USERS_FILEPATH.exists() and not overwrite:
        raise FileExistsError(f'{USERS_FILEPATH} exists and overwrite=False')
    
    with open(USERS_FILEPATH, 'w') as f:
        json.dump(user_index, f, indent=2)
        print('Created users file at:', str(USERS_FILEPATH))

def placeholder_userdata(populate=False):
    user_index = [
         {"id": 0, "firstName": "Cloney","authorInitial": "b0", "username": "cloneus", "displayName": "Cloneus", "isBot": True},
    ]
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

            user_index.append(placeholder_user)
    return user_index

    

# if k and v: return dict mapping all user's k -> v
# if k and no v: return dict of dicts indexed by k where values are each user's whole data dict (including k)
# if v and no k: return list of values, all user's v in no particular order
# if neither k nor v: return list of dicts of all user's data dicts
# @functools.cache
def get_users(get: typing.Literal['dname','initial','fname','uname','bot','id']|None = None, 
              by: typing.Literal['dname','initial','fname','uname','id']|None = None, 
              *, include_bot:bool=False, user_index: dict = None,
              ) -> dict[str,] | dict[str, dict[str,]] | list | list[dict[str,]]:
    '''Reindex and filter user data index.

    aliases: dname -> displayName; initial -> authorInitial; fname -> firstName; uname -> username; bot -> isBot

    Args:
        get: The value to select from each user's data. If None, return unfiltered user data
        by: The key to index user data by. If None, return a list instead of a dict
        include_bot: If True, include Cloneus bot as first entry in the returned data
        user_index: The user data index to use. If None, will use global default (config/users.json) 

    Returns:
        dict: { by : user_index[get] } if both `by` and `get`
        dict[dict]: { by : user_index } if `by` and not `get`
        list: user_index[get] if `get` and not `by`
        list[dict]: user_index if neither `get` nor `by`

    Raises:
        ValueError: if `by` has missing or duplicate values
    '''
    
    if user_index is None:
        global USER_INDEX
        
        if USER_INDEX is None:
            USER_INDEX = _read_user_index()
    
        user_index = USER_INDEX
    
    
    if not include_bot:
        user_index = user_index[1:]
    else:
        assert user_index[0]['isBot'], 'Cloneus bot not in user data!'
        user_index = user_index[:] # shallow copy just for consistency with exclude
        #user_index = [bot_data, *user_index[1:]]
        
    
    alias = {'dname':'displayName', 'initial':'authorInitial', 'fname':'firstName', 'uname':'username', 'bot':'isBot', 'id':'id'}
    

    if by:
        # this could go later, but it *feels* better earlier
        _key = alias.get(by, by)
        _vals = [u[_key] for u in user_index]
        if not all(_vals):
            raise ValueError(f'Cannot index by {by!r}. Not all users assigned a value for {by!r}.')
        if len(set(_vals)) != len(_vals):
            raise ValueError(f'Cannot index by {by!r}. Values for {by!r} are not unique across users.')


    if by is None and get is None:
        return user_index
    
    if get and not by:
        val = alias.get(get, get)
        return [u[val] for u in user_index]

    
    key = alias.get(by, by)

    if by and not get:
        return {u[key]: u for u in user_index}
    
    # idx_by and select
    val = alias.get(get, get)
    return {u[key]: u[val] for u in user_index}




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

def check_author_initials(user_initials:list[str] = None):
    if user_initials is None:
        user_initials = get_users('initial')
    assert len(set([i.lower() for i in user_initials])) == len(user_initials), 'Each user must be assigned a unique, case-insensitive initial or char+num(s).'

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
def create_default_initials(first_names:list[str]=None, display_names:list[str]=None, usernames:list[str] = None):
    '''Get initials for each user based name
    
    priorty: 
        1: unique initial first names
        2: unique initial display names
        3: unique initial usernames
        4: first names
        5: display names
        6: usernames
    '''
    
    if first_names is None:
        first_names = []
    if display_names is None:
        display_names = []
    if usernames is None:
        usernames = []

    i_to_fname = {}
    i_to_dname = {}
    i_to_uname = {}

    if any(first_names):
        i_to_fname = assign_initials(first_names)
        if ''.join(i_to_fname.keys()).isalpha():
            return i_to_fname,'fname'
        
    if any(display_names):
        i_to_dname = assign_initials(display_names)
        if ''.join(i_to_dname.key()).isalpha():
            return i_to_dname,'dname'
    
    if any(usernames):
        i_to_uname = assign_initials(usernames)
        if ''.join(i_to_uname.key()).isalpha():
            return i_to_uname,'uname'
                
    if i_to_fname:
        return i_to_fname, 'fname'
    
    if i_to_dname:
        return i_to_dname, 'dname'
    
    if i_to_uname:
        return i_to_uname, 'uname'
    
    raise ValueError('Need to specify at least one of `first_names`, `display_names`, or `usernames`')