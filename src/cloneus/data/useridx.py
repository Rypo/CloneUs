import re
import json
import copy
import hashlib
from typing import Literal
import warnings
import itertools
from collections import defaultdict

import unidecode

from cloneus.types import cpaths 

USERS_FILEPATH = cpaths.ROOT_DIR/'config/users.json'

USER_INDEX = None

DEFAULT_CLONEUS_USER = {
    'id': 000000000000000000,     # replace in config/users.json (used by server)
    'firstName': 'Cloney',        # Not used.
    'authorInitial': 'b0',        # Do Not Change. Special assignment to differentiate from other bots. 
    'username': 'cloneus',        # replace in config/users.json (optional)
    'displayName': 'Cloneus',     # replace in config/users.json (used by server)
    'isBot': True
}

def user_index_exists() -> bool:
    '''Returns True if config/user.json exists and contains valid json parsable content, else False'''
    try:
        with USERS_FILEPATH.open('r') as f:
            return any(json.load(f))
    except (FileNotFoundError,json.JSONDecodeError):
        return False

def _read_user_index():
    try:
        with open(USERS_FILEPATH,'r') as f:
            USER_INDEX: list[dict] = json.load(f)
            return USER_INDEX
    except FileNotFoundError as e:
        raise FileNotFoundError('Missing config/users.json. You must run `scripts/build.py` or otherwise create users.json to proceed.')

def write_user_index(user_index:dict,  overwrite:bool=False):        
    if user_index_exists() and not overwrite:
        raise FileExistsError(f'{USERS_FILEPATH} exists and overwrite=False')
    
    with open(USERS_FILEPATH, 'w') as f:
        json.dump(user_index, f, indent=2)
        print('Saved user index file:', str(USERS_FILEPATH))


def get_cloneus_user(allow_default:bool=True) -> dict[str,]:
    '''Return cloneus bot entry from config/user.json if set, otherwise use DEFAULT_CLONEUS_USER
    
    Args:
        allow_default: If False, raise exception if cloneus bot not in config/user.json or file does not exist

    Raises:
         AssertionError: cloneus bot not in config/user.json
         FileNotFoundError: config/user.json does not exist
    '''
    if not allow_default:
        return get_users(include_bot=True)[0]
    
    try:
        bot_record = get_users(include_bot=True)[0]
    except (AssertionError, FileNotFoundError):
        bot_record = DEFAULT_CLONEUS_USER
    return bot_record

# if k and v: return dict mapping all user's k -> v
# if k and no v: return dict of dicts indexed by k where values are each user's whole data dict (including k)
# if v and no k: return list of values, all user's v in no particular order
# if neither k nor v: return list of dicts of all user's data dicts
# @functools.cache
def get_users(get: Literal['dname','initial','fname','uname','bot','id']|None = None, 
              by:  Literal['dname','initial','fname','uname','id']|None = None, 
              *,  user_index: list[dict] = None, include_bot:bool=False,
              ) -> dict[str,] | dict[str, dict[str,]] | list | list[dict[str,]]:
    '''Reindex and filter user data index.

    aliases: dname -> displayName; initial -> authorInitial; fname -> firstName; uname -> username; bot -> isBot

    Args:
        get: The value to select from each user's data. If None, return unfiltered user data
        by: The key to index user data by. If None, return a list instead of a dict
        user_index: The user data index to use, any returned dicts WILL be mutable. If None, will use a deep copy of global default (config/users.json)
        include_bot: If True, include Cloneus bot as first entry in the returned data
         

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
    
        user_index = copy.deepcopy(USER_INDEX)
    
    user_index = user_index[:] # shallow copy so we don't pop off bot from orig source
    bot_data = next(filter(lambda u: u['authorInitial']=='b0', user_index), None)
    
    # remove from current arbitrary index
    if bot_data is not None:
        user_index.remove(bot_data) 
    # insert back at head of list if include_bot
    if include_bot:
        assert bot_data is not None, 'Cloneus bot not in user index!'
        user_index.insert(0, bot_data) 

        
    
    alias = {'dname':'displayName', 'initial':'authorInitial', 'fname':'firstName', 'uname':'username', 'bot':'isBot', 'id':'id'}
    

    if by:
        # this could go later, but it *feels* better earlier
        _key = alias.get(by, by)
        _vals = [u[_key] for u in user_index]
        if any(v is None for v in _vals):
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
# USER_DATA['USERS']: get_users(include_bot=False)


def fake_author_id(name:str):
    asciibyte_name = ascii(name).encode('utf-8')
    b16_enc_hexname = int(hashlib.md5(asciibyte_name, usedforsecurity=False).hexdigest(), 16)
    return int(b16_enc_hexname**(1/2)) # sqrt just to make it shorter


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
    

def assign_initials(names:list[str]) -> dict[str,str]:
    '''Assign each name a unique initial based on its first alphabetical character. If a name has no alpha chars, defaults to "x". 
    
    If all names cannot be represented by a unique alphabetical character, all initials will be given a digit incremented by 1 to disambiguate.

    Args:
        names: List of names to assign initials to

    Returns:
        dict mapping each name to its assigned initial
    '''
    snames = sorted(set(names), key=str.lower)
    # take the first alpha letter from the decoded name, if none, default is "x"
    initials = [next(filter(str.isalpha, unidecode.unidecode(name)), 'x').lower() for name in snames]
    
    # if they are all unique, no digits required.
    if len(set(initials)) == len(snames):
        return dict(zip(snames, initials))
    
    dd = defaultdict(lambda: 1)
    uinitials = []

    for c in initials:
        uinitials.append(f'{c}{dd[c]}')
        dd[c]+=1
    
    return dict(zip(snames, uinitials))



def create_default_initials(user_index:list[dict], priority_order:tuple[Literal['fname','dname','uname'], ...]=('fname','dname','uname')) -> tuple[dict[str, str], Literal['fname','dname','uname']]:
    '''Get initials for each user based name. 
    
    Initial assignment priority is given by priority_order, but digitless assignments take precendence over order.
    
    e.g.
        If order is fname, dname, uname and first names map to a1,g1,a2 and usernames map to k,n,c, username mapping is used.
    '''
    # TODO: allow partial initial assignment. 
    if isinstance(priority_order, str):
        priority_order = (priority_order, )
    
    name_ini_maps = []
    for namekey in priority_order:
        names = get_users(namekey, user_index=user_index, include_bot=False)
        if all(names):
            name_to_ini = assign_initials(names)
            # if can be assigned char with out digits, stop early
            if ''.join(name_to_ini.keys()).isalpha():
                return name_to_ini,namekey
            name_ini_maps.append((name_to_ini,namekey))
    # raise ValueError('Need to specify at least one of `first_names`, `display_names`, or `usernames`')
    # if none can be represented by only a char, return first tried since it was highest priorty
    return name_ini_maps[0]

def update_initials(user_index:list[dict], priority_order:tuple[Literal['fname','dname','uname'], ...]=('fname','dname','uname'), overwrite_existing:bool=True) -> None:
    '''Updates initials IN PLACE. Will never overwrite cloneus user special initial entry.'''
    if overwrite_existing or not all(get_users('initial', user_index=user_index,  include_bot=False)):
        name_to_ini,namekey = create_default_initials(user_index, priority_order = priority_order)
        user_index_by_name = get_users(by=namekey, user_index=user_index, include_bot=False)
        
        for name,ini in name_to_ini.items():
            # if not ud[name]['authorInitial']:
            user_index_by_name[name]['authorInitial'] = ini


def _to_username(name:str):
    if not name.isprintable():
        username = ascii(name).strip("'")
    else:
        username = unidecode.unidecode(name).replace(' ','_')
    return username.strip()

def _new_user(display_name:str=None, first_name:str=None, username:str=None):
    # first name does not need to be set and will never be filled
    # username -> display_name | first_name -> display_name
    # ascii(display_name) -> username | ascii(first_name) -> username
    if display_name is None:
        if username is not None:
            display_name = username 
        else:
            display_name = first_name

    if username is None:
        username = _to_username(display_name)

    return {'id': fake_author_id(username), 'firstName': first_name, 'authorInitial':None, # filled after
            'username':username, 'displayName':display_name, 'isBot':False}
    
def new_user_index(display_names:list[str]=None, first_names:list[str]=None, usernames:list[str]=None):
    '''Create a temporary user index from a list of names. 
    
    If multiple name lists passed, they are zipped. Ensure correct ordering between lists.'''

    assert any([display_names is not None, first_names is not None, usernames is not None]), 'Need at least one of `display_names`, `first_names`, `usernames`'
    
    names = [n if n is not None else [] for n in [display_names, first_names, usernames]]
    user_index = [_new_user(d,f,u) for d,f,u in itertools.zip_longest(*names)]
    
    update_initials(user_index)
    
    if (clobot := get_cloneus_user()) not in user_index:
        user_index = [clobot, *user_index]
    
    return user_index

# TODO: Figure out a better place to put this

def format_author_tag(user_display_name:str, author_tag:str, *, insert_raw:bool=False, user_index: list[dict] = None):
    if insert_raw:
        return author_tag.format(author=user_display_name, lauthor=user_display_name, fname=user_display_name)
    
    return author_tag.format(author=user_display_name, lauthor=user_display_name.lower(), fname=get_users('fname', 'dname', user_index=user_index).get(user_display_name,user_display_name))
