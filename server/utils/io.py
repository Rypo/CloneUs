import re
import ast
import typing
import datetime
import subprocess
from enum import Enum
from pathlib import Path

import more_itertools
from omegaconf import OmegaConf


class RunLevel(Enum):
    '''Represents the last four constituents parts of a checkpoint path location.
    i.e. path/to/.../cloneus/runs/full/MODEL/DATASET/RUNNAME/CHECKPOINT
    '''
    MODEL = -4
    DATASET = -3
    RUNNAME = -2
    CHECKPOINT = -1

# def gb_part(candidates, part, part_max=5):
#     return dict(more_itertools.groupby_transform(candidates, lambda c: c.parts[part], None, (lambda ncands: gb_part(ncands, part+1) if part < part_max else list(ncands))))

# TODO: group_level != model seems pointless. Needs rework. Setting group_level to anything else with return_type=dict will clobber values
def gb_part(checkpoint_paths:list[Path], group_level:typing.Literal['model','dataset','runname','checkpoint']='model', final_level:typing.Literal['model','dataset','runname','checkpoint']='runname', return_type=list):
    '''Recursively group by subparts of checkpoint run paths.

    Args:
        checkpoint_paths: Flat list of for absolute or relative checkpoint paths
        group_level: Directory level to group checkpoints by
        final_level: Level to stop nesting groups and return a list of remaining paths
        return_type: Output structure. If list, groups are lists of tuples. If dict, groups are dicts of dicts
            e.g. [(part, [(part, [...,])]] or {part: {part: {part: [...,]}}}
    '''
    part = RunLevel[group_level.upper()].value if isinstance(group_level, str) else group_level
    max_depth = RunLevel[final_level.upper()].value if isinstance(final_level, str) else final_level
    if part > max_depth:
        return list(checkpoint_paths)
    return return_type(more_itertools.groupby_transform(checkpoint_paths, lambda c: c.parts[part], None, lambda ncands: gb_part(ncands, part+1, max_depth, return_type)))

def print_basic_ckpttree(candidates:list[Path], group_level='model', final_level='runname'):
    '''../runs/full/ - skip
    mistral-7b-i4/
     chunk4h_eos/
      cnk8192-cosine-wu0.1-lora_a64_r32_d0.5_kgqvoud/
       checkpoint-380'''
    spc=' '
    for model,datasets in gb_part(candidates, group_level, final_level, return_type=list):
        print(f'{model}')
        for dataset,modelbases in datasets:
            print(f'{spc*2}- {dataset}')
            for modelbase, checkpoints in modelbases:
                print(f'{spc*4}* {modelbase}')
                for checkpoint in checkpoints:
                    print(f'{spc*6}+ {checkpoint.name}')
                    #subchecks = sorted([*checkpoint.rglob('merged'), *checkpoint.rglob('awq')])
                    #for sub in subchecks:
                    #    print(f'{spc*8}- {sub.name}')

    
def fmt_checkpoints(model_basedir:Path, checkpoints: list[Path], rel_to=None, wrapcodeblock=False):
    optstring = f'{model_basedir.relative_to(rel_to) if rel_to else model_basedir.name}'

    #checkp_opts = '\n'.join(['- {}'.format(str(o).replace(str(model_basedir)+'/','')) for o in cmatchs])
    checkp_opts = ''.join(['\n- {}'.format(o.relative_to(model_basedir)) for o in checkpoints])
    if wrapcodeblock:
        return f'```\n{optstring+checkp_opts}\n```'
    
    return optstring,checkp_opts


def _find_checkpoints(model_basedir: Path, rel_to=None, md_format=True, wrapcodeblock=False):
    cmatchs = sorted([*model_basedir.rglob('checkpoint*')])
    
    if md_format:
        return fmt_checkpoints(model_basedir, cmatchs, rel_to, wrapcodeblock)

    return model_basedir,cmatchs

def find_checkpoints(search_path: Path, include_pattern:str|re.Pattern=None, exclude_pattern:str|re.Pattern=None, require_config:bool=True, ):

    ckpt_paths = sorted([c for c in search_path.rglob('checkpoint*') 
                         if (not require_config) or (c.parent/'config.yaml').exists()
                         ])
    if include_pattern:
        irgx = re.compile(include_pattern, re.I)
        ckpt_paths = [c for c in ckpt_paths if irgx.search(str(c))]

    if exclude_pattern:
        ergx = re.compile(exclude_pattern, re.I)
        ckpt_paths = [c for c in ckpt_paths if not ergx.search(str(c))]

    return ckpt_paths

class PersistentStorage:
    def __init__(self, pstore_file=None) -> None:
        if pstore_file is None:
            import config.settings as settings
            pstore_file = settings.CONFIG_DIR/'persistent_storage.yaml'
        
        self.pstore_file = pstore_file
        
        try:
            self.persistent_storage = OmegaConf.load(self.pstore_file)
        except FileNotFoundError:
            self.persistent_storage = {self.edate: {'youtube_quota': 0, 'changelog_shown': False}}
            self.save()
        self.edate = f"entry_{datetime.datetime.now().strftime('%Y_%m_%d')}"
        self.persistent_storage = OmegaConf.load(self.pstore_file)
        self.today_store = self.load()

    def init_entry(self, entry_datekey:str, keys:list[str]=None):
        if entry_datekey not in self.persistent_storage:
            if keys is None:
                last_entry_date = list(self.persistent_storage.keys())[-1]
                keys = self.persistent_storage[last_entry_date]
            self.persistent_storage[entry_datekey] = dict.fromkeys(keys)


    def load(self):
        self.init_entry(self.edate)

        return self.persistent_storage[self.edate]

    def save(self):
        OmegaConf.save(self.persistent_storage, self.pstore_file)

    def update(self, **kwargs):
        self.today_store.update(**kwargs)
        self.save()

    def get(self, key, default=None):
        # Yes, this is actually correct given how we initialize
        # since doing dict.fromkeys, it inits to null (None), so, .get is finding it and returning None
        value = self.today_store.get(key, default)
        return value if value is not None else default



def tail(f:str|Path, n:int, offset:int=0):
    # https://stackoverflow.com/a/136280
    try:
        out = subprocess.run(["tail", "-n", f'{n+offset}', str(f)], capture_output=True, check=True, text=True)
    except subprocess.CalledProcessError as e:
        l1 = Path(f).open('r').readline() # try to read a line for a better Error message if missing
        raise e # else raise processError
    
    end = -offset if offset else None
    return out.stdout.splitlines()[:end]



def _resub_parse_flags(match:re.Match):    
    flags = match.group(1).strip('><').split()
    flag_cls_name = flags.pop(0)
    
    return str({k:ast.literal_eval(v) for k,v in [x.split('=') for x in flags]})

def parse_log_entry(log_entry:str):
    # todo: unslop the ai regex
    pattern = re.compile(
        r"INFO\s+\[(?P<datetime>[^\]]+)\]\s+cmds\s+:\s+\((?P<logging_func>[^,]+)"
        r"(?:,\s*(?P<cog_name>[^)]+))?\)-\s+\[(?P<status>[A-Z]+)\]\s+(?P<user_display_name>\w+)"
        r"\((?P<username>\w+)\)\s+command\s+\"(?P<cmd_call>[^\" ]+)\s+\((?P<cmd_name>[^)]+)\)\"\s+args:(?P<args>.*)", re.I
    )
    flags_pattern = re.compile(r'''(?<=['"]flags['"]: )(<[^>]+>)''', re.I)
    
    match = pattern.match(log_entry)
    if not match:
        return
    
    data = match.groupdict()
    
    # parse special flags obj into standard str(dict)
    args_str = flags_pattern.sub(_resub_parse_flags, data.pop('args').strip())
    
    try:
        cmd_args, cmd_kwargs = ast.literal_eval(args_str)
    except (SyntaxError, ValueError):
        cmd_args, cmd_kwargs = [], {}
    
    # Add parsed arguments back into data
    data['cmd_args'] = cmd_args
    data['cmd_kwargs'] = cmd_kwargs
    
    data['raw'] = log_entry
    return data