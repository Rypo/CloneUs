import datetime
from pathlib import Path

import more_itertools
from omegaconf import OmegaConf

import config.settings as settings

# def gb_part(candidates, part, part_max=5):
#     return dict(more_itertools.groupby_transform(candidates, lambda c: c.parts[part], None, (lambda ncands: gb_part(ncands, part+1) if part < part_max else list(ncands))))

def gb_part(candidates, part, part_max=5):
    '''groupby subparts of paths recursively
    Note: structure [(part, [(part, ...)]] can be {part: {part: {part: []}}} if change recurse-case list -> dict
    '''
    if part > part_max:
        return list(candidates)
    return list(more_itertools.groupby_transform(candidates, lambda c: c.parts[part], None, lambda ncands: gb_part(ncands, part+1)))

def print_basic_ckpttree(candidates, init_part=3, part_max=5):
    '''../runs/full/ - skip
    mistral-7b-i4/
     chunk4h_eos/
      cnk8192-cosine-wu0.1-lora_a64_r32_d0.5_kgqvoud/
       checkpoint-380'''
    spc=' '
    for model,datasets in gb_part(candidates, init_part, part_max):
        print(f'{model}')
        for dataset,modelbases in datasets:
            print(f'{spc*2}- {dataset}')
            for modelbase, checkpoints in modelbases:
                print(f'{spc*4}* {modelbase}')
                for checkpoint in checkpoints:
                    print(f'{spc*6}+ {checkpoint.name}')
                    subchecks = sorted([*checkpoint.rglob('merged'), *checkpoint.rglob('awq')])
                    for sub in subchecks:
                        print(f'{spc*8}- {sub.name}')

    
def fmt_chkpts(model_basedir:Path, checkpoints: list[Path], rel_to=None, wrapcodeblock=False):
    optstring = f'{model_basedir.relative_to(rel_to) if rel_to else model_basedir.name}'

    #checkp_opts = '\n'.join(['- {}'.format(str(o).replace(str(model_basedir)+'/','')) for o in cmatchs])
    checkp_opts = ''.join(['\n- {}'.format(o.relative_to(model_basedir)) for o in checkpoints])
    if wrapcodeblock:
        return f'```\n{optstring+checkp_opts}\n```'
    
    return optstring,checkp_opts


def find_checkpoints(model_basedir: Path, rel_to=None, md_format=True, wrapcodeblock=False):
    cmatchs = sorted([*model_basedir.rglob('checkpoint*'), *model_basedir.rglob('checkpoint*/merged'), *model_basedir.rglob('checkpoint*/merged/awq')])
    if md_format:
        return fmt_chkpts(model_basedir, cmatchs, rel_to, wrapcodeblock)

    return model_basedir,cmatchs


class PersistentStorage:
    def __init__(self, pstore_file=None) -> None:
        self.pstore_file = pstore_file if pstore_file is not None else settings.CONFIG_DIR/'persistent_storage.yaml'
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
