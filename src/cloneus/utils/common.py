import gc
import time
from pathlib import Path
import itertools
from contextlib import contextmanager

import torch

import safetensors.torch as sft
from huggingface_hub import snapshot_download, hf_hub_download

def release_memory():
    torch.cuda.empty_cache()
    gc.collect()


@contextmanager
def timed(label:str=''):
    print(f'BEGIN: {label}')
    t0 = time.perf_counter()
    try:
        yield
    finally:
        te = time.perf_counter()
        print(f'{label} : {te-t0:0.2f}s')


def batched(iterable, n:int):
    '''https://docs.python.org/3/library/itertools.html#itertools.batched'''
    # batched('ABCDEFG', 3) → ABC DEF G
    
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := list(itertools.islice(iterator, n)):
        yield batch


def read_state_dict(model_id_or_path:str|Path, subfolder:str|None=None, local_files_only:bool = False, allow_empty:bool = False, device:str|int='cpu') -> dict[str, torch.Tensor]:
    '''Read sharded safetensors files into a unified state dict from a remote HF repo or local filepath.
    
    Args:
        model_id_or_path: The name of the HF repo or local filepath.
        subfolder: The subfolder within the remote repo. Ignored if `model_id_or_path` is a local filepath.
        local_files_only: Whether to only look for files locally. If False, will try to download from HF hub.
        allow_empty: Whether to raise an error if no weights are found.
        device: The device to load the tensors onto. 
        
        Returns:
            A unified state dict.
    '''
    
    if (model_path := Path(model_id_or_path)).exists():
        if model_path.is_file():
            return sft.load_file(model_path, device)
        
        weights_loc = model_path
        
    else:
        if subfolder:
            weights_loc = Path(snapshot_download(repo_id=model_id_or_path, allow_patterns=f"{subfolder}/*", local_files_only=local_files_only)).joinpath(subfolder)
        else:
            weights_loc = Path(snapshot_download(repo_id=model_id_or_path, local_files_only=local_files_only))
    
    state_dict = {}
    for shard in sorted(Path(weights_loc).glob('*.safetensors')):
        state_dict.update(sft.load_file(shard, device))

    if not state_dict and not allow_empty:
        raise RuntimeError('Failed to read state dict data from given path')

    return state_dict