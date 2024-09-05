import gc
import time
import itertools
from contextlib import contextmanager

import torch

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