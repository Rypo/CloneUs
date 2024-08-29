import gc
import time
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