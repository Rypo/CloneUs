import enum
import asyncio
import functools
import concurrent.futures
import threading
# import queue
from collections.abc import Generator, AsyncGenerator, Coroutine

def async_wrap_default(func):
    @functools.wraps(func)
    async def run(*args, loop=None, executor=None, **kwargs):
        if loop is None:
            loop = asyncio.get_event_loop()
        if executor is None:
            executor = get_global_executor()
        pfunc = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(executor, pfunc)
    return run


# https://stackoverflow.com/questions/69092420/is-there-a-way-to-forcibly-kill-a-thread-in-python
_executor = None
_executor_alt = None

def stop_global_executors(wait=True, cancel_futures=False):
    global _executor,_executor_alt
    if _executor is not None:
        _executor.shutdown(wait=wait, cancel_futures=cancel_futures)
    if _executor_alt is not None:
        _executor_alt.shutdown(wait=wait, cancel_futures=cancel_futures)
    _executor = None
    _executor_alt = None

def get_global_executor(primary=True):
    if primary:
        global _executor
        if _executor is None:
            _executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix='tpe-PRIMARY-')
        return _executor
    else:
        global _executor_alt
        if _executor_alt is None:
            _executor_alt = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix='tpe-ALT-')
        return _executor_alt



async def print_async_env_info(loop=None):
    all_tasks = [tsk.get_name() for tsk in asyncio.all_tasks(loop)]
    
    print("Thread: {tid}({tname}), event loop: {loop_id}, Active Task: {cur_task}\nAll Tasks:{all_task}".format(
              tid=threading.get_ident(), tname=threading.current_thread().name, 
              loop_id=id(asyncio.get_running_loop()), 
              cur_task=asyncio.current_task(loop).get_name(), all_task=all_tasks))


# https://docs.python.org/3/library/asyncio-dev.html#concurrency-and-multithreading
# https://docs.python.org/3/library/asyncio-task.html#running-in-threads
# https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.run_in_executor
def wrap_async_executor(f_py=None, use_alternate_executor:bool=False):
    # https://stackoverflow.com/a/60832711
    assert callable(f_py) or f_py is None
    def _decorator(func):
    # https://stackoverflow.com/questions/43241221/how-can-i-wrap-a-synchronous-function-in-an-async-coroutine?noredirect=1&lq=1
    
        @functools.wraps(func)
        async def run(*args, loop=None, executor=None, **kwargs):
            if loop is None:
                loop = asyncio.get_running_loop()
            
            pfunc = functools.partial(func, *args, **kwargs)
            
            if executor is None:
                executor = get_global_executor(not use_alternate_executor)

            #await print_async_env_info(loop)
            
            return await loop.run_in_executor(executor, pfunc)
        return run
    return _decorator(f_py) if callable(f_py) else _decorator


class Sentinel(enum.Enum):
    ITER_COMPLETE = enum.auto()

def _queue_item_ready(semaphore:asyncio.Semaphore):
    # increase the count of available items in the queue so that it can be read in an async-friendly way:
    semaphore.release()

def _enqueue_generator(sync_gen:Generator, loop:asyncio.AbstractEventLoop, queue:asyncio.Queue, semaphore:asyncio.Semaphore):
    def wrapper(*args, **kwargs):
        #for item in (sync_gen() if callable(sync_gen) else sync_gen):
        for item in sync_gen:
            queue.put_nowait(item)
            loop.call_soon_threadsafe(_queue_item_ready, semaphore)
    
    wrapper()
    queue.put_nowait(Sentinel.ITER_COMPLETE)
    loop.call_soon_threadsafe(_queue_item_ready, semaphore)
    return


async def async_gen(sync_gen:Generator):
    """Create an async generator from a synchronous generator"""
    # https://stackoverflow.com/a/77441403
    # see also: 
    # https://gist.github.com/monkeytruffle/36a57792bad4f0c507c6be8111526ec6
    # https://stackoverflow.com/questions/77441269/how-to-convert-a-python-generator-to-async-generator
    # https://stackoverflow.com/questions/31623194/asyncio-two-loops-for-different-i-o-tasks/62631135#62631135
    
    executor = get_global_executor(primary=True) 
    # Do NOT use alt exec. Need to block the primary executor so other cmd calls don't interfere
    
    loop = asyncio.get_running_loop()
    semaphore = asyncio.Semaphore(0)
    async_queue = asyncio.Queue()
    
    loop.run_in_executor(executor, _enqueue_generator, sync_gen, loop, async_queue, semaphore)

    async def qstep():
        await semaphore.acquire() # releassed during enqueue
        return await async_queue.get() 
    
    while (item := await qstep()) != Sentinel.ITER_COMPLETE:
        yield item