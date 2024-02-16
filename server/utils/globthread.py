import asyncio
import functools
import concurrent.futures

def async_wrap_default(func):
    @functools.wraps(func)
    async def run(*args, loop=None, executor=None, **kwargs):
        if loop is None:
            loop = asyncio.get_event_loop()
        pfunc = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(executor, pfunc)
        #return loop.run_in_executor(executor, pfunc)
    return run


# NOTE: Not sure if this level of care is needed for managing 1 thread, but it FINALLY works. 
# https://stackoverflow.com/questions/69092420/is-there-a-way-to-forcibly-kill-a-thread-in-python
THREAD = None

def stop_global_thread(wait=True, cancel_futures=False):
    global THREAD
    if THREAD is not None:
        THREAD.shutdown(wait=wait, cancel_futures=cancel_futures)
    THREAD = None

def get_global_thread():
    global THREAD
    if THREAD is None:
        THREAD = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    return THREAD

def async_wrap_thread(func):
    # https://stackoverflow.com/questions/43241221/how-can-i-wrap-a-synchronous-function-in-an-async-coroutine?noredirect=1&lq=1
    # https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.run_in_executor
    @functools.wraps(func)
    async def run(*args, loop=None, executor=None, **kwargs):
        if loop is None:
            loop = asyncio.get_event_loop()
        pfunc = functools.partial(func, *args, **kwargs)
        #with THREAD as pool:
        global_thread = get_global_thread()
        return await loop.run_in_executor(global_thread, pfunc)
        #return await loop.run_in_executor(executor, pfunc)
    return run
