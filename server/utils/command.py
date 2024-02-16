

import functools
from discord.ext import commands

def check_up(attr_name:str, msg:str='❗ Model not loaded.'):
    def inner_wrap(f):
        @functools.wraps(f)
        async def wrapper(self, ctx: commands.Context, *args, **kwargs):
            if getattr(self, attr_name).is_ready:
                return await f(self, ctx, *args, **kwargs)
            
            return await ctx.send(msg)
                
        return wrapper
    return inner_wrap