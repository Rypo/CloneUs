import typing
from collections import  defaultdict

import discord
from discord.ext import commands

from utils import text as text_utils



class MessageManager:
    autonomous: bool = False

    def __init__(self, bot, n_init_messages:int, message_cache_limit:int):
        '''Handler for chat message history. Importantly, does NOT have commands'''
        self.bot = bot
        self.message_caches = defaultdict(list) # priorty queue? set? heapq? bisect? (time, data)
        self.base_message_cache = []
        self.n_init_messages = n_init_messages
        self.message_cache_limit = message_cache_limit # (msgs/4hr: median=24, mean=35.5)

        #self.check_vram_every = 5
        #self.last_perc_vram_used = 0.0
        #self.message_add_count = 0
        self.last_caches = {}
        self.default_channel: discord.TextChannel = None
        

    async def set_default(self, channel: discord.TextChannel):
        if self.default_channel is None:
            self.default_channel = channel
            self.message_caches.setdefault(channel.id, await self.get_history(channel))
    
    def get_mcache(self, message_ctx: discord.Message|commands.Context|int):
        cache_id = message_ctx if isinstance(message_ctx, int) else message_ctx.channel.id
        return self.message_caches[cache_id]
    
    def set_mcache(self, message_ctx: discord.Message|commands.Context|int, new_message_cache: list[discord.Message]):
        cache_id = message_ctx if isinstance(message_ctx, int) else message_ctx.channel.id
        self.message_caches[cache_id] = new_message_cache
    
    async def add_message(self, message: discord.Message, force_auto_response=False):
        message_cache = self.get_mcache(message)
        message_cache.append(message)
        if self.autonomous or force_auto_response:
            await message.add_reaction('🔜')
        
        # NOTE: ordering seems to only get messed up when manually calling /*bot in autoreply mode
        # but, sorting here is still more efficent than in text transforms.
        self.set_mcache(message_cache[-1], sorted(set(message_cache), key=lambda m: m.created_at))
        self.trim_context(message_cache)
    
    async def replace_message(self, old_message: discord.Message, new_message: discord.Message|list[discord.Message]):
        '''replace a message in context with one (or more for partitioned) messages'''
        message_cache = self.get_mcache(old_message)
        msg_idx = message_cache.index(old_message)
        if isinstance(new_message, list):
            message_cache.pop(msg_idx)
            new_message.reverse()
            for nmsg in new_message:
                message_cache.insert(msg_idx, nmsg)
        else:
            message_cache[msg_idx] = new_message

        #if self.autonomous or force_auto_response:
        #    await message.add_reaction('🔜')
        
        # NOTE: ordering seems to only get messed up when manually calling /*bot in autoreply mode
        # but, sorting here is still more efficent than in text transforms.   
    
    def get_mcache_subset(self, message:discord.Message, inclusive=True):
        '''Returns a slice of context up to message'''
        mcache=self.get_mcache(message)
        msg_idx = mcache.index(message)
        msg_idx += 1 if inclusive else 0 # yes, i could just implict cast, but clever code is ass-bite code
        return mcache[:msg_idx]

    def remove_message(self, message: discord.Message):
        self.get_mcache(message).remove(message)
    
    def clear_mcache(self, message_cache: typing.Literal['default','base','all']|discord.TextChannel):
        self.last_caches = {'default': self.get_mcache(self.default_channel.id).copy(), 'base': self.base_message_cache.copy()}
        if isinstance(message_cache, discord.TextChannel):
            return self.message_caches[message_cache.id].clear()
            
        if message_cache == 'all':
            self.get_mcache(self.default_channel.id).clear()
            self.base_message_cache.clear()
        elif message_cache == 'base':
            self.base_message_cache.clear()
        elif message_cache == 'default':
            self.get_mcache(self.default_channel.id).clear()
        
    def restore_caches(self):
        if self.last_caches:
            self.set_mcache(self.default_channel.id, self.last_caches['default'])
            self.base_message_cache = self.last_caches['base']
            self.last_caches = {}

    def trim_context(self, message_cache, limit=None):
        #self.message_add_count+=1
        
        n_messages = len(message_cache)

        # if self.message_add_count % self.check_vram_every == 0:
        #     cur,total = get_gpu_memory()
        #     percent_used = cur/total
        #     print(f'vram usage: {percent_used}')
        #     if percent_used >= 0.8 and (percent_used > self.last_perc_vram_used*1.1): # 10% buffer
        #         limit = int(n_messages*0.8) # reduce by 20%
        #         print(f'Messages Trimmed: {n_messages} -> {limit}')
        #         self.last_perc_vram_used = percent_used
        if limit is None:
            limit = self.message_cache_limit
        
        if n_messages > limit:
            self.set_mcache(message_cache[-1], message_cache[-limit:])
            #message_cache = message_cache[-limit:]

    
    async def get_history(self, channel: discord.TextChannel) -> typing.List[discord.Message]:
        """Get a slice of chat history."""        
        print('Getting initial context from message history...')
        
        #messages = [message async for message in ctx.channel.history(limit=self.history_limit*3, oldest_first=False)][::-1]
        messages = []
        n=0
        # Search 3x the history_limit for usable messages 
        async for message in channel.history(limit=self.n_init_messages*3, oldest_first=False):
            if text_utils.message_filter(message):
                messages.append(message)
                n+=1
            if n>=self.n_init_messages:
                break
        # Need to reverse because we want the most recent messages, but in cronological order
        messages = messages[::-1]
        print('Initial messages: ',len(messages))
        return messages



    async def check_mention(self, message: discord.Message):
        if not message.author.bot:
            if self.bot.user.mentioned_in(message) and message.clean_content.startswith(f'@{self.bot.user.name}'):
                await message.channel.send('Hello')

