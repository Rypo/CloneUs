import os
import gc
import re
import typing
import asyncio
import datetime

import functools
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands, tasks

import torch

from cloneus.data import roles

from server.config import settings

from server.utils import  io as io_utils
from server.utils.globthread import async_wrap_thread, stop_global_thread
#from ..views.contextview import PagedChangelogView, MessageContextView
from server.views import contextview as cview

from server.managers.txtman import CloneusManager
from server.managers.msgman import MessageManager

from .gen.textgen import TextGen


bot_logger = settings.logging.getLogger('bot')
model_logger = settings.logging.getLogger('model')
cmds_logger = settings.logging.getLogger('cmds')
event_logger = settings.logging.getLogger('event')


async def record_usage(ctx):
    print(ctx.author, 'used', ctx.command, 'at', ctx.message.created_at)


def release_memory():
    torch.cuda.empty_cache()
    gc.collect()


def check_up(attr_name:str, msg:str='❗ Model not loaded.'):
    def inner_wrap(f):
        @functools.wraps(f)
        async def wrapper(self, ctx: commands.Context, *args, **kwargs):
            if getattr(self, attr_name).is_ready:
                return await f(self, ctx, *args, **kwargs)
            
            return await ctx.send(msg)
                
        return wrapper
    return inner_wrap


class AICore(commands.Cog):
    '''Central Command, the AI choreographer if you will..'''
    
    def __init__(self, bot: commands.Bot, load_nowait=False):
        self.bot = bot
        self._load_nowait = load_nowait

        self.emojis = self.bot.guilds[0].emojis
        self.last_cleanse = datetime.datetime.fromtimestamp(0)
        self._start_time = datetime.datetime.now()
        self._changelog_shown = False
        self._channel=self.bot.get_channel(settings.CHANNEL_ID)

        self.pstore = io_utils.PersistentStorage()
        
        
        self.clomgr = CloneusManager(self.bot,)
        self.msgmgr = MessageManager(self.bot, n_init_messages=15, message_cache_limit=31)
                
        # very close between these two.
        #self.init_model = settings.RUNS_DIR/'solar-10b-inst-hermes2/chunk135h/cnk4096-cosine-wu0.03-lora_a32_r16_d0.0_udovkqg/checkpoint-3500'
        #self.init_model = settings.RUNS_DIR/'solar-10b-inst-hermes2/chunk135h/cnk4096-cosine-wu0.03-lora_a32_r16_d0.0_ovugdkq/checkpoint-4500'
        self.cog_textgen = TextGen(self.bot, pstore=self.pstore, clomgr=self.clomgr, msgmgr=self.msgmgr)
        

        
    async def cog_load(self):
        #self.pstore.update(youtube_quota = self.pstore.get('youtube_quota', 0))
        await self.bot.wait_until_ready()
        #await self.bot.add_cog(TextGen(self.bot, pstore=self.pstore, clomgr=self.clomgr, msgmgr=self.msgmgr), override=True) #load_nowait=self._load_nowait
        await self.bot.add_cog(self.cog_textgen, override=True) #load_nowait=self._load_nowait
        
        self.autoloader.start()
        self.cleaner.start()

    async def cog_unload(self):
        await self.bot.wait_until_ready()
        #self.pstore.update(youtube_quota = self.youtube_quota)
        #await self.txtdown(self._channel, announce=False)
        self.autoloader.cancel()
        self.cleaner.cancel()
        #self.bot.tree.remove_command(self.ctx_menu.name, type=self.ctx_menu.type)
        await self.bot.remove_cog('TextGen')
        
        stop_global_thread()


    @tasks.loop(minutes=1.0)
    async def autoloader(self):
        #if not self.message_caches[self._channel.id]:
        #    self.message_caches[self._channel.id] = await self.get_history(self._channel)
        #self.message_caches.setdefault(self._channel.id, await self.get_history(self._channel))
        await self.msgmgr.set_default(self._channel)
        #self.clomgr._preload(self.init_model, gconfig_fname='best_generation_config.json')
        # if self._load_nowait:
        #     await self.txtup(self._channel, announce=False, load=True)
        #     self.autoloader.stop()
        
        uptime = (datetime.datetime.now()-self._start_time).total_seconds()
        if uptime >= (30*1):
        #if self.clomgr.status=='down' and self.clomgr.run_count == 0 and uptime >= (30*1):
            #await self.up(self._channel, announce=False)
            if settings.SHOW_CHANGELOG and not self.pstore.get('changelog_shown'):
                await self.changelog(self._channel)
                
                self.pstore.update(changelog_shown=True)
                
            self.autoloader.stop()

    @tasks.loop(minutes=5.0)
    async def cleaner(self):
        #print('RunCount:',self.run_count)
        if self.clomgr.run_count > 0:
            inactive_duration = (datetime.datetime.now(datetime.timezone.utc) - self.bot.cached_messages[-1].created_at)
            
            if inactive_duration > datetime.timedelta(minutes=5):
                if self.clomgr.last_run > self.last_cleanse:
                    release_memory()
                    self.last_cleanse = datetime.datetime.now()



    @cleaner.before_loop
    @autoloader.before_loop
    async def before_task(self):
        await self.bot.wait_until_ready()


    async def check_mention(self, message: discord.Message):
        if not message.author.bot:
            if self.bot.user.mentioned_in(message) and message.clean_content.startswith(f'@{self.bot.user.name}'):
                await message.channel.send('Hello')


    # @commands.Cog.listener('on_message')
    # async def on_message(self, message: discord.Message):
        # Moved to TextGen
    
    # @commands.Cog.listener('on_reaction_add')
    # async def on_reaction_add(self, reaction: discord.Reaction, user: discord.User):
        # Moved to TextGen

    @commands.Cog.listener('on_raw_message_delete')
    async def on_raw_message_delete(self, payload: discord.RawMessageDeleteEvent):
        if (message := payload.cached_message) is not None:
            try:
                self.msgmgr.remove_message(message)
            except ValueError:
                print('Message not in context')
            print(f'Message Deleted: {message.author} - "{message.content}"')
            event_logger.info('(on_raw_message_delete)'
                             '- [DELETE] {a.display_name}({a.name}) message {message.content!r}'.format(a=message.author, message=message))
        else:
            print('Message not cached, cannot remove from context.')

    
    @commands.Cog.listener('on_message_edit')
    async def on_message_edit(self, before:discord.Message, after:discord.Message):
        if not before.author == self.bot.user:
            try:
                await self.msgmgr.replace_message(before, after)
            except ValueError:
                print('Message edit ignored')
                pass
            event_logger.info('(on_message_edit)'
                            '- [EDIT] {user.display_name}({user.name}) UPDATE message {b.content!r} TO {a.content!r}'.format(user=before.author, b=before, a=after))


    @commands.Cog.listener('on_thread_create')
    async def on_thread_create(self, thread: discord.Thread):
        #members = await thread.fetch_members()
        #membernames = [self.bot.get_user(m.id).name for m in members]
        #if not self.message_caches[thread.id]
        await thread.join()
        if thread.starter_message:
            self.msgmgr.add_message(thread.starter_message)
        #await thread.send(f'found thread: {thread.id}, cat: {thread.category}, members: {membernames}, messages: {self.message_caches[thread.id]}')

    
    async def cog_before_invoke(self, ctx: commands.Context) -> None:
        # first 2 args are self and ctx
        pos_args = [a for a in ctx.args[2:] if a]
        print(f'before invoke [{datetime.datetime.now()}]', ctx.author.display_name, ctx.prefix, ctx.command.name, ctx.invoked_with, pos_args, ctx.kwargs)
        
        event_logger.debug(f'(cog_before_invoke)'
                         '- [CALL] {a.display_name}({a.name}) command "{c.prefix}{c.invoked_with} ({c.command.name})" args:({args}, {c.kwargs})'.format(a=ctx.author, c=ctx, args=pos_args))
        
    async def cog_after_invoke(self, ctx: commands.Context) -> None:
        #print('after invoke', datetime.datetime.now())
        cmd_status = 'FAIL' if ctx.command_failed else 'PASS'
        pos_args = [a for a in ctx.args[2:] if a]

        cmds_logger.info(f'(cog_after_invoke, {self.qualified_name})'
                         '- [{stat}] {a.display_name}({a.name}) command "{c.prefix}{c.invoked_with} ({c.command.name})" args:({args}, {c.kwargs})'.format(stat=cmd_status, a=ctx.author, c=ctx, args=pos_args))


    @commands.command(name='changes')
    async def changelog(self, ctx: commands.Context):
        '''Show most recent patch notes'''
        # TODO: changelog button (show more, show less)

        clog = settings.RES_DIR.joinpath('changelog.md').read_text()
        clogs = clog.split('------')
        paged_changelog = cview.PagedChangelogView(clogs, timeout=None)
        await paged_changelog.send(ctx)

    
    @commands.command('status')
    async def status_report(self, ctx: commands.Context):
        '''Check bot status.'''
        # ✅⚠❌🛑💯🟥🟨🟩⬛✔🗯💭💬👁‍🗨🗨
        
        #gconf_settings = self.model.get_genconf(verbose=False)
        #vram_use, vram_total = get_gpu_memory()
        statuses = [
            *self.clomgr.list_status(self.pstore.get('youtube_quota',0)),
            ('Message Context', len(self.msgmgr.get_mcache(ctx)), f' / {self.msgmgr.message_cache_limit}'), # Note: this show the _unmerged_ length.
            #('Messages bot cached', len(self.bot.cached_messages)),
            ('','---',''),
            *self.cog_textgen.argsettings(ctx)
        ]
        msg = '\n'.join(f'{label}: **{desc}**{post}' if label else desc for label, desc, post in statuses)
        await ctx.send(msg)




async def setup(bot):
    load_nowait = bool(os.getenv('EAGER_LOAD',False))
    
    await bot.add_cog(AICore(bot, load_nowait=load_nowait))