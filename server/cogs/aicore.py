import os
import gc
import copy
import typing
import asyncio
import logging
import datetime
import functools

import discord
from discord import app_commands
from discord.ext import commands, tasks

import torch

import config.settings as settings

from utils import io as io_utils, globthread
from utils.reporting import StatusItem, vram_usage
from views import contextview as cview


from .textgen import TextGen
from .imagegen import ImageGen

from run import BotUs

bot_logger = logging.getLogger('bot')
event_logger = logging.getLogger('server.event')
cmds_logger = logging.getLogger('server.cmds')



def release_memory():
    torch.cuda.empty_cache()
    gc.collect()


def parse_interaction_data(opts:list[dict], qualname = ''):
    '''Recursively extract interaction command args from interaction.data'''
    if isinstance(opts, dict):
        opts = [opts]
    
    kwargs = {}
    for sublvl in opts:
        # subcommand/subcommand group
        if sublvl.get('type', -1) in [1,2]: 
            return parse_interaction_data(sublvl['options'], f'{qualname} {sublvl["name"]}'.strip())
        else:
            kwargs[sublvl['name']] = sublvl['value']
    
    return {'qualified_name': qualname, 'kwargs': kwargs}


async def reconstruct_flags(ctx:commands.Context, flags_clsname:str, flag_kwargs:dict[str, typing.Any]):
    '''Manually reconstruct the flags from a serializable form'''
    import cmds.flags

    Flags: commands.FlagConverter = getattr(cmds.flags, flags_clsname)
    Flags = await Flags._construct_default(ctx)
    
    for k,v in flag_kwargs.items():
        setattr(Flags, k, v)

    return Flags


class AICore(commands.Cog):
    '''Central Command, the AI choreographer if you will..'''
    
    def __init__(self, bot: BotUs, load_nowait=False):
        self.bot = bot
        self._load_nowait = load_nowait

        self.emojis = self.bot.guilds[0].emojis
        self.last_cleanse = datetime.datetime.fromtimestamp(0)
        self._start_time = datetime.datetime.now()
        self._changelog_shown = False
        self._channel = self.bot.get_channel(settings.CHANNEL_ID)

        self.pstore = self.bot.pstore #self.cog_textgen.pstore
        self.txt_gen: TextGen = None # TextGen(self.bot, pstore=self.pstore, clomgr=self.clomgr, msgmgr=self.msgmgr)
        self.img_gen: ImageGen = None
        
        # https://github.com/Rapptz/discord.py/issues/7823#issuecomment-1086830458
        self.run_ctx_menu = app_commands.ContextMenu(name='▶ Replay CMD', callback=self._cm_run_cmd) # 🔄 '▶ Run CMD' | '▶ Replay Command'
        self.bot.tree.add_command(self.run_ctx_menu)#, override=True)
        
        self._log_extra = {'cog_name': self.qualified_name}
        
        
        
    async def cog_load(self):
        await self.bot.wait_until_ready()
        
        release_memory()
        self.wait_for_cogs.start()

        #self.autoloader.start()
        #self.cleaner.start()

    async def cog_unload(self):
        await self.bot.wait_until_ready()

        self.wait_for_cogs.cancel()
        self.autoloader.cancel()
        self.cleaner.cancel()
        self.bot.tree.remove_command(self.run_ctx_menu.name, type=self.run_ctx_menu.type)

        release_memory()
        globthread.stop_global_executors()

    @tasks.loop(seconds=10.0)
    async def wait_for_cogs(self):
        # make sure other cogs are loaded before assigning
        if all(cog in self.bot.cogs for cog in ['TextGen','ImageGen']):
            self.txt_gen = self.bot.get_cog('TextGen')
            self.img_gen = self.bot.get_cog('ImageGen')
            self.autoloader.start()
            self.cleaner.start()
            self.wait_for_cogs.cancel()
        

    @tasks.loop(minutes=1.0)
    async def autoloader(self):
        #await self.txt_gen.msgmgr.set_default(self._channel)
        
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
        if self.txt_gen.clomgr.run_count > 0:
            inactive_duration = (datetime.datetime.now(datetime.timezone.utc) - self.bot.cached_messages[-1].created_at)
            
            if inactive_duration > datetime.timedelta(minutes=5):
                if self.txt_gen.clomgr.last_run > self.last_cleanse:
                    release_memory()
                    self.last_cleanse = datetime.datetime.now()


    @wait_for_cogs.before_loop
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
                self.txt_gen.msgmgr.remove_message(message)
            except ValueError:
                pass
            print(f'Message Deleted: {message.author} - "{message.content}"')
            event_logger.info('[DELETE] {a.display_name}({a.name}) message {message.content!r}'.format(
                a=message.author, message=message), extra=self._log_extra)

    @commands.Cog.listener('on_message_edit')
    async def on_message_edit(self, before:discord.Message, after:discord.Message):
        if before.author != self.bot.user:
            try:
                await self.txt_gen.msgmgr.replace_message(before, after)
            except ValueError:
                print('Message edit ignored')
                
            event_logger.info('[EDIT] {user.display_name}({user.name}) UPDATE message {b.content!r} TO {a.content!r}'.format(
                user=before.author, b=before, a=after), extra=self._log_extra)


    @commands.Cog.listener('on_thread_create')
    async def on_thread_create(self, thread: discord.Thread):
        await thread.join()
        if thread.starter_message:
            await self.txt_gen.msgmgr.add_message(thread.starter_message)
        #await thread.send(f'found thread: {thread.id}, cat: {thread.category}, members: {membernames}, messages: {self.message_caches[thread.id]}')

    
    async def cog_before_invoke(self, ctx: commands.Context) -> None:
        # first 2 args are self and ctx
        pos_args = [a for a in ctx.args[2:] if a]
        #print(f'before invoke [{datetime.datetime.now()}]', ctx.author.display_name, ctx.prefix, ctx.command.name, ctx.invoked_with, pos_args, ctx.kwargs)
        cmds_logger.debug('[CALL] {a.display_name}({a.name}) command "{c.prefix}{c.invoked_with} ({c.command.qualified_name})" args:({args}, {c.kwargs})'.format(
            a=ctx.author, c=ctx, args=pos_args), extra=self._log_extra)
        
    async def cog_after_invoke(self, ctx: commands.Context) -> None:
        cmd_status = 'FAIL' if ctx.command_failed else 'PASS'
        # first 2 args are self and ctx
        pos_args = [a for a in ctx.args[2:] if a] 
        cmds_logger.info('[{stat}] {a.display_name}({a.name}) command "{c.prefix}{c.invoked_with} ({c.command.qualified_name})" args:({args}, {c.kwargs})'.format(
            stat=cmd_status, a=ctx.author, c=ctx, args=pos_args), extra=self._log_extra)


    async def _cm_run_cmd(self, interaction: discord.Interaction, message: discord.Message):
        # print('message.type', message.type)
        # bot plain text message: type = default
        # user !cmd: type = default
        if message.author.bot and message._interaction is None:
            return await interaction.response.send_message("Replay your original command message, not bot's reply.", ephemeral=True)
        
        ctx = await self.bot.get_context(message)

        if not ctx.valid and message.interaction_metadata is None:
           return await interaction.response.send_message("Whatever you clicked on ain't look much like a command to me.", ephemeral=True)


        await interaction.response.defer(thinking=True)
        await asyncio.sleep(1)
        
        
        ictx = await self.bot.get_context(interaction)
        
        #print('CTX VALID:', ctx.valid)
        if ctx.valid:
            #try:
            await ictx.invoke(ctx.command, *ctx.args, **ctx.kwargs)
                #await ctx.reinvoke(call_hooks=True)
            #except ValueError as e:
            #    await interaction.followup.send("Whatever you clicked on ain't look much like a command to me.")
        else:
            #parse_interaction_data(self.bot.interaction_cache[int(inter_id)])
            try:
                cached_idata = copy.deepcopy(self.bot.cmd_cache[message.interaction_metadata.id])
            except KeyError as e:
                cmds_logger.error(f'Missing interaction key: {interaction!s} | {message!s}', exc_info=e, extra=self._log_extra)
                return await interaction.followup.send("Couldn't find any command to replay in that message. Could be that the message:\n"
                                                       "(1) is not a command   (2) is too old   (3) originally failed   (4) is haunted")
            
            # point back to original data to allow replaying a replay of a replay ...
            self.bot.cmd_cache[interaction.id] = self.bot.cmd_cache[message.interaction_metadata.id]

            args = cached_idata['args']
            kwargs = cached_idata['kwargs']

            # manually reconstruct the flags from their serializable form
            if (flags := kwargs.pop('flags', None)) is not None:
                Flags = await reconstruct_flags(ictx, flags['FlagsClsName'], flags['flag_kwargs'])
                kwargs['flags'] = Flags
            
            # command resignment necessary or it will fail in drawUI because ContextMenu has no cog 
            ictx.command = self.bot.get_command(cached_idata['qualified_name'])
            
            await ictx.invoke(ictx.command, *args, **kwargs)
            


    @commands.command(name='changes')
    async def changelog(self, ctx: commands.Context):
        '''Show most recent patch notes'''

        clog = settings.RES_DIR.joinpath('changelog.md').read_text()
        clogs = clog.split('------')
        paged_changelog = cview.PagedChangelogView(clogs, timeout=None)
        await paged_changelog.send(ctx)

    
    @commands.command('status')
    async def status_report(self, ctx: commands.Context):
        '''Get an abridged system, text, and image model status report.'''
        # ✅⚠❌🛑💯🟥🟨🟩⬛✔🗯💭💬👁‍🗨🗨🟢🔴
        
        vram_used, vram_total = vram_usage()
        system_status = [
            StatusItem('Latency', f'{round(self.bot.latency * 1000)}ms'),
            StatusItem('vRAM usage', f'{vram_used:,}', f'MiB / {vram_total:,}MiB'),
        ]
        text_header = '\n### Text ' + ('\🟢' if self.txt_gen.clomgr.is_ready else '\🔴') + '\n'
        image_header = '\n### Image ' + ('\🟢' if self.img_gen.igen.is_ready else '\🔴') + '\n'

        text_status = self.txt_gen.status_list(keep_advanced=False, ctx=ctx)
        image_status = self.img_gen.status_list(keep_advanced=False)
        msg = (
            '## Status\n'
            + '\n'.join(sstat.md() for sstat in system_status)
            + text_header
            + '\n'.join(tstat.md() for tstat in text_status)
            + image_header
            + '\n'.join(istat.md() for istat in image_status)
        )

        await ctx.send(msg)


async def setup(bot: BotUs):
    load_nowait = bool(os.getenv('EAGER_LOAD',False))            
    await bot.add_cog(AICore(bot, load_nowait=load_nowait))
    # A lot of folks tend to sync before loading their extensions… which means their commands aren’t loaded into their CommandTree yet.
    # - https://about.abstractumbra.dev/discord.py/2023/01/30/app-command-basics.html#caveats
    