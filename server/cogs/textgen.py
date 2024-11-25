import os
import gc
import re
import yaml
import random
import itertools
import typing
import asyncio
import datetime
import logging
import functools
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands, tasks

from omegaconf import OmegaConf
import torch

from cloneus.data import useridx

import config.settings as settings
from utils import text as text_utils, io as io_utils
from utils.command import check_up
from utils.reporting import StatusItem
from utils.globthread import wrap_async_executor, stop_global_executors

from views import contextview as cview
from cmds import (
    flags as cmd_flags, 
    transformers as cmd_tfms, 
    choices as cmd_choices
)
from managers.txtman import CloneusManager
from managers.msgman import MessageManager

from run import BotUs

from .gen.gencfg import SetTextConfig


cmds_logger = logging.getLogger('server.cmds')
event_logger = logging.getLogger('server.event')


class TextGen(commands.Cog, SetTextConfig):
    '''Imitate a user or chat with the base model.'''
    def __init__(self, bot: BotUs):
        self.bot = bot
        self.clomgr = CloneusManager(bot,)
        self.msgmgr = MessageManager(bot, n_init_messages=15, message_cache_limit=31)
        self.pstore = self.bot.pstore 
        self.ctx_menu = app_commands.ContextMenu(name='🔂 Redo (Text)', callback=self._cm_redo,)
        self.bot.tree.add_command(self.ctx_menu, override=True)
        
        self.active_model_kwargs = {'checkpoint_path':settings.ACTIVE_MODEL_CKPT, 'gen_config':'best_generation_config.json', 'dtype': None, 'attn_implementation': None}
        self.task_prompts = self._read_task_prompts()

        self._log_extra = {'cog_name': self.qualified_name}
        self._load_lock = asyncio.Lock()
        self._model_load_commands = set(
            [self.ctx_menu.name,'pbot','ubot','mbot','xbot', 'ask','chat', 'reword', ])

    def _read_task_prompts(self, filename='prompts.yaml'):
        with open(settings.CONFIG_DIR/'text'/filename) as f:
            task_prompts = yaml.load(f, yaml.SafeLoader)
        return task_prompts

    @property
    def youtube_quota(self):
        active_usage = self.clomgr.yt_session_quota
        return self.pstore.get('youtube_quota',0)+active_usage

    async def cog_load(self):
        await self.bot.wait_until_ready()
        self.pstore.update(youtube_quota = self.pstore.get('youtube_quota', 0))
        #self.bot.tree.add_command(self.ctx_menu)
        await self.msgmgr.set_default(self.bot.get_channel(settings.CHANNEL_ID))
        
        self.clomgr._preload(**self.active_model_kwargs)
        

    async def cog_unload(self):
        await self.bot.wait_until_ready()
        self.pstore.update(youtube_quota = self.youtube_quota)
        await self.txtdown(self.bot.get_channel(settings.CHANNEL_ID))
        print('TextGen - Goodbye')
        self.bot.tree.remove_command(self.ctx_menu.name, type=self.ctx_menu.type)
        stop_global_executors()
    
    async def cog_before_invoke(self, ctx: commands.Context) -> None:
        if not self.clomgr.is_ready:
            if ctx.command.name in self._model_load_commands:
                await self.check_defer(ctx)
                msg = await self.txtup(ctx.channel)

    async def cog_after_invoke(self, ctx: commands.Context) -> None:
        #print('after invoke', datetime.datetime.now())
        cmd_status = 'FAIL' if ctx.command_failed else 'PASS'
        pos_args = [a for a in ctx.args[2:] if a]

        cmds_logger.info('[{stat}] {a.display_name}({a.name}) command "{c.prefix}{c.invoked_with} ({c.command.qualified_name})" args:({args}, {c.kwargs})'.format(
            stat=cmd_status, a=ctx.author, c=ctx, args=pos_args), extra=self._log_extra)



    @commands.Cog.listener('on_message')
    async def on_message(self, message: discord.Message):

        #await self.check_mention(message)
        message_cache = self.msgmgr.get_mcache(message)
        context_updated = False
        if text_utils.message_filter(message, allow_bot=False):
            # because bot messages are added to context elsewhere, want them all filtered out.
            if message not in message_cache[-5:]:
                print(f'on_message - added: {message.content!r}')
                
                await self.msgmgr.add_message(message)
                context_updated=True
                
                if self.auto_reply_mode and message.author in self.auto_reply_enabled_users: # != '', False, None
                    await self.auto_respond(message_cache)

        #human_proc = self.auto_reply_mode and context_updated        
        
        mstat,lvl = ("ADD", logging.INFO) if context_updated else ("SKIP", logging.DEBUG)
        event_logger.log(lvl, '[{mstat}] {a.display_name}({a.name}) message {message.content!r}'.format(
            mstat=mstat, a=message.author, message=message), extra=self._log_extra)




    async def _react_thumb_down(self, reaction: discord.Reaction):
        try:
            self.msgmgr.remove_message(reaction.message)
        except ValueError:
            print('Message not in context')
        await reaction.message.delete()
    
    async def _react_repeat_one(self, reaction: discord.Reaction):
        try:
            ctx = await self.bot.get_context(reaction.message)
            await self.redo(ctx, reaction.message)
        except ValueError:
            print('Message not in context')

    @commands.Cog.listener('on_reaction_add')
    async def on_reaction_add(self, reaction: discord.Reaction, user: discord.User):
        if not user.bot:
            if reaction.message.author == self.bot.user:
                if str(reaction.emoji) == '👎':
                    await self._react_thumb_down(reaction)
                elif str(reaction.emoji) == '🔂':
                    await self._react_repeat_one(reaction)
                    
        else:
            # bot added reaction to bot authored message ONLY
            if (self.auto_reply_mode and reaction.message.author == self.bot.user and str(reaction.emoji) == '🔜'):
                await self.auto_respond(self.msgmgr.get_mcache(reaction.message))
                await reaction.remove(self.bot.user)

        
        event_logger.info(f'[REACT] {user.display_name}({user.name}) ADD "{reaction.emoji}" TO '
                         '{a.display_name}({a.name}) message {reaction.message.content!r}'.format(a=reaction.message.author, reaction=reaction), extra=self._log_extra)
                        

    async def check_defer(self, ctx: commands.Context):
        needs_defer = False
        if ctx.interaction and not ctx.interaction.response.is_done():
            await ctx.defer()
            await asyncio.sleep(1)
            needs_defer = True
        
        return needs_defer
    
    def status_list(self, keep_advanced:bool=True, ctx: commands.Context = None):
        if ctx is None:
            ctx = self.msgmgr.default_channel
        status = [
            *self.clomgr.list_status(),
            StatusItem('','','---'),
            StatusItem('Message context', len(self.msgmgr.get_mcache(ctx)), f' / {self.msgmgr.message_cache_limit}'), # Note: this show the _unmerged_ length.
            StatusItem('YouTube quota', self.youtube_quota, ' / 10000',), # self.clomgr.yt_session_quota+self.pstore.get('youtube_quota', 0)
            #('Messages bot cached', len(self.bot.cached_messages)),
            #StatusItem('','','---'),
            *self.argsettings()
        ]
        return [s for s in status if keep_advanced or not s.advanced]

    @commands.command('tstatus', aliases=['tstat'])
    async def tstatus_report(self, ctx: commands.Context):
        '''Full text model status report.'''
        msg = '\n'.join(stat.md() for stat in self.status_list(keep_advanced=True, ctx=ctx))
        await ctx.send(msg)

    #@commands.cooldown(1, 10, commands.BucketType.guild)    
    @commands.command(name='txtup', aliases=['textup','tup'])
    async def txtup(self, ctx: commands.Context, reload: bool=False):
        """Fire up the text generation model"""
        
        was_called = hasattr(ctx,'command') and ctx.command.name=='txtup'
        msg = None
        async with self._load_lock:
            if not self.clomgr.is_ready or reload:
                msg = await ctx.send('Powering up....',  silent=True)
                await self.clomgr.load(**self.active_model_kwargs)
            
            await self.bot.report_state('chat', ready=True)
            
        if msg is not None: 
            await msg.edit(content='Imitation Engine Fired Up 🔥')
        elif was_called:
            await self.tstatus_report(ctx)

        
    
    #@commands.cooldown(1, 10, commands.BucketType.guild)   
    @commands.command(name='txtdown', aliases=['textdown','tdown'])
    async def txtdown(self, ctx: commands.Context):
        '''Disable the text generation model'''
        
        was_called = hasattr(ctx,'command') and ctx.command.name=='txtdown'
        
        
        await self.clomgr.unload()
        # doesn't clear unless these are here for some reason
        torch.cuda.empty_cache()
        gc.collect()

        if was_called:
            await ctx.send('Ahh... sweet release ...', delete_after=5)
        
        await self.bot.report_state('chat', ready=False)
        
        #release_memory()
    
    @commands.hybrid_command(name='gcsave')
    async def save_gen_config(self, ctx: commands.Context, name: str = None, ):
        '''Save the current generation config settings for later use.
        
        Args:
            name: the name to save as. If None, will use current datetime. 
        '''
        gc_dir = (self.clomgr.clo.path_data.checkpoint_path/'gen_configs')
        
        if name is None:
            tnow = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
            name = f'{self.clomgr.clo.gen_mode}_{tnow}'
        
        name = re.sub('\W+','_', name[:100].removesuffix('.json'))
        
        self.clomgr.clo.save_genconfig(filepath=(gc_dir/name).with_suffix('.json'))
        
        return await ctx.send(f'Saved current generation config settings to {name}')
    
    async def savedgc_autocomplete(self, interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
        gc_dir = (self.clomgr.clo.path_data.checkpoint_path/'gen_configs')
        confs = ['default'] + ([g.stem.casefold() for g in gc_dir.iterdir()] if gc_dir.exists() else [])
        
        return [app_commands.Choice(name=c, value=c) for c in confs if current.casefold() in c]

    @commands.hybrid_command(name='gcload')
    @app_commands.autocomplete(name=savedgc_autocomplete)
    async def load_saved_gen_config(self, ctx: commands.Context, name: str):
        '''Load saved generation config settings.
        
        Args:
            name: the name of config to load. 
        '''
        gc_dir = (self.clomgr.clo.path_data.checkpoint_path/'gen_configs')

        if name == 'default':
            self.clomgr.clo.set_genconfig(save_on_change=False, preset='ms')
        else:
            gc = self.clomgr.clo.load_genconfig((gc_dir/name).with_suffix('.json'), self.clomgr.clo.path_data)
            delt = self.clomgr.update_genconfig(gc.to_dict())#clo.set_genconfig(save_on_change=False, preset='ms')
        
        await ctx.send(f'Loaded GenConfig: {name!r}\n{delt}')

    @commands.hybrid_command(name='showmodels')
    async def show_models(self, ctx: commands.Context, name_filter: str = None):
        '''Show a list of available chat models
        
        You can pass `.`, `..`, or `...` to see models in the same family as the currently active model.
        Args:
            name_filter: filters down to models that match this name
        '''
        if name_filter and all([c == '.' for c in name_filter]):
            cur_model_path = self.clomgr.clo.path_data.checkpoint_path.parent
            if name_filter == '.': # current run dir - checkpoint level
                cur_model_path = cur_model_path
            elif name_filter == '..': # current data-run dirs - runs level
                cur_model_path = cur_model_path.parent
            elif name_filter == '...': # current model-data-run dirs - data level
                cur_model_path = cur_model_path.parent.parent
            name_filter = cur_model_path.relative_to(settings.RUNS_DIR).as_posix()

        ckpt_list = self.clomgr.modelview_data(name_filter, remove_empty=True)
        if not ckpt_list:
            return await ctx.send(f'No models found matching {name_filter!r}')
        ppview = cview.PrePagedView(ckpt_list, timeout=180)

        msg = 'All Valid Model Options' if name_filter is None else f'Model options matching {name_filter!r}'
        # if len(matches)>1:
        #     ppview = cview.PrePagedView(self.clomgr.modelview_data(name_filter=model_runname, remove_empty=True))
        #     #matchemb = discord.Embed(title="Matching Options", description="".join(io_utils.find_checkpoints(m, allmodelpath) for m in matches))
        #     return await ppview.send(ctx,f'Multiple matches found for "{model_runname}", specify harder')#, embed=matchemb)

        # if checkpoint_name is None:
        #     ckpts = io_utils.find_checkpoints(model_basedir)
        #     emb=discord.Embed(title='Options', description=io_utils.fmt_checkpoints(model_basedir, ckpts, md_format=True, wrapcodeblock=True))
        #     return await ctx.send('Active Model Checkpoints', embed=emb)
        return await ppview.send(ctx, msg)
        
    @commands.hybrid_command(name='switchmodel')
    async def switch_model(self, ctx: commands.Context, 
                           modelpath_filter: str = None, 
                           dtype: typing.Literal['bfloat16','float16']=None, 
                           attn_implementation: typing.Literal["eager", "sdpa", "flash_attention_2"]=None):
        '''Change the underlying text model.
        
        Args:
            modelpath_filter: the path or segment of path to the model (part after Model: --- when call !status)
            dtype: model dtype to load/switch to
            attn_implementation: type of attention to use/switch to
        '''
    
        if modelpath_filter is None:
            if dtype or attn_implementation:
                await self.check_defer(ctx)
                self.active_model_kwargs.update(dtype=dtype, attn_implementation=attn_implementation)

                return await self.txtup(ctx, reload=True)
            
            return await self.show_models(ctx)
        elif all([c == '.' for c in modelpath_filter]):
            return await self.show_models(ctx, modelpath_filter)

        if (exact_path:=settings.RUNS_DIR/modelpath_filter).exists() and (exact_path.parent/'config.yaml').exists():
            # in case a full path is passed, use exactly. 
            # Prevents issues with matching substring checkpoints like checkpoint-100, checkpoints-1000, checkpoints-10000
            full_model_path = exact_path
        else:
            matches = sorted([p for p in settings.RUNS_DIR.rglob('*checkpoint*') if (p.parent/'config.yaml').exists() and modelpath_filter in str(p)])
            if len(matches) != 1:
                return await self.show_models(ctx, name_filter=modelpath_filter)

            full_model_path = matches[0]
            
        msg = await ctx.send(f'Switching text model ... `{full_model_path.relative_to(settings.RUNS_DIR)}`')
        
        self.active_model_kwargs.update(checkpoint_path=full_model_path, dtype=dtype, attn_implementation=attn_implementation)
        return await self.txtup(ctx, reload=True)


    @commands.hybrid_command(name='sayas')
    @app_commands.choices(author=cmd_choices.AUTHOR_DISPLAY_NAMES)
    async def sayas(self, ctx, author: str, message:str):
        """Say something verbatim as someone in chat and add it to the context
        
        Args:
            author: Person to speak as 
            message: Text that the author should say verbatim
        """
        msg = await ctx.send(f"[{author}] {message}")
        await self.msgmgr.add_message(msg, force_auto_response=True)
    
    @commands.command(name='hide', aliases=['h'])
    async def hide(self, ctx: commands.Context):
        '''Prevent a message from being added to the context'''
        # I think this can just be a noop because messages prefixed with ! are ignored anyway.
        return

    @commands.hybrid_command(name='clear')
    @app_commands.choices(context=[app_commands.Choice(name = c, value=c) for c in ['all', 'base', 'discord']])
    async def clear(self, ctx: commands.Context, context:str = 'all'):
        '''Clears all messages from the context and start with a clean slate.
        
        Args:
            context: which chat history to clear. a/all, b/base, or d/discord  
        '''
        bot_name = useridx.get_users('dname', include_bot = True)[0]
        msg = ''
        context = context.lower()
        if context.startswith('a'):
            self.msgmgr.clear_mcache('all') 
            msg = f'{bot_name} and Base context cleared. Got nuffin chief.'
        elif context.startswith('b'):
            self.msgmgr.clear_mcache('base') 
            msg = 'Base context cleared.'
        elif context.startswith('d'):
            self.msgmgr.clear_mcache('default')
            msg = f'{bot_name} Context cleared. FEED ME.'
        
        torch.cuda.empty_cache()
        gc.collect()
        await ctx.send(msg)
    
    @commands.command(name='unclear')
    async def unclear(self, ctx: commands.Context):
        '''Restore all messages removed from the last !clear'''
        
        if not self.msgmgr.last_caches:
            return await ctx.send('Nothing to restore. Chats never cleared.')

        self.msgmgr.restore_caches()
        
        await ctx.send('All chats restored.')

    @commands.command(name='badbot', aliases=['bb'])
    async def badbot(self, ctx: commands.Context):
        '''Deletes the bot's most recent message and removes from context.'''
        last_bot_msg = await discord.utils.find(
            lambda m: (m.author == self.bot.user) and m.content.startswith('['), # BRACKET DEPENDENCY 
            ctx.channel.history(limit=self.msgmgr.message_cache_limit))
        try:
            self.msgmgr.remove_message(last_bot_msg)
        except ValueError:
            print('Message not in context')

        print('DELETED:', last_bot_msg.content)
        await last_bot_msg.edit(content='💥')
        await last_bot_msg.delete(delay=1)
        
    
    @commands.hybrid_command(name='viewctx')
    @app_commands.choices(context=[app_commands.Choice(name = c, value=c) for c in [ 'base', 'discord']])
    async def viewctx(self, ctx, context:str = 'discord', raw: bool = False):
        '''Show all in-context messages. See what bot sees.
        
        Args:
            context: which chat history to view. b/base or d/discord (default: discord)
            raw: View as close to what the model sees as possible, formatting be damned (default: False)
        '''
        if context.lower().startswith('b'):
            
            role_cycle = itertools.cycle(['user','assistant'])
            llm_input_messages = [(next(role_cycle), msg) for msg in self.msgmgr.base_message_cache]
        else:
            llm_input_messages = text_utils.llm_input_transform(self.msgmgr.get_mcache(ctx), user_aliases=self.user_aliases)
            if raw:
                llm_text = self.clomgr.clo.to_text_input(llm_input_messages)
                split_on = self.clomgr.clo.tokenizer.eos_token if self.clomgr.clo.stop_criteria is None else self.clomgr.clo.stop_criteria[0].words[0]
                llm_input_messages = [(i,pmsg+split_on) for i,pmsg in enumerate(filter(None, llm_text.split(split_on))) if pmsg]
            

        pagination_view = cview.MessageContextView(llm_input_messages, items_per_page=10, raw=raw)#data, timeout=None)
        
        await pagination_view.send(ctx)


    #@bot.tree.context_menu(name="🔂 Redo (Text)")
    async def _cm_redo(self, interaction: discord.Interaction, message: discord.Message):
        if not message.author.bot:
            return await interaction.response.send_message('No touchy non-bot messages.', ephemeral=True)
        
        await interaction.response.defer(thinking=True, ephemeral=True)
        await asyncio.sleep(1)
        #try:
        await self.redo(interaction, message, author=None, seed_text=None, _needsdefer=False)
        await interaction.delete_original_response()

    #@check_up('clomgr', '❗ Text model not loaded. Call `!txtup`')
    async def redo(self, ctx:commands.Context, message: discord.Message, author:str=None, seed_text:str=None, _needsdefer=True):
        #ctx = await self.bot.get_context(message)
        if author is None:
            try:
                author = text_utils.extract_author(message.content)
            except ValueError:
                await ctx.send(f'Can only re-roll imitation messages i.e. "[Author] words I say..."', ephemeral=True)
        try:
            mcache_slice = self.msgmgr.get_mcache_subset(message, inclusive=False)
        except ValueError:
            await ctx.send("Can't re-roll. Message not in context.", ephemeral=True)

        await self.check_defer(ctx)
        # if _needsdefer:
        #     await ctx.defer()
        #     await asyncio.sleep(1)
        
        async with (ctx.channel.typing(), self.bot.busy_status(activity='chat')):
            #message.reply()
            sent_messages = await self.clomgr.pipeline(message, mcache_slice, [author], seed_text, ('stream_one' if self.streaming_mode else 'gen_one'))

            await self.msgmgr.replace_message(message, sent_messages)

    @commands.hybrid_command(name='pbot')
    #@check_up('clomgr', '❗ Text model not loaded. Call `!txtup`')
    @app_commands.choices(auto_mode=cmd_choices.AUTO_MODES)
    async def pbot(self, ctx: commands.Context, 
                      auto_mode: str ='rbest', # typing.Literal['rbest','irbest','top', 'urand']
                      author_initials: app_commands.Transform[str,cmd_tfms.AuthorInitialsTransformer]=None,
                      seed_text:str=None):
        """Probabilistically select a random bot to respond with. 
        
        Args:
            auto_mode: Method for automatically choosing the author (default: rbest)
                rbest  = Random weighted selection (p = p)
                irbest = Random inverse weighted selection (p = 1-p)
                urand  = Uniform random selection (p = 1/5)
                top    = Most probable author (p = 1)
            author_initials: Unordered sequence of author initials (no spaces). Restricts selection to those authors.
            seed_text: Text to start off the bot's response with

        """
        
        await self.check_defer(ctx)
        author_candidates = None
        
        if author_initials:
            author_candidates = [useridx.get_users('dname', by='initial')[i] for i in author_initials]

            if len(author_candidates) == 1:
                return await self.anybot(ctx, author_candidates[0], seed_text=seed_text, _needsdefer=False)
                
        
        next_author = await self.clomgr.predict_author(self.msgmgr.get_mcache(ctx), auto_mode, author_candidates)
        await self.anybot(ctx, next_author, seed_text=seed_text, _needsdefer=False)

    async def auto_respond(self, message_cache):
        if self.auto_reply_mode in ['rbest','irbest','urand','top']:
            next_author = await self.clomgr.predict_author(message_cache, self.auto_reply_mode, self.auto_reply_candidates)
        else: 
            # <author_initial>bot. e.g.: j1bot, abot, qbot, d11bot
            next_author = useridx.get_users('dname', by='initial')[self.auto_reply_mode.replace('bot','')]
    
        ctx = await self.bot.get_context(message_cache[-1])
        await self.anybot(ctx, next_author, seed_text=None)
    
    #@check_up('clomgr', '❗ Text model not loaded. Call `!txtup`')
    async def streambot(self, ctx: commands.Context, author:str, seed_text:str,*, _needsdefer=True):        
        await self.check_defer(ctx)
        # if _needsdefer:
        #     await ctx.defer()
        #     await asyncio.sleep(1)
        
        #author_tag_prefix = f"[{author}] " + ((seed_text + ' ') if seed_text else '')
        #msg = await ctx.send(author_tag_prefix)
        
        async with (ctx.channel.typing(), self.bot.busy_status(activity='chat')):
            sent_messages = await self.clomgr.pipeline(ctx, self.msgmgr.get_mcache(ctx), [author], seed_text, 'stream_one')
            for msg in sent_messages:
                await self.msgmgr.add_message(msg)

    #@check_up('clomgr', '❗ Text model not loaded. Call `!txtup`')
    async def batch_streambot(self, ctx: commands.Context, authors:list[str], seed_text:str):        
        await self.check_defer(ctx)
        # await ctx.defer()
        # await asyncio.sleep(1)
        async with (ctx.channel.typing(), self.bot.busy_status(activity='chat')):
            sent_messages = await self.clomgr.pipeline(ctx, self.msgmgr.get_mcache(ctx), authors, seed_text, 'stream_batch')
            for msg in sent_messages:
                await self.msgmgr.add_message(msg)
                
    #@check_up('clomgr', '❗ Text model not loaded. Call `!txtup`')
    async def anybot(self, ctx: commands.Context, author: str, seed_text=None, *, _needsdefer=True):
        """Generalist bot generator."""
        
        if self.streaming_mode:
            return await self.streambot(ctx, author=author, seed_text=seed_text, _needsdefer=_needsdefer)
        await self.check_defer(ctx)    
        # if _needsdefer:
        #     await ctx.defer()
        #     await asyncio.sleep(1)
        
        self.clomgr.tts_mode = self.tts_mode
        async with (ctx.channel.typing(), self.bot.busy_status(activity='chat')):
            sent_messages = await self.clomgr.pipeline(ctx, self.msgmgr.get_mcache(ctx), [author], seed_text, 'gen_one')
            for msg in sent_messages:
                await self.msgmgr.add_message(msg)

    @commands.hybrid_command(name='ubot')
    @app_commands.choices(user=cmd_choices.AUTHOR_DISPLAY_NAMES)
    async def user_bot(self, ctx, user: str, seed_text: str = None ):
        """Call a bot by the authors username 
        
        Args:
            user: the username of bot author
            seed_text: Text to start off responses with.
        """
        
        if self.streaming_mode:
            return await self.streambot(ctx, user, seed_text)
        
        return await self.anybot(ctx, user, seed_text=seed_text)

    @commands.hybrid_command(name='mbot')
    #@check_up('clomgr', '❗ Text model not loaded. Call `!txtup`')
    async def initials_bot(self, ctx: commands.Context, author_initials: app_commands.Transform[str, cmd_tfms.AuthorInitialsTransformer], *, seed_text: str=None):
        """Call one or more bots using author initials
        
        Args:
            author_initials: Unordered sequence of 1+ author initials.
            seed_text: Text to start off responses with.
        """
        
        authors = [useridx.get_users('dname', by='initial')[i] for i in author_initials]
        
        if len(authors)==1:
            return await self.anybot(ctx, authors[0], seed_text=seed_text)
        
        if self.streaming_mode:
            return await self.batch_streambot(ctx, authors, seed_text)
            
        await self.check_defer(ctx)
        # await ctx.defer()
        # await asyncio.sleep(1)

        self.clomgr.tts_mode = self.tts_mode
        async with (ctx.channel.typing(), self.bot.busy_status(activity='chat')):
            sent_messages = await self.clomgr.pipeline(ctx, self.msgmgr.get_mcache(ctx), authors, seed_text, 'gen_batch')
            for msg in sent_messages:
                await self.msgmgr.add_message(msg)
            
    @commands.command(name='author_initial', aliases=useridx.get_users('initial'))
    async def author_initial_commands(self, ctx: commands.Context, *, seed_text:str = None):
        """(pseudo-command). Call a bot by initial. e.g: `!a seed some text`.
        
        Args:
            seed_text: Text to start off response with.
        """
        
        if ctx.invoked_with == ctx.command.name:
            return await ctx.send('Command should not be called directly. Use `!<initial> [SEED_TEXT]`. Options: '+' '.join(f'`!{i}`' for i in useridx.get_users('initial')))
        
        print(ctx.prefix, ctx.invoked_with, ctx.args[2:], ctx.kwargs)
        author_initial = ctx.invoked_with
        await self.anybot(ctx, useridx.get_users('dname', by='initial')[author_initial], seed_text=seed_text)

    @commands.hybrid_command(name='xbot')
    #@check_up('clomgr', '❗ Text model not loaded. Call `!txtup`')
    async def xbot(self, ctx, author: commands.Range[str, 1, 32], seed_text:typing.Optional[str] = None):
        """BYOName, but keep it alphanumeric
        
        Args:
            author: A custom discord username.
            seed_text: Text to start off response with.
        """
        await self.anybot(ctx, author, seed_text=seed_text)




    @commands.hybrid_command(name='ask')
    #@check_up('clomgr', '❗ Text model not loaded. Call `!txtup`')
    async def ask(self, ctx: commands.Context, prompt:str, *, system_msg:str=None):
        """Send a prompt to the underlying, un-fintuned base model.
        
        tips: 
        - use a high max_new_tokens (like 1024)
        - enable streaming if you don't like waiting
        - keep generation config simple. 


        Args:
            prompt: The text to send the chat bot. Can be a question or a instruction or otherwise.
            system_prompt: A guidance statement for the model. Alters how it responds to the prompt.
        """

        if system_msg is None:
            system_msg = self.default_system_msg
        await self.check_defer(ctx)
        # await ctx.defer()
        # await asyncio.sleep(1)
        
        self.clomgr.tts_mode = self.tts_mode
        async with (ctx.channel.typing(), self.bot.busy_status(activity='chat')):
            if self.streaming_mode:
                sent_messages = await self.clomgr.base_streaming_generate(ctx, prompt, system_msg)
            else:
                sent_messages = await self.clomgr.base_generate(ctx, prompt, system_msg)
        return sent_messages    
    
    @commands.hybrid_command(name='chat')            
    #@check_up('clomgr', '❗ Text model not loaded. Call `!txtup`')
    async def chat(self, ctx: commands.Context, prompt:str, *, system_msg:str=None):
        """Have a conversation with the underlying, un-fintuned base model.
        
        tips: 
        - use a high max_new_tokens (like 1024)
        - enable streaming if you don't like waiting
        - keep generation config simple. 


        Args:
            prompt: The next message to send the chat bot.
            system_prompt: A guidance statement for the model. Alters how it responds to the prompt.
        """
        if system_msg is None:
            system_msg = self.default_system_msg
        
        await self.check_defer(ctx)
        # await ctx.defer()
        # await asyncio.sleep(1)
        self.msgmgr.base_message_cache.append(prompt)
        self.clomgr.tts_mode = self.tts_mode
        async with (ctx.channel.typing(), self.bot.busy_status(activity='chat')):
            if self.streaming_mode:
                sent_messages = await self.clomgr.base_streaming_generate(ctx, self.msgmgr.base_message_cache, system_msg)
            else:
                sent_messages = await self.clomgr.base_generate(ctx, self.msgmgr.base_message_cache, system_msg)
            
        # Re-join any splits from the 2000 char limit 
        # TODO: Watch for any missing spaces
        sent_text = ''.join([m.clean_content for m in sent_messages])
        self.msgmgr.base_message_cache.append(sent_text)
        return sent_text
    
    # provide 4 creative rewritings of this text to image prompt, be detailed and highly creative but keep try to stay true to 
    @commands.hybrid_command(name='reword')
    async def reword_t2i_prompt(self, ctx:commands.Context, *, prompt:str):
        '''Reword an image prompt to add details for use in /draw or /redraw

        Args:
            prompt: The prompt to enrich with details
        '''
        formatted_prompt = "\n".join(self.task_prompts['reword']['enhanced_prompt']).format(prompt=prompt)
        sent_messages = await self.ask(ctx, formatted_prompt, system_msg=None)
        return ''.join([m.clean_content for m in sent_messages])
        #return formatted_prompt


async def setup(bot:BotUs):
    #load_nowait = bool(os.getenv('EAGER_LOAD',False))
    await bot.add_cog(TextGen(bot))#, load_nowait=load_nowait))