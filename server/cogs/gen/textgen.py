import os
import gc
import re
import random
import itertools
import typing
import asyncio
import datetime

import functools
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands, tasks

from omegaconf import OmegaConf
import torch

from cloneus.data import roles

import config.settings as settings
from utils import text as text_utils, io as io_utils
from utils.command import check_up

from utils.globthread import async_wrap_thread, stop_global_thread

from views import contextview as cview
from cmds import (
    flags as cmd_flags, 
    transformers as cmd_tfms, 
    choices as cmd_choices
)
from managers.txtman import CloneusManager
from managers.msgman import MessageManager

from run import BotUs

from .gencfg import SetConfig

# Determine if initials based commmands can be used.
AUTHOR_INITALS = list(roles.initial_to_author)
ICMD_ENABLED = any(AUTHOR_INITALS)

#model_logger = settings.logging.getLogger('model')
cmds_logger = settings.logging.getLogger('cmds')
event_logger = settings.logging.getLogger('event')


class TextGen(commands.Cog, SetConfig):
    '''Imitate a user or chat with the base model.'''
    def __init__(self, bot: BotUs, pstore:io_utils.PersistentStorage, clomgr: CloneusManager, msgmgr:MessageManager):
        self.bot = bot
        self.clomgr = clomgr
        self.msgmgr = msgmgr
        self.pstore = pstore #io_utils.PersistentStorage()
        self.ctx_menu = app_commands.ContextMenu(name='🔂 Redo (Text)', callback=self._cm_redo,)
        self.bot.tree.add_command(self.ctx_menu, override=True)
        self.init_model = settings.ACTIVE_MODEL_CKPT

    @property
    def youtube_quota(self):
        active_usage = self.clomgr.yt_session_quota
        return self.pstore.get('youtube_quota',0)+active_usage

    async def cog_load(self):
        await self.bot.wait_until_ready()
        self.pstore.update(youtube_quota = self.pstore.get('youtube_quota', 0))
        #self.bot.tree.add_command(self.ctx_menu)
        self.clomgr._preload(self.init_model, gen_config='best_generation_config.json')
        

    async def cog_unload(self):
        await self.bot.wait_until_ready()
        self.pstore.update(youtube_quota = self.youtube_quota)
        await self.txtdown(self._channel, announce=False)
        print('TextGen - Goodbye')
        self.bot.tree.remove_command(self.ctx_menu.name, type=self.ctx_menu.type)
        stop_global_thread()

    async def cog_after_invoke(self, ctx: commands.Context) -> None:
        #print('after invoke', datetime.datetime.now())
        cmd_status = 'FAIL' if ctx.command_failed else 'PASS'
        pos_args = [a for a in ctx.args[2:] if a]

        cmds_logger.info(f'(cog_after_invoke, {self.qualified_name})'
                         '- [{stat}] {a.display_name}({a.name}) command "{c.prefix}{c.invoked_with} ({c.command.name})" args:({args}, {c.kwargs})'.format(stat=cmd_status, a=ctx.author, c=ctx, args=pos_args))



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

        mstat = "ADD" if context_updated else "SKIP"
        event_logger.info('(on_message)'
                        '- [{mstat}] {a.display_name}({a.name}) message {message.content!r}'.format(mstat=mstat, a=message.author, message=message))




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

        
        event_logger.info('(on_reaction_add)'
                         f'- [REACT] {user.display_name}({user.name}) ADD "{reaction.emoji}" '
                         'TO {a.display_name}({a.name}) message {reaction.message.content!r}'.format(a=reaction.message.author, reaction=reaction))
                        





    @commands.cooldown(1, 10, commands.BucketType.guild)    
    @commands.command(name='txtup', aliases=['textup','tup'])
    async def txtup(self, ctx: commands.Context, announce: bool = True, load: bool=True):
        """Fire up the model"""
        await self.bot.wait_until_ready()

        if announce: 
            msg = await ctx.send('Powering up....',  silent=True)
        
        if load and not self.clomgr.is_ready:
            await self.clomgr.load(self.init_model, gen_config='best_generation_config.json')
            
        await self.bot.change_presence(**settings.BOT_PRESENCE['chat'])
            
        if announce: 
            await msg.delete()
            await ctx.send('Imitation Engine Fired Up 🔥')
        
    
    @commands.cooldown(1, 10, commands.BucketType.guild)   
    @commands.command(name='txtdown', aliases=['textdown','tdown'])
    async def txtdown(self, ctx: commands.Context, announce: bool = True, unload: bool=True):
        '''Disable the bot's brain power'''
        await self.bot.wait_until_ready()
        
        if unload:
            await self.clomgr.unload()
            # doesn't clear unless these are here for some reason
            torch.cuda.empty_cache()
            gc.collect()

        if announce:
            await ctx.send('Ahh... sweet release ...', delete_after=5)
        
        await self.bot.change_presence(**settings.BOT_PRESENCE['ready'])
        await self.bot.wait_until_ready()
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
        
        name = re.sub('\w+','_', name[:100].removesuffix('.json'))+'.json'
        
        self.clomgr.clo.save_genconfig(filepath=gc_dir/name)
        
        return await ctx.send(f'Saved current generation config settings to {name}')
    
    async def savedgc_autocomplete(self, interaction: discord.Interaction, current: str) -> list[app_commands.Choice[str]]:
        gc_dir = (self.clomgr.clo.path_data.checkpoint_path/'gen_configs')
        confs = ['default'] + ([g.name for g in gc_dir.iterdir()] if gc_dir.exists() else [])
        
        return [app_commands.Choice(name=c, value=c) for c in confs if current.lower() in c.lower()]

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
            self.clomgr.clo.load_genconfig(gc_dir/name, self.clomgr.clo.path_data)
        
        await ctx.send(f'Loaded GenConfig: {name!r}')

    @commands.hybrid_command(name='showmodels')
    async def show_models(self, ctx: commands.Context, name_filter: str = None):
        '''Show a list of available models
        
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
                await ctx.defer()
                await self.txtdown(ctx, False, False)
                
                await self.clomgr.load(self.clomgr.clo.path_data.checkpoint_path, dtype=dtype, attn_implementation=attn_implementation)
                await self.txtup(ctx, False, False)
                #await self.status_report(ctx) # TODO: FIX 
                #await self.up(ctx, True)
                return await ctx.send(content='Done.', delete_after=3)
            
            return await self.show_models(ctx)
        elif all([c == '.' for c in modelpath_filter]):
            return await self.show_models(ctx, modelpath_filter)
            
        matches = sorted([p for p in settings.RUNS_DIR.rglob('*checkpoint*') if (p.parent/'config.yaml').exists() and modelpath_filter in str(p)])
        if len(matches) != 1:
            #print(matches)
            return await self.show_models(ctx, name_filter=modelpath_filter)

        full_model_path = matches[0]
            
        msg = await ctx.send(f'Switching to model: {full_model_path.relative_to(settings.RUNS_DIR)} ...')
        
        await self.txtdown(ctx, False, False)
        await self.clomgr.load(full_model_path, dtype=dtype, attn_implementation=attn_implementation)
        await self.txtup(ctx, False, False)
        await msg.edit(content='Done.', delete_after=3)
        #await self.status_report(ctx) # TODO: FIX
        #await self.down(ctx, False)
        #await self.up(ctx, True)

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
        msg = ''
        context = context.lower()
        if context.startswith('a'):
            self.msgmgr.clear_mcache('all') 
            msg = f'{roles.BOT_NAME} and Base context cleared. Got nuffin chief.'
        elif context.startswith('b'):
            self.msgmgr.clear_mcache('base') 
            msg = 'Base context cleared.'
        elif context.startswith('d'):
            self.msgmgr.clear_mcache('default')
            msg = f'{roles.BOT_NAME} Context cleared. FEED ME.'
        
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
                split_on = self.clomgr.clo.tokenizer.eos_token if self.clomgr.clo.stop_criteria is None else self.clomgr.clo.cfg.postfix
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
        msg = await interaction.followup.send('Done.', silent=True, ephemeral=True, wait=True)
        await msg.delete(delay=1)

    @check_up('clomgr', '❗ Text model not loaded. Call `!txtup`')
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
            await ctx.send("Can't re-roll. Message not in context.")

        if _needsdefer:
            await ctx.defer()
            await asyncio.sleep(1)
        
        async with (ctx.channel.typing(), self.bot.writing_status()):
            #message.reply()
            sent_messages = await self.clomgr.pipeline(message, mcache_slice, [author], seed_text, ('stream_one' if self.streaming_mode else 'gen_one'))

            await self.msgmgr.replace_message(message, sent_messages)

    @commands.hybrid_command(name='pbot')
    @check_up('clomgr', '❗ Text model not loaded. Call `!txtup`')
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
        
        await ctx.defer()
        await asyncio.sleep(1)
        author_candidates = None
        
        if author_initials:
            author_candidates = [roles.initial_to_author[i] for i in author_initials]

            if len(author_candidates) == 1:
                return await self.anybot(ctx, author_candidates[0], seed_text=seed_text, _needsdefer=False)
                
        
        next_author = await self.clomgr.predict_author(self.msgmgr.get_mcache(ctx), auto_mode, author_candidates)
        await self.anybot(ctx, next_author, seed_text=seed_text, _needsdefer=False)

    async def auto_respond(self, message_cache):
        if self.auto_reply_mode in ['rbest','irbest','urand','top']:
            next_author = await self.clomgr.predict_author(message_cache, self.auto_reply_mode, self.auto_reply_candidates)
        else: 
            # <author_initial>bot. e.g.: j1bot, abot, qbot, d11bot
            next_author = roles.initial_to_author[self.auto_reply_mode.replace('bot','')]
    
        ctx = await self.bot.get_context(message_cache[-1])
        await self.anybot(ctx, next_author, seed_text=None)
    
    @check_up('clomgr', '❗ Text model not loaded. Call `!txtup`')
    async def streambot(self, ctx: commands.Context, author:str, seed_text:str,*, _needsdefer=True):        
        if _needsdefer:
            await ctx.defer()
            await asyncio.sleep(1)
        
        #author_tag_prefix = f"[{author}] " + ((seed_text + ' ') if seed_text else '')
        #msg = await ctx.send(author_tag_prefix)
        
        async with (ctx.channel.typing(), self.bot.writing_status()):
            sent_messages = await self.clomgr.pipeline(ctx, self.msgmgr.get_mcache(ctx), [author], seed_text, 'stream_one')
            for msg in sent_messages:
                await self.msgmgr.add_message(msg)

    @check_up('clomgr', '❗ Text model not loaded. Call `!txtup`')
    async def batch_streambot(self, ctx: commands.Context, authors:list[str], seed_text:str):        
        await ctx.defer()
        await asyncio.sleep(1)
        async with (ctx.channel.typing(), self.bot.writing_status()):
            sent_messages = await self.clomgr.pipeline(ctx, self.msgmgr.get_mcache(ctx), authors, seed_text, 'stream_batch')
            for msg in sent_messages:
                await self.msgmgr.add_message(msg)
                
    @check_up('clomgr', '❗ Text model not loaded. Call `!txtup`')
    async def anybot(self, ctx: commands.Context, author: str, seed_text=None, *, _needsdefer=True):
        """Generalist bot generator."""
        
        if self.streaming_mode:
            return await self.streambot(ctx, author=author, seed_text=seed_text, _needsdefer=_needsdefer)
            
        if _needsdefer:
            await ctx.defer()
            await asyncio.sleep(1)
        
        self.clomgr.tts_mode = self.tts_mode
        async with (ctx.channel.typing(), self.bot.writing_status()):
            sent_messages = await self.clomgr.pipeline(ctx, self.msgmgr.get_mcache(ctx), [author], seed_text, 'gen_one')
            for msg in sent_messages:
                await self.msgmgr.add_message(msg)


    @commands.hybrid_command(name='ibot', enabled=ICMD_ENABLED, hidden=(not ICMD_ENABLED))
    @check_up('clomgr', '❗ Text model not loaded. Call `!txtup`')
    async def initials_bot(self, ctx: commands.Context, author_initials: app_commands.Transform[str, cmd_tfms.AuthorInitialsTransformer], *, seed_text: str=None):
        """Call one or more bots using author initials
        
        Args:
            author_initials: Unordered sequence of 1+ author initials.
            seed_text: Text to start off responses with.
        """
        
        authors = [roles.initial_to_author[i] for i in author_initials]
        
        if len(authors)==1:
            return await self.anybot(ctx, authors[0], seed_text=seed_text)
        
        if self.streaming_mode:
            return await self.batch_streambot(ctx, authors, seed_text)
            
        
        await ctx.defer()
        await asyncio.sleep(1)

        self.clomgr.tts_mode = self.tts_mode
        async with (ctx.channel.typing(), self.bot.writing_status()):
            sent_messages = await self.clomgr.pipeline(ctx, self.msgmgr.get_mcache(ctx), authors, seed_text, 'gen_batch')
            for msg in sent_messages:
                await self.msgmgr.add_message(msg)
            
    @commands.command(name='author_initial', enabled=ICMD_ENABLED, hidden=(not ICMD_ENABLED), aliases=AUTHOR_INITALS)
    async def author_initial_commands(self, ctx: commands.Context, *, seed_text:str = None):
        """(pseudo-command). Call a bot by initial. e.g: `!a seed some text`.
        
        Args:
            seed_text: Text to start off response with.
        """
        
        if ctx.invoked_with == ctx.command.name:
            return await ctx.send('Command should not be called directly. Use `!<initial> [SEED_TEXT]`. Options: '+' '.join(f'`!{i}`' for i in AUTHOR_INITALS))
        
        print(ctx.prefix, ctx.invoked_with, ctx.args[2:], ctx.kwargs)
        author_initial = ctx.invoked_with
        await self.anybot(ctx, roles.initial_to_author[author_initial], seed_text=seed_text)

    @commands.hybrid_command(name='xbot')
    @check_up('clomgr', '❗ Text model not loaded. Call `!txtup`')
    async def xbot(self, ctx, author: commands.Range[str, 1, 32], seed_text:typing.Optional[str] = None):
        """BYOName, but keep it alphanumeric
        
        Args:
            author: A custom discord username.
            seed_text: Text to start off response with.
        """
        await self.anybot(ctx, author, seed_text=seed_text)




    @commands.hybrid_command(name='ask')
    @check_up('clomgr', '❗ Text model not loaded. Call `!txtup`')
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
        await ctx.defer()
        await asyncio.sleep(1)
        
        self.clomgr.tts_mode = self.tts_mode
        async with (ctx.channel.typing(), self.bot.writing_status()):
            if self.streaming_mode:
                sent_messages = await self.clomgr.base_streaming_generate(ctx, prompt, system_msg)
            else:
                sent_messages = await self.clomgr.base_generate(ctx, prompt, system_msg)
            
    
    @commands.hybrid_command(name='chat')            
    @check_up('clomgr', '❗ Text model not loaded. Call `!txtup`')
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
        
        await ctx.defer()
        await asyncio.sleep(1)
        self.msgmgr.base_message_cache.append(prompt)
        self.clomgr.tts_mode = self.tts_mode
        async with (ctx.channel.typing(), self.bot.writing_status()):
            if self.streaming_mode:
                sent_messages = await self.clomgr.base_streaming_generate(ctx, self.msgmgr.base_message_cache, system_msg)
            else:
                sent_messages = await self.clomgr.base_generate(ctx, self.msgmgr.base_message_cache, system_msg)
            
        # Re-join any splits from the 2000 char limit 
        # TODO: Watch for any missing spaces
        self.msgmgr.base_message_cache.append(''.join([m.clean_content for m in sent_messages]))
        # for msg in sent_messages:
        #     self.msgmgr.base_message_cache.append(msg)
    

    @commands.hybrid_command(name='reword')
    async def reword_t2i_prompt(self, ctx:commands.Context, prompt:str):
        '''Reword a image prompt to add details for use in /draw or /redraw

        Args:
            prompt: The base simple prompt to add details to
        '''
        # https://github.com/TencentQQGYLab/ELLA?tab=readme-ov-file#1-caption-upscale
        instr = ('Please generate the long prompt version of the short one according to the given examples. '
                'Long prompt version should consist of 3 to 5 sentences. Long prompt version must specify the color, shape, texture or spatial relation of the included objects. '
                'DO NOT generate sentences that describe any atmosphere!!!')
        
        samples = [
            {'short':'A calico cat with eyes closed is perched upon a Mercedes.',
            'long':("a multicolored cat perched atop a shiny black car. the car is parked in front of a building with wooden walls and a green fence. "
                    "the reflection of the car and the surrounding environment can be seen on the car's glossy surface.")
            },
            {'short':"A boy sitting on a chair holding a video game remote.",
            'long':("a young boy sitting on a chair, wearing a blue shirt and a baseball cap with the letter 'm'. he has a red medal around his neck and is holding a white game controller. "
                    "behind him, there are two other individuals, one of whom is wearing a backpack. to the right of the boy, there's a blue trash bin with a sign that reads 'automatic party'.")
            },
            {'short':"man on a bank of the water fishing.",
            'long':("a serene waterscape where a person, dressed in a blue jacket and a red beanie, stands in shallow waters, fishing with a long rod. "
                    "the calm waters are dotted with several sailboats anchored at a distance, and a mountain range can be seen in the background under a cloudy sky.")
            },
            {'short':"A kitchen with a cluttered counter and wooden cabinets.",
            'long':("a well-lit kitchen with wooden cabinets, a black and white checkered floor, and a refrigerator adorned with a floral decal on its side. "
                    "the kitchen countertop holds various items, including a coffee maker, jars, and fruits.")
            },
            {'short': prompt, 'long':""}
        ]
        
        few_shots = '\n\n'.join([f"Short: {s['short']}\nLong: {s['long']}" for s in samples])
        formatted_prompt = f'{instr}\n\n{few_shots}'
        return await self.ask(ctx, formatted_prompt, system_msg='')
        #return formatted_prompt


# async def setup(bot):
#     load_nowait = bool(os.getenv('EAGER_LOAD',False))
    
#     await bot.add_cog(TextGen(bot, load_nowait=load_nowait))