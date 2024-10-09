import re
import gc
import datetime
import subprocess
from pathlib import Path
from functools import cached_property
import random
import typing

import torch
import discord
from discord import app_commands
from discord.ext import commands, tasks

import config.settings as settings
from utils import text as text_utils, io as io_utils

from utils.globthread import wrap_async_executor

from cloneus import Cloneus
from cloneus.data import useridx
from cloneus.plugins import youtube

def get_gpu_memory():
    try:
        memfree,memtotal = torch.cuda.mem_get_info()
        vram_total = round(memtotal / (1024**2))
        vram_used = round((memtotal-memfree) / (1024**2))
        
        return vram_used, vram_total
    except (subprocess.CalledProcessError, IndexError, ValueError) as e:
        print(e)
        return 0,0


model_logger = settings.logging.getLogger('model')


class CloneusManager():
    def __init__(self, bot):
        '''Handler/Middleware for all interactions with Cloneus class and between discord. Importantly, does NOT have commands'''
        self.bot = bot
        self.all_checkpoints = io_utils.find_checkpoints(settings.RUNS_DIR, require_config=True) # .relative_to(settings.ROOT_DIR)
        # self._load_nowait = load_nowait
        self.clo = None
        self.ytm = youtube.YouTubeManager(enabled=True)
        
        self.tts_mode = False
        self.emojis = self.bot.guilds[0].emojis
        #self.status = 'down'
        self.last_run = None
        self.run_count = 0
        self.model_randomization:dict = {}

        self.banned_words: list[str] = []
        self.weighted_words: list[tuple[str,float]] = []
        self.user_aliases: dict[str,str] = None

    @property
    def yt_session_quota(self):
        return self.ytm.quota_usage if self.clo else 0
    
    @property
    def gen_config(self):
        return self.clo.gen_config
    
    @property
    def is_ready(self):
        return self.clo is not None and self.clo.model is not None
        #return self.status == 'up'
    
    @property
    def hot_swappable_checkpoints(self) -> list[Path] | list:
        if self.is_ready:
            return [c for c in self.all_checkpoints if self.clo.path_data.base_model_alias in str(c)]
        return []
    
    def list_status(self, stored_yt_quota=0):
        '''Check bot status.'''
        # ✅⚠❌🛑💯🟥🟨🟩⬛✔🗯💭💬👁‍🗨🗨
        gconf_settings = self.clo.get_genconfig(verbose=False)
        gconf_settings.pop('ban_words_ids',None)
        gconf_settings.pop('sequence_bias',None)
        vram_use, vram_total = get_gpu_memory()
        model_name,checkpoint='',''
        if self.clo:
            model_name = self.clo.path_data.run_path.relative_to(settings.RUNS_DIR)
            checkpoint = self.clo.path_data.checkpoint_name
        statuses = [
            ('Bot status', 'Up' if self.is_ready else 'Down', " ✔" if self.is_ready else " ✖"),
            ('Model', model_name, f"/{checkpoint}"),
            ('Tag placement', self.clo.cfg.tag_placement, f""),
            ('Flash_dtype', f'{self.clo.cfg.attn_implementation}', f' - {self.clo.torch_dtype}'),
            ('vRAM usage', f'{vram_use:,}MiB', f' / {vram_total:,}MiB'),
            ('YouTube quota', self.yt_session_quota+stored_yt_quota,' / 10000',),
            ('Latency', f'{round(self.bot.latency * 1000)}ms',''),
            ('Generation mode', self.clo.gen_mode,''),
            ('', '\n'.join([f'- {k}: {v}' for k,v in gconf_settings.items()]),''),
            
        ]
        if self.banned_words:
            statuses.append(('Word Bans','', self.banned_words))
        if self.weighted_words:
            statuses.append(('Word Weights','',self.weighted_words))

        return statuses
    
    @wrap_async_executor
    def unload(self):
        prekwargs = {'checkpoint_path':self.clo.path_data.checkpoint_path, 'gen_config':self.clo.gen_config, 
                     'dtype':self.clo.cfg.dtype, 'attn_implementation':self.clo.cfg.attn_implementation}
        if self.clo is not None:
            self.clo.unload_model(partial=False)
        #self.status = 'down'
        self._preload(**prekwargs)

    def _preload(self, checkpoint_path: str|Path, gen_config=None, dtype:str=None, attn_implementation:typing.Literal["eager", "sdpa", "flash_attention_2"]=None):
        if self.clo is None or self.clo.model is None:
            self.clo = Cloneus.from_pretrained(checkpoint_path, gen_config=gen_config, ytm=self.ytm, dtype=dtype, attn_implementation=attn_implementation, load=False)

    @wrap_async_executor
    def load(self, checkpoint_path: str|Path, gen_config=None, dtype:str=None, attn_implementation:typing.Literal["eager", "sdpa", "flash_attention_2"]=None ):
        if self.clo is None:
            self.clo = Cloneus.from_pretrained(checkpoint_path, gen_config=gen_config, ytm=self.ytm, dtype=dtype, attn_implementation=attn_implementation, load=True)
        else:
            self.clo = self.clo.swap_model(checkpoint_path, gen_config=gen_config, dtype=dtype, attn_implementation=attn_implementation, )
        
        #self.status = 'up'
        
        esc_authtags = [re.escape(useridx.format_author_tag(u, self.clo.cfg.author_tag)) for u in useridx.get_users('dname')]
        self.RE_ANY_USERTAG = re.compile(r'(^{}){}'.format('|'.join(esc_authtags), self.clo.cfg.tag_sep), re.MULTILINE) # NOTE: will NOT work if decide to use UPPER or lower case names
        
        model_logger.info(f'Using model:\n - {str(self.clo.path_data.checkpoint_path)} - ({self.clo.torch_dtype} / {self.clo.cfg.attn_implementation})')
        model_logger.info(f'Generation mode init: "{self.clo.gen_mode}"\n - {self.clo.get_genconfig(verbose=True)}\n')
        gc.collect()
    
    async def load_random_model(self, fast_proba=0.5):
        # TODO: determine if swapping between 2 different mistral models is faster mistral->llama
        # e.g. OpenHeremes -> mistral-instruct-v1 faster than llama2-13b -> mistral7b 
        # If so, can factor that in to probablities as well.
        
        # model_ckpt_groups = io_utils.gb_part(self.all_checkpoints, 'model', 'model')
        # total_ckpts = sum(len(cg) for m,cg in model_ckpt_groups)
        # for modelname,ckptgrp in model_ckpt_groups: print((modelname, len(ckptgrp), len(ckptgrp)/total_ckpts))

        gconfig = self.gen_config
        fast_ckpts = self.hot_swappable_checkpoints
        if random.random() < fast_proba and len(fast_ckpts) > 1:
            options = fast_ckpts #[settings.RUNS_DIR/m['ckpt'] for m in settings.TRAINED_MODELS]
        else:
            options = self.all_checkpoints
        
        next_model = random.choice(options)
        # aqlm causes memory to run out and break things
        while 'aqlm' in str(next_model):
            next_model = random.choice(options)

        print(f'CHANGE PLACES: {next_model.relative_to(settings.RUNS_DIR)}')
        await self.load(next_model, gen_config=gconfig)
        #self.clo.gen_config = gconfig
        #print(f'CHANGE PLACES: {next_model}')

    def modelview_data(self, name_filter=None, remove_empty=True):
        '''Format checkpoints for use in list/switch model view'''
        pages = {}
        inc_pat = (f'.*{name_filter}.*' if name_filter else None)
        rel_runs_dir = settings.RUNS_DIR#.relative_to(settings.ROOT_DIR) # runs/full
        all_ckpts = io_utils.find_checkpoints(rel_runs_dir, include_pattern=inc_pat, require_config=True)
        ckpt_grps = io_utils.gb_part(all_ckpts, 'model', 'runname', list)
        
        for model_name, dataset_runs in ckpt_grps:
            title=model_name
            pages[title] = []
            for dataset_name, run_sets in dataset_runs:
                field_title = dataset_name
                data_segments = []
                for run_name, ckptlist in run_sets:
                    ckpt_md_list = ''.join(['\n- {}'.format(o.name) for o in sorted(ckptlist, key=lambda c: int(c.name.split('-')[1]) ) ])
                    md_list_segment = f'{run_name}{ckpt_md_list}'
                    data_segments.append(md_list_segment)
                    
                    #print(model_name, dataset_name, run_name, len(md_list_segment))
                    
                data = '\n\n'.join(data_segments)
                fdata = f'```\n{data}\n```'
                
                pages[title].append((field_title, fdata))
                #print('fdata len:',len(fdata))
                    
        if remove_empty:
            return [(k,v) for k,v in pages.items() if v]
        
        return list(pages.items())
    
    
    def set_guidance_phrase(self, author:str, phrase:str, guidance_scale:float=None):
        prev_phrase = self.clo.guidance_phrases[author]
        self.clo.guidance_phrases[author] = phrase
        if guidance_scale is not None:
            self.update_genconfig({'guidance_scale':guidance_scale})
        return f"Updated {author}'s vibe: {prev_phrase} -> {phrase}"
        
    def update_wordlists(self, banned_words:list[str]|typing.Literal['CLEAR']=None, weighted_words:list[tuple[str,float]]|typing.Literal['CLEAR']=None):
        gcon_update = {}
        if banned_words is not None:
            if banned_words == 'CLEAR':
                self.banned_words = []
                gcon_update['bad_words_ids']=None
            else:
                self.banned_words = sorted(set(self.banned_words+banned_words))
                gcon_update['bad_words_ids']=self.clo.encode_wordslist(self.banned_words)

        if weighted_words is not None:
            if weighted_words == 'CLEAR':
                self.weighted_words = []
                gcon_update['sequence_bias']=None
            else:
                self.weighted_words = sorted(set(self.weighted_words+weighted_words))
                gcon_update['sequence_bias']=self.clo.encode_wordslist(self.weighted_words)

        
        return self.update_genconfig(gcon_update)
        


    def update_genconfig(self, new_config: dict):
        """Set a Generation Configuration value. Changes """
        
        changed = self.clo.set_genconfig(**new_config)
        
        if changed:
            model_logger.info(f'Generation args update\n - {self.clo.get_genconfig(verbose=True)}\n')
            update_message = ("Updated: "+ ', '.join(f"`{a}: ({changed[a]['prev']} -> {changed[a]['new']})`" for a in changed))
        else:
            update_message = ("No Δ: " + ', '.join(f"`{k}: {v}`" for k,v in self.clo.get_genconfig(False).items()))
        
        return update_message

    @wrap_async_executor
    def predict_author(self, message_cache:list[discord.Message],  autoreply_mode: str, author_candidates: list[str]=None) -> str:
        
        llm_input_messages = text_utils.llm_input_transform(message_cache, do_filter=False, user_aliases=self.user_aliases)
        auth_cands = useridx.get_users('dname') if not author_candidates else author_candidates
        author_probas = self.clo.author_probabilities(llm_input_messages, authors=auth_cands)
        
        model_logger.info(f'Autobot Probablities: {author_probas}')
        if author_candidates:
            author_probas = [(a,p) for (a,p) in author_probas if a in author_candidates]
            psum = sum(p for a,p in author_probas)
            author_probas = [(a,p/psum) for a,p in author_probas]
            print('->',author_probas)
    
        if autoreply_mode == 'top':
            author_choice,top_prob = author_probas[0]
            
        
        authors,probas = list(zip(*author_probas))

        match autoreply_mode:
            case 'rbest':
                author_choice = random.choices(authors,probas)[0]
            case 'irbest':
                author_choice = random.choices(authors,[1-p for p in probas])[0]
            case 'urand':
                author_choice = random.choices(authors, None)[0]
        
        if not author_choice.isalnum():  
            # failsafe against empty/punctuation brackets
            author_choice = random.choice(auth_cands)

        return author_choice
    

    @wrap_async_executor
    def _base_generate(self, text_inputs:list[str], system_prompt:str = None):
        """Generate a response."""
        input_text, model_output, input_length, output_len = self.clo.base_generate(text_inputs, system_prompt, return_tuple=True)
        return input_text, model_output, input_length, output_len
    
    async def base_generate(self, ctx, text_inputs:list[str], system_prompt:str = None):
        """Generate a response."""
        #text_input, _ = self.prepare_llm_inputs(text_input, None)
        print(text_inputs)
        input_text, model_output, input_length, output_len = await self._base_generate(text_inputs, system_prompt)
        log_input_text, log_discord_out, output_lengths = input_text, model_output, [output_len]
        
        await self.log_and_bookkeep(input_length, log_input_text, output_lengths, log_discord_out)
        
        messages = await self.send_collect(ctx, [model_output], char_limit=2000)
        
        return messages
    
    @wrap_async_executor
    def _base_streaming_generate(self, text_inputs:list[str], system_prompt:str = None):

        chunk_idx = 0
        generated_text, chunk_len, tick = "", 0, 0
        for new_word in self.clo.base_stream_generate(text_inputs, system_prompt):
            if new_word:
                wordlen = len(new_word)
                if chunk_len+wordlen >=2000:
                    completed_text, completed_i = generated_text, chunk_idx 
                    generated_text, chunk_len, tick = "", 0, 0 # reset
                    chunk_idx+=1
                    
                    yield completed_i,completed_text
                    
                generated_text += new_word
                chunk_len+=wordlen

                if tick % 3 == 0:
                    yield chunk_idx, generated_text 
                tick+=1
        print() # force a nl because of end=''
        yield chunk_idx,generated_text

    
    async def base_streaming_generate(self, ctx:commands.Context, text_inputs:list[str], system_prompt:str = None):
        # llm_input_messages, seed_text = self.prepare_llm_inputs(message_cache, seed_text)
        messages = []
        #author = authors[0]
        #author_tag_prefix = f"[{author}] " + ((seed_text + ' ') if seed_text else '')
        author_tag_prefix = '​'
        msg = await ctx.send(author_tag_prefix)
        messages.append(msg)
        last_i = 0
        for cidx,updated_content in await self._base_streaming_generate(text_inputs, system_prompt):                
            #msg = await msg.edit(content=author_tag_prefix+updated_content)
            if last_i != cidx:
                msg = await ctx.send(updated_content)
                messages.append(msg)
                last_i = cidx
            
            messages[cidx] = await messages[cidx].edit(content=updated_content)
            
        input_text, model_output, input_length, output_length = self.clo.last_streamed(batch=False, return_tuple=True)
        discord_outs = [model_output]
        
        #msg = await msg.edit(content=discord_outs[0])
        
        #messages.append(msg)
        
        log_input_text, log_discord_out, output_lengths = input_text, discord_outs[0], [output_length]
        await self.log_and_bookkeep(input_length, log_input_text, output_lengths, log_discord_out)

        return messages

    # NOTE: this WORKS
    @wrap_async_executor
    def generate(self, llm_input_messages: list[tuple], author:str, seed_text:str):
        """Generate a response."""
        input_text, model_output, input_length, output_len = self.clo.generate(llm_input_messages, (author, seed_text), return_tuple=True)
        return input_text, model_output, input_length, output_len
        
    @wrap_async_executor
    def batch_generate(self, llm_input_messages: list[tuple], authors: list[str], seed_text: str):
        """Generate a batch of response."""        
        base_input_text, author_prompts, model_outputs, input_length, output_lengths = self.clo.batch_generate(llm_input_messages, authors, seed_text, return_tuple=True)
        return base_input_text, author_prompts, model_outputs, input_length, output_lengths

    
    @wrap_async_executor
    def streaming_generate(self, llm_input_messages: list[tuple], author: str, seed_text: str):
        generated_text = ""
        tick=0
        for new_word in self.clo.stream_generate(llm_input_messages, (author, seed_text)):
            if new_word:
                print(new_word, end='')
                
                generated_text+=new_word
                tick+=1
                if tick % 3 == 0:
                    yield generated_text
        print() # force a nl because of end=''
        yield generated_text
        #author_tag_prefix = f"[{author}] " + ((seed_text + ' ') if seed_text else '')
        #for updated_content in self._stream_helper(self.rai.stream_generate(llm_input_messages, (author, seed_text))):
        #    yield updated_content

        
    @wrap_async_executor
    def streaming_batch_generate(self, llm_input_messages: list[tuple], authors: list[str], seed_text: str):
        generated_text, last_i, tick = "", 0, 0
        for i,new_word in self.clo.stream_batch_generate(llm_input_messages, authors, seed_text):
            if new_word:
                if i != last_i:
                    print('\n'+('-'*50))#+'\n')
                    completed_text, completed_i = generated_text, last_i # store for yield
                    
                    generated_text, last_i, tick = "", i, 0 # reset
                    
                    yield completed_i,completed_text
                    
                print(new_word, end='')
                
                generated_text+=new_word
                tick+=1
                if tick % 3 == 0:
                    yield i,generated_text
        
        #for i,updated_content in self._stream_batch_helper(self.rai.stream_batch_generate(llm_input_messages, authors, seed_text)):
        #    yield i,updated_content


    async def log_and_bookkeep(self, input_length, input_text, output_length, discord_out):
        model_logger.info(f'[MODEL_INPUT({input_length})]\n' + input_text + '\n')
        model_logger.info(f'[DISCORD_OUTPUT({output_length})]\n' + discord_out + f'\n{"-"*200}\n')
        self.last_run = datetime.datetime.now()
        self.run_count+=1


    def prepare_llm_inputs(self, message_cache:list[discord.Message], seed_text:str|None):
        llm_input_messages = text_utils.llm_input_transform(message_cache, do_filter=False, user_aliases=self.user_aliases)
        seed_text = text_utils.process_seedtext(seed_text)
        
        return llm_input_messages, seed_text

    def to_discord_output(self, model_output, author, seed_text):
        model_output = text_utils.splitout_tag(model_output, self.RE_ANY_USERTAG)
        if self.clo.cfg.postfix and self.clo.cfg.postfix in model_output:
            print('WARNING: postfix detected in model_output, removing')
            model_output = model_output.replace(self.clo.cfg.postfix, '')
        # space at the end of a sentence encodes a special token (28705). Shouldn't pass space in seed text or results are sub optimal  
        text_out = (seed_text + ' ' + model_output) if seed_text else model_output
        text_out = self.ytm.decode(text_out)
        
        llm_output = text_utils.llm_output_transform(text_out, self.emojis)
        
        llm_output=llm_output.strip() # strip for models that use \n\n in chat template (llama3)
        
        discord_out = f'[{author}] {llm_output}'

        discord_out = text_utils.fix_mentions(discord_out, self.bot )

        return discord_out
    

    async def send_collect(self, ctx: commands.Context, discord_outs:list[str], char_limit: int=2000) -> list[discord.Message]:
        new_messages = []
        for discord_out in discord_outs:
            if len(discord_out) > char_limit:
                for sub_dout in text_utils.split_message(discord_out, char_limit):
                    msg = await ctx.send(sub_dout, tts=self.tts_mode)
                    new_messages.append(msg)
            else:
                msg = await ctx.send(discord_out, tts=self.tts_mode)
                new_messages.append(msg)
        
        return new_messages
    
    async def edit_collect(self, msg: discord.Message, discord_outs:list[str], char_limit: int=2000) -> list[discord.Message]:
        new_messages = []
        for i,discord_out in enumerate(discord_outs):
            if len(discord_out) > char_limit:
                for j,sub_dout in enumerate(text_utils.split_message(discord_out, char_limit)):
                    if j==0:
                        msg = await msg.edit(content=sub_dout)
                    else:
                        msg = await msg.reply(content=sub_dout)
                    new_messages.append(msg)
            else:
                if i==0:
                    msg = await msg.edit(content=discord_out)
                else:
                    msg = await msg.reply(content=discord_out)
                new_messages.append(msg)
        
        return new_messages
    
    def batch_log_format(self, base_input_text, author_prompts, discord_outs):
        log_input_text = base_input_text + '\n'.join([f'REPR: {ap!r}' for ap in author_prompts])
        log_discord_out = '\n'.join([f'REPR: {ap!r}' for ap in discord_outs])
        return log_input_text, log_discord_out

    async def pipeline(self, 
                       ctx: commands.Context|discord.Message,
                       message_cache:list[discord.Message], 
                       authors: list[str], 
                       seed_text: str, 
                       gen_type: typing.Literal['gen_one', 'gen_batch', 'stream_one','stream_batch']
                       ) -> list[discord.Message]:
        
        llm_input_messages, seed_text = self.prepare_llm_inputs(message_cache, seed_text)
        messages = []

        if isinstance(authors, str):
            authors = [authors]
        
        # redo_msg = None
        # if isinstance(ctx, discord.Message):
        #     redo_msg = ctx
        #     ctx = self.bot.get_context(ctx)
        edit_mode = isinstance(ctx, discord.Message)

        async def route_content(ctx, content):
            msg = await ctx.edit(content=content) if edit_mode else await ctx.send(content)
            return msg

        
        if gen_type == 'gen_one':
            author = authors[0]
            input_text, model_output, input_length, output_length  = await self.generate(llm_input_messages, author, seed_text=seed_text)
            discord_outs = [self.to_discord_output(model_output, author, seed_text)]
            log_input_text, log_discord_out, output_lengths = input_text, discord_outs[0], [output_length]
            
            
        elif gen_type == 'gen_batch':
            base_input_text, author_prompts, model_outputs, input_length, output_lengths = await self.batch_generate(llm_input_messages, authors, seed_text=seed_text)
            discord_outs = [self.to_discord_output(model_output, author, seed_text) for model_output,author in zip(model_outputs, authors)]

            log_input_text, log_discord_out = self.batch_log_format(base_input_text, author_prompts, discord_outs)
            
            
        elif gen_type == 'stream_one':
            author = authors[0]
            author_tag_prefix = f"[{author}] " + ((seed_text + ' ') if seed_text else '')
            # msg = await ctx.send(author_tag_prefix)
            msg = await route_content(ctx,author_tag_prefix)
            
            for updated_content in await self.streaming_generate(llm_input_messages, author, seed_text):                
                msg = await msg.edit(content=author_tag_prefix+updated_content)
                
            input_text, model_output, input_length, output_length = self.clo.last_streamed(batch=False ,return_tuple=True)
            discord_outs = [self.to_discord_output(model_output, author, seed_text)]
            
            msg = await msg.edit(content=discord_outs[0])
            messages.append(msg)
            
            log_input_text, log_discord_out, output_lengths = input_text, discord_outs[0], [output_length]
            
        
        elif gen_type == 'stream_batch':
            msg = None
            
            author_tag_prefixes = [f"[{author}] " + ((seed_text + ' ') if seed_text else '') for author in authors]
            messages = [await ctx.send(at_prefix) for at_prefix in author_tag_prefixes]

            for auth_idx, updated_content in await self.streaming_batch_generate(llm_input_messages, authors, seed_text):
                messages[auth_idx] = await messages[auth_idx].edit(content=author_tag_prefixes[auth_idx]+updated_content)


            base_input_text, author_prompts, model_outputs, input_length, output_lengths = self.clo.last_streamed(batch=True, return_tuple=True)
            discord_outs = [self.to_discord_output(model_output, author, seed_text) for model_output,author in zip(model_outputs, authors)]
            for msg, discord_out in zip(messages, discord_outs):
                msg = await msg.edit(content=discord_out)
                messages.append(msg)
            
            log_input_text, log_discord_out = self.batch_log_format(base_input_text, author_prompts, discord_outs)
            
        
        await self.log_and_bookkeep(input_length, log_input_text, output_lengths, log_discord_out)
        
        if not messages:
            if edit_mode:
                messages = await self.edit_collect(ctx, discord_outs, char_limit=2000) 
            else: 
                #await self.send_collect(ctx, discord_outs, char_limit=2000)
                messages = await self.send_collect(ctx, discord_outs, char_limit=2000)
        
        if self.model_randomization and random.random() < self.model_randomization['probability']:
            if self.model_randomization['announce']:
                await ctx.send('🧠🔀', delete_after=2, silent=True)
            await self.load_random_model(fast_proba=self.model_randomization['fast_proba'])

        return messages

