import os
import gc
import re
import io
import math
import time
import random

import typing
import asyncio
import datetime
import functools
from pathlib import Path

import discord
from discord import app_commands
from discord.app_commands.translator import locale_str
from discord.ext import commands, tasks

from diffusers.utils import load_image, make_image_grid

from PIL import Image

from managers import imgman
import config.settings as settings

from cmds import transformers as cmd_tfms, choices as cmd_choices
from utils.command import check_up
from utils.globthread import stop_global_thread
from views import redrawui
from run import BotUs

bot_logger = settings.logging.getLogger('bot')
model_logger = settings.logging.getLogger('model')
cmds_logger = settings.logging.getLogger('cmds')
event_logger = settings.logging.getLogger('event')

# Resource: https://github.com/CyberTimon/Stable-Diffusion-Discord-Bot/blob/main/bot.py

IMG_DIR = settings.SERVER_ROOT/'output'/'imgs'
PROMPT_FILE = IMG_DIR.joinpath('_prompts.txt')


def prompt_to_filename(prompt, ext='png'):
    tstamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    fname = tstamp+'_'+(re.sub('[^\w -]+','', prompt).replace(' ','_')[:100])+f'.{ext}'
    return fname

def save_image_prompt(image: Image.Image, prompt:str):
    fname = prompt_to_filename(prompt, 'png')
    out_imgpath = IMG_DIR/fname

    image.save(out_imgpath)
    
    with PROMPT_FILE.open('a') as f:
        f.write(f'{fname} : {prompt!r}\n')

    return out_imgpath

async def send_imagebytes(ctx:commands.Context, image:Image.Image, prompt:str):
    with io.BytesIO() as imgbin:
        image.save(imgbin, 'PNG')
        imgbin.seek(0)
        msg = await ctx.send(file=discord.File(fp=imgbin, filename='image.png',  description=prompt)) 
    return msg


def extract_image_url(message: discord.Message, verbose=False):
    """read image url from message"""
    
    if message.embeds:
        if verbose: print('embeds_url:', message.embeds[0].url)
        return message.embeds[0].url
    elif message.attachments:
        #img_filename = message.attachments[0].filename
        if verbose: print('attach_url:',message.attachments[0].url)
        return message.attachments[0].url

    return None
        

async def read_attach(ctx: commands.Context):
    try:
        attach = ctx.message.attachments[0]
        print(attach.url)
        print(f'Image dims (WxH): ({attach.width}, {attach.height})')
        
        image = Image.open(io.BytesIO(await attach.read())).convert('RGB')
        return image
    except IndexError as e:
        await ctx.send('No image attachment given!')
        return




class ImageGen(commands.Cog): #commands.GroupCog, group_name='img'
    '''Suite of tools for generating images.'''
    
    def __init__(self, bot: BotUs):
        self.bot = bot
        self.igen = imgman.JuggernautXLLightningManager()
        self.ctx_menu = app_commands.ContextMenu(name='🎨 Redraw (Image)', callback=self._cm_redraw,)
        self.bot.tree.add_command(self.ctx_menu)
        
        
    async def cog_unload(self):
        await self.bot.wait_until_ready()
        await self.igen.unload_pipeline()
        self.bot.tree.remove_command(self.ctx_menu.name, type=self.ctx_menu.type)
        stop_global_thread()
    
    async def cog_after_invoke(self, ctx: commands.Context) -> None:
        cmd_status = 'FAIL' if ctx.command_failed else 'PASS'
        pos_args = [a for a in ctx.args[2:] if a]
        
        cmds_logger.info(f'(cog_after_invoke, {self.qualified_name})'
                         '- [{stat}] {a.display_name}({a.name}) command "{c.prefix}{c.invoked_with} ({c.command.name})" args:({args}, {c.kwargs})'.format(stat=cmd_status, a=ctx.author, c=ctx, args=pos_args))
    
    
    async def _cm_redraw(self, interaction: discord.Interaction, message: discord.Message):
        # https://github.com/Rapptz/discord.py/issues/7823#issuecomment-1086830458
        ctx = await self.bot.get_context(interaction)
        await self._reaction_redraw(ctx, message, ephemeral=True)
    
    @check_up('igen', '❗ Drawing model not loaded. Call `!imgup`')
    async def _reaction_redraw(self, ctx:commands.Context, message: discord.Message, ephemeral=False):
        
        img_url = extract_image_url(message, True)
        if img_url is None:
            return await message.reply('No see image 🙈', mention_author=False)
        
        #print(f'IMG URL: {img_url}\nIMG FILE: {img_filename}')
        #print(f'IMG URL: {img_url}')
        await message.remove_reaction('🖼️',self.bot.user)

        mv = redrawui.RedrawUIView(self.igen.model_name)
        msg = await mv.send(ctx=ctx, image_url=img_url, image_filename=None, ephemeral=ephemeral)
        await mv.wait()
        print(mv.data)
        #message.to_reference().reply()
        #ctx.send()
        #ctx = await self.bot.get_context(message)#msg)
        #self.bot.get_command
        ctx = await self.bot.get_context(message)#msg)
        await self.redraw(ctx, image_url=img_url, prompt=mv.data['prompt'], 
                          steps=mv.data['steps'],
                          strength=mv.data['strength'],
                          neg_prompt=mv.data['neg_prompt'],
                          guidance=mv.data['guidance'],
                          stage_mix=mv.data['stage_mix'],
                          refine_strength=mv.data['refine_strength'],
                          )
        
        
        #await reaction.remove(self.bot.user)
        await mv.defered_interaction.followup.send(mv.data['prompt'])
        await msg.delete()

    @commands.Cog.listener('on_raw_reaction_add')
    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        if discord.utils.get(self.bot.cached_messages, id=payload.message_id):
            return
        
        if not payload.member.bot:
            if str(payload.emoji) == '🖼️': # 🖌
                channel = self.bot.get_channel(payload.channel_id)
                msg = await discord.utils.get(channel.history(limit=100), id=payload.message_id)
                ctx = await self.bot.get_context(msg)
                #ctx = await self.bot.get_context(reaction.message)
                await self._reaction_redraw(ctx, message=msg)

    @commands.Cog.listener('on_reaction_add')
    async def on_reaction_add(self, reaction: discord.Reaction, user: discord.User):
        if not user.bot:
            if str(reaction.emoji) == '🖼️': # 🖌
                ctx = await self.bot.get_context(reaction.message)
                await self._reaction_redraw(ctx, message=reaction.message)

    async def _temporary_reaction(self, message: discord.Message, emoji:str, delete_after=60.0):
        await message.add_reaction(emoji)
        def check(reaction, user):
            return not user.bot and str(reaction.emoji) == emoji 
        try:
            reaction, user = await self.bot.wait_for('reaction_add', timeout=delete_after, check=check)
        except asyncio.TimeoutError:
            print('No response in time. Removing')
            await message.remove_reaction('🖼️', self.bot.user)
    
    @commands.Cog.listener('on_message')
    async def on_message(self, message: discord.Message):
        if not message.author.bot:
            if self.igen.is_ready:
                try:
                    if (img_url:=extract_image_url(message)) is not None:
                        img=load_image(img_url).load()
                    #await message.add_reaction('🖼️')
                    #await self._temporary_reaction(message, emoji='🖼️', delete_after=60.0)

                except Exception as e:
                    print(e)
                        
    @commands.command(name='iseed', aliases=['imgseed', 'imageseed'])
    async def iseed(self, ctx: commands.Context, seed:int = None):
        self.igen.set_seed(seed=seed)
        return await ctx.send(f'Global image seed set to {seed}. Welcome to the land of {"non-" if seed is None else ""}determinisim')
    
    @commands.command(name='iinfo', aliases=['imginfo', 'imageinfo'])
    async def iinfo(self, ctx: commands.Context, seed:int = None):
        model_alias = imgman.AVAILABLE_MODELS[self.igen.model_name]['desc']
        msg = model_alias + '\n' + f'Image Seed: {self.igen.global_seed}'
        msg += self.igen.config.to_md()
        #msg += f'\nImage seed: {self.igen.global_seed}'
        return await ctx.send(msg)
    
    @commands.command(name='imgup', aliases=['iup','imageup'])
    async def imgup(self, ctx: commands.Context):
        '''Loads in the image generation model'''
        msg = None
        if not self.igen.is_ready:
            msg = await ctx.send('Warming up drawing skills...', silent=True)
            await self.igen.load_pipeline()
        
        await self.bot.change_presence(**settings.BOT_PRESENCE['draw'])

        
        model_alias = imgman.AVAILABLE_MODELS[self.igen.model_name]['desc']
        complete_msg = f'{model_alias} all fired up'
        complete_msg += self.igen.config.to_md()

        return await ctx.send(complete_msg) if msg is None else await msg.edit(content = complete_msg)
        
    
    @commands.command(name='imgdown', aliases=['idown','imagedown'])
    async def imgdown(self, ctx: commands.Context):
        '''Unloads the image generation model'''
        await self.igen.unload_pipeline()
        await self.bot.change_presence(**settings.BOT_PRESENCE['ready'])
        await ctx.send('Drawing disabled.')
            
    @commands.hybrid_command(name='artist')
    @app_commands.choices(model=cmd_choices.IMAGE_MODELS)
    async def artist(self, ctx: commands.Context, model: app_commands.Choice[str]):
        '''Unloads the image generation model'''
        if self.igen.model_name != model.value:
            await self.igen.unload_pipeline()
            
            self.igen = imgman.AVAILABLE_MODELS[model.value]['manager']
            return await self.imgup(ctx)
        elif not self.igen.is_ready:
            return await self.imgup(ctx)
        else:
            await ctx.send(f'{model.name} already up')
        
    
    @commands.command(name='optimize', aliases=['compile'])
    @check_up('igen', '❗ Drawing model not loaded. Call `!imgup`')
    async def optimize(self, ctx: commands.Context):
        '''Compiles the model to make it go brrrr (~20% faster)'''

        if self.igen.is_compiled:
            return await ctx.send('Already going brrrrr')
        
        msg = await ctx.send("Sit tight. This'll take 2-4 minutes...")

        await ctx.defer()
        await asyncio.sleep(1)
        async with (ctx.channel.typing(), self.bot.writing_status('Compiling...')):
            await self.igen.compile_pipeline()
            await msg.edit(content='🏎 I am SPEED. 🏎 ')


        
    # @commands.hybrid_command(name='qdraw')
    # @check_up2('igen', '❗ Drawing model not loaded. Call `!imgup`')
    # async def qdraw(self, ctx: commands.Context, prompt:str, *,
    #                steps: int = None, 
    #                neg_prompt: str = None, 
    #                guidance: float = 7.0, 
    #                stage_mix: app_commands.Transform[float, cmd_tfms.PercentTransformer] = None, 
    #                refine_strength: app_commands.Transform[float, cmd_tfms.PercentTransformer] = 30.0):
    #     """
    #     Same as `draw` but trades quality for speed (2-3x faster). Generate an image from a text prompt description.
    #     """

    #     self.igen.dc_fastmode(enable=True, img2img=False)
    #     await self.draw(ctx, prompt, steps=steps, neg_prompt=neg_prompt, guidance=guidance, stage_mix=stage_mix, refine_strength=refine_strength)
    #     self.igen.dc_fastmode(enable=False, img2img=False)


    @commands.hybrid_command(name='draw')
    @check_up('igen', '❗ Drawing model not loaded. Call `!imgup`')
    async def draw(self, ctx: commands.Context, prompt:str, *,
                   steps: int = None, 
                   neg_prompt: str = None, 
                   guidance: float = None, 
                   orient: typing.Literal['square', 'portrait'] = None,
                   refine_strength: app_commands.Transform[float, cmd_tfms.PercentTransformer] = 30.0, 
                   stage_mix: app_commands.Transform[float, cmd_tfms.PercentTransformer] = None, 
                   fast:bool=False):
        """
        Generate an image from a text prompt description.

        Args:
            prompt: A description of the image to be generated.
            steps: Num iters to run. Increase = ⬆Quality, ⬆Run Time. Default=40 (Turbo: Default=2).
            
            orient: Image shape (aspect ratio). square w=h = 1:1. portrait w<h = 13:19. Default='square' (Turbo ignores). 
            neg_prompt: Description of what you DON'T want. Usually comma sep list of words. Default=None (Turbo ignores).
            guidance: Guidance scale. Increase = ⬆Prompt Adherence, ⬇Quality, ⬇Creativity. Default=10.0 (Turbo ignores).
            refine_strength: Refinement stage intensity. 0=Alter Nothing. 100=Alter Everything. Default=30 (Turbo ignores).
            stage_mix: Percent of `steps` for Base before Refine stage. ⬇Quality, ⬇Run Time. Default=None (Turbo ignores).
            fast: Trades image quality for speed - about 2-3x faster. Default=False (Turbo ignores).
        """
        
        await ctx.defer()
        await asyncio.sleep(1)
        async with self.bot.writing_status(presense_done='draw'):
            self.igen.dc_fastmode(enable=fast, img2img=False)
            image = await self.igen.generate_image(prompt, steps, negative_prompt=neg_prompt, guidance_scale=guidance, denoise_blend=stage_mix, refine_strength=refine_strength, orient=orient)
            await send_imagebytes(ctx, image, prompt)
            #self.igen.dc_fastmode(enable=False, img2img=False)
        
        out_imgpath = save_image_prompt(image, prompt)
        #await ctx.send(file=discord.File(out_imgpath))

    @commands.hybrid_command(name='redraw')
    @check_up('igen', '❗ Drawing model not loaded. Call `!imgup`')
    async def redraw(self, ctx: commands.Context, image_url: str, prompt: str, *, 
                     steps: int = None, 
                     strength: app_commands.Transform[float, cmd_tfms.PercentTransformer] = 55.0, 
                     neg_prompt: str = None, 
                     guidance: float = None, 
                     stage_mix: app_commands.Transform[float, cmd_tfms.PercentTransformer] = None, 
                     refine_strength: app_commands.Transform[float, cmd_tfms.PercentTransformer] = 30.0, 
                     fast:bool=False
                     ):
        """
        Remix an image from a text prompt and image url.

        Args:
            image_url: image URL. Square = Best results. Ideal size= 1024x1024 (Turbo ideal= 512x512).
            prompt: A description of the image to be generated.
            steps: Num of iters to run. Increase = ⬆Quality, ⬆Run Time. Default=50 (Turbo: Default=4).
            strength: How much to change input image. 0 = Change Nothing. 100=Change Completely. Default=55.

            neg_prompt: Description of what you DON'T want. Usually comma sep list of words. Default=None (Turbo ignores).
            guidance: Guidance scale. Increase = ⬆Prompt Adherence, ⬇Quality, ⬇Creativity. Default=10.0 (Turbo ignores).
            stage_mix: Percent of `steps` for Base before Refine stage. ⬇Quality, ⬇Run Time. Default=None (Turbo ignores).
            refine_strength: Refinement stage intensity. 0=Alter Nothing. 100=Alter Everything. Default=30 (Turbo ignores).
            fast: Trades image quality for speed - about 2-3x faster. Default=False (Turbo ignores).
        """
        image = load_image(image_url).convert('RGB')
                
        await ctx.defer()
        await asyncio.sleep(1)
        
        async with self.bot.writing_status(presense_done='draw'):
            self.igen.dc_fastmode(enable=fast, img2img=False)
            image = await self.igen.regenerate_image(image=image, prompt=prompt, 
                                                     steps=steps, strength=strength, negative_prompt=neg_prompt, guidance_scale=guidance, denoise_blend=stage_mix, refine_strength=refine_strength)
            await send_imagebytes(ctx, image, prompt)
            #self.igen.dc_fastmode(enable=False, img2img=False)
        out_imgpath = save_image_prompt(image, prompt)

    @commands.command(name='disco')
    async def disco(self, ctx: commands.Context, duration: int = 32):
        """💯 PARTY HARD 💯. but also responsibly within the message rate limit of discord servers"""
        # https://discordpy.readthedocs.io/en/stable/api.html#discord.Client.wait_for
        # use wait for to stop

        
        msg = await send_imagebytes(ctx, Image.new("RGB", (350, 350), (0, 0, 0)), None)
        for i in range(duration):
            with io.BytesIO() as imgbin:
                Image.new("RGB", (350, 350), tuple(random.choices(range(256), k=3))).save(imgbin, 'JPEG')
                imgbin.seek(0)
                await msg.edit(attachments=[discord.File(fp=imgbin, filename=f'image.jpg')])
                # if i==0:
                #     msg = await ctx.send(file=discord.File(fp=imgbin, filename=f'image.jpg'))
                # else:
                #     await msg.edit(attachments=[discord.File(fp=imgbin, filename=f'image.jpg')])
            await asyncio.sleep(0.42)
        
        
            

async def setup(bot: BotUs):
    igen = ImageGen(bot)

    await bot.add_cog(igen)




