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

from cmds import transformers as cmd_tfms, choices as cmd_choices, flags as cmd_flags
from utils.command import check_up
from utils.globthread import stop_global_thread
from views import imageui
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
        #view = redrawui.DrawUIView(ctx)
        #msg = await view.send(ctx, discord.File(fp=imgbin, filename='image.png',  description=prompt))
        msg = await ctx.send(file=discord.File(fp=imgbin, filename='image.png',  description=prompt))#, view=redrawui.DrawUIView()) 
    return msg

def imgbytes_file(image:Image.Image, prompt:str):
    with io.BytesIO() as imgbin:
        image.save(imgbin, 'PNG')
        imgbin.seek(0)
        
        return discord.File(fp=imgbin, filename='image.png',  description=prompt)
        

def clean_discord_urls(url, verbose=False):
    if not isinstance(url, str) or 'discordapp' not in url:
        return url
    
    clean_url = url.split('format=')[0].rstrip('&=?')
    if verbose:
        print(f'old discord url: {url}\nnew discord url: {clean_url}')
    return clean_url

def extract_image_url(message: discord.Message, verbose=False):
    """read image url from message"""
    url = None
    if message.embeds:
        url = message.embeds[0].url
        if verbose: print('embeds_url:', url)
        
    elif message.attachments:
        url = message.attachments[0].url
        #img_filename = message.attachments[0].filename
        if verbose: print('attach_url:', url)
        

    return clean_discord_urls(url)
        

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
        self.igen = imgman.JuggernautXLLightningManager(offload=True)

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
    
    @commands.command(name='iseed', aliases=['imgseed', 'imageseed'])
    async def iseed(self, ctx: commands.Context, seed:int = None):
        '''Set image model seed for deterministic output'''
        self.igen.set_seed(seed=seed)
        return await ctx.send(f'Global image seed set to {seed}. Welcome to the land of {"non-" if seed is None else ""}determinisim')
    
    @commands.command(name='iinfo', aliases=['imginfo', 'imageinfo'])
    async def iinfo(self, ctx: commands.Context):
        '''Display image model info'''
        model_alias = imgman.AVAILABLE_MODELS[self.igen.model_name]['desc']
        msg = model_alias + '\n' + f'Image Seed: {self.igen.global_seed}'
        msg += self.igen.config.to_md()
        
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
    async def artist(self, ctx: commands.Context, model: app_commands.Choice[str], offload: bool=True):
        '''Loads in the image generation model
        
        Args:
            model: Image model name
            offload: If True, model will be moved off GPU when not in use to save vRAM 
        '''
        if self.igen.model_name != model.value:
            await self.igen.unload_pipeline()
            
            self.igen = imgman.AVAILABLE_MODELS[model.value]['manager'](offload=offload)
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


    @commands.hybrid_command(name='draw')
    @check_up('igen', '❗ Drawing model not loaded. Call `!imgup`')
    async def _draw(self, ctx: commands.Context, prompt:str,  flags: cmd_flags.DrawFlags):
        """
        Generate an image from a text prompt description.

        Args:
            prompt: A description of the image to be generated.
            steps: Num iters to run. Increase = ⬆Quality, ⬆Run Time. Default varies.

            no: Negative prompt. What to exclude from image. Usually comma sep list of words. Default=None.
            aspect: Image aspect ratio (shape). square w=h = 1:1. portrait w<h = 13:19. Default='square'. 
            
            guide: Guidance scale. Increase = ⬆Prompt Adherence, ⬇Quality, ⬇Creativity. Default varies.
            hdsteps: High Definition steps. If > 0, image is upscaled 1.5x and refined. Default=0. Usually < 3.
            hdstrength: HD steps strength. 0=Alter Nothing. 100=Alter Everything. Ignored if hdsteps=0.
            dblend: Percent of `steps` for Base before Refine stage. ⬇Quality, ⬇Run Time. Default=None (SDXL Only).
            fast: Trades image quality for speed - about 2-3x faster. Default=False (Turbo ignores).
        """
        # flags can't be created by drawUI, so need to separate out draw/redraw functionality
        return await self.draw(ctx, prompt, 
                               steps = flags.steps, 
                               negative_prompt = flags.no, 
                               guidance_scale = flags.guide, 
                               aspect = flags.aspect, 
                               refine_steps = flags.hdsteps, 
                               refine_strength = flags.hdstrength, 
                               denoise_blend = flags.dblend, 
                               fast = flags.fast,
                               )
        
    async def draw(self, ctx: commands.Context, prompt:str, *,
                   steps: int = None, 
                   negative_prompt: str = None, 
                   guidance_scale: float = None, 
                   aspect: typing.Literal['square', 'portrait', 'landscape'] = None,
                   refine_steps: int = 0,
                   refine_strength: float = None, 
                   denoise_blend: float|None = None, 
                   fast:bool=False):
        
        has_view = ctx.interaction.response.is_done()
        if not has_view:
            await ctx.defer()
            await asyncio.sleep(1)
        
        async with self.bot.writing_status(presense_done='draw'):
            self.igen.dc_fastmode(enable=fast, img2img=False)
            image, fwkg = await self.igen.generate_image(prompt, steps, negative_prompt=negative_prompt, guidance_scale=guidance_scale, aspect=aspect, 
                                                         refine_steps=refine_steps, refine_strength=refine_strength, denoise_blend=denoise_blend,)
            #await send_imagebytes(ctx, image, prompt)
            #image_file = imgbytes_file(image, prompt)
            out_imgpath = save_image_prompt(image, prompt)
            if not has_view:
                view = imageui.DrawUIView(fwkg, timeout=5*60)
                msg = await view.send(ctx, image, out_imgpath)
                
        return image, out_imgpath

    @commands.hybrid_command(name='redraw')
    @check_up('igen', '❗ Drawing model not loaded. Call `!imgup`')
    async def _redraw(self, ctx: commands.Context, imgfile: discord.Attachment, prompt: str, flags: cmd_flags.RedrawFlags):
        """
        Remix an image from a text prompt and image.

        Args:
            imgfile: image attachment. Square = Best results. Ideal size= 1024x1024 (Turbo ideal= 512x512).
            prompt: A description of the image to be generated.
            steps: Num of iters to run. Increase = ⬆Quality, ⬆Run Time. Default=50 (Turbo: Default=4).
            strength: How much to change input image. 0 = Change Nothing. 100=Change Completely. Default=55.

            no: What to exclude from image. Usually comma sep list of words. Default=None.
            aspect: Image aspect ratio (shape). If None, will pick nearest to imgfile.

            guide: Guidance scale. Increase = ⬆Prompt Adherence, ⬇Quality, ⬇Creativity. Default varies.
            hdsteps: High Definition steps. If > 0, image is upscaled 1.5x and refined. Default=0. Usually < 3.
            hdstrength: HD steps strength. 0=Alter Nothing. 100=Alter Everything. Ignored if hdsteps=0.
            dblend: Percent of `steps` for Base before Refine stage. ⬇Quality, ⬇Run Time. Default=None (SDXL Only).
            fast: Trades image quality for speed - about 2-3x faster. Default=False (Turbo ignores).
        """
        
        return await self.redraw(ctx, imgfile, prompt, 
                                 steps = flags.steps, 
                                 strength = flags.strength, 
                                 negative_prompt = flags.no, 
                                 guidance_scale = flags.guide, 
                                 aspect = flags.aspect, 
                                 refine_steps = flags.hdsteps, 
                                 refine_strength = flags.hdstrength,
                                 denoise_blend = flags.dblend, 
                                 fast = flags.fast
                                 )

    async def redraw(self, ctx: commands.Context, imgfile: discord.Attachment, prompt: str, *, 
                     steps: int = None, 
                     strength: float = None, 
                     negative_prompt: str = None, 
                     guidance_scale: float = None, 
                     aspect: typing.Literal['square', 'portrait', 'landscape'] = None,
                     refine_steps: int = 0,
                     refine_strength: float = None,
                     denoise_blend: float = None, 
                     fast:bool=False):
        
        # this may be passed a url string in drawUI config
        image_url = clean_discord_urls(imgfile.url if isinstance(imgfile,discord.Attachment) else imgfile)
        image = load_image(image_url).convert('RGB')
        
        needs_view = False
        if not ctx.interaction.response.is_done():
            await ctx.defer()
            await asyncio.sleep(1)
            needs_view = True
        
        async with self.bot.writing_status(presense_done='draw'):
            self.igen.dc_fastmode(enable=fast, img2img=False)
            image, fwkg = await self.igen.regenerate_image(image=image, prompt=prompt, steps=steps, 
                                                           strength=strength, negative_prompt=negative_prompt, 
                                                           guidance_scale=guidance_scale, aspect=aspect, refine_steps=refine_steps,
                                                           refine_strength=refine_strength, denoise_blend=denoise_blend,)
            
            #image_file = imgbytes_file(image, prompt)
            out_imgpath = save_image_prompt(image, prompt)
            if needs_view:
                view = imageui.DrawUIView(fwkg, timeout=5*60)
                msg = await view.send(ctx, image, out_imgpath)
            
        #out_imgpath = save_image_prompt(image, prompt)
        return image, out_imgpath


        

async def setup(bot: BotUs):
    igen = ImageGen(bot)

    await bot.add_cog(igen)




