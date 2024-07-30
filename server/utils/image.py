import re
import io

import random
import typing
import asyncio
import datetime
from pathlib import Path
from urllib.error import HTTPError


import discord
from discord import app_commands
from discord.ext import commands

from diffusers.utils import load_image, make_image_grid

import requests
from PIL import Image, UnidentifiedImageError
import imageio.v3 as iio # pip install -U "imageio[ffmpeg, pyav]" # ffmpeg: mp4, pyav: trans_png
import pygifsicle # sudo apt install gifsicle


import config.settings as settings

IMG_DIR = settings.SERVER_ROOT/'output'/'imgs'
PROMPT_FILE = IMG_DIR.joinpath('_prompts.txt')

USER_AGENTS = [ # https://github.com/microlinkhq/top-user-agents/blob/master/src/desktop.json
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) obsidian/1.4.14 Chrome/114.0.5735.289 Electron/25.8.1 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 OPR/112.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 11.6; rv:92.0) Gecko/20100101 Firefox/92.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) obsidian/1.6.3 Chrome/120.0.6099.291 Electron/28.3.3 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
]

def batched(iterable, n:int):
    '''https://docs.python.org/3/library/itertools.html#itertools.batched'''
    # batched('ABCDEFG', 3) → ABC DEF G
    import itertools
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := list(itertools.islice(iterator, n)):
        yield batch


def prompt_to_filename(prompt, ext='png'):
    tstamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    fname = tstamp+'_'+(re.sub('[^\w -]+','', prompt).replace(' ','_')[:100])+f'.{ext}'
    return fname

def save_gif_prompt(frames:list[Image.Image], prompt:str, optimize:bool=True):
    fname = prompt_to_filename(prompt, 'gif')
    out_imgpath = IMG_DIR/fname
    iio.imwrite(out_imgpath, frames, extension='.gif', loop=0)
    if optimize:
        pygifsicle.optimize(out_imgpath)#, 'tmp-o.gif')
    with PROMPT_FILE.open('a') as f:
        f.write(f'{fname} : {prompt!r}\n')
    return out_imgpath

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

def to_bytes_file(image:Image.Image, prompt: str):
    filename = f'image_{hash(image.tobytes())}.png' 
    with io.BytesIO() as imgbin:
        image.save(imgbin, 'PNG')
        imgbin.seek(0)
        
        return discord.File(fp=imgbin, filename=filename, description=prompt)

def to_bfile(image:Image.Image, filename: str=None, description:str=None, ):
    #if filename is None:
    #    filename = f'image_{hash(image.tobytes())}.png'
    if filename is None:
        filename = tempfile.NamedTemporaryFile(suffix=".WebP").name
    with io.BytesIO() as imgbin:
        image.save(imgbin, 'WebP')#'JPEG')
        imgbin.seek(0)
        
        return discord.File(fp=imgbin, filename=filename, description=description)


async def try_send_gif(msg:discord.Message, out_imgpath: Path, prompt: str, view: discord.ui.View = None):
    try:
        msg = await msg.edit(content='', attachments=[discord.File(fp=out_imgpath, filename=out_imgpath.name,  description=prompt)], view=view)
    except discord.errors.HTTPException as e:
        if e.status == 413:
            msg  = await msg.edit(content=f"Uh oh, plate is overflowing. Trying to scrape some off...")
            pygifsicle.optimize(out_imgpath)
            msg = await msg.edit(content='', attachments=[discord.File(fp=out_imgpath, filename=out_imgpath.name,  description=prompt)], view=view)
        else:
            raise e
    return msg

def tenor_fix(url: str):
    if 'https/media.tenor.com' in url: # https://images-ext-1.discordapp.net/external/sW67YUaWQx_lnwJE5_TP2p3GMBAXbehBhrxzrSFn4tA/https/media.tenor.com/aUz-N2QvBOsAAAPo/the-isle-evrima.mp4
        outlink = 'https://' + url.split('https/')[-1]
        return outlink
    elif url.startswith('https://tenor.com/view/'):# in url: # https://tenor.com/view/the-isle-evrima-kaperoo-quality-assurance-hypno-gif-25376214
        #channel.history(limit=100, oldest_first=False)
        #discord.utils.find()
        #for emb in message.embeds:
            #pprint.pprint(emb.to_dict()['video']['url'])
        raise ValueError('Incorrect Tenor URL format: Long form')
    elif url.startswith('https://tenor.com/') and url.endswith('.gif'): # https://tenor.com/bSDFW.gif'
        raise ValueError('Incorrect Tenor URL format: Short form')
    return url


def clean_discord_urls(url:str, verbose=False):
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

def is_animated(image_url:str):
    try:
        image_url = tenor_fix(image_url)
        img_props = iio.improps(image_url)
            # transparent png is batch but n_images = 0
        return img_props.is_batch and img_props.n_images > 1 
    except HTTPError as e: # urllib.error.HTTPError: HTTP Error 403: Forbidden
        print(e)
        return False
    # transparent png is batch but n_images = 0
    #return img_props.is_batch and img_props.n_images > 1 

def load_images(image_uri:str, result_type:typing.Literal['PIL','np']|None = None):
    '''Fetch an image, UA spoof if necessary

    default return type if None:
        animated image: numpy array
        non-animated image: PIL.Image
    '''
    try:
        image = iio.imread(image_uri)
    except HTTPError as e:
        print(e)
        rsp = requests.get(image_uri, stream=True, headers={"User-Agent": random.choice(USER_AGENTS)})
        rsp.raise_for_status()
        image = iio.imread(rsp.raw.read())
    
    if image.ndim < 4 or image.shape[0] < 2:
        # hwc or alpha-hwc
        if result_type != 'np': # None or PIL
            image = Image.fromarray(image, mode="RGB")
    elif result_type == 'PIL':
        # gif
        image = [Image.fromarray(frame, mode="RGB") for frame in image]

    return image