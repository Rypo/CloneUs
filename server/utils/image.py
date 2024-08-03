import re
import io
import random
import typing
import asyncio
import datetime
import tempfile
import itertools
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
THUMB_DIR = IMG_DIR.parent/'thumbnails'
THUMB_DIR.mkdir(exist_ok=True)

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
    
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := list(itertools.islice(iterator, n)):
        yield batch


def prompt_to_filename(prompt:str, suffix:str='png'):
    tstamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    fname = tstamp+'_'+(re.sub('[^\w -]+','', prompt).replace(' ','_')[:100])+f'.{suffix}'
    return fname

def save_gif_prompt(frames:list[Image.Image], prompt:str, optimize:bool=False):
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

def to_thumbnails(images:list[Image.Image], max_size:tuple[int,int]=(256,256), out_filestems: list[str]=None) -> list[Image.Image]|list[Path]:
    out_imgs = [image.copy() for image in images]
    out_paths = []
    
    
    for i,img in enumerate(out_imgs):
        img.thumbnail(max_size, Image.Resampling.BOX) # [img.reduce(4)]
        if out_filestems:
            out_path = THUMB_DIR/f'{out_filestems[i]}_{i}.jpg'
            img.save(out_path, "JPEG", optimize=True)
            out_paths.append(out_path)
    
    if out_filestems:
        return out_paths
    
    return out_imgs


def to_bytes_file(image:Image.Image, prompt: str, ext:typing.Literal['PNG','WebP','JPEG']='PNG', **kwargs):
    sfx = 'jpg' if ext == 'JPEG' else ext.lower()
        
    #filename = f'image_{hash(image.tobytes())}.{sfx}' 
    filename = prompt_to_filename(prompt, suffix=sfx)
    with io.BytesIO() as imgbin:
        image.save(imgbin, format=ext, **kwargs)
        imgbin.seek(0)
        
        return discord.File(fp=imgbin, filename=filename, description=prompt)

def to_bfile(image:Image.Image, filestem: str=None, description:str=None, ext: typing.Literal['PNG','WebP','JPEG'] = 'WebP', **kwargs):
    sfx = f'.{ext.lower()}'
    
    filename = filestem+sfx if filestem is not None else tempfile.NamedTemporaryFile(suffix=sfx).name
        
    with io.BytesIO() as imgbin:
        image.save(imgbin, format=ext, **kwargs)
        imgbin.seek(0)
        
        return discord.File(fp=imgbin, filename=filename, description=description)


def impath_to_file(img_path:Path|str, description:str=None):
    img_path = Path(img_path)
    return discord.File(fp=img_path, filename=img_path.name, description=description)

def to_discord_file(image:Image.Image|str|Path, filestem:str=None, description:str=None, ext:typing.Literal['PNG','WebP','JPEG']=None, **kwargs):
    if isinstance(image, Image.Image):
        assert filestem or ext, 'must specify at least one of `filestem` or `ext` when using Image objects'
        return to_bfile(image=image, filestem=filestem, description=description, ext=ext, **kwargs)
    
    return impath_to_file(img_path=image, description=description)


async def try_send_gif(msg:discord.Message, gif_filepath: Path, prompt: str, view: discord.ui.View = None):
    size_limit = msg.guild.filesize_limit if msg.guild is not None else discord.utils.DEFAULT_FILE_SIZE_LIMIT_BYTES # 25*(1024**2)#2.5e7 # 25MB for non nitro
    gif_bytes = Path(gif_filepath).stat().st_size
    print(f'GIF SIZE: {gif_bytes/1e6:0.3f} MB')
    
    if gif_bytes <= size_limit:
        msg = await msg.edit(content='', attachments=[discord.File(fp=gif_filepath, filename=gif_filepath.name,  description=prompt)], view=view)
    else:
        msg = await msg.edit(content=f"Hwoo boy, that's a lotta pixels. Attempting compression...")
        pygifsicle.optimize(gif_filepath)
        try:
            msg = await msg.edit(content='', attachments=[discord.File(fp=gif_filepath, filename=gif_filepath.name,  description=prompt)], view=view)
        except discord.errors.HTTPException as e:
            if e.status == 413:
                # msg  = await msg.edit(content=f"Uh oh, plate is overflowing. Trying to scrape some off...")
                msg  = await msg.edit(content=f"Still too big, maybe tone it down a bit next time.")
            else:
                raise e
    return msg

def tenor_fix(url: str):
    if not isinstance(url, str) or 'tenor' not in url:
        return url
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
    # GIF: https://media.discordapp.net/attachments/.../XYZ.gif?ex=...&is=...&=&width=837&height=837
    # JPG: https://media.discordapp.net/attachments/.../.../XYZ.jpg?ex=...&is=...&hm=...&=&format=webp&width=396&height=836
    clean_url = url.split('&=&')[0] # gifs don't have a "format=", but both gifs and images have "&=&"
    clean_url = clean_url.split('format=')[0].rstrip('&=?')
    
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

def img_bytestream(image_url:str, random_ua:bool=True):
    headers = {"User-Agent": random.choice(USER_AGENTS)} if random_ua else None
    rsp = requests.get(image_url, stream=True, headers=headers)
    rsp.raise_for_status()
    return rsp.raw


def is_animated(image_url:str):
    try:
        image_url = tenor_fix(image_url)
        img_props = iio.improps(img_bytestream(image_url).read())
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
        image = iio.imread(img_bytestream(image_uri).read())
    
    if image.ndim < 4 or image.shape[0] < 2:
        # hwc or alpha-hwc
        if result_type != 'np': # None or PIL
            image = Image.fromarray(image, mode="RGB")
    elif result_type == 'PIL':
        # gif
        image = [Image.fromarray(frame, mode="RGB") for frame in image]

    return image