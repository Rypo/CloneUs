import re
import io
import typing
import asyncio
import logging
import datetime
import tempfile
from pathlib import Path
from urllib.error import HTTPError

import discord
from discord.ext import commands

import numpy as np
from PIL import Image
import imageio.v3 as iio # pip install -U "imageio[ffmpeg, pyav]" # ffmpeg: mp4, pyav: trans_png # imageio.plugins.freeimage.download()
from diffusers.utils import load_image, make_image_grid

import config.settings as settings
from . import http as http_util

logger = logging.getLogger('pconsole') 

IMG_DIR = settings.SERVER_ROOT/'output'/'imgs'
PROMPT_FILE = IMG_DIR.joinpath('_prompts.txt')
THUMB_DIR = IMG_DIR.parent/'thumbnails'
IMG_DIR.mkdir(parents=True, exist_ok=True)
THUMB_DIR.mkdir(exist_ok=True)


def prompt_to_filename(prompt:str, ext:typing.Literal['PNG','WebP','JPEG', 'GIF', 'MP4']='PNG', bidx:int=None):
    suffix=f'.{ext.lower()}'
    tstamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    prefix = tstamp+'_'
    if bidx is not None:
        prefix += f'{bidx:02d}_'
    fname = prefix+(re.sub('[^\w -]+','', prompt).replace(' ','_')[:100])+suffix
    return fname

def prompt_to_fpath(prompt:str, ext:typing.Literal['PNG','WebP','JPEG', 'GIF', 'MP4']='WebP', bidx:int=None, *, writable:bool=False):
    fname = prompt_to_filename(prompt, ext=ext, bidx=bidx)
    
    if writable:
        date = datetime.date.today().strftime('%Y%m%d')
        (IMG_DIR/date).mkdir(exist_ok=True, parents=True)
        out_imgpath = IMG_DIR/date/fname
        return out_imgpath
    return Path(fname)

def save_animation_prompt(frames:list[Image.Image], prompt:str, ext: typing.Literal['GIF','MP4']='MP4', optimize:bool=False, imgmeta:dict=None):
    out_imgpath = prompt_to_fpath(prompt, ext=ext, bidx=None, writable=True)
    sfx = f'.{ext.lower()}'
    if imgmeta is None:
        imgmeta = {}
    
    if ext == 'GIF':
        iio.imwrite(out_imgpath, frames, extension=sfx, loop=0, duration=imgmeta.get('duration', None))
        if optimize:
            import pygifsicle # sudo apt install gifsicle
            pygifsicle.optimize(out_imgpath)#, 'tmp-o.gif')
    else:
        iio.imwrite(out_imgpath, frames, extension=sfx, fps=imgmeta.get('fps', 10))
    
    with PROMPT_FILE.open('a') as f:
        f.write(f'{out_imgpath.name} : {prompt!r}\n')
    
    return out_imgpath

def save_image_prompt(image: Image.Image, prompt:str, ext:typing.Literal['PNG','WebP','JPEG']='PNG', bidx:int=None, **kwargs):
    out_imgpath = prompt_to_fpath(prompt, ext=ext, bidx=bidx, writable=True)

    image.save(out_imgpath, format=ext, **kwargs)
    
    with PROMPT_FILE.open('a') as f:
        f.write(f'{out_imgpath.name} : {prompt!r}\n')

    return out_imgpath

def to_thumb(image:Image.Image, max_size:tuple[int,int]=(256,256)):
    img = image.copy()
    img.thumbnail(max_size, Image.Resampling.BOX)
    return img

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

def trim_desc(description: str|None, max_len: int = 1000) -> str|None:
    if description and len(description) >= max_len: # will error if len desc > 1024
        description = description[:(max_len-1)] + '...'
    return description

def to_bytes_file(image:Image.Image, prompt: str, ext:typing.Literal['PNG','WebP','JPEG']='PNG', **kwargs):
    filename = prompt_to_filename(prompt, ext=ext)
    with io.BytesIO() as imgbin:
        image.save(imgbin, format=ext, **kwargs)
        imgbin.seek(0)
        
        return discord.File(fp=imgbin, filename=filename, description=trim_desc(prompt))

def to_bfile(image:Image.Image, filestem: str=None, description:str=None, ext: typing.Literal['PNG','WebP','JPEG'] = 'WebP', spoiler:bool=False, **kwargs):
    sfx = f'.{ext.lower()}'
    
    # discord thinks all .webp are animated. Rename as png, but keep webp all performance benefits
    if sfx == '.webp':
        sfx = '.png'
    
    filename = filestem+sfx if filestem is not None else tempfile.NamedTemporaryFile(suffix=sfx).name
        
    with io.BytesIO() as imgbin:
        image.save(imgbin, format=ext, **kwargs)
        imgbin.seek(0)
        
        return discord.File(fp=imgbin, filename=filename, description=trim_desc(description), spoiler=spoiler)


def impath_to_file(img_path:Path|str, description:str=None):
    return discord.File(fp=Path(img_path), filename=img_path.name, description=trim_desc(description))

def to_discord_file(image:Image.Image|str|Path, filestem:str=None, description:str=None, ext:typing.Literal['PNG','WebP','JPEG']=None, **kwargs):
    if isinstance(image, Image.Image):
        assert filestem or ext, 'must specify at least one of `filestem` or `ext` when using Image objects'
        return to_bfile(image=image, filestem=filestem, description=description, ext=ext, **kwargs)
    
    return impath_to_file(img_path=image, description=description)

def animation_to_bfile(image_frames:np.ndarray|list[Image.Image], filestem: str=None, description:str=None, ext: typing.Literal['GIF','MP4','WebP']='GIF', spoiler:bool=False, **kwargs):
    sfx = f'.{ext.lower()}'
    filename = filestem+sfx if filestem is not None else tempfile.NamedTemporaryFile(suffix=sfx).name
    
    with io.BytesIO() as outbin:
        if ext == 'MP4':
            # https://imageio.readthedocs.io/en/stable/_autosummary/imageio.plugins.ffmpeg.html#parameters-for-writing
            iio.imwrite(outbin, image_frames, extension=sfx, fps=kwargs.pop('fps', 10), **kwargs)
        else:
            iio.imwrite(outbin, image_frames, extension=sfx, loop=kwargs.pop('loop', 0), **kwargs)
        
        outbin.seek(0)
        return discord.File(fp=outbin, filename=filename, description=trim_desc(description), spoiler=spoiler)



async def read_attach(ctx: commands.Context):
    try:
        attach = ctx.message.attachments[0]
        print(attach.url)
        print(f'Image dims (WxH): ({attach.width}, {attach.height})')
        
        image = Image.open(io.BytesIO(await attach.read())).convert('RGB')
        return image
    except IndexError as e:
        return await ctx.send('No image attachment given!')


async def is_animated(image_url:str) -> bool:
    try:
        img_props = iio.improps(await http_util.aget_bytestream(image_url))
        # transparent png is batch but n_images = 0
        return img_props.is_batch and img_props.n_images > 1 
    except HTTPError as e: # urllib.error.HTTPError: HTTP Error 403: Forbidden
        print(e)
        return False

def gif_transparency_fix(imgbytes:bytes|Image.Image, bg_color:int|tuple[int,int,int] = 50):
    if isinstance(imgbytes, bytes):
        # https://en.wikipedia.org/wiki/GIF#Animated_GIF
        colormap = imgbytes[int('D', 16):int('30D', 16)]
        trns_idx = int(str(imgbytes[int('326', 16)]),16)
        # https://pillow.readthedocs.io/en/stable/_modules/PIL/ImagePalette.html#ImagePalette
        trns_rgb = tuple(colormap[trns_idx*3: (trns_idx+1)*3]) # 3 = R,G,B
        imgarr = iio.imread(imgbytes)
    elif isinstance(imgbytes, Image.Image):
        trns_idx = imgbytes.info['transparency']
        trns_rgb = {v:k for k,v in imgbytes.palette.colors.items()}[trns_idx]
        imgarr = np.array(imgbytes)

    imgarr[imgarr == trns_rgb] = bg_color
    return imgarr

def image_fix(image:np.ndarray, animated:bool=False, transparency:bool=False):
    if image.ndim == 2: # grayscale
        return image[:, :, None].repeat(3, -1) # copy 3x, HW->HWC
    if not animated and image.ndim > 3:
        image = image[0] # gifs -> take first frame
    if not transparency and image.shape[-1] > 3: # flatten transparency
        bg = Image.fromarray(np.full_like(image, [50,50,50,255])) # [50,51,57,255] # cool gray
        image = Image.alpha_composite(bg, Image.fromarray(image),).convert('RGB')
        image = np.array(image)

    return image

def convert_imgarr(image:np.ndarray, result_type:typing.Literal['PIL','np']|None = None) -> np.ndarray|Image.Image:
    if image.ndim < 4 or image.shape[0] < 2:
        # hwc or alpha-hwc
        if result_type != 'np': # None or PIL
            image = Image.fromarray(image)#, mode="RGB") # adding mode="RGB" will corrupt images with an alpha channel
    elif result_type == 'PIL':
        # gif
        image = [Image.fromarray(frame) for frame in image]

    return image

async def aload_image(image_uri:str, result_type:typing.Literal['PIL','np']|None = None):
    '''Asynchronously fetch an image, UA spoof if necessary

    default return type if None:
        animated image: numpy array
        non-animated image: PIL.Image
    '''
    simage_uri = str(image_uri)
    
    if simage_uri.startswith("http://") or simage_uri.startswith("https://"):
        try:
            imbytes = await http_util.aget_bytestream(image_uri, random_ua=True)
        except Exception as e:
            logger.warning(f'async fetch failed, trying sync. {str(e)}')
            imbytes = http_util.get_bytestream(image_uri, random_ua=False)
        
        # If both fail, will have uncaught exception. TODO: Better option?
        imeta = iio.immeta(imbytes)
        
        if ('duration' in imeta and 'transparency' in imeta): # animated gif with transparency
            image = gif_transparency_fix(imbytes, bg_color=50)
        else:
            image = iio.imread(imbytes)
    else:
        imeta = iio.immeta(image_uri)
        image = iio.imread(image_uri)
    
    image = convert_imgarr(image, result_type)

    return image, imeta



def print_image_info(image:Image.Image):
    # https://pillow.readthedocs.io/en/stable/reference/Image.html#image-attributes
    print('Info:', image.info)
    print('is_animated:', getattr(image, "is_animated", False))
    print('n_frames:', getattr(image, "n_frames", -1))
    print('has_transparency_data:', getattr(image, "has_transparency_data", None))
    print('format:', image.format)
    print('mode:', image.mode)
    print('size:', image.size)
    print('palette:', image.palette)