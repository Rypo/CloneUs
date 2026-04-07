import re
import copy
import typing
import urllib.parse
import mimetypes
import functools

import requests
import imageio.v3 as iio # pip install -U "imageio[ffmpeg, pyav]" # ffmpeg: mp4, pyav: trans_png

from cloneus.types import cpaths

DEFAULT_HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36'} # use static UA unless dynamic determined necessary

RE_FILENAME_UNSAFE = re.compile('[^.\w -]+')

@functools.cache
def resolve_mime_type(url:str):
    mtype, enc = mimetypes.guess_type(url) # Or: https://github.com/ahupp/python-magic
    
    if mtype is None:
        try:
            if (resp := requests.head(url, headers=DEFAULT_HEADERS)).ok:
                mtype = resp.headers.get('Content-Type')
        except:
            pass
    
    if mtype:
        return mtype.split('/')[0]

def categorize_media_urls(urls: list[str], valid_formats: tuple[str] = None):
    if valid_formats is None:
        valid_formats = []
    
    # only classify modalities the model understands. Other urls are ignored. TODO: other urls -> tools if tool use?
    media_types = {fmt: [] for fmt in valid_formats if fmt not in ['text', 'tools']} # | {'other': []}
    
    if not media_types:
        return {}
    
    for url in urls:
        if (media_format := resolve_mime_type(url)) in media_types:
            media_types[media_format].append(url)

    return media_types

@functools.cache
def fetch_local_uri(url: str, item_type: typing.Literal['image','video']):
    filename = urllib.parse.urlsplit(url).path.rsplit('/',1)[-1]
    filename = urllib.parse.unquote(filename)
    filename = RE_FILENAME_UNSAFE.sub('', filename).replace(' ', '_')
    filepath = cpaths.DATA_DIR.joinpath('.cache', item_type, filename)
    filepath = filepath.with_stem(filepath.stem[:88])
    if not filepath.exists():
        rsp = requests.get(url, headers=DEFAULT_HEADERS)
        iio.imwrite(filepath, iio.imread(rsp.content))
    
    return filepath.as_uri()
    
def urls_to_local_files(conversations: list[dict]|list[list[dict]], as_uri: bool = False):
    # conversations = copy.deepcopy(conversations) # TODO: mutate okay?
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for item in message["content"]:
                    item_type = item["type"]
                    if item_type in ("image", "image_url", "video") and item[item_type].startswith('http'):
                        local_uri = fetch_local_uri(item[item_type], item_type.split('_')[0]) # "image_url" -> "image"
                        item[item_type] = local_uri if as_uri else local_uri.removeprefix('file://')
    return conversations