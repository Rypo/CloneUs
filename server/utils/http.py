import re
import random
import typing
import asyncio
import datetime
import tempfile
import functools
from urllib.error import HTTPError

import discord

import orjson
import aiohttp
import requests
from bs4 import BeautifulSoup, SoupStrainer

import config.settings as settings

RE_URL = re.compile(r"(?:https?:\/\/)?(?:www\.)?[-\w@:%.\+~#=]{2,256}\.[a-z]{2,6}\b[-\w@:%\+.~#?&/=)(]*", re.I+re.M) # modified from:https://regexr.com/3e6m0, saved as https://regex101.com/r/RiTYBH/2
UA_FILE = settings.RES_DIR/'user_agents.json'

def _read_user_agents():
    # https://github.com/microlinkhq/top-user-agents/blob/master/src/desktop.json
    try:
        iso_today = datetime.datetime.today().isocalendar()
        iso_mtime = datetime.datetime.fromtimestamp(UA_FILE.stat().st_mtime).isocalendar()
        # simple check for weekly update, e.g. 2025.52 < 2026.01
        update_ua_file = iso_mtime.year + iso_mtime.week/100 < iso_today.year + iso_today.week/100
    except FileNotFoundError:
        update_ua_file = True

    if update_ua_file:
        resp = requests.get('https://raw.githubusercontent.com/microlinkhq/top-user-agents/refs/heads/master/src/desktop.json')
        resp.raise_for_status()
        UA_FILE.write_bytes(orjson.dumps(resp.json()))
    
    return orjson.loads(UA_FILE.read_bytes())

USER_AGENTS = _read_user_agents()
USER_AGENT = random.choice(USER_AGENTS) # we don't really need a new UA every fetch for most applications, any one will do


@functools.cache
def valid_url(url:str):
    try:
        return requests.head(url, headers={"User-Agent": USER_AGENT}).ok
    except:
        return False

def get_bytestream(url:str, random_ua:bool=True):
    headers = {"User-Agent": random.choice(USER_AGENTS)} if random_ua else None
    rsp = requests.get(url, stream=True, headers=headers)
    rsp.raise_for_status()
    return rsp.raw.read()

async def aget_bytestream(url:str, random_ua:bool=True):
    headers = {"User-Agent": random.choice(USER_AGENTS)} if random_ua else None
    async with aiohttp.ClientSession(headers=headers, raise_for_status=True) as session:
        async with session.get(url) as resp:
            return await resp.read()


def extract_all_urls(msg: discord.Message):
    text_content_urls = RE_URL.findall(msg.clean_content)
    attachment_urls = attached_media_urls(msg)

    # tack on scheme if missing
    urls = [(url if url.startswith('http') else 'https://'+url) for url in attachment_urls + text_content_urls]
    # make sure url not hallucinated or inaccessible
    valid_urls = filter(valid_url, urls)
    # strip off discord image formatting url params, convert redirects tenors links to .mp4
    cleaned_urls = [normalize_media_url(url, tenor_mp4_url = True) for url in valid_urls]
    return list(dict.fromkeys(cleaned_urls))


def attached_media_urls(message: discord.Message):
    """Extract image urls from message attachments/embeds"""
    urls = []
    if message.embeds:
        urls.extend([emb.url for emb in message.embeds])
        
    if message.attachments:
        urls.extend([att.url for att in message.attachments])

    return urls

def normalize_media_url(url:str, tenor_mp4_url:bool):
    if not isinstance(url, str):
        return url
    
    if 'discordapp' in url:
        # GIF: https://media.discordapp.net/attachments/.../XYZ.gif?ex=...&is=...&=&width=837&height=837
        # JPG: https://media.discordapp.net/attachments/.../.../XYZ.jpg?ex=...&is=...&hm=...&=&format=webp&width=396&height=836
        url = url.split('&=&')[0] # gifs don't have a "format=", but both gifs and images have "&=&"
        #clean_url = re.sub(r'&(?:width|height)=\d*','',url).strip('&') + ('&=&format=webp&quality=lossless' if '&format=' not in clean_url else '')
        url = url.split('format=')[0].rstrip('&=?')

    if 'tenor' in url and tenor_mp4_url:
        # https://images-ext-1.discordapp.net/external/sW67...4tA/https/media.tenor.com/aUz-N2QvBOsAAAPo/the-isle-evrima.mp4 -> https://media.tenor.com/aUz-N2QvBOsAAAPo/the-isle-evrima.mp4
        if 'https/media.tenor.com' in url: 
            url = 'https://' + url.split('https/')[-1]

        # (https://tenor.com/view/the-isle-evrima-kaperoo-quality-assurance-hypno-gif-25376214 | https://tenor.com/bSDFW.gif ) -> https://media.tenor.com/aUz-N2QvBOsAAAPo/the-isle-evrima.mp4
        elif not url.endswith('.mp4'):
            url = get_tenor_links(url)['mp4']
    
    return url


@functools.cache
def get_tenor_links(url:str):
    resp = requests.get(url)
    # resp.raise_for_status()
    return {
        'src_url': url,
        'url': resp.url, # redirected url if shortlink (e.g. https://tenor.com/bcdEB.gif -> https://tenor.com/view/scooby-doo-dancing-sailing-gif-15266411)
        'mp4': BeautifulSoup(resp.content, features="html.parser", parse_only=SoupStrainer('meta', property="og:video")).select_one('[content$=".mp4"]').attrs.get('content')
    }

async def aget_tenor_links(url:str, random_ua: bool = True):
    headers = {"User-Agent": random.choice(USER_AGENTS)} if random_ua else None
    async with aiohttp.ClientSession(headers=headers, raise_for_status=True) as session:
        async with session.get(url) as resp:
            return {
                'src_url': url,
                'url': resp.url.human_repr(), # redirected url if shortlink (e.g. https://tenor.com/bcdEB.gif -> https://tenor.com/view/scooby-doo-dancing-sailing-gif-15266411)
                'mp4': BeautifulSoup(await resp.read(), features="html.parser", parse_only=SoupStrainer('meta', property="og:video")).select_one('[content$=".mp4"]').attrs.get('content')
            }
        
def extract_tenor_data(url:str, detailed:bool = True):
    resp = requests.get(url)
    doc = BeautifulSoup(resp.content, features="html.parser",)

    img_primary = doc.body.select_one('img[alt][fetchpriority="high"]')
    
    data_extract = {
        'source_url': url,
        'dest_url': resp.url,
        'video_url': doc.head.select_one('meta[property="og:video"][content$=".mp4"]').attrs.get('content',''),
        'keywords': doc.head.select_one('meta[name="keywords"]').attrs.get('content',''),
        'title': doc.head.select_one('meta[property="og:title"]').attrs.get('content','').replace('Discover & Share GIFs','').strip('- '),
        'alt_text': img_primary.attrs.get('alt'),
        'gif_url': img_primary.attrs.get('src'),
        'detailed_metadata': {},
    }
    
    if not detailed:
        return data_extract
    # keep_info = ['id','legacy_info','created','content_description', 'long_title', 'itemurl','url','tags','shares']

    json_payload = orjson.loads(doc.body.select_one("script#store-cache").contents[0])
    
    target_metadata = next(iter(json_payload['gifs']['byId'].values()))['results'][0]
    related_items = next(iter(json_payload['gifs']['related'].values()))['results']
    
    target_metadata['mp4'] = target_metadata.pop('media_formats')['mp4']
    for r in related_items:
        r['mp4'] = r.pop('media_formats')['mp4']

    target_metadata['related_items'] = related_items
    # related_metadata = [{k:r[k] for k in keep_info}|{'mp4':r['media_formats']['mp4']} for r in related_items]

    data_extract['detailed_metadata'] = target_metadata

    return data_extract


def tenor_search(query: str, limit: int = 8, media_filter: str = 'minimal'):
    # base url for most formats: 'https://media.tenor.com/'
    media_format_suffix = {  
        'gif': 'C',         # 'https://media1.tenor.com/m/'
        'gifpreview': 'e',
        'mediumgif': 'd',   # 'https://media1.tenor.com/m/'
        'mp4': 'o',
        'tinygif': 'M',
        'tinymp4': '1',
        'tinywebp': '1',
        'tinywebp_transparent': 'm',
        'webm': 's',
        'webp': 'x',
        'webp_transparent': 'l'
    }
    # https://tenor.com/gifapi/documentation#endpoints-search
    base_url = "https://g.tenor.com/v1/search"
    params = {'q': query, 'key':'LIVDSRZULELA', 'limit':limit, 'media_filter': media_filter}
    rsp = requests.get(base_url, params=params)

    return rsp.json()