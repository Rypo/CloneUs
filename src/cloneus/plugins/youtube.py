import os
import re
#import json
import random
import logging
import warnings
import typing
from pathlib import Path
from dataclasses import dataclass, field
import orjson
import more_itertools
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
import requests

from googleapiclient.discovery import build as gbuild, Resource
from dotenv import load_dotenv

from cloneus.types import cpaths

logger = logging.getLogger(__name__)

YOUTUBE_DATA_DIR = cpaths.DATA_DIR / 'youtube'
YOUTUBE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# # https://stackoverflow.com/questions/3717115/regular-expression-for-youtube-links
RE_YOUTUBE_URL_ID = re.compile(r'(?P<url>(?:https?://)?(?:www\.|m\.)?youtu(?:\.be/|be.com/\S*(?:watch|embed|shorts)(?:(?:(?=/[-a-zA-Z0-9_]{11,}(?!\S))/)|(?:\S*v=|v/)))(?P<video_id>[-a-zA-Z0-9_]{11,})\S*)', re.I)

# private (uploader only): fileDetails, processingDetails, suggestions
PARTS = ['snippet', 'contentDetails', 'status', 'statistics', 'topicDetails', 'paidProductPlacementDetails']

NFFIELDS = '''items(
etag, id, 
snippet(publishedAt, channelId, title, channelTitle, categoryId, tags, description),
contentDetails(definition, duration, licensedContent, caption, contentRating(ytRating)),
status,
statistics(commentCount, likeCount, viewCount),
topicDetails,
paidProductPlacementDetails
)'''.replace('\n','').replace(' ','')

VIDEO_CATEGORY_MAP = {
    # 0: 'Unknown',
    1: 'Film & Animation',
    2: 'Autos & Vehicles',
    10: 'Music',
    15: 'Pets & Animals',
    17: 'Sports',
    18: 'Short Movies',
    19: 'Travel & Events',
    20: 'Gaming',
    21: 'Videoblogging',
    22: 'People & Blogs',
    23: 'Comedy',
    24: 'Entertainment',
    25: 'News & Politics',
    26: 'Howto & Style',
    27: 'Education',
    28: 'Science & Technology',
    29: 'Nonprofits & Activism',
    30: 'Movies',
    31: 'Anime/Animation',
    32: 'Action/Adventure',
    33: 'Classics',
    34: 'Comedy',
    35: 'Documentary',
    36: 'Drama',
    37: 'Family',
    38: 'Foreign',
    39: 'Horror',
    40: 'Sci-Fi/Fantasy',
    41: 'Thriller',
    42: 'Shorts',
    43: 'Shows',
    44: 'Trailers'
}



#YT_TEMPLATE = '<youtube id="{video_id}" title="{title}" channel="{channel}" topics="{topics}">'
#YT_TEMPLATE = '<youtube title="{title}" channel="{channel}" topics="{topics}">'
YT_TEMPLATE = '<youtube id="{video_id}" title="{title}">'
RE_YT_TEMPLATE = re.compile(r'< *youtube ([^>]+) *>',) # This should be safe since angle brackets "<>" aren't allowed in titles: https://youtu.be/q_wdIc9W7ZU

#ALT_YT_TEMPLATE = '{{youtube id="{video_id}" title="{title}" channel="{channel}" topics="{topics}"}}'
#ALT_YT_TEMPLATE = '/youtube(id="{video_id}", title="{title}", channel="{channel}", topics="{topics}")'

def quick_check(video_id: str) -> dict:
    # https://www.reddit.com/r/learnprogramming/comments/7ioxra/how_to_check_if_a_youtube_link_is_valid_or_not/
    # (['title', 'author_name', 'author_url', 'type', 'height', 'width', 'version', 'provider_name', 'provider_url', 'thumbnail_height', 'thumbnail_width', 'thumbnail_url', 'html'])
    resp = requests.get(f'https://www.youtube.com/oembed?format=json&url=https://www.youtube.com/watch?v={video_id}')
    if resp.ok:
        data = resp.json()
        return {'video_id': video_id, 'title': data['title'], 'channel': data['author_name']}
    return {}

@dataclass
class YouTubeVideo:
    video_id: str
    title: str
    channel: str
    description: str = ''
    publish_time: str = ''
    topics: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    query: str = ''
    #raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_search_item(cls, search_item:dict, query:str):
        video_id = search_item['id']['videoId']

        _snippet: dict[str,] = search_item['snippet']
        title = _snippet['title']
        channel = _snippet['channelTitle']

        description = _snippet['description']
        publish_time = _snippet['publishTime']

        return cls(video_id, title, channel, description, publish_time, topics=[], tags=[], query=query)#, raw=search_item)

    @classmethod
    def from_video_item(cls, video_item:dict, query='', filter_tags=True):
        video_id = video_item['id']

        _snippet: dict[str,] = video_item['snippet']#.get('snippet', dict())
        title = _snippet['title'] #_snippet.get('title','')
        channel = _snippet['channelTitle'] #_snippet.get('channelTitle','')

        # --- Everything below this line can be optional ---
        description = _snippet.get('description','')
        publish_time = _snippet.get('publishTime','')

        tags = _snippet.get('tags',[])
        
        cat_id = int(_snippet.get('categoryId',0))
        category = VIDEO_CATEGORY_MAP.get(cat_id,'')

        
        topic_caturls = video_item.get('topicDetails', dict()).get('topicCategories',[])
        topic_cats = [t.split('/')[-1] for t in topic_caturls] # ['https://en.wikipedia.org/wiki/Electronic_music'] -> ['Electronic_music']

        topics = list(dict.fromkeys([category, *topic_cats])) # Unique no sort

        if filter_tags:
            _titlechannel = f'{title} - {channel}'.lower()
            # keep tags that contain words not included in the video title or channel
            tags = list(filter(lambda tag: any(map(lambda w: w not in _titlechannel, tag.lower().split())), tags))  # [:3]

        return cls(video_id, title, channel, description, publish_time, topics, tags, query=query)
    

    def to_url(self):
        return f'https://www.youtube.com/watch?v={self.video_id}'
    
    def to_template(self, max_topics=None):
        return YT_TEMPLATE.format(video_id=self.video_id, title=self.title, channel=self.channel, topics=', '.join(self.topics[:max_topics])) # ', '.join(self.tags)
    
    
def read_video_data(data_filepath: str|Path) -> list[dict]:
    video_data = []

    with open(data_filepath, 'rb') as f:
        for i,line in enumerate(f.read().splitlines()):
            try:
                video_data.append(orjson.loads(line))
            except Exception as e:
                print(i,e)

    return video_data

def write_video_data(result: dict[str, ], query:str, filepath:Path):
    with open(filepath, 'ab') as f:
        f.write(orjson.dumps({'query': query, 'result': result}))
        f.write(b'\n')
    
    logger.info(f"wrote {len(result['items'])} items to {filepath.name} file")

class YouTubeVideoCollection(dict[str,YouTubeVideo]):
    
    @classmethod
    def from_files(cls, video_data_filepath:str|Path, search_data_filepath:str|Path):
        video_data_filepath = Path(video_data_filepath)
        search_data_filepath = Path(search_data_filepath)

        for filepath in [video_data_filepath, search_data_filepath]: 
            filepath.touch()
            
        video_data = read_video_data(video_data_filepath)
        
        ytv_collection = {}
        for video_items in video_data:
            query = video_items['query']
            result = video_items['result']
            
            for item in result['items']:
                video_id = item['id']
                ytv = YouTubeVideo.from_video_item(item, query=query, filter_tags=True)
                # keep higher information entry
                if query:
                    ytv_collection.update({video_id: ytv})
                else:
                    ytv_collection.setdefault(video_id, ytv)

        return cls(ytv_collection)
    

    def add_result_items(self, result_items:list[dict], query:str) -> list[str, YouTubeVideo]:
        '''add parsed result items to collection, return a list of video_ids'''
        
        video_ids = []
        for item in result_items: #result['items']:
            video_id = item['id']
            ytv = YouTubeVideo.from_video_item(item, query=query, filter_tags=True)
            # We don't want to clobber items with higher information
            # For example, if got the same video twice, but once with a query and once without
            # we want to keep the query no mater what
            
            # TODO: UNHANDLED CASE - two different queries return the same video
            # -- this is likely very common, but code has no way of handling it
            if query:
                self.update({video_id: ytv})
            else:
                self.setdefault(video_id, ytv)
            
            video_ids.append(video_id)

        return video_ids
    
    
    def find_all(self, predicate: typing.Callable[[YouTubeVideo], bool]) -> list[YouTubeVideo]:
        return list(filter(predicate, self.values()))
    
    def find(self, predicate: typing.Callable[[YouTubeVideo], bool], default=None) -> YouTubeVideo | None:
        return next(filter(predicate, self.values(),), default)
        




def parse_template_string(template_string: str) -> dict[str,str]:
    '''Takes a templated youtube string <youtube title="..." channel="..." ...>
    and returns a dict(title="...", channel...)
    '''
    # '''Takes a templated youtube string <youtube id="..." title="..." ...>
    # and returns a dict(id="...", title="...", ...)
    # '''
    kv_strings = template_string.split('" ') # 'id="-1qju6V1jLM', 'channel="ZMOONCHILD live',....
    # NOTE: this assumes perfect formatting... it won't be
    matched_params = {} # dict.fromkeys(['id','title','channel','topics'])
    
    for kvs in kv_strings:
        try:
            k,v = kvs.replace('"','').split('=')
            matched_params[k.strip()] = v.strip()
        except Exception as e:
            logger.error(f"Couldn't parse: {kvs!r}", exc_info=e)

    return matched_params
    


# TODO: Use stored raw search results to build a structure that maps query: video_ids, or query: {video_id: id, relevance_rank: idx}

class YouTubeManager:
    def __init__(self, video_data_file='video_data.jsonl', search_data_file='search_data.jsonl', invalid_ids_file='invalid_ids.txt', allow_fetch=True, enabled=True):
        self.enabled = enabled
        self.allow_fetch = allow_fetch
        self.quota_usage = 0

        self.video_data_filepath  = YOUTUBE_DATA_DIR.joinpath(video_data_file)
        self.search_data_filepath = YOUTUBE_DATA_DIR.joinpath(search_data_file)
        self.invalid_ids_filepath = YOUTUBE_DATA_DIR.joinpath(invalid_ids_file)

        for filepath in [self.video_data_filepath, self.search_data_filepath, self.invalid_ids_filepath]:
            filepath.touch()
        
        self.video_index = YouTubeVideoCollection.from_files(self.video_data_filepath, self.search_data_filepath)
        self.invalid_ids = set(self.invalid_ids_filepath.read_text().splitlines())

        if not os.getenv('YOUTUBE_API_KEY'):
            if self.enabled:
                self.enabled = False
                warnings.warn('"YOUTUBE_API_KEY" Environment variable not set. YouTube links will not be encoded.', RuntimeWarning)
        
        if self.enabled:
            yt = gbuild('youtube', 'v3', developerKey=os.getenv('YOUTUBE_API_KEY'))
            self.ytv = yt.videos()
            self.yts = yt.search()
    

    def write_invalid_ids(self, new_invalid_ids: str | list[str]):
        '''writes to invalid_ids file and updates self.invalid_ids'''
        if not new_invalid_ids: 
            return
        
        if isinstance(new_invalid_ids, str):
            new_invalid_ids = [new_invalid_ids]
        
        self.invalid_ids.update(new_invalid_ids)

        with open(self.invalid_ids_filepath, 'a') as f:
            f.write('\n'.join(new_invalid_ids) + '\n')
            # f.writelines([ii + '\n' for ii in new_invalid_ids])

        logger.info(f'added {len(new_invalid_ids)} ids to known invalid_ids')

    
    def encode(self, text:str, allow_fetch:bool = None) -> str:
        if not self.enabled: 
            return text
        
        if 'youtu' not in text.lower():
            return text
                
        allow_fetch = self.allow_fetch if allow_fetch is None else allow_fetch

        def _encode_re_helper(match_obj: re.Match) -> str:
            video_id = match_obj.groupdict()['video_id']
            if video_id in self.invalid_ids:
                return ''
            if (videos := self.get_video_metadata(video_id, query='', allow_fetch=allow_fetch)):
                return videos[0].to_template()
            if allow_fetch:
                logger.info(f'No record found for: {video_id}')
                self.write_invalid_ids(video_id)
            return ''
        
        
        text = RE_YOUTUBE_URL_ID.sub(_encode_re_helper, text)
        
        return text

    def decode(self, text:str, allow_fetch:bool = None) -> str:
        if not self.enabled:
            return text
        
        if '<youtube' not in text.lower():
            return text

        allow_fetch = self.allow_fetch if allow_fetch is None else allow_fetch
        
        def _decode_re_helper(match_obj: re.Match) -> str:
            template_str = match_obj.group(1)
            matched_params = parse_template_string(template_str)
            
            video_id = matched_params.get('id','')
            title = matched_params.get('title','')
            channel = matched_params.get('channel','')
            
            try:
                if len(title) >= 3:
                    return self.get_search_results(title, allow_fetch=allow_fetch)[0].to_url() # return top/first search result
                elif video_id and (video := self.video_index.get(video_id)): 
                    return video.to_url() # unless in collection, assume video_id is a hallucinated. Not worth querying. NOTE: could quick_check
                elif len(channel) >= 3:
                    return self.get_search_results(channel, allow_fetch=allow_fetch)[0].to_url() # return top/first search result
                else:
                    raise RuntimeError("YouTube parser failure")
            except Exception as e:
                logger.error(f'Metadata extraction failed. Using random video. Matched: {template_str!r}', exc_info=e)
                random.seed(template_str) # seed repeated parse consistency (Specific concern with Clonues.author_probabilties->Cloneus.to_seeded_text)
                return random.choice(list(self.video_index.values())).to_url()
                
    
        text = RE_YT_TEMPLATE.sub(_decode_re_helper, text)
            
        return text

    def get_video_metadata(self, video_ids:list[str], query: str | None, allow_fetch:bool = None) -> list[YouTubeVideo]:
        allow_fetch = self.allow_fetch if allow_fetch is None else allow_fetch
        if isinstance(video_ids, str):
            video_ids = [video_ids]
        
        # filter out any known invalid ids, preserve relevance sort order
        video_id_order = {k: i for i,k in enumerate(filter(lambda t: t not in self.invalid_ids, video_ids))}
        
        ids_to_lookup = list(video_id_order.keys())
        
        output = []

        for v_id in video_id_order:
            if (vid_data := self.video_index.get(v_id)):
                output.append(vid_data) # this makes final sort required
                ids_to_lookup.remove(v_id)
        
        if not ids_to_lookup: 
            return output # all ids present in collection
        
        if not allow_fetch:
            logger.warning('IDs not index and `allow_fetch=False`:', ids_to_lookup)
            return output
            
        
        for video_ids_chunk in more_itertools.chunked(ids_to_lookup, 50):
            v_idstr = ','.join(video_ids_chunk)

            video_response_data = self._fetch_videos(v_idstr)

            if video_response_data['items']:
                write_video_data(video_response_data, query, self.video_data_filepath)
                video_ids = self.video_index.add_result_items(video_response_data['items'], query=query)
                
                for v_id in video_ids:
                    output.append(self.video_index.get(v_id))
                
                # any ids excluded from result were not returned by api, and therefore not valid
                if (new_invalid_ids := set(video_ids_chunk)-set(video_ids)):
                    self.write_invalid_ids(new_invalid_ids)
                               
        # preserve original sort order
        output = sorted(output, key=lambda ytv: video_id_order[ytv.video_id])
        return output

    def get_search_results(self, query:str, allow_fetch:bool = None) -> list[YouTubeVideo]:
        allow_fetch = self.allow_fetch if allow_fetch is None else allow_fetch
        
        if not query:
            logger.warning('Search attempted with empty query')
            return []
        # If there is an *exact* match for query, return from collection
        if (cached_results := self.video_index.find_all(lambda v: query in [v.title,v.query])):
            return cached_results
        
        search_results = []
        
        if allow_fetch:
            search_response_data = self._fetch_search(query)

            if search_response_data['items']:
                write_video_data(search_response_data, query, self.search_data_filepath)
                video_ids = [item['id']['videoId'] for item in search_response_data['items'] if item['id']['kind']=='youtube#video'] # omit youtube#playlist, youtube#channel
                
                # Only API cost +1 to get detailed metadata for all search results now  
                search_results = self.get_video_metadata(video_ids, query=query, allow_fetch=allow_fetch)

        return search_results


    
    def _fetch_videos(self, video_idstr:str) -> dict[str, ]:
        # https://developers.google.com/youtube/v3/docs/videos/list
        resp = self.ytv.list(part=PARTS, id=video_idstr, fields=NFFIELDS)
        result = resp.execute()
        self.quota_usage+=1
        logger.info(f'API CALL: fetch_videos({video_idstr}) | QUOTA_USAGE: {self.quota_usage}')
        return result   
    
    def _fetch_search(self, query:str) -> dict[str, ]:
        # https://developers.google.com/youtube/v3/docs/search/list
        resp = self.yts.list(part='snippet', safeSearch='none', maxResults=50, q=query, order='relevance', type='video')
        result = resp.execute()
        self.quota_usage+=100
        logger.info(f'API CALL: fetch_search("{query}") | QUOTA_USAGE: {self.quota_usage}')
        return result
    

