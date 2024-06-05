import os
import re
#import json
import random
import warnings
from dataclasses import dataclass, field
import ujson as json
import more_itertools
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
import requests

from googleapiclient.discovery import build as gbuild, Resource
from dotenv import load_dotenv

from cloneus.types import cpaths


YOUTUBE_DATA_DIR = cpaths.DATA_DIR / 'youtube'

# # https://stackoverflow.com/questions/3717115/regular-expression-for-youtube-links
RE_YOUTUBE_URL_ID = re.compile(r'(?P<url>(?:https?://)?(?:www\.|m\.)?youtu(?:\.be/|be.com/\S*(?:watch|embed|shorts)(?:(?:(?=/[-a-zA-Z0-9_]{11,}(?!\S))/)|(?:\S*v=|v/)))(?P<video_id>[-a-zA-Z0-9_]{11,})\S*)', re.I)

PARTS = ['snippet', 'contentDetails', 'statistics', 'topicDetails']

NFFIELDS = '''items(
etag, id, 
snippet(publishedAt, channelId, title, channelTitle, categoryId, tags, description),
contentDetails(definition, duration, licensedContent), 
statistics(commentCount, likeCount, viewCount),  
topicDetails)'''.replace('\n','').replace(' ','')

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
    def from_search_item(cls, search_item, query):
        video_id = search_item['id']['videoId']

        _snippet = search_item['snippet']
        title = _snippet['title']
        channel = _snippet['channelTitle']

        description = _snippet['description']
        publish_time = _snippet['publishTime']

        return cls(video_id, title, channel, description, publish_time, topics=[], tags=[], query=query)#, raw=search_item)

    @classmethod
    def from_video_item(cls, video_item, query='', filter_tags=True):
        video_id = video_item['id']

        _snippet = video_item['snippet']#.get('snippet', dict())
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

        return cls(video_id, title, channel, description, publish_time, topics, tags, query=query)#, raw=video_item)
    

    def to_url(self):
        return f'https://www.youtube.com/watch?v={self.video_id}'
    
    def to_template(self, max_topics=None):
        return YT_TEMPLATE.format(video_id=self.video_id, title=self.title, channel=self.channel, topics=', '.join(self.topics[:max_topics])) # ', '.join(self.tags)
    
    # def to_dict(self, parsed=False):
    #     if parsed:
    #         data = self.__dict__.copy()
    #         data.pop('raw')
    #         return data

    #     return self.raw
    
    


class YouTubeVideoCollection(dict):
    #def __init__(self, collection: dict[str, YouTubeVideo]):
    #    super().__init__(collection)

    # @classmethod
    # def from_search(cls, search_results:dict, query:str):
    #     return cls([YouTubeVideo.from_search_item(item, query) for item in search_results['items']], raw=search_results)
    
    # @classmethod
    # def from_idquery(cls, idquery_results:dict):
    #     return cls([YouTubeVideo.from_video_item(item, filter_tags=True) for item in idquery_results['items']], raw=idquery_results)
    
    # @classmethod
    # def from_dict(cls, ndict):
    #    return cls(ndict)
    
    # @classmethod
    # def reindex(self, key='video_id'):
    #     return self.from_dict({ytv.__getattribute__(key): ytv for ytv in self.values()})
    
    def find_all(self, predicate):
        return list(filter(predicate, self.values()))
    
    def find(self, predicate, default=None):
        #return more_itertools.first_true(self.collection, default=default, pred=predicate)
        return next(filter(predicate, self.values(),), default)
        


# def chunk_unique(items, max_chunk_size=50):
#     items = np.unique(items)
#     if items.shape[0] <= max_chunk_size:
#         return [items]
#     ngroup, nrem = divmod(items.shape[0], max_chunk_size)
#     chunk_splits = np.split(items[:-nrem], ngroup) + [items[-nrem:]]
#     return chunk_splits

def fast_check(video_id):
    # https://www.reddit.com/r/learnprogramming/comments/7ioxra/how_to_check_if_a_youtube_link_is_valid_or_not/
    # (['title', 'author_name', 'author_url', 'type', 'height', 'width', 'version', 'provider_name', 'provider_url', 'thumbnail_height', 'thumbnail_width', 'thumbnail_url', 'html'])
    resp = requests.get(f'https://www.youtube.com/oembed?format=json&url=https://www.youtube.com/watch?v={video_id}')
    if resp.ok:
        data = resp.json()
        return {'video_id': video_id, 'title': data['title'], 'channel': data['author_name']}
    return {}


def parse_template_string(template_string: str):
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
            print(f"Couldn't parse: {kvs}\n{e}")

    return matched_params
    

    
class YouTubeManager:
    def __init__(self, video_data_file='video_data.jsonl', search_data_file='search_data.jsonl', invalid_ids_file='invalid_ids.txt', allow_fetch=True, enabled=True):
        self.enabled = enabled
        self.allow_fetch = allow_fetch
        self.quota_usage = 0

        self.invalid_ids = set()
        self.video_index = YouTubeVideoCollection()

        if not os.getenv('YOUTUBE_API_KEY'):
            if self.enabled:
                self.enabled = False
                warnings.warn('"YOUTUBE_API_KEY" Environment variable not set. YouTube links will not be encoded.', RuntimeWarning)
        
        if self.enabled:
            yt = gbuild('youtube', 'v3', developerKey=os.getenv('YOUTUBE_API_KEY'))
            
            self.ytv = yt.videos()
            self.yts = yt.search()
        
            self.video_data_filepath = YOUTUBE_DATA_DIR.joinpath(video_data_file)
            self.search_data_filepath = YOUTUBE_DATA_DIR.joinpath(search_data_file)
            self.invalid_ids_filepath = YOUTUBE_DATA_DIR.joinpath(invalid_ids_file)    
            
            self.video_index = YouTubeVideoCollection()
            self._build_initial_index()

            self.invalid_ids = self.read_invalid_ids()


    def _build_initial_index(self):
        YOUTUBE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.video_data_filepath.touch()
        self.search_data_filepath.touch()
        self.invalid_ids_filepath.touch()
        raw_vid_index = self.read_video_data(result_type='lookup')

        for video_items in raw_vid_index:
            query = video_items['query']
            result = video_items['result']
            self.parse_result(result, query)
        
    def read_invalid_ids(self):
        with open(self.invalid_ids_filepath, 'r') as f:
            invalid_ids = f.read().splitlines()
        return set(invalid_ids)
    
    def write_invalid_ids(self, new_invalid_ids):
        '''writes to invalid_ids file and updates self.invalid_ids'''
        if isinstance(new_invalid_ids, str):
            new_invalid_ids = [new_invalid_ids]
        with open(self.invalid_ids_filepath, 'a') as f:
            f.writelines([ii + '\n' for ii in new_invalid_ids])

        self.invalid_ids.update(new_invalid_ids)
        print(f'added {len(new_invalid_ids)} ids to known invalid_ids')


    def write_video_data(self, result, query, result_type):
        filepath = self.search_data_filepath if result_type=='search' else self.video_data_filepath
        with open(filepath, 'a') as f:
            json.dump({'query': query, 'result': result}, f)
            f.write('\n')
        
        print(f"wrote {len(result['items'])} items to {result_type} file")

    def read_video_data(self, result_type='lookup'):
        filepath = self.search_data_filepath if result_type=='search' else self.video_data_filepath
        with open(filepath, 'r') as f:
            video_data = list(map(json.loads, f.read().splitlines()))
        return video_data
    

    def parse_result(self, result, query):
        '''Parse a result and update self.video_index'''
        vidx = {}
        for item in result['items']:
            viddata = YouTubeVideo.from_video_item(item, query=query, filter_tags=True)
            vidx[viddata.video_id] = viddata
        # We don't want to clobber items with higher information
        # For example, if got the same video twice, but once with a query and once without
        # we want to keep the query no mater what
        if query:
            self.video_index.update(vidx)
        else:
            # if in index, assume equal or greater information and don't update
            # TODO: UNHANDLED CASE - two different queries return the same video
            # -- this is likely a very common occurance, but code has no way of handling it at the moment
            # -- will need major revisions to accomodate
            for k,v in vidx.items():
                self.video_index.setdefault(k, v)
        
        return vidx
    
    def encode(self, text, allow_fetch=None):
        if not self.enabled:
            return text
        allow_fetch = self.allow_fetch if allow_fetch is None else allow_fetch
        if 'youtu' in text.lower():
            text = RE_YOUTUBE_URL_ID.sub(lambda m: self._encode_re_helper(m, allow_fetch=allow_fetch), text)
        return text

    def decode(self, text, allow_fetch=None, return_matchdict=False):
        if not self.enabled:
            return text
        allow_fetch = self.allow_fetch if allow_fetch is None else allow_fetch
        if '<youtube' in text:
            text = RE_YT_TEMPLATE.sub(lambda m: self._decode_re_helper(m, allow_fetch=allow_fetch), text)
            if return_matchdict:
                matched_params = parse_template_string(RE_YT_TEMPLATE.search(text))
                return text, matched_params

        return text
    
    def _encode_re_helper(self, match_obj: re.Match, allow_fetch):
        video_id = match_obj.groupdict()['video_id']
        if video_id in self.invalid_ids:
            return ''
        videos = self.get_videos(video_id, query='', allow_fetch=allow_fetch)
        if not videos:
            if allow_fetch:
                print('NO RECORD FOUND FOR:',video_id)
                self.write_invalid_ids(video_id)
            return ''
        return videos[0].to_template()

    def _decode_re_helper(self, match_obj: re.Match, allow_fetch):
        template_str = match_obj.group(1)
        matched_params = parse_template_string(template_str)
        
        #video_id = matched_params.get('id','')
        title = matched_params.get('title','')
        #channel = matched_params.get('channel','')

        # there are two competing interests here.
        # If we are decoding a known valid url (i.e user:youtube.com/... -> <youtube ...> -> youtube.com/)
        # wait, would that ever even happen?
        # If this function will ONLY be called on generated templates, then we can remove the cache_video look up
        # but if not, then would be spamming api calls for no reason.
        #if (cached_video := self.video_index.get(video_id)):
        #    return cached_video.to_url()
        
        # assume video_id is a fake if not cached, so don't both querying
       
        if len(title) >= 3:
            videos, from_cache = self.get_search(title, allow_fetch=allow_fetch)
            pick = videos[0] if not from_cache else random.choice(videos)
            return pick.to_url()
            
        # elif len(channel) >= 3:
        #     videos, from_cache = self.get_search(channel, allow_fetch=allow_fetch)
        #     pick = random.choice(videos)
        #     return pick.to_url()
        else:
            # TODO: can we do better than just randomly picking something?
            print('USING RANDOM VIDEO')
            rand_id = random.choice(list(self.video_index.keys()))
            return self.video_index[rand_id].to_url()



    def get_videos(self, video_ids:list[str], query:str, allow_fetch=None) -> list[YouTubeVideo]: # (should it return a list of YouTubeVideo or {'viDeoID': YouTubeVideo})??
        allow_fetch = self.allow_fetch if allow_fetch is None else allow_fetch
        if isinstance(video_ids, str):
            video_ids = [video_ids]

        # filter out any known invalid ids
        video_ids = (set(video_ids)-self.invalid_ids)
        found_ids = set()
        output = []

        for vid in video_ids:
            if (vdata:=self.video_index.get(vid)):
                output.append(vdata)
                found_ids.add(vid)
        
        video_ids -= found_ids
        
        if not video_ids:
            return output
        
        if not allow_fetch:
            print('WARNING: some ids not index and `allow_fetch=False`:', video_ids)
            return output
            
        # TODO: consider moving write logic outside of the loop so don't do multiple rounds of io
        for video_ids_chunk in more_itertools.chunked(video_ids, 50):
            v_idstr = ','.join(video_ids_chunk)

            new_videos_result = self._fetch_videos(v_idstr)
            if new_videos_result['items']:
                self.write_video_data(new_videos_result, query, result_type='lookup')
                new_vididxs = self.parse_result(new_videos_result, query=query)
                # any ids excluded from result were not returned by api, and therefore not valid
                if (new_invalid_ids := set(video_ids_chunk)-set(new_vididxs.keys())):
                    self.write_invalid_ids(new_invalid_ids)
                output.extend(list(new_vididxs.values()))

        return output

    def get_search(self, query, allow_fetch=None):
        allow_fetch = self.allow_fetch if allow_fetch is None else allow_fetch
        # LOGIC FOR lookup by query/title goes here
        #cache_output = []
        cache_output = self.video_index.find_all(lambda v: query in [v.title,v.query])
        from_cache = True
        
        if cache_output:
            return cache_output, from_cache
        
        new_output = []
        from_cache = False
        
        if allow_fetch:
            new_search_result = self._fetch_search(query)
            if new_search_result['items']:
                self.write_video_data(new_search_result, query, result_type='search')
                new_video_ids = [item['id']['videoId'] for item in new_search_result['items'] if item['id']['kind']=='youtube#video']
                # NOTE: these are not _strictly_ new outputs since some of the ids may be cached already
                new_output = self.get_videos(new_video_ids, query=query, allow_fetch=allow_fetch)

        return new_output, from_cache


    
    def _fetch_videos(self, video_idstr:str):
        resp = self.ytv.list(part=PARTS, id=video_idstr, fields=NFFIELDS)
        result = resp.execute()
        self.quota_usage+=1
        print(f'API CALL: fetch_videos({video_idstr}) | QUOTA_USAGE: {self.quota_usage}')
        return result   
    
    def _fetch_search(self, query):
        resp = self.yts.list(part='snippet', safeSearch='none', maxResults=50, q=query)
        result = resp.execute()
        self.quota_usage+=100
        print(f'API CALL: fetch_search("{query}") | QUOTA_USAGE: {self.quota_usage}')
        return result
    

