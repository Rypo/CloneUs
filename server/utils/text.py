import re
import typing
import datetime
import more_itertools
from dataclasses import dataclass

import discord

import config.settings as settings
# https://support.discord.com/hc/en-us/articles/12620128861463-New-Usernames-Display-Names#h_01GXPQABMYGEHGPRJJXJMPHF5C
RE_USER_TAG = re.compile(r'^\[[\w ]{1,32}\]') # Line starts with [authorDisplayName] will match a space
RE_EMOJI_RAW = re.compile(r'<a?(:\w+:)\d+>')

def extract_author(msg_content: str):
    '''Return the name in the leading brackets "[John] ..." -> "John"
    
    DISCORD STYLE TAGS ONLY (simple [alphanum] tags)
    '''
    if msg_content.startswith('['):
        return msg_content.split(']', maxsplit=1)[0].strip('[')
    raise ValueError('Message has no author tag.')

def find_emoji(emoji_name: str, emojis: list[discord.Emoji]) -> (discord.Emoji|str):
    """Find an emoji by name."""
    emoji = discord.utils.find(lambda e: e.name == emoji_name, emojis)
    if emoji is None:
        emoji = f':{emoji_name}:'
    return emoji

def fix_mentions(message_str:str, bot):
    if '@' in message_str:
        for m in re.findall(r'(@\w+)', message_str):
            if m and m not in ['@everyone', '@here']:
                if usr := bot.get_guild(settings.GUILDS_ID_INT).get_member_named(m.strip('@')):
                    message_str = message_str.replace(m, usr.mention)

    return message_str

def message_filter(msg: discord.Message, allow_bot:bool=True) -> bool:
    """Filter out user commands messages and non-context bot messages.
    
    Args:
        allow_bot: If False, will filter out ALL bot messages, including message generations.
    """

    author = msg.author
    content = msg.clean_content

    # filter empty strings and filter out command calls
    if not content or content.startswith('!'):
        return False
    
    # filter all bot output other than user immitation
    if author.bot:
        if allow_bot:
            match = RE_USER_TAG.search(content)
            return (match is not None)
        else:
            return False
    
    return True

def filter_messages(messages: list[discord.Message]) -> list[discord.Message]:
    return list(filter(message_filter, messages))


@dataclass
class Msg:
    user: str
    message: str
    created_at: datetime.datetime

    @property
    def user_msg(self) -> tuple[str,str]:
        return (self.user, self.message)

def _parse_message(msg: discord.Message, user_aliases:dict[str,str]=None) -> Msg:
    """Get author display_name and message content """
    #return discord.utils.get(settings.GUILDS_ID.members, id=id)
    content = msg.clean_content
    # replace emoji <:name:id> with :name:
    content = RE_EMOJI_RAW.sub(r'\1', content)
    #content =  re.sub(r'<a?(:\w+:)\d+>', r'\1', content)
    # replace 2+ \n with a single one to avoid parsing confusion
    content = re.sub(r'\n{2,}','\n',content)

    # Fake the author name for bots output message
    if msg.author.bot:
        # This *shouldn't* fail because of the message filter
        # [USER:dave] -- llm input, [Dave] -- discord bot output
        author_tag = RE_USER_TAG.search(msg.clean_content).group()
        author = author_tag.strip('][')
        content = content.replace(author_tag, '').strip()
    else:
        author = msg.author.display_name
        if user_aliases:
            author = user_aliases.get(author,author)
    
    return Msg(author, content, msg.created_at)


def merge_messages(user_content_times:list[Msg], merge_minutes=7) -> list[tuple[str,str]]:
    merged_msgs = []
    for m in user_content_times:
        if not merged_msgs:
            merged_msgs.append(m)
        else:
            pm = merged_msgs[-1]
            if m.user==pm.user and (m.created_at-pm.created_at).total_seconds() <= (merge_minutes*60):
                merged_msgs[-1] = Msg(m.user, pm.message + '\n' + m.message, m.created_at)
            else:
                merged_msgs.append(m)

    return [m.user_msg for m in merged_msgs]


def process_seedtext(seed_text):
    bs2 = re.escape('\\')
    return '' if seed_text is None else re.sub(f' *{bs2} *','\n', seed_text)


def splitout_tag(model_output, RE_ANY_USERTAG:re.Pattern):
    # if '[/INST]' in model_output:
    #     print('WARNING: "[/INST]" found in model_output, returning first split')
    #     print(f'Raw Model output:\n {model_output!r}')
    #     escinst = re.escape('[/INST]')
    #     model_output = re.split(f' *{escinst}',model_output)[0]

    if len(usplits := RE_ANY_USERTAG.split(model_output)) > 1:
        print('WARNING: Multiple tags found in model_output, returning first split')
        print(f'Raw Model output:\n {model_output!r}')
        model_output =  usplits[0]

    return model_output

def llm_input_transform(messages: list[discord.Message], do_filter=False, user_aliases:dict[str,str]=None) -> list[tuple[str,str]]:
    """filter, if needed, sorts by created_at, merges consecutive author messages returns [(author, message), ...]"""
    if do_filter:
        messages = filter_messages(messages)
    
    user_content_times = [_parse_message(m, user_aliases) for m in messages]
    author_messages = merge_messages(user_content_times)
    
    return author_messages


def llm_output_transform(output: str, emojis) -> str:
    """Replace :emoji: string with discord Emoji objects

    Transform LLM output to Discord output."""
    # replace emoji :name: with <:name: emoji>
    output = re.sub(r':(\w+):',  lambda m: f'{find_emoji(m.group(1), emojis)}', output)

    return output


def cast_string_value(value):
    if not isinstance(value, str):
        return value
    if value[0].isnumeric():
        value = float(value) if '.' in value else int(value)
    elif value.lower() in ['true', 'false']:
        value = True if value.title() == 'True' else False
    elif value.lower() == 'none':
        value = None
    return value


def split_message(message:str, max_len=2000):
    for msgpart in more_itertools.constrained_batches(message.split(' '), max_size=max_len, get_len=lambda g: len(' '.join(g))):
        yield ' '.join(msgpart)
    #[' '.join(msgpart) for msgpart in list(more_itertools.constrained_batches(message.split(' '), max_size=max_len, get_len=lambda g: len(' '.join(g))))]
        

