import re
import typing
import discord
from discord import app_commands
from discord.ext import commands

from cloneus.data import useridx

class WordListTransformer(app_commands.Transformer):
    '''Converts as comma separated string into a list of strings or tuples if has word:val format'''
    # https://discordpy.readthedocs.io/en/latest/interactions/api.html#discord.app_commands.Transformer
    async def transform(self, interaction: discord.Interaction, wordlist: str) -> list[str]:
        # (x, _, y) = value.partition(',')
        words = re.split(', ?', wordlist)
        words = [w.replace('"','') for w in words]
        if all(':' in w for w in words):
            words = [w.split(':') for w in words]
            words = [(word, float(weight)) for word,weight in words]
        return words    
    

class AuthorInitialsTransformer(app_commands.Transformer):
    '''Check that all chars an author initial and return lower case str'''
    all_author_initials: str = useridx.get_users('initial')
    async def transform(self, interaction: discord.Interaction, author_initials: str) -> list[str]:
        if author_initials and not all(i.lower() in self.all_author_initials for i in useridx.parse_initials(author_initials)):
            raise commands.BadArgument(f'all characters should be one of {self.all_author_initials!r}')
        return useridx.parse_initials(author_initials)


class PercentTransformer(app_commands.Transformer):
    '''Rescales from 1-100 to 0.01-1.00 if not already in range'''
    @property
    def min_value(self): return 0.0
    @property
    def max_value(self): return 100.0
    @property
    def type(self): return discord.AppCommandOptionType.number

    async def transform(self, interaction: discord.Interaction, value: float) -> float:  # commands.Range[float, 0, 100]
        if value:
            value = float(value) 
            if value>1.0:
                value/=100.0
        return value

def percent_transform(value: float|None):
    '''Rescales from 1-100 to 0.01-1.00 if not already in range'''
    if value:
        value = float(value) 
        if value>1.0:
            value/=100.0
    return value

def dash_to_empty(vstring:str|None):
    '''Convert a single "-" into an empty string. Avoids autofill if None.'''
    if vstring in ['-']:
        return ""
    return vstring

#PercentTransform = typing.NewType('PercentTransform', app_commands.Transform[float, PercentTransformer])
# https://github.com/Rapptz/discord.py/blob/bd402b486cc12f0c1bf7377fd65f2fe0a8fabd73/discord/app_commands/transformers.py#L514

class VibeTransformer(app_commands.Transformer):
    async def transform(self, ctx: commands.Context, vibe_phrase:str) -> tuple[str, float]:
        """Set a vibe phrase for an author. 
        
        examples: 
            "here is one way to do"+0.8
            another way that works+0.3
            another way gere-2.3
            this will-not-4.3
            "this will-work"-4.3
        
        Args:
            author: Who's vibe.
            phrase: vibe, motto, mantra, words to live by, catch phrase
            strength: how strong is the vibe. Neg values = opposite vibe
        """
        default_intensity = 0.2

        if '"' in vibe_phrase:
            phrase,intensifier = vibe_phrase.rsplit('"')
            phrase = phrase.strip('"')
            if '+' in intensifier:
                intensity = float(intensifier.strip('+'))
            elif '-' in intensifier:
                intensity = -float(intensifier.strip('-'))
            else:
                intensity = default_intensity

        elif '+' in vibe_phrase:
            phrase,intensity = vibe_phrase.rsplit('+', 1)
            intensity = float(intensity)
        elif '-' in vibe_phrase:
            phrase,intensity = vibe_phrase.rsplit('-', 1)
            intensity = -float(intensity)
        else:
            phrase = vibe_phrase
            intensity = default_intensity
        
        if intensity > 1.0:
            raise commands.BadArgument(f'Vibe Check Failed. Intensity must be <= 1.0 (found: {intensity})')

        scale = 1 - intensity
        return (phrase, scale)

