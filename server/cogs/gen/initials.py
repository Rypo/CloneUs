import typing

import discord
from discord import app_commands
from discord.ext import commands

from cloneus.data import roles

AUTHOR_INITALS = list(roles.initial_to_author)
ICMD_ENABLED = any(AUTHOR_INITALS)



class InitialsMixin:
    
    @commands.hybrid_command(name='ibot', enabled=ICMD_ENABLED, hidden=(not ICMD_ENABLED))
    @app_commands.choices(author_initial = [app_commands.Choice(name=i, value=i) for i in AUTHOR_INITALS])
    async def initials_bot(self, ctx: commands.Context, author_initial:commands.Range[str, 1, 1], *, seed_text:typing.Optional[str] = None):
        """Call bot by author inital
        
        Args:
            author_initial: Designated initial for the author to respond as.
            seed_text: Text to start off with.
        """
        
        await self.anybot(ctx, roles.initial_to_author[author_initial], seed_text=seed_text)
    # https://gist.github.com/AbstractUmbra/a9c188797ae194e592efe05fa129c57f#file-07-extension_with_optional_group-py
    
    ########################## ADD YOUR CUSTOM SHORT COMMANDS HERE ##########################
    # EXAMPLE:



    # @commands.hybrid_command(name='sbot')
    # async def sbot(self, ctx: commands.Context, *, seed_text:typing.Optional[str] = None):
    #     """Chime in with Sambot
        
    #     Args:
    #         seed_text: Text to start off with.
    #     """
        
    #     await self.anybot(ctx, roles.initial_to_author['s'], seed_text=seed_text)



    ##########################################################################################