import re
import json
import random
import typing
import discord
from discord import app_commands
from discord.ext import commands 

from ..config import settings
from ..views import redrawui


class Utility(commands.Cog):
    '''Random utility commands. Some /slash, some !bang'''
    def __init__(self, bot):
        self.bot = bot
        self.voicelines = None
        
    @commands.command()
    async def ping(self, ctx):
        """A simple command that returns latency."""
        await ctx.send(f'Pong! {round(self.bot.latency * 1000)}ms')


    @commands.command()
    async def roll(self, ctx, dice: str):
        """Rolls a dice in NdN format."""
        try:
            rolls, limit = map(int, dice.split('d'))
        except Exception:
            await ctx.send('Format has to be in NdN! (e.g. 1d6)', emphemeral=True)
            return

        result = ', '.join(str(random.randint(1, limit)) for r in range(rolls))
        await ctx.send(result)

    @commands.command(aliases=['ss'])
    async def shortstraw(self, ctx: commands.Context):
        """See who amongus draws the short straw."""
       
        ss_user = random.choice([*filter(lambda m: not m.bot and m.status != discord.Status.offline, ctx.guild.members)]).name
        await ctx.send(f'{ss_user} drew the short straw. RIP')
    
    @commands.command(aliases=['ls'])
    async def longstraw(self, ctx: commands.Context):
        """See who amongus draws the looong straw."""
       
        ss_user = random.choice([*filter(lambda m: not m.bot and m.status != discord.Status.offline, ctx.guild.members)]).display_name
        await ctx.send(f'{ss_user} drew the **Long** straw. Gratz')
    
    @commands.command(description='For when you wanna settle the score some other way')
    async def choose(self, ctx, *choices: str):
        """Randomly chooses between multiple choices."""
        await ctx.send(random.choice(choices))

    @commands.command()
    async def tic(self, ctx: commands.Context):
        """Starts a tic-tac-toe game with yourself."""
        await ctx.send('Tic Tac Toe: X goes first', view=redrawui.TicTacToe())



async def setup(bot):
    await bot.add_cog(Utility(bot))