import os
import typing
import asyncio
from contextlib import asynccontextmanager
#import uvloop
import discord
from discord import app_commands
from discord.ext import commands

from cloneus.data import roles

import config.settings as settings



DESC = '''A bot that fills in when your friends go AWOL (and other things).

Commands are either prefixed with `!` or `/`.
Commands that accept arguments can be called with either prefix.
'''


logger = settings.logging.getLogger('bot')

DYN_GLOB = '*.py'  if settings.TESTING else '[!_]*.py'

class BotUs(commands.Bot):
    def __init__(self):
        super().__init__(
            command_prefix='!',#commands.when_mentioned_or('!'), 
            description=DESC, 
            intents=discord.Intents.all(), 
            case_insensitive=True,
            allowed_mentions = discord.AllowedMentions(everyone=False, roles=True, users=True, replied_user=True),
            status=discord.Status.do_not_disturb,
        )    
    
    async def on_ready(self):
        print('ready')
        logger.info(f'Logged in as {self.user} (ID: {self.user.id})')
        await self.toggle_extensions(extdir='cogs', state='on')

        # This copies the global commands over to your guild.
        self.tree.copy_global_to(guild=settings.GUILDS_ID)
        await self.tree.sync() #await self.tree.sync(guild=settings.GUILDS_ID)

    async def gsync(self, guild:discord.Guild, spec: typing.Literal["~", "*", "^"] = "*"):
        # https://gist.github.com/AbstractUmbra/a9c188797ae194e592efe05fa129c57f#sync-command-example
        if spec == "~":
            synced = await self.tree.sync(guild=guild)
        elif spec == "*":
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
        elif spec == "^":
            self.tree.clear_commands(guild=guild)
            await self.tree.sync(guild=guild)
            synced = []
        else:
            synced = await self.tree.sync()
        return synced

    @asynccontextmanager
    async def writing_status(self, presence_busy:str='busy', presense_done:str='ready'):        
        await self.change_presence(**settings.BOT_PRESENCE.get(presence_busy, 
                                                               settings.custom_presense(status_name='idle', state=presence_busy)))
        try:
            yield 
        finally:
            await self.change_presence(**settings.BOT_PRESENCE.get(presense_done, 
                                                                   settings.custom_presense(status_name='online', state=presense_done)))
            

    async def toggle_extensions(self, extdir:typing.Literal['appcmds','cogs','views'], state='on'):
        for ext_file in settings.SERVER_ROOT.joinpath(extdir).glob(DYN_GLOB):
            if ext_file.stem != '__init__':
                if state=='on':
                    await self.load_extension(f'{extdir}.{ext_file.stem}')
                elif state=='off':
                    await self.unload_extension(f'{extdir}.{ext_file.stem}')
                elif state=='reload':
                    await self.reload_extension(f'{extdir}.{ext_file.stem}')
                else:
                    raise ValueError('Unknown state:', state)

    async def add_temporary_reaction(self, message: discord.Message, emoji:str, delete_after:float=60.0):
        await message.add_reaction(emoji)
        def check(reaction, user):
            return not user.bot and str(reaction.emoji) == emoji 
        try:
            reaction, user = await self.wait_for('reaction_add', timeout=delete_after, check=check)
        except asyncio.TimeoutError:
            print('No response in time. Removing')
            await message.remove_reaction(emoji, self.user)

    async def on_command_error(self, ctx: commands.Context, error: commands.CommandError):
        """The event triggered when an error is raised while invoking a command.
        Parameters
        ------------
        ctx: commands.Context
            The context used for command invocation.
        error: commands.CommandError
            The Exception raised.
        """
        if isinstance(error, commands.CommandNotFound):
            return
        elif isinstance(error, commands.MissingRequiredArgument):
            await ctx.send(f'You are missing a required argument: `{error.param.name}`')
        elif isinstance(error, commands.BadArgument):
            await ctx.send(f'Bad argument: `{error}`')
        elif isinstance(error, commands.NoPrivateMessage):
            try:
                await ctx.author.send(f'{ctx.command} can not be used in Private Messages.')
            except discord.HTTPException:
                pass
        elif isinstance(error, commands.CheckFailure):
            await ctx.send('You do not have permission to use this command.')
        elif isinstance(error, commands.CommandOnCooldown):
            await ctx.send(f'Command on cooldown ({round(error.retry_after, 2)}s).', delete_after=error.retry_after)
        elif isinstance(error, commands.DisabledCommand):
            await ctx.send(f'{ctx.command} has been disabled.')
        elif isinstance(error, commands.CommandInvokeError):
            logger.error(f'In {ctx.command.qualified_name}:', exc_info=error.original)
            await ctx.send(f'An error occurred in {ctx.command.qualified_name}: {error.original}')
        elif isinstance(error, commands.NotOwner):
            await ctx.send(f'You do not own this bot.')
        else:
            logger.error(f'In {ctx.command.qualified_name}:', exc_info=error.original)
            await ctx.send(f'An error occurred: {error}')


def main():
    cog_list = [c.stem.lower() for c in settings.COGS_DIR.glob(DYN_GLOB) if c.stem != '__init__']

    bot = BotUs()

    @bot.tree.command(name='reload', description='Reload a set of commands')
    @app_commands.choices(cog=[app_commands.Choice(name=cog, value=cog) for cog in cog_list])
    async def reload(interaction: discord.Interaction, cog: str):
        """Reload a command set."""
        await bot.reload_extension(f'server.cogs.{cog.lower()}')
        await interaction.response.send_message(f'Reloaded: {cog}', ephemeral=True, delete_after=1)
    

    @bot.command(name='restart')
    async def restart(ctx: commands.Context):
        '''Reload all command extentions'''
        os.environ['EAGER_LOAD'] = '1'
        await ctx.send('Restarting...', delete_after=3)
        await bot.toggle_extensions('cogs', 'reload')
        os.environ.pop('EAGER_LOAD')


    @bot.command(name='gsync')
    #@commands.guild_only()
    @commands.is_owner()
    async def sync_treecmds(ctx: commands.Context, spec: typing.Literal["~", "*", "^"] = None) -> None:
        
        synced = await bot.gsync(ctx.guild, spec=spec)
        
        synced_md = '\n- ' + '\n- '.join([s.name for s in synced])
        await ctx.send(f"Synced {len(synced)} commands {'globally' if spec is None else 'to the current guild.'}{synced_md}")
        return
    
    bot.run(settings.BOT_TOKEN)


if __name__ == '__main__':
    settings._init_dirs()
    settings.setup_logging()
    roles.check_author_initials()
    main()