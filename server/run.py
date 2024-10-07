import os
import sys
import time
import signal
import psutil
import typing
import asyncio
import logging
import datetime
from contextlib import asynccontextmanager
import uvloop
import discord
from discord import app_commands
from discord.ext import commands

# NOTE: Do not import from cloneus in this file. It seems to break logging in the package.
#from cloneus.data import useridx

import config.settings as settings

import ujson as json

DESC = '''A bot that fills in when your friends go AWOL (and other things).

Commands are either prefixed with `!` or `/`.
Commands that accept arguments can be called with either prefix.
'''


logger = logging.getLogger('bot')
struct_logger = logging.getLogger('struct_cmds')
DYN_GLOB = '*.py'  if settings.TESTING else '[!_]*.py'

def hard_reset():
    """Restarts the current program, with file objects and descriptors cleanup"""
    # https://stackoverflow.com/a/33334183
    # see also:
    # https://gist.github.com/plieningerweb/39e47584337a516f56da105365a2e4c6
    # https://stackoverflow.com/questions/72715121/how-to-restart-a-python-script
    # https://stackoverflow.com/questions/11329917/restart-python-script-from-within-itself
    
    os.environ['REBOOT_REQUESTED'] = '1'
    
    try:
        p = psutil.Process(os.getpid())
        #time.sleep(3.0)
        #p.send_signal(signal.SIGINT)
        for handler in p.open_files():# + p.net_connections():
            os.close(handler.fd)
        p.send_signal(signal.SIGINT)

    except Exception as e:
        logger.error(e, exc_info=e)
    #finally:
        # python = sys.executable
        # os.execl(python, python, *sys.argv)

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
        self._operational_state = {
            'chat': False,
            'draw': False
        }
    
    async def on_ready(self):
        print('ready')
        logger.info(f'Logged in as {self.user} (ID: {self.user.id})')
        await self.toggle_extensions(extdir='cogs', state='on')
        if os.environ.pop('MESSAGE_ON_START', None):
            await self.get_channel(settings.CHANNEL_ID).send('I. am. REBORN.')
    
    def _get_presence(self):
        text_up = self._operational_state['chat']
        image_up = self._operational_state['draw']
        # 000 - busy,chat,draw
        # https://docs.python.org/3/howto/enum.html#flag
        if text_up and image_up:
            presence = settings.BOT_PRESENCE['chat_draw']
        elif text_up:
            presence = settings.BOT_PRESENCE['chat']
        elif image_up:
            presence = settings.BOT_PRESENCE['draw']
        else:
            presence = settings.BOT_PRESENCE['down']
        return presence

    async def report_state(self, activity: typing.Literal['chat','draw'], ready:bool):
        state_changed = self._operational_state[activity] != ready
        self._operational_state[activity] = ready
        if state_changed:
            presence = self._get_presence()
            await self.change_presence(**presence)
        

    @asynccontextmanager
    async def busy_status(self, activity:typing.Literal['chat','draw']|str):
        '''Async context manager. Temorarily change bot presence while working.
        
        Args:
            activity (['chat', 'draw'] | str): message to show in presence. If 'chat' or 'draw' use default setting for that activity.
        '''
        if activity in ['chat','draw']:
            presense = settings.BOT_PRESENCE.get('busy_'+activity)
        else:
            presense = settings.custom_presense(status_name='idle', state=activity)
        
        await self.change_presence(**presense)
        try:
            yield 
        finally:
            await self.change_presence(**self._get_presence())
            

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

    async def on_app_command_completion(self, interaction:discord.Interaction, command:app_commands.Command):
        # app_commands.ContextMenu
        data = {
            'log_time': datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S%z'),
            'dev': settings.TESTING,
        }

        idata = {
            'interaction.data':interaction.data,
            'interaction.id':interaction.id,
            'interaction.created_at':interaction.created_at.strftime('%Y-%m-%d %H:%M:%S%z'),
            'interaction.type':interaction.type,
            'interaction.user':{'id':interaction.user.id, 'name':interaction.user.name, 'global_name':interaction.user.global_name},
            'interaction.guild_id':interaction.guild_id,
            'interaction.channel_id':interaction.channel_id,
            'interaction.application_id':interaction.application_id,
            
            'interaction.context_types':interaction.context.to_array(),
            'interaction.extras':interaction.extras,
            #'interaction.command': interaction.command.to_dict(self.tree),
            #'interaction.command_failed':interaction.command_failed,
        }
        data.update(idata)

        if interaction.message is not None:
            i_msg_data = {
                'interaction.message': {
                    'id':interaction.message.id,
                    'type':interaction.message.type,
                    'created_at':interaction.message.created_at.strftime('%Y-%m-%d %H:%M:%S%z'),
                    'author.id':interaction.message.author.id,
                    'application_id':interaction.message.application_id,
                    'content':interaction.message.content,
                    'pinned':interaction.message.pinned,
                    'webhook_id':interaction.message.webhook_id,
                    'attachments': [attachment.to_dict() for attachment in interaction.message.attachments],
                    'reactions': [{'reaction': str(react), 'count':react.count} for react in  interaction.message.reactions],
                    'components': [component.to_dict() for component in interaction.message.components],
                    'mentions': [{'user_id':user.id} for user in interaction.message.mentions]
            }}
            data.update(i_msg_data)
        # 'command': command.to_dict(self.tree),
        #param_types = dict(subcommand = 1, subcommand_group = 2, string = 3, integer = 4, boolean = 5, user = 6, channel = 7, role = 8, mentionable = 9, number = 10, attachment = 11,)
        cmd_data = {
            'command.module':command.module,
            'command.name': command.name,
            'command.description': command.description,
            'command.parameters': [{'name': param.name, 
                                    'type': param.type, 
                                    'default': (str(param.default) if param.required else param.default), 
                                    'required': param.required} 
                                    for param in command.parameters],
            #'command': command.to_dict(self.tree), 
            #'options': [param.to_dict() for param in self._params.values()],
            'command.extras':command.extras,
        }

        data.update(cmd_data)
        
        #data.update(ctx_data)
        struct_logger.info(json.dumps(data))
        

async def main(bot=None):
    cog_list = [c.stem.lower() for c in settings.COGS_DIR.glob(DYN_GLOB) if c.stem != '__init__']
    if bot is None:
        bot = BotUs()

    @bot.tree.command(name='reload', description='Reload a set of commands')
    @app_commands.choices(cog=[app_commands.Choice(name=cog, value=cog) for cog in cog_list])
    async def reload(interaction: discord.Interaction, cog: str):
        """Reload a command set."""
        await interaction.response.defer(ephemeral=True, thinking=True)
        await bot.reload_extension(f'cogs.{cog.lower()}')
        #await interaction.response.send_message(f'Reloaded: {cog}', ephemeral=True, delete_after=1)
        msg = await interaction.followup.send(f'Reloaded: {cog}', ephemeral=True, wait=True)
        await msg.delete(delay=1)
    
    
    @bot.command(name='restart', aliases=['reboot'])
    async def restart(ctx: commands.Context, spec: typing.Literal["-h", "-s",]='-h'):
        '''Nuke and Pave. All state will be lost.
        
        Args:
            spec: type of reboot (options "-h" ard, "-s" oft)
                "-h" - hard restart, complete shutdown and reboot
                "-s" - soft restart, attempt to reload without shutting down everything
        '''
        if spec == '-s':
            os.environ['EAGER_LOAD'] = '1'
            msg = await ctx.send('Attempting soft reboot..')
            await bot.toggle_extensions('cogs', 'reload')
            os.environ.pop('EAGER_LOAD')
            await msg.edit('Done.', delete_after=3)
        elif spec == '-h':
            await ctx.send('⚠️ **Reboot Request Received** ⚠️ Razing and Rebuilding to Remedy RuhRos...')
            await bot.toggle_extensions('cogs', 'off')
            hard_reset()

    @bot.command(name='gsync', aliases=['sync'])
    #@commands.guild_only()
    @commands.is_owner()
    async def sync_treecmds(ctx: commands.Context, spec: typing.Literal["+", "-", "~","."] = None) -> None:
        '''Attempt to synchronize bot commands.

        Use when:
            1. Not seeing commands that should be available.
            2. Command has outdated parameters/functionality.
            2. Seeing duplicates in app command auto completions. 

        Args:
            spec: Changes how commands are updated (options {"+", "-", "~", "."})
                "+" - copy global commands to guild, sync guild
                "-" - clear guild commands, sync guild
                "~" - sync global commands
                "." - sync guild commands
                None - sync global and guild commands
        '''
        # https://gist.github.com/AbstractUmbra/a9c188797ae194e592efe05fa129c57f#sync-command-example
        # - https://gist.github.com/AbstractUmbra/a9c188797ae194e592efe05fa129c57f#syncing-gotchas
        await bot.wait_until_ready()
        if (guild := ctx.guild) is None:
            guild = bot.get_guild(settings.GUILDS_ID_INT)
            
        
        print(guild, bot.guilds)
        
        #synced = await bot.gsync(guild, spec=spec)
        synced = []
        if spec == "+": 
            bot.tree.copy_global_to(guild=guild) # this causes duplicatation after sync
            synced = await bot.tree.sync(guild=guild)
        elif spec == "-":
            bot.tree.clear_commands(guild=guild) # removes duplicate command but needs sync anyway, no point in clear global
            synced = await bot.tree.sync(guild=guild)
        elif spec == "~":
            synced = await bot.tree.sync(guild=None)
        elif spec == ".":
            synced = await bot.tree.sync(guild=guild)
        else:
            # Double sync: clears dupes, adds new, removes old. It is the way. 
            
            global_synced = await bot.tree.sync(guild=None)
            # At this point, commands have not changed. 
            await bot.wait_until_ready()
            bot.tree.copy_global_to(guild=guild)
            
            guild_synced = await bot.tree.sync(guild=guild)
            # At this point, commands are synced, updated but are duplicated.
            bot.tree.clear_commands(guild=guild)
            
            await bot.wait_until_ready()
            guild_synced += await bot.tree.sync(guild=guild)
            # At this point, commands are synced, updated, and de-duplicated

            synced = set(global_synced+guild_synced)
        
        #synced_md = '\n- ' + '\n- '.join([s.name,s. for s in synced]) if synced else ''
        synced_md = '\n'.join([f'- {s.name} <{"guild" if s.guild_id else "global"}>' for s in sorted(synced, key=lambda x: x.name)]) if synced else ''
        guild_cnt = synced_md.count("guild")
        global_cnt = synced_md.count("global")
        if spec is None:
            where = f': ({global_cnt}) global and ({guild_cnt}) guild "{guild.name}".'
        elif spec == '~':
            where = 'globally.'
        elif spec == '.':
            where = f'to the current guild ({guild.name}).'
        else:
            where = ''
        print(synced_md)
        return await ctx.send(f"Synced {len(synced)} commands {where}".strip())
        
    async with bot:
        await bot.start(settings.BOT_TOKEN)
    #bot.run(settings.BOT_TOKEN)

def run_main():
    DEBUG = None
    bot = BotUs()
    
    try:
        uvloop.run(main(bot), debug=DEBUG)
    except KeyboardInterrupt:
        for task in asyncio.all_tasks(bot.loop):
           print(task.get_name())
           task.cancel()
        
        if 'REBOOT_REQUESTED' in os.environ:
            logger.info(f'REBOOT_REQUESTED. Running: {sys.argv}')
            os.environ.pop('REBOOT_REQUESTED')
            os.environ['MESSAGE_ON_START'] = '1'
            
            python = sys.executable
            os.execl(python, python, *sys.argv)
        
        # nothing to do here
        # `asyncio.run` handles the loop cleanup
        # and `self.start` closes all sockets and the HTTPClient instance.
        return

if __name__ == '__main__':
    #from threading import Thread
    settings._init_dirs()
    settings.setup_logging()
    
    #useridx.check_author_initials()
    # main()
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    run_main()