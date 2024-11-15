import os
import sys
import time
import signal
import discord.ext.tasks
import psutil
import typing
import asyncio
import logging
import argparse
import datetime
from contextlib import asynccontextmanager
import ujson as json
import uvloop
import discord
from discord import app_commands
from discord.ext import commands

# NOTE: Do not import from cloneus in this file. It seems to break logging in the package.
#from cloneus.data import useridx


import discord.ext
import utils.io as io_utils

DESC = '''A bot that fills in when your friends go AWOL (and other things).

Commands are either prefixed with `!` or `/`.
Commands that accept arguments can be called with either prefix.
'''


logger = logging.getLogger('bot')
struct_logger = logging.getLogger('struct_cmds')
cmdcache_logger = logging.getLogger('cmd_cacher')
# DYN_GLOB = '*.py'  if settings.TESTING else '[!_]*.py'

async def hard_reset(defer_handling=True):
    """Restarts the current program, with file objects and descriptors cleanup"""
    # https://stackoverflow.com/a/33334183
    # see also:
    # https://gist.github.com/plieningerweb/39e47584337a516f56da105365a2e4c6
    # https://stackoverflow.com/questions/72715121/how-to-restart-a-python-script
    # https://stackoverflow.com/questions/11329917/restart-python-script-from-within-itself
    
    os.environ['REBOOT_REQUESTED'] = '1'
    
    
    
    if defer_handling:
        # for task in asyncio.all_tasks(asyncio.get_running_loop()):
        #     if task.cancel():
        #         print('CANCELED TASK:', task.get_name(), task)
        #await asyncio.sleep(2.0) # give time for tasks to catchup
        raise RuntimeError('defer handling')
    else:
        try:
            p = psutil.Process(os.getpid())
            #time.sleep(3.0)
            # Closing openfiles 
            # TODO: try to determine which of these causes the crontab closing issue
            for handler in p.open_files():# + p.net_connections():
                print(handler.path)
                os.fsync(handler.fd)
                os.close(handler.fd)
            
        except Exception as e:
            logger.error(e, exc_info=e)

        p.send_signal(signal.SIGINT)
        #finally:
        #    os.environ['MESSAGE_ON_START'] = '1'
        #    python = sys.executable
        #    os.execl(python, python, *sys.argv)

def cmd_cache_init(cache_filepath:str, n:int = 100):
    last_n_cmds = io_utils.tail(cache_filepath, n=n)
    cmd_cache = {}
    
    for cmd_str in last_n_cmds:
        cmd = json.loads(cmd_str)['cmd']
        cmd_cache[cmd.pop('interaction_id')] = cmd
    return cmd_cache



class BotUs(commands.Bot):
    def __init__(self, cogs_list:list[str], extention_glob:str = '[!_]*.py'):
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
        self.cogs_list = cogs_list
        self._extention_glob = extention_glob
        
        self.pstore = io_utils.PersistentStorage(settings.CONFIG_DIR/'persistent_storage.yaml')
        self.cmd_cache = cmd_cache_init(settings.LOGS_DIR/'cmd_cache.jsonl', n=100)
    
    async def on_ready(self):
        print('ready')
        logger.info(f'Logged in as {self.user} (ID: {self.user.id})')
        await self.toggle_extensions(extdir='cogs', state='on')
        if os.environ.pop('MESSAGE_ON_START', None):
            await self.get_channel(settings.CHANNEL_ID).send('I. am. REBORN.', silent=True)
    
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
            

    async def toggle_extensions(self, extdir:typing.Literal['appcmds','cogs','views'], state:typing.Literal['on','off','reload']='on'):
        
        if extdir == 'cogs':
            ext_filestems = self.cogs_list
        else:
            ext_filestems = [ext_file.stem for ext_file in settings.SERVER_ROOT.joinpath(extdir).glob(self._extention_glob) if ext_file.stem != '__init__']
        
        state_func = {'on': self.load_extension, 'off': self.unload_extension, 'reload': self.reload_extension}[state]
        
        for ext_stem in ext_filestems:
            await state_func(f'{extdir}.{ext_stem}')


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

    async def on_command_completion(self, ctx:commands.Context):
        # this will only trigger on hybrid commands
        if ctx.interaction is not None:
            interaction = ctx.interaction
            args = ctx.args
            kwargs = ctx.kwargs.copy() # doesn't really matter if mutate ctx since command is done by the time this is called

            # NOTE: for this to work, all commands must always refer to the flags argument as "flags". But at least then don't need to iterate over values
            if (flags := kwargs.pop('flags', None)):
                kwargs['flags'] = {'FlagsClsName': flags.__class__.__name__, 'flag_kwargs': {k:v for k,v in flags}}

            cmd_attrs = {'qualified_name': ctx.command.qualified_name, 'args':args, 'kwargs':kwargs, } #  'namespace': ctx.interaction.namespace
            self.cmd_cache[interaction.id] = cmd_attrs
            
            # this is what we will use to rebuild cache on start
            data = {'cmd': {'interaction_id': interaction.id, **cmd_attrs}}
            
            # the rest is just for logging purposes
            data['interaction'] = {
                'user_global_name': interaction.user.global_name,
                'user_name': interaction.user.name,
                'user_id': interaction.user.id,
                'type': interaction.type,
                'created_at':interaction.created_at.strftime('%Y-%m-%d %H:%M:%S%z'),
                'guild_id':interaction.guild_id,
                'channel_id':interaction.channel_id,
            }

            data['metadata'] = {
                'command_failed': ctx.command_failed,
                'module': ctx.command.module,
                'cog_name': ctx.command.cog_name,
                'message_id': ctx.message.id,
                
                'log_time': datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S%z')
            }

            cmdcache_logger.info(json.dumps(data))
        

    async def on_app_command_completion(self, interaction:discord.Interaction, command:app_commands.Command|app_commands.ContextMenu):
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
        if isinstance(command, app_commands.Command):
            cmd_data = {
                'command.module':command.module,
                'command.name': command.name,
                'command.qualified_name': command.qualified_name,
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
        

async def main(args, bot=None):
    
    extention_glob = '*.py' if args.test else '[!_]*.py'
    cog_list = [c.stem.lower() for c in settings.COGS_DIR.glob(extention_glob) if c.stem != '__init__']
    if args.beta:
        cog_list += ['beta.' + c.stem.lower() for c in (settings.COGS_DIR/'beta').glob(extention_glob) if c.stem != '__init__']
    if bot is None:
        bot = BotUs(cog_list, extention_glob)


    @bot.tree.command(name='reload', description='Reload a set of commands')
    @app_commands.choices(cog=[app_commands.Choice(name=cog.split('.')[-1], value=cog) for cog in cog_list])
    async def reload(interaction: discord.Interaction, cog: str):
        """Reload a command set."""
        await interaction.response.defer(ephemeral=True, thinking=True)
        await bot.reload_extension(f'cogs.{cog.lower()}')
        #await interaction.response.send_message(f'Reloaded: {cog}', ephemeral=True, delete_after=1)
        msg = await interaction.followup.send(f'Reloaded: {cog}', ephemeral=True, wait=True)
        await msg.delete(delay=1)
    
    @bot.command(name='kill')
    async def kill(ctx: commands.Context):
        '''Destroy the bot, by any means necessary. UNRECOVERABLE.'''
        print('KILL CALL')
        await ctx.send('☠️')
        await bot.change_presence(activity=discord.ActivityType.unknown, status=discord.Status.invisible)
        
        p = psutil.Process(os.getpid())
        for task in asyncio.all_tasks(bot.loop):
           print('CANCEL:', task.get_name(), task)
           task.cancel()
        
        p.send_signal(signal.SIGTERM)


    @bot.command(name='restart', aliases=['reboot'])
    async def restart(ctx: commands.Context, spec: typing.Literal["-h", "-s",]='-h'):
        '''Hard reset the bot. All state will be lost.
        
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
            os.environ['REBOOT_REQUESTED'] = '1'
            raise KeyboardInterrupt('Restart')
            # try:
            #     await hard_reset(defer_handling=True)
            # except RuntimeError:
            #     raise KeyboardInterrupt('Restart')

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



def get_cli_args():
    parser = argparse.ArgumentParser(description='run discord bot')
    run_mode_group = parser.add_mutually_exclusive_group(required=True)
    run_mode_group.add_argument('--live', action='store_true', help='run on live server')
    run_mode_group.add_argument('--test', action='store_true', help='run on test server (if configured)')
    parser.add_argument('--non-interactive', action='store_true', help='disable "human features" like color logging to improve crontab compatibility')
    parser.add_argument('--stable', dest='beta', action='store_false', help='exclude unstable beta cogs')
    return parser.parse_args()

def run_main(args):
    #extention_glob = '*.py' if args.test else '[!_]*.py'
    DEBUG = True
    #bot = BotUs(extention_glob)
    
    # signal.signal(signal.SIGINT, signal_handler)
    try:
        uvloop.run(main(args), debug=DEBUG)
    # except (KeyboardInterrupt, SystemExit):
    except KeyboardInterrupt:
        print('Keyint')
        #bot.loop.close()
        # for task in asyncio.all_tasks(bot.loop):
        #   if task.cancel():
        #       print('CANCELED TASK:', task.get_name(), task)
           
        
        # nothing to do here
        # `asyncio.run` handles the loop cleanup
        # and `self.start` closes all sockets and the HTTPClient instance.
        return
    finally:
        if os.environ.pop('REBOOT_REQUESTED', None):
            print(f'REBOOT_REQUESTED. Running: {sys.executable} {sys.argv}')
            
            os.environ['MESSAGE_ON_START'] = '1'
            
            python = sys.executable
            os.execl(python, python, *sys.argv)

if __name__ == '__main__':
    
    cli_args = get_cli_args()
    if cli_args.test:
        os.environ['TESTING'] = '1'
    if cli_args.non_interactive:
        os.environ['DISCORD_SESSION_INTERACTIVE'] = '0'
    
    # Only import settigns AFTER parsing cli. 
    # TODO: make settings less auto run.
    import config.settings as settings
    settings._init_dirs()
    settings.setup_logging()
    
    #useridx.check_author_initials()
    # main()
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    run_main(cli_args)