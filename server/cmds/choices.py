from discord import app_commands

from cloneus.data import useridx
import config.settings as settings

from managers import imgman

AUTHOR_DISPLAY_NAMES = [app_commands.Choice(name=n,value=n) for n in useridx.get_users('dname')]

MODEL_GENERATIONS = [app_commands.Choice(name=m['desc'], value=m['name']) for m in settings.BEST_MODELS]
MODEL_YEARS = [app_commands.Choice(name=m['years'], value=m['years']) for m in settings.YEAR_MODELS]+[app_commands.Choice(name='random', value='random')]
AUTO_MODES = [app_commands.Choice(name=mode, value=mode) for mode in ['off','rbest','irbest','urand','top', ]]

#CHAT_CONTEXTS = [app_commands.Choice(name = c, value=c) for c in ['all', 'base', 'discord']]
IMAGE_MODELS = [app_commands.Choice(name=v['desc'], value=k) for k,v in imgman.AVAILABLE_MODELS.items()]