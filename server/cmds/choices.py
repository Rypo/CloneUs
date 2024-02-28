from discord import app_commands

from cloneus.data import roles
import config.settings as settings


AUTHOR_DISPLAY_NAMES = [app_commands.Choice(name=n,value=n) for n in roles.author_display_names]

MODEL_GENERATIONS = [app_commands.Choice(name=m['desc'], value=m['name']) for m in settings.TRAINED_MODELS]

AUTO_MODES = [app_commands.Choice(name=mode, value=mode) for mode in ['off','rbest','irbest','urand','top', ]]

#CHAT_CONTEXTS = [app_commands.Choice(name = c, value=c) for c in ['all', 'base', 'discord']]