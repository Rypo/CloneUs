import os
import re
import json
import random
import typing
from pathlib import Path
import logging
import logging.handlers
from logging.config import dictConfig

import discord
from colorama import Fore, Back, Style
from omegaconf import OmegaConf
from dotenv import load_dotenv


SERVER_ROOT = Path(__file__).parent.parent

ROOT_DIR = SERVER_ROOT.parent

RUNS_DIR = ROOT_DIR / 'runs/full'

CONFIG_DIR = SERVER_ROOT / 'config'
COGS_DIR = SERVER_ROOT / 'cogs'
RES_DIR = SERVER_ROOT / 'res'

DISCORD_SESSION_INTERACTIVE = int(os.getenv('DISCORD_SESSION_INTERACTIVE', 1))
TESTING = int(os.getenv('TESTING', 0))
print(f'Testing: {bool(TESTING)}, Interactive: {bool(DISCORD_SESSION_INTERACTIVE)}', )
SHOW_CHANGELOG = not TESTING

PFX = 'DEV_' if TESTING else ''

LOGS_DIR = SERVER_ROOT / 'logs' / ('dev' if TESTING else 'prod')


load_dotenv(ROOT_DIR.joinpath('.env'))

BOT_TOKEN = os.getenv(PFX+'BOT_TOKEN')

GUILDS_ID_INT = int(os.getenv(PFX+'GUILDS_ID'))
GUILDS_ID = discord.Object(id=GUILDS_ID_INT)
CHANNEL_ID = int(os.getenv(PFX+'CHANNEL_ID'))

def _init_dirs():
    (LOGS_DIR/'daily').mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

def _read_models(models_file='models.json'):
    with open(CONFIG_DIR/models_file, 'r') as f:
        trained_models = json.load(f)
    return trained_models

ALL_MODELS = _read_models(models_file='models.json')
BEST_MODELS = ALL_MODELS['BEST']
YEAR_MODELS = ALL_MODELS['ERA']

ACTIVE_MODEL_CKPT = RUNS_DIR/BEST_MODELS[-1]["ckpt"].split('runs/full/')[-1] # Use the last added model by default, split just in case



BOT_PRESENCE = {
    'ready': dict(status=discord.Status.online, activity=discord.Activity(type=discord.ActivityType.listening, name='yo Commands')), # Listening to...
    'chat': dict(status=discord.Status.online, activity=discord.Activity(type=discord.ActivityType.custom, name='💬 Imma Chat Man',state='💬 Imma Chat Man')),
    'draw': dict(status=discord.Status.online, activity=discord.Activity(type=discord.ActivityType.custom, name='🎨 Sketch-y boi', state='🎨 Sketch-y boi')),
    'chat_draw': dict(status=discord.Status.online, activity=discord.Activity(type=discord.ActivityType.custom, name='🎨💬 WordArt™', state='🎨💬 WordArt™')),
    'busy_chat': dict(status=discord.Status.idle, activity=discord.Game("Imitation Games...")), # Playing...
    'busy_draw': dict(status=discord.Status.idle, activity=discord.Game("with finger paints...")), # Playing...
    'down': dict(status=discord.Status.do_not_disturb, activity=discord.Game("Dead")) # Playing...
}

def custom_presense(status_name: typing.Literal['online','idle','do_not_disturb'], state: str):
    return {'status':getattr(discord.Status, status_name), 'activity': discord.Activity(type=discord.ActivityType.custom, name=state, state=state)}

class ColorFormatter(discord.utils._ColourFormatter):
    # https://github.com/Rapptz/discord.py/blob/576ab269e84b4be80fff3091e56bac42fa0aa552/discord/utils.py#L1262
    LEVEL_COLORS = [
        (logging.DEBUG, Back.BLACK+Style.BRIGHT), #'\x1b[40;1m'),
        (logging.INFO, Fore.BLUE+Style.BRIGHT), #'\x1b[34;1m'),
        (logging.WARNING, Fore.YELLOW+Style.BRIGHT ),#'\x1b[33;1m'),
        (logging.ERROR, Fore.RED),
        (logging.CRITICAL, Back.RED), #'\x1b[41m'),
    ]

    fncol = Back.BLACK+Fore.GREEN+Style.DIM #Fore.MAGENTA
    
    FMT_STRING = ('[' + '{color}' + '%(levelname)-5s' + Style.RESET_ALL
              + Fore.BLACK+Style.BRIGHT + ' @ ' + '%(asctime)s' + Style.RESET_ALL + ']'
              #+ '_' + Fore.MAGENTA + '%(module)s' + Style.RESET_ALL + '_'
              + '(' + fncol + '%(clsName)s' + '%(funcName)s' + Style.RESET_ALL + ')' + ' %(message)s')
    
    FORMATS = None

    def __init__(self, fmt: str | None = None, datefmt: str | None = None, style: typing.Literal['%','{','$'] = "%", validate: bool = True, *, defaults: os.Mapping[str, typing.Any] | None = None) -> None:
        super().__init__(fmt, datefmt, style, validate, defaults=defaults)
        self.FORMATS = self.set_level_formats()

    
    def set_level_formats(self, fmt_string:str = None, level_colors:tuple[int, str]=None):
        if level_colors is not None:
            self.LEVEL_COLORS = level_colors
        if fmt_string is not None:
            self.FMT_STRING = fmt_string
        
        formats = {level: logging.Formatter(self.FMT_STRING.format(color=color), ('%H:%M:%S' if level < logging.WARNING else '%Y-%m-%d %H:%M:%S'))
        for level, color in self.LEVEL_COLORS}
        self.FORMATS = formats
        return self.FORMATS
    

class TimedDirRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    def namer(self, default_name):
        # logs/dev/foo.log.YYYY-mm-dd -> logs/dev/daily/foo_YYYY-mm-dd.log
        pdir, fname = os.path.split(default_name)
        stem, sfx, date = fname.rsplit('.', maxsplit=2)
        new_name = os.path.join(pdir, 'daily', f'{stem}_{date}.{sfx}')
        
        return new_name


class RoutingFileHandler(logging.Handler):#logging.handlers.TimedRotatingFileHandler):
    HANDLERS: dict[str, logging.handlers.BaseRotatingHandler] = {}

    # https://pyformat.info/
    GFMTR = logging.Formatter("{levelname:10s} [{asctime:10s}] {name:10s} : ({funcName}, {cog_name}) - {message}", style='{', defaults={'cog_name':'COG'})
    VFMTR = logging.Formatter("{levelname:10s} - {asctime} - {module:15s} : {message}", style='{')
    
    for route in ['cmds','event','model']:
        _rhand = TimedDirRotatingFileHandler(filename=str(LOGS_DIR/f"{route}.log"), when='midnight')
        
        _rhand.setFormatter(GFMTR)
        HANDLERS[route] = _rhand
    
    for route in ['infos']:
        _rhand = logging.handlers.RotatingFileHandler(filename=str(LOGS_DIR/f"{route}.log"), mode='w') # Note: this never actually rotates, maxBytes=0.
        _rhand.setFormatter(VFMTR)
        HANDLERS[route] = _rhand
    
    def route_record(self, record:logging.LogRecord):
        if record.funcName.startswith('on_'):
            return self.HANDLERS['event']
        elif record.funcName.startswith('cog_'):
            return self.HANDLERS['cmds']
        elif 'managers' in record.pathname:
            return self.HANDLERS['model']
        
        return self.HANDLERS['infos']

    def emit(self, record:logging.LogRecord):
        # first, check extra={'route':'...'}, fallback name suffix (e.g. server.cmds)
        route = getattr(record, 'route', record.name.split('.')[-1])
        # route plain server without .sfx to infos
        if route=='sever':
            route = 'infos'
        # if neither in extra nor name in routes, attempt to infer
        handler = self.HANDLERS.get(route, self.route_record(record))
        
        handler.emit(record)

class RoutingFilter(logging.Filter):
    def __init__(self, dest:typing.Literal['model','event','cmds'], name: str = "") -> None:
        super().__init__(name)
        self.dest = dest
        self.filter_map = {
            'model': self._model_like,
            'event': self._event_like,
            'cmds': self._cmds_like,
        }

    def _event_like(self, record:logging.LogRecord):
        return record.funcName.startswith('on_')
    def _cmds_like(self, record:logging.LogRecord):
        return record.funcName.startswith('cog_')
    def _model_like(self, record:logging.LogRecord):
        print(f'{record.module=}, {record.filename=}, {record.pathname=}')
        return 'managers' in record.pathname

    def filter(self, record:logging.LogRecord):
        name_filter = super().filter(record)
        
        # if set via extra={'route':'...'}
        if (route:=getattr(record, 'route', None)) is not None:
            print('RECORD GETATTR ROUTE:', route)
            return name_filter and route==self.dest
        
        # if name ends with [cmds,model,event] e.g. server.cmds
        if (route:=record.name.split('.')[-1]) in self.filter_map:
            print('RECORD NAME SPLIT ROUTE:', route)
            return name_filter and route==self.dest# self.filter_map[route](record)
        
        print('RECORD DEST DEFAULT:', self.dest)
        # use the assigned default
        return name_filter and self.filter_map[self.dest](record)

class MaxSeverityFilter(logging.Filter):
    def __init__(self, name: str = "", max_level=logging.WARNING) -> None:
        super().__init__(name)
        self.max_level = max_level

    def filter(self, record):
        return super().filter(record) and record.levelno <= self.max_level

LOGGING_CONFIG = {
    "version": 1,
    "disabled_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "%(levelname)-10s - %(asctime)s - %(module)-15s : %(message)s"
        },
        "standard": {
            "format": "%(levelname)-10s - %(name)-15s : %(message)s" 
            #"format": "%(levelname)-10s [%(asctime)-10s] %(name)-10s : %(message)s"
        },
        "colorized": {
            '()': lambda: ColorFormatter(defaults={'clsName':''})
        },
        "plain": {
            "format": '[%(levelname)-5s @ %(asctime)s]'+'(%(clsName)s'+'%(funcName)s)'+' %(message)s',
            "datefmt": '%Y-%m-%d %H:%M:%S'
        },
        "struct": {
            'format': '%(message)s'
        },
    },
    "filters": {
        "warnings_and_below": {
            "()" : lambda: MaxSeverityFilter('server', logging.WARNING),
        },
    },
    "handlers": {
        "console2": {
            "level": "WARNING",
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
        "color_console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "colorized",
        },
        "plain_console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "plain",
        },
        "file": {
            "level": "INFO",
            "class": "logging.FileHandler",
            "filename": LOGS_DIR/"infos.log",
            "mode": "w",
            "formatter": "verbose",
        },
        "routefile":{
            '()': RoutingFileHandler,#lambda: RoutingFileHandler(logging.INFO),
            "level": "INFO",
            "filters":['warnings_and_below'],
            
        },
        "errfile": {
            "level": "ERROR",
            "class": "logging.FileHandler",
            "filename": str(LOGS_DIR/"errors.log"),
            "mode": "w",
            "formatter": "verbose",
        },

        # structured
        "cmds_jsonl": {
            "level": "DEBUG",
            "class": "logging.FileHandler",
            "filename": LOGS_DIR/"struct_cmds.jsonl",
            "mode": "a",
            "formatter": "struct",
        },
        "cmd_cache_jsonl": {
            "level": "DEBUG",
            "class": "logging.FileHandler",
            "filename": LOGS_DIR/"cmd_cache.jsonl",
            "mode": "a",
            "formatter": "struct",
        },
    },
    "loggers": {
       
        "server": {"handlers": ["errfile","routefile"], "level": "INFO", "propagate": False},

        "pconsole": {"handlers": (["color_console"] if DISCORD_SESSION_INTERACTIVE else ["plain_console"]), # , 
                     "filters": [], "level": "DEBUG", "propagate": False},
        
        "struct_cmds": {"handlers": ["cmds_jsonl"], "level": "DEBUG", "propagate": False},
        "cmd_cacher": {"handlers": ["cmd_cache_jsonl"], "level": "DEBUG", "propagate": False},

        "discord": {
            "handlers": ["console2", "file"],
            "level": "INFO",
            "propagate": False,
        },
    },
}


def setup_logging():
    #move_logs_lts("model.log", "cmds.log", "event.log")
    #dictConfig(LOGGING_CONFIG)

    old_factory = logging.getLogRecordFactory()
    
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        clsName = ''
        
        if hasattr(record.args, 'get'):
            if (cls := record.args.get('cls', '')):
                clsName = cls.__class__.__name__+'.'
        
        record.clsName = clsName
        return record
    
    logging.setLogRecordFactory(record_factory)
    dictConfig(LOGGING_CONFIG)