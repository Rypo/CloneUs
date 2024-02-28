import os
import re
import json
import random
import typing
from pathlib import Path
import logging
from logging.config import dictConfig

import discord

from omegaconf import OmegaConf
from dotenv import load_dotenv

TESTING = bool(os.getenv('TESTING', 0))
print(f'Testing: {TESTING}')
SHOW_CHANGELOG = not TESTING

PFX = 'DEV_' if TESTING else ''

SERVER_ROOT = Path(__file__).parent.parent

ROOT_DIR = SERVER_ROOT.parent

RUNS_DIR = ROOT_DIR / 'runs/full'

CONFIG_DIR = SERVER_ROOT / 'config'
COGS_DIR = SERVER_ROOT / 'cogs'
APPS_DIR = SERVER_ROOT / 'appcmds'
VIEWS_DIR = SERVER_ROOT / 'views'

LOGS_DIR = SERVER_ROOT / 'logs' / ('dev' if TESTING else 'prod')
RES_DIR = SERVER_ROOT / 'res'

load_dotenv(ROOT_DIR.joinpath('.env'))

BOT_TOKEN = os.getenv(PFX+'BOT_TOKEN')

GUILDS_ID_INT = int(os.getenv(PFX+'GUILDS_ID'))
GUILDS_ID = discord.Object(id=GUILDS_ID_INT)
CHANNEL_ID = int(os.getenv(PFX+'CHANNEL_ID'))

def _init_dirs():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

def _read_models(models_file='models.json'):
    with open(CONFIG_DIR/models_file, 'r') as f:
        trained_models = json.load(f)
    return trained_models

TRAINED_MODELS = _read_models(models_file='models.json')
ACTIVE_MODEL_CKPT = RUNS_DIR/TRAINED_MODELS[-1]["ckpt"] # Use the last added model by default





BOT_PRESENCE = {
    'ready': dict(status=discord.Status.online, activity=discord.Activity(type=discord.ActivityType.listening, name='yo Commands')), # Listening to...
    'chat': dict(status=discord.Status.online, activity=discord.Activity(type=discord.ActivityType.custom, name='💬 Imma Chat Man',state='💬 Imma Chat Man')),
    'draw': dict(status=discord.Status.online, activity=discord.Activity(type=discord.ActivityType.custom, name='🎨 Sketch-y boi', state='🎨 Sketch-y boi')),
    'busy': dict(status=discord.Status.idle, activity=discord.Game("Imitation Games...")), # Playing...
    'down': dict(status=discord.Status.do_not_disturb, activity=discord.Game("Dead")) # Playing...
}

def custom_presense(status_name: typing.Literal['online','idle','do_not_disturb'], state: str):
    return {'status':getattr(discord.Status, status_name), 'activity': discord.Activity(type=discord.ActivityType.custom, name=state, state=state)}

# discord.utils._ColourFormatter.FORMATS
# https://github.com/Rapptz/discord.py/blob/576ab269e84b4be80fff3091e56bac42fa0aa552/discord/utils.py#L1262

LOGGING_CONFIG = {
    "version": 1,
    "disabled_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "%(levelname)-10s - %(asctime)s - %(module)-15s : %(message)s"
        },
        "standard": {
            "format": "%(levelname)-10s - %(name)-15s : %(message)s"
        },
        "model": {
            "format": "%(levelname)-10s [%(asctime)-10s] %(name)-10s : %(message)s"
        },
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
        "console2": {
            "level": "WARNING",
            "class": "logging.StreamHandler",
            "formatter": "standard",
        },
        "file": {
            "level": "INFO",
            "class": "logging.FileHandler",
            "filename": LOGS_DIR/"infos.log",
            "mode": "w",
            "formatter": "verbose",
        },
        "modelfile": {
            "level": "INFO",
            "class": "logging.FileHandler",
            "filename": LOGS_DIR/"model.log",
            "mode": "w",
            "formatter": "model",
        },
        "cmdsfile": {
            "level": "INFO",
            "class": "logging.FileHandler",
            "filename": LOGS_DIR/"cmds.log",
            "mode": "w",
            "formatter": "model",
        },
        "eventfile": {
            "level": "INFO",
            "class": "logging.FileHandler",
            "filename": LOGS_DIR/"event.log",
            "mode": "w",
            "formatter": "model",
        },
        "errfile": {
            "level": "ERROR",
            "class": "logging.FileHandler",
            "filename": LOGS_DIR/"errors.log",
            "mode": "w",
            "formatter": "verbose",
        },
    },
    "loggers": {
        "bot": {"handlers": ["console"], "level": "INFO", "propagate": False},
        "model": {"handlers": ["modelfile"], "level": "INFO", "propagate": False},
        "cmds": {"handlers": ["cmdsfile"], "level": "INFO", "propagate": False},
        "event": {"handlers": ["eventfile"], "level": "INFO", "propagate": False},
        "discord": {
            "handlers": ["console2", "file"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

def move_logs_lts(*logfilesnames):
    '''Move last run's logs to a daily and merge all file'''
    for logfile in logfilesnames:
        mlogs: Path = LOGS_DIR/logfile
        
        if mlogs.exists():
            plogs = mlogs.read_text()

            if (match := re.search(r'\[(\d{4}-\d{2}-\d{2})',plogs)):
                date = match.group(1)
                with open(mlogs.parent.joinpath('daily', f'{mlogs.stem}_{date}{mlogs.suffix}'), 'a') as f:
                    f.write(plogs)
            
            with open(mlogs.parent.joinpath(f'all_{logfile}'), 'a') as f:
                f.write(plogs)

move_logs_lts("model.log", "cmds.log", "event.log")

dictConfig(LOGGING_CONFIG)

