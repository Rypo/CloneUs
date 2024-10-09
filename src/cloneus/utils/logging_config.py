"""
Common logging module
"""
# Adapted from: 
# https://github.com/axolotl-ai-cloud/axolotl/blob/main/src/axolotl/logging_config.py
# https://github.com/Rapptz/discord.py/blob/master/discord/utils.py

import os
import sys
import typing
import logging
from logging.config import dictConfig

from colorama import Fore, Back, Style, init


class ColorfulFormatter(logging.Formatter):
    """
    Formatter to add coloring to log messages by log type
    """

    LEVEL_COLORS = [
        (logging.DEBUG, Back.BLACK+Style.BRIGHT),
        (logging.INFO, Fore.BLUE+Style.BRIGHT), 
        (logging.WARNING, Fore.YELLOW+Style.BRIGHT), 
        (logging.ERROR, Fore.RED),
        (logging.CRITICAL, Back.RED),
    ]

    FORMATS: dict[int, logging.Formatter] = None

    # def __init__(self, fmt: str | None = None, datefmt: str | None = None, style: typing.Literal['%','{','$'] = "%", validate: bool = True, *, defaults: os.Mapping[str, typing.Any] | None = None) -> None:
    #     super().__init__(fmt, datefmt, style, validate, defaults=defaults)
    #     self.FORMATS = self._set_level_formats()

    def _set_level_formats(self):
        level_time = ('{color}' + '%(levelname)s' + Style.RESET_ALL 
                      + Fore.BLACK+Style.BRIGHT + ' @ ' + '%(asctime)s' + Style.RESET_ALL)
        
        basic_loc = Fore.MAGENTA + '%(filename)s:%(funcName)s' + Style.RESET_ALL
        
        verbose_loc = '%(name)s.%(funcName)s:%(lineno)d'
        
        msg = '%(message)s'
        
        self.FORMATS = {}
        for level, color in self.LEVEL_COLORS:
            if level in [logging.INFO, logging.WARNING]:
                loc = basic_loc
                date_fmt = '%H:%M:%S'
            else:
                loc = verbose_loc
                date_fmt = '%Y-%m-%d %H:%M:%S'
            
            lvl_time = level_time.format(color=color)
            
            fmt_string = f'[{lvl_time}]({loc}) {msg}'

            self.FORMATS[level] = logging.Formatter(fmt_string, datefmt=date_fmt)


    def format(self, record):
        if self.FORMATS is None:
            self._set_level_formats()
        formatter = self.FORMATS.get(record.levelno)
        if formatter is None:
            formatter = self.FORMATS[logging.DEBUG]

        # Override the traceback to always print in red
        if record.exc_info:
            text = formatter.formatException(record.exc_info)
            record.exc_text = Fore.RED + text + Style.RESET_ALL #'\x1b[31m{text}\x1b[0m'

        output = formatter.format(record)

        # Remove the cache layer
        record.exc_text = None
        return output
    
    
DEFAULT_LOGGING_CONFIG: dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] [PID:%(process)d] %(message)s",
        },
        "colorful": {
            "()": ColorfulFormatter,
            #"format": "[%(asctime)s] [%(levelname)s] [%(name)s.%(funcName)s:%(lineno)d] [PID:%(process)d] [RANK:%(rank)d] %(message)s",
        },
        "breif": {
            "format": "(CloneUs:%(name)s.%(filename)s):%(levelname)s: %(message)s", # [%(levelname)-5s @ %(asctime)s]
            #"datefmt": "%Y-%m-%d %H:%M:%S"
        },
    },
    "filters": {},
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "filters": [],
            "stream": sys.stdout,
        },
        "color_console": {
            "class": "logging.StreamHandler",
            "formatter": "colorful",
            "filters": [],
            "stream": sys.stdout,
        },
        "brief_console": {
            "class": "logging.StreamHandler",
            "formatter": "breif",
            "filters": [],
            "stream": sys.stdout,
        },
    },
    "root": {"handlers": ["console"], "level": os.getenv("LOG_LEVEL", "INFO")},
    "loggers": {
        "cloneus": {
            "handlers": ["color_console"],
            "level": "DEBUG",
            "propagate": False,
        },
        "scripts": {
            "handlers": ["brief_console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}


def setup_logging():
    """Setup default logging configuration"""
    print('LOGGING SETUP CALLED')
    init()  # Initialize colorama
    dictConfig(DEFAULT_LOGGING_CONFIG)