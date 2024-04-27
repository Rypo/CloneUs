from dotenv import load_dotenv as _load_dotenv

from .cloneus import Cloneus

from cloneus.core import paths as cpaths

_load_dotenv(cpaths.ROOT_DIR/'.env')