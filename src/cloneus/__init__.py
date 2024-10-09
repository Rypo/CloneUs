from dotenv import load_dotenv as _load_dotenv
from .types import cpaths
from .utils.logging_config import setup_logging
from .cloneus import Cloneus

_load_dotenv(cpaths.ROOT_DIR/'.env')
setup_logging()