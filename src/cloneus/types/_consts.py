from pathlib import Path as _Path
from ._types import CloneusPaths as _CloneusPaths


__ROOT_DIR = _Path(__file__).parent.parent.parent.parent

cpaths = _CloneusPaths(
    ROOT_DIR = __ROOT_DIR, 
    RUNS_DIR = __ROOT_DIR / 'runs/full', 
    DATA_DIR = __ROOT_DIR / 'data'
)


__all__ = ['cpaths']