import typing
from pathlib import Path
from collections.abc import Sequence


# trunc_input_text, out_text, input_len, output_len
#GenerationOutput = tuple[str, str, int, int]

# trunc_context, author_prompts, out_texts, input_len, output_lens
#BatchedGenerationOutput = tuple[str, list[str], list[str], int, list[int]]

class GenerationOutput(typing.NamedTuple):
    input_text: str
    output_text: str
    input_len: int
    output_len: int


class BatchGenerationOutput(typing.NamedTuple):
    input_context: str 
    author_prompts: list[str]
    output_texts: list[str]
    input_len: int 
    output_lens: list[int]


class CloneusPaths(typing.NamedTuple):
    ROOT_DIR: Path
    RUNS_DIR: Path
    DATA_DIR: Path



__all__ = ['GenerationOutput', 'BatchGenerationOutput', 'CloneusPaths']