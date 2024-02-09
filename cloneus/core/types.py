import typing
from collections.abc import Sequence


# trunc_input_text, out_text, input_len, output_len
GenerationOutput = tuple[str, str, int, int]

# trunc_context, author_prompts, out_texts, input_len, output_lens
BatchedGenerationOutput = tuple[str, list[str], list[str], int, list[int]]

