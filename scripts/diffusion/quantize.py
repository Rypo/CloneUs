import gc
import time
import argparse
from pathlib import Path
import torch; torch.backends.cuda.matmul.allow_tf32 = True

from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import T5EncoderModel, BitsAndBytesConfig, QuantoConfig, AutoModelForTextEncoding
from optimum import quanto

from cloneus import cpaths


class QuantizedFluxTransformer2DModel(quanto.QuantizedDiffusersModel):
    base_class = FluxTransformer2DModel

class QuantizedModelForTextEncoding(quanto.QuantizedTransformersModel):
    auto_class = AutoModelForTextEncoding


def quant_and_save(qtype:str, quants_dirpath:Path, skip:list[str] = None, delete_old: bool = True, model_id:str="black-forest-labs/FLUX.1-schnell"):
    flux_variant = 'schnell' if 'schnell' in model_id.lower() else 'dev'

    text_enc_model_id = "black-forest-labs/FLUX.1-schnell"
    # text encoder is identical for schnell/dev, avoid downloading twice
    
    if skip is None:
        skip = []
    
    skip_options = {'te2':'text_encoder_2', 'transformer':'transformer'}
    skip = [skip_options[s] for s in skip]
    
    quant_options={'i4':'qint4', 'i8':'qint8', 'f8':'qfloat8'}
    weight_qtype = quant_options[qtype]
    
    quant_base_dir = quants_dirpath / weight_qtype  # extras/quantized/flux/{qint4,qint8,qfloat8}

    quant_variant_dir = quant_base_dir / flux_variant # extras/quantized/flux/{qint4,qint8,qfloat8}/{schnell,dev}

    text_encoder2_out_dir = quant_base_dir / 'text_encoder_2' # extras/quantized/flux/{qint4,qint8,qfloat8}/text_encoder_2
    transformer_out_dir = quant_variant_dir / 'transformer' # extras/quantized/flux/{qint4,qint8,qfloat8}/{schnell,dev}/transformer

    text_encoder2_out_dir.mkdir(parents=True, exist_ok=True)
    transformer_out_dir.mkdir(parents=True, exist_ok=True)

    text_encoder_exclude = None
    transformer_exclude = None
    
    if weight_qtype == 'qint4':
        transformer_exclude = ["proj_out", "x_embedder", "norm_out", "context_embedder"]
        
    
    print('Quantized location:', quant_base_dir)

    # subjectively, it seems like pre-cast to bf16 benefits qint, but is determental to qfloat.
    # However, for qint4, pre-cast means the first call of .to(cuda) will take an exorbonate amount of time.
    # it seems like it is just requantizing. It's likely that qint4 is not intended to be used like this and quanto is correcting it by just quantizing again 
    # this is further supported by the fact that you'll get a same dype error (got BFloat16 and Float) once you try to run inference 
    # dtype_kwarg = {} if weight_qtype == 'qfloat8' else {'torch_dtype': torch.bfloat16}
    dtype_kwarg = {} if weight_qtype != 'qint8' else {'torch_dtype': torch.bfloat16}
    
    if 'text_encoder_2' not in skip:
        if delete_old:
            for file in text_encoder2_out_dir.iterdir():
                file.unlink()
        
        text_encoder = T5EncoderModel.from_pretrained(text_enc_model_id, subfolder="text_encoder_2", use_safetensors=True, **dtype_kwarg)#.to(0, dtype=torch.bfloat16)
        
        print('Quantizing Text Encoder...')
        quanto.quantize(text_encoder, weights=weight_qtype, exclude=text_encoder_exclude)
        quanto.freeze(text_encoder)
        
        
        print(f'Saving Text Encoder... ({text_encoder.dtype})')
        text_encoder = QuantizedModelForTextEncoding(text_encoder) 
        text_encoder.save_pretrained(text_encoder2_out_dir)
        print('Quantized text_encoder saved:', text_encoder2_out_dir.as_posix())

        del text_encoder
        gc.collect()
        torch.cuda.empty_cache()
    
    if 'transformer' not in skip:
        if delete_old:
            for file in transformer_out_dir.iterdir():
                file.unlink()

        transformer = FluxTransformer2DModel.from_pretrained(model_id, subfolder="transformer", use_safetensors=True, **dtype_kwarg)#, torch_dtype=torch.bfloat16) # low_cpu_mem_usage=True,
        
        print('Quantizing Transformer...')
        quanto.quantize(transformer, weights=weight_qtype, exclude=transformer_exclude)
        quanto.freeze(transformer)
        
        print(f'Saving Transformer... ({transformer.dtype})')
        transformer = QuantizedFluxTransformer2DModel(transformer)
        transformer.save_pretrained(transformer_out_dir)
        print('Quantized transformer saved:', transformer_out_dir.as_posix())

def get_cli_args():
    parser = argparse.ArgumentParser(description='Quantize Flux', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-q','--qtype', choices=['i4', 'i8', 'f8'], default='i8', help='quant type. i4 (qint4), i8 (qint8), or f8 (qfloat8)')
    parser.add_argument('-m','--model', choices=['dev', 'schnell'], default='schnell', help='huggingface model variant. Only Flux supported')
    parser.add_argument('-x','--exclude', choices=['te2', 'transformer'], nargs='*', help='skip quantization for these model components. Any existing will not be removed')

    parser.add_argument('--quants-dir', type=Path, default=None, help='base out directory for quantized files. Defaults to PROJECT_ROOT/extras/quantized/flux')
    parser.add_argument('--keep-old', action='store_true', help='do not delete previous quants before overwriting')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_cli_args()
    
    quant_path = args.quants_dir
    model_alias={
        "schnell":"black-forest-labs/FLUX.1-schnell",
        "dev": "black-forest-labs/FLUX.1-dev"
    }
    
    model_id = model_alias[args.model]

    if quant_path is None:
        quant_path = cpaths.ROOT_DIR / f'extras/quantized/flux/'
        
        # flux/{qint4,qint8,qfloat8}/text_encoder_2/
        # flux/{qint4,qint8,qfloat8}/dev/transformer/
        # flux/{qint4,qint8,qfloat8}/schnell/transformer/

    # USAGE:    
    # python scripts/diffusion/quantize.py -q i8 -m dev -x te2

    quant_and_save(args.qtype, quants_dirpath=quant_path, skip=args.exclude, delete_old=(not args.keep_old), model_id=model_id)
    print('Quantized files saved:',quant_path.as_posix())