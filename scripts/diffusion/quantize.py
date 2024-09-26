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

def release_memory():
    gc.collect()
    torch.cuda.empty_cache()

def quant_and_save(qtype:str, quants_dirpath:Path, skip:list[str] = None, delete_old: bool = True, model_id:str="black-forest-labs/FLUX.1-schnell"):
    flux_variant = 'schnell' if 'schnell' in model_id.lower() else 'dev'

    text_enc_model_id = "black-forest-labs/FLUX.1-dev"
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
    
    print('Quantized location:', quant_base_dir)

    text_encoder_exclude = None
    transformer_exclude = None
    
    if weight_qtype == 'qint4':
        transformer_exclude = ["proj_out", "x_embedder", "norm_out", "context_embedder"]
    

    # subjectively, it seems like pre-cast to bf16 benefits qint, but is determental to qfloat. That said, source weights are bf16.
    # I can't see how arbitrarily upcasting to fp32 would make the quantization better. We wouldn't be adding any useful information.
    # For qint4, however, pre-cast means the first call of .to(cuda) will take an exorbonate amount of time (~240s).
    # It's likely that qint4 is not intended to be used like this and quanto is correcting by just running quantization again. 
    # This is further supported by the fact that you'll get a same dype error (got BFloat16 and Float) once you try to run inference 
    # Attempting to cast i4 to bf16 at any point post-quantization yield "ValueError: The dtype of a QBitsTensor cannot be changed" 
    # - https://github.com/huggingface/optimum-quanto/blob/601dc193ce0ed381c479fde54a81ba546bdf64d1/optimum/quanto/tensor/qbits/qbits_ops.py#L52
    torch_dtype = torch.bfloat16
    vram_reqs = (10, 20) #gb required to fit model in gpu for faster quantization

    #if weight_qtype == 'qint4':
    #   torch_dtype = torch.float32
    #   vram_reqs = (20, 40)
    
    memfree,memtotal = torch.cuda.mem_get_info()
    vram_gb = round(memtotal / (1024**3))
    
    te2_device = 'cuda' if vram_gb > vram_reqs[0] else None
    transformer_device = 'cuda' if vram_gb > vram_reqs[1] else None
    

    if 'text_encoder_2' not in skip:
        if delete_old:
            for file in text_encoder2_out_dir.iterdir():
                file.unlink()
        
        text_encoder = T5EncoderModel.from_pretrained(text_enc_model_id, subfolder="text_encoder_2", use_safetensors=True, torch_dtype=torch_dtype).to(te2_device)
        
        print('Quantizing Text Encoder using {}...'.format('cuda' if te2_device else 'cpu'))
        quanto.quantize(text_encoder, weights=weight_qtype, exclude=text_encoder_exclude)
        quanto.freeze(text_encoder)
        release_memory()

        print(f'Saving Text Encoder... ({text_encoder.dtype})')
        text_encoder = QuantizedModelForTextEncoding(text_encoder) 
        text_encoder.save_pretrained(text_encoder2_out_dir)
        print('Quantized text_encoder saved:', text_encoder2_out_dir.as_posix())

        del text_encoder
        release_memory()
    
    if 'transformer' not in skip:
        if delete_old:
            for file in transformer_out_dir.iterdir():
                file.unlink()

        transformer = FluxTransformer2DModel.from_pretrained(model_id, subfolder="transformer", use_safetensors=True, torch_dtype=torch_dtype).to(transformer_device)
        
        print('Quantizing Transformer using {}...'.format('cuda' if transformer_device else 'cpu'))
        quanto.quantize(transformer, weights=weight_qtype, exclude=transformer_exclude)
        quanto.freeze(transformer)
        release_memory()
        
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
    # NOTE: After discovering how fast quantization is on cuda, they only reasons you should bother saving to disk are
    # if you have < 20 gb of vram or you want to delete the bf16 weights after quantization to save disk space.
    # Otherwise, it's barely faster to load from disk vs quantize on the fly. Save ~ 10 seconds (20 vs 30)
    # Additionally, qint4 will cause headaches saving to disk. Life is easier with live quantization.

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