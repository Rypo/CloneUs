import gc
import os
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AwqConfig
from peft import AutoPeftModelForCausalLM
import awq
from awq import AutoAWQForCausalLM
from awq.models.base import BaseAWQForCausalLM
from unsloth import FastLanguageModel

from cloneus.inference import load


@torch.inference_mode()
def awq_from_checkpoint(model_path, awq_outpath=None):
    if awq_outpath is None:
        (Path(model_path)/'awq').mkdir(exist_ok=True)
        awq_outpath = (Path(model_path)/'awq')
    
    awq_outpath = str(awq_outpath)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        #quantization_config=quantization_config,
        device_map="auto",
        torch_dtype='auto',
        attn_implementation="flash_attention_2",
        use_cache=True
    )
    model = model.merge_and_unload()
    model.half()
    #model.save_pretrained(outdir)
    #tokenizer.save_pretrained(outdir)
    torch.cuda.empty_cache()
    gc.collect()
    quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4 , "version":"gemm"} # "gemm", "gemv", "marlin", "gemv_fast"
    quantization_config = AwqConfig(
        bits=quant_config["w_bit"],
        group_size=quant_config["q_group_size"],
        zero_point=quant_config["zero_point"],
        version=quant_config["version"].lower(),
    )
    model_type = model.config.to_dict()['model_type']
    
    AnyAWQForCausalLM = awq.models.auto.AWQ_CAUSAL_LM_MODEL_MAP[model_type]
    model: BaseAWQForCausalLM = AnyAWQForCausalLM(
        model=model, 
        model_type=model_type, 
        is_quantized=False, 
        config=model.config, 
        quant_config=quantization_config, 
        processor=None
    )
    
    # model = AutoAWQForCausalLM.from_pretrained(
    #     model_path, 
    #     low_cpu_mem_usage=True, 
    #     torch_dtype='auto', 
    #     device_map='auto', 
    #     use_cache=False, 
    #     attn_implementation='flash_attention_2'
    # )
    # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # quantization_config = AwqConfig(
    #         bits=quant_config["w_bit"],
    #         group_size=quant_config["q_group_size"],
    #         zero_point=quant_config["zero_point"],
    #         version=quant_config["version"].lower(),
    #         exllama_config={"version":2, "max_input_len": 8192, "max_batch_size": 8}
    # )
    
    model.quantize(tokenizer, quant_config=quant_config)

    model.save_quantized(awq_outpath)
    tokenizer.save_pretrained(awq_outpath)
    print('Saved to:',awq_outpath)


# https://colab.research.google.com/drive/1HzZH89yAXJaZgwJDhQj9LqSBux932BvY#scrollTo=KE8xjwlL8DnA
@torch.inference_mode()
def awq_from_merged(merged_dirpath, quant_config, awq_outpath=None):
    merged_dirpath = Path(merged_dirpath)
    if awq_outpath is None:
        ckpt_name = merged_dirpath.name.replace('-merged','')
        awq_outpath = merged_dirpath.with_name(ckpt_name+'-awq')
    
    awq_outpath = str(awq_outpath)

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(merged_dirpath, safetensors=True, device_map='cuda:0', low_cpu_mem_usage=True, 
                                               use_cache=False, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2') # torch_dtype='auto',
    tokenizer = AutoTokenizer.from_pretrained(merged_dirpath, trust_remote_code=True)

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config)
    # the pretrained transformers model is stored in the model attribute + we need to pass a dict
    quantization_config = AwqConfig(
        bits=quant_config["w_bit"],
        group_size=quant_config["q_group_size"],
        zero_point=quant_config["zero_point"],
        version=quant_config["version"].lower(),
        # exllama_config={"version":2, "max_input_len": 8192, "max_batch_size": 8}
    ).to_dict()
    # model.model.config.quantization_config = quantization_config # should be automatic now
    # Save quantized model
    model.save_quantized(awq_outpath, safetensors=True)
    tokenizer.save_pretrained(awq_outpath)

@torch.inference_mode()
def unsloth_save_gguf(checkpoint_dirpath:str, quantization_method="q4_k_m", gguf_outpath=None):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dirpath)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = checkpoint_dirpath,
        max_seq_length = tokenizer.model_max_length,
        dtype = None,
        load_in_4bit = True,
    )
    #from unsloth.save import save_to_gguf, unsloth_save_pretrained_gguf
    checkpoint_dirpath = Path(checkpoint_dirpath)
    
    if gguf_outpath is None:
        gguf_outpath = checkpoint_dirpath.with_name(checkpoint_dirpath.name+'-gguf')
    
    model.save_pretrained_gguf(gguf_outpath, tokenizer, quantization_method = quantization_method)
    return outpath

def get_parser():
    parser = argparse.ArgumentParser(description='Quantize or merge a trained model checkpoint.')
    parser.add_argument('checkpoint_path', metavar='PATH', type=str, 
                        help='path/to/runs/model/.../checkpoint-xxxx')
    parser.add_argument('--merge', default=False, action='store_true', 
                        help='merge peft model and save the weights')
    parser.add_argument('--awq', default=False, action='store_true', 
                        help='quantize a model with AWQ')
    parser.add_argument('--gguf', metavar='QUANT_METHOD', type=str, default=False, nargs='?',
                        help="quantize a model with unsloth's gguf utility")
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    ckpt_path = Path(args.checkpoint_path)
    
    merge_path = ckpt_path.with_name(ckpt_path.name+'-merged')
    awq_path   = ckpt_path.with_name(ckpt_path.name+'-awq')
    gguf_path  = ckpt_path.with_name(ckpt_path.name+'-gguf')

    has_merged = merge_path.exists() and any(merge_path.iterdir())
    
    if args.gguf is None: # empty flag passed
        args.gguf = 'q4_k_m'

    if args.gguf: # no flag = False
        print('GGUF Quantization Method:',args.gguf)
        outpath = unsloth_save_gguf(ckpt_path, quantization_method=args.gguf, gguf_outpath=gguf_path)

    if args.merge:
        mergepath = load.load_merge_save(ckpt_path, args.mergedir)
        has_merged = True
    
    if args.awq:
        if has_merged:
            print('awqing...')
            quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4 , "version":"GEMM"}
            awq_from_merged(merge_path, quant_config, awq_outdir=awq_path)
        else:
            awq_from_checkpoint(ckpt_path, awq_outpath = awq_path)

    if not any([args.merge, args.awq, args.gguf]):
        print('No flags set. No action taken.')
