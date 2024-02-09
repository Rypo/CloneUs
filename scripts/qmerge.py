import gc
import os
import re
import json
import random
import datetime
import argparse
from pathlib import Path

from tqdm.auto import tqdm
from omegaconf import OmegaConf

import torch
import datasets
import tokenizers
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, AwqConfig

from cloneus.inference import load

# https://colab.research.google.com/drive/1HzZH89yAXJaZgwJDhQj9LqSBux932BvY#scrollTo=KE8xjwlL8DnA
def awq_quant(merged_dirpath, quant_config, awq_outdir='awq'):
    quant_outpath = (Path(merged_dirpath)/awq_outdir).as_posix()


    # Load model
    model = AutoAWQForCausalLM.from_pretrained(merged_dirpath, safetensors=True, device_map='cuda:0')#, low_memory=True)
    tokenizer = AutoTokenizer.from_pretrained(merged_dirpath, trust_remote_code=True)

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config)#quant_config)
    # the pretrained transformers model is stored in the model attribute + we need to pass a dict
    quantization_config = AwqConfig(
        bits=quant_config["w_bit"],
        group_size=quant_config["q_group_size"],
        zero_point=quant_config["zero_point"],
        version=quant_config["version"].lower(),
    ).to_dict()
    model.model.config.quantization_config = quantization_config
    # Save quantized model
    model.save_quantized(quant_outpath, safetensors=True)
    tokenizer.save_pretrained(quant_outpath)


def get_parser():
    parser = argparse.ArgumentParser(description='Merge a trained model checkpoint.')
    parser.add_argument('checkpoint_path', metavar='PATH', type=str, help='path/to/runs/model/.../checkpoint-xxxx')
    parser.add_argument('--force-remerge',  default=False, action='store_true', help='remerge the weights even if a merge subdirectory exists')
    parser.add_argument('--awq', default=False, action='store_true', help='quantize merged model with AWQ')
    parser.add_argument('--mergedir', type=str, default='merged', help='subdir name to put merged files in')
    parser.add_argument('--awqdir', type=str, default='awq', help='subdir name to put awq quantized files in')
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    ckpt_dir = Path(args.checkpoint_path)
    #print(args.mergedir,ckpt_dir.as_posix(), args.mergedir not in ckpt_dir.as_posix())
    do_merge = ((not ckpt_dir.joinpath(args.mergedir).exists()) or args.force_remerge)
    do_awq = args.awq

    if do_merge:
        print('merging...')
        mergepath = load.load_merge_save(ckpt_dir, args.mergedir)
    if do_awq:
        print('awqing...')
        merge_path = ckpt_dir.joinpath(args.mergedir)
        quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4 , "version":"GEMM"}
        awq_quant(merge_path, quant_config, awq_outdir=args.awqdir)