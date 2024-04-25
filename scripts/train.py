import gc
import os
import argparse
from omegaconf import OmegaConf
from dotenv import load_dotenv
import torch
from peft import LoraConfig, LoftQConfig, AutoPeftModelForCausalLM
from safetensors.torch import load_model as load_model_safetensors, save_model as save_model_safetensors

import cloneus.training.trainer as mtrain
import cloneus.training.model as mllm
import cloneus.training.evaluation as meval
from cloneus.data import dataset, tokenization, roles
from cloneus.core import paths as cpaths

def safe_train(trainer:mtrain.Trainer, checkpoint_path=None):
    torch.cuda.empty_cache()
    gc.collect()

    try:
        trainer.train(checkpoint_path)
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
        if trainer.state.global_step:
            mtrain.create_resumable_save(trainer)

def write_first_batches(trainer, batchsample_dir='_tmp'):
    (cpaths.ROOT_DIR/batchsample_dir).mkdir(exist_ok=True)
    with open(cpaths.ROOT_DIR/batchsample_dir/'sample_evalbatch.txt', 'w') as f:
        f.writelines(mtrain.get_batch(trainer, train=False))
    with open(cpaths.ROOT_DIR/batchsample_dir/'sample_trainbatch.txt', 'w') as f:
        f.writelines(mtrain.get_batch(trainer, train=True))

def verify_config(cfg):
    if 'fname' in cfg.author_tag:
        # Check that all users have assigned firstName if using "fname" in tag
        for dispname, fname in roles.author_to_fname.items():
            if fname is None:
                raise KeyError(f'users.json missing firstName for "{dispname}". Add firstName for all users or remove "fname" from `author_tag` in train_config.yaml')
    if cfg.chat_template_format == 'chatml' and cfg.tag_sep != '\n':
        print('NOTE: for chat_template_format=chatml, tag_sep must = \\n. Setting tag_sep=\\n')
        cfg.tag_sep = '\n'
    if cfg.flashattn_lib == 'unsloth': 
        if cfg.quant_method != 'bnb4':
            raise ValueError('for flashattn_lib=unsloth, only quant_method=bnb4 is supported')
        if cfg.lora_use_dora:
            raise ValueError('Unsloth does not support DoRA training. Set lora_use_dora: false to use unsloth')
        if cfg.lora_target_modules == 'all-linear':
            print('NOTE: Unsloth does not support target_modules=all-linear, setting to default: qkvogud')
            cfg.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            

def main(args):
    # Decent example for HfArgumentParser
    # https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py
    
    config_filepath = args.config if args.config else cpaths.ROOT_DIR/'config'/'train'/'train_config.yaml'
    cfg = OmegaConf.load(config_filepath)
    
    
    resume_ckpt = None
    # command line resume path takes precedence over config
    if args.resume:
        resume_ckpt = args.resume
        resume_cfgpath = os.path.join(os.path.dirname(args.resume), 'config.yaml')
    elif cfg.resume_from_checkpoint:
        resume_ckpt = cfg.resume_from_checkpoint
        resume_cfgpath = (cpaths.ROOT_DIR/cfg.resume_from_checkpoint).parent/'config.yaml'
    
    if resume_ckpt:
        cfg = OmegaConf.load(resume_cfgpath)
        print(f'Resuming training from: {resume_ckpt}')
        cfg.resume_from_checkpoint = os.path.relpath(resume_ckpt, cpaths.ROOT_DIR)
    
    verify_config(cfg)

    model_map = {
        'NousResearch/Llama-2-7b-hf':'llama2-7b-i4', # foundation
        'mistralai/Mistral-7B-v0.1':'mistral-7b-i4', # foundation
        'mistralai/Mistral-7B-Instruct-v0.1':'mistral-inst-v01-7b-i4', # instruct
        'mistralai/Mistral-7B-Instruct-v0.2':'mistral-inst-v02-7b-i4', # instruct
        'teknium/OpenHermes-2.5-Mistral-7B':'mistral-inst-OpenHermes2.5', # chatml
        'NousResearch/Llama-2-13b-hf':'llama2-13b-i4', # foundation
        'TheBloke/Llama-2-13B-GPTQ':'llama2-13b-gptq', # foundation
        'TinyLlama/TinyLlama-1.1B-Chat-v1.0':'tinyllama1b-chat-v1', # chat (Zephyr)
        'NousResearch/Nous-Hermes-2-SOLAR-10.7B':'solar-10b-inst-hermes2', # chatml
        'ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf':'mixtral-8x7b-aqlm-2bit', # foundation
        'ISTA-DASLab/Mixtral-8x7B-Instruct-v0_1-AQLM-2Bit-1x16-hf': 'mixtral-inst-8x7b-aqlm-2bit', # instruct
        'NousResearch/Nous-Hermes-2-Mistral-7B-DPO': 'mistral-7b-hermes2-dpo', # chatml
        #'solidrust/Nous-Hermes-2-Mistral-7B-DPO-AWQ':'mistral-7b-hermes2-dpo-awq'
        'alpindale/Mistral-7B-v0.2-hf': 'mistral-7b-v2', # foundation
        'unsloth/Hermes-2-Pro-Mistral-7B-bnb-4bit': 'mistral-7b-hermes2-pro-4bit', # chatml (with tools)
        'rhysjones/phi-2-orange-v2':'phi2-orange-v2', # chatml
        'unsloth/llama-3-8b-bnb-4bit': 'llama3-8b-4bit',
        'unsloth/llama-3-8b-Instruct-bnb-4bit': 'llama3-8b-instruct-4bit',
        'meta-llama/Meta-Llama-3-8B': 'llama3-8b',
        'meta-llama/Meta-Llama-3-8B-Instruct': 'llama3-8b-instruct'
        
        
        # Add aliases for new models here
    }

    model_name = model_map.get(cfg.model_id, cfg.model_id.split('/')[-1])  
    
    if resume_ckpt:
        peft_config = LoraConfig.from_pretrained(resume_ckpt)
        peft_config.inference_mode = False
    else:
        target_modules = (cfg.lora_target_modules if isinstance(cfg.lora_target_modules, str) else list(cfg.lora_target_modules)) # all-linear
        peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=target_modules, 
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode = False,
            use_rslora=cfg.lora_use_rslora,
            init_lora_weights=('loftq' if cfg.lora_use_loftq else True),
            #loftq_config=
            use_dora=cfg.lora_use_dora
    )
    
    # This does nothing currently.
    num_custom_tokens = tokenization.apply_special_tokens(tokenizer=None, custom_tokens=cfg.custom_tokens, pad_vocab_to=cfg.pad_vocab_to)

        
    if cfg.dataset.name == 'chunkh': # append hours_between_session
        hbs = cfg.dataset.hours_between_sessions
        cfg.dataset.name += str(hbs) if isinstance(hbs, int) else ''.join(map(str,hbs))
    
    base_outdir = cpaths.RUNS_DIR/f'{model_name}/{cfg.dataset.name}'
    train_args = mtrain.create_args(
        base_outdir,
        peft_config,
        chunk_size=cfg.chunk_size,
        batch_size=cfg.batch_size,
        gradient_accumulation_steps = cfg.gradient_accumulation_steps,
        n_custom_tokens=num_custom_tokens,
        attn_implementation=cfg.attn_implementation,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        tf32=cfg.tf32,
        #bf16_full_eval=cfg.bf16, #  faster and save memory but can harm metric values (default: False)
        num_train_epochs=cfg.num_epochs,
        warmup_ratio=cfg.warmup_ratio,
        warmup_steps=cfg.warmup_steps,
        learning_rate=cfg.learning_rate,
        save_strategy=('epoch' if isinstance(cfg.dataset.hours_between_sessions, int) else 'steps'), #-- TODO Think about better solution
        save_steps=cfg.save_steps,
        #eval_steps=cfg.eval_steps,
        logging_steps=cfg.logging_steps, # appears that trl fixed the iteration number issue (5 if cfg.use_sft_trainer else cfg.logging_steps), 
        
        optim=cfg.optimizer,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler, 
        neftune_noise_alpha=cfg.neftune_noise_alpha,
        group_by_length=(cfg.group_by_length and not cfg.use_sft_trainer),
        weight_decay=cfg.weight_decay,
        custom_scheduler=cfg.custom_scheduler,
        logging_first_step=True,
        #torch_compile=True,
        save_only_model = False, # if True (default False), can't resume training from checkpoint. Doesn't store the optimizer, scheduler & rng state. Must use from_pretrained if True
        resume_from_checkpoint=resume_ckpt,
    )

    model, tokenizer = mllm.model_tokenizer_from_config(peft_config, cfg)
    
    
    cfg.ctx_len = cfg.chunk_size
    cfg.has_custom_tokens=(num_custom_tokens is not None and num_custom_tokens > 0)
    cfg.dtype = 'bfloat16' if cfg.bf16 else 'float16'
    cfg.fprompt = None
    cfg.base_dir = train_args.output_dir.replace(str(cpaths.ROOT_DIR/'runs/full/'),'').strip('/')

    if cfg.prompt.template:
        name_mapping = ', '.join(roles.format_author_tag(author, cfg.author_tag) for author in roles.author_display_names)
        cfg.prompt.name_mapping = name_mapping
        cfg.fprompt = cfg.prompt.template.format(name_mapping=name_mapping, task=cfg.prompt.task)

    data_file_path = cpaths.ROOT_DIR/cfg.dataset.chatlog_csv

    if cfg.dataset.train_jsonl and cfg.dataset.eval_jsonl:
        dset = dataset.dataset_tc_files(tokenizer, maxlen=4096, train_jsonl=cfg.train_jsonl, eval_jsonl=cfg.eval_jsonl) 
    elif cfg.dataset.name == 'ungrouped_eos':
        dset = dataset.dataset_ungrouped(data_file_path, tokenizer, cfg, text_only=False)
    elif cfg.dataset.name == 'chunk_maxlen':
        dset = dataset.dataset_all_chunks(data_file_path, tokenizer, cfg)
    
    elif cfg.tag_placement == 'replace_role':
        dset = dataset.author_roletags_dataset(data_file_path, tokenizer, cfg)
    elif cfg.tag_placement == 'content_prefix':
        dset = dataset.ua_tags_dataset(data_file_path, tokenizer, cfg)    
    else:
        dset = dataset.dataset_timechunk(data_file_path, tokenizer, cfg, text_only=False)
    
    
    callbacks = [] # [GenerationCallback(20), FullSaveCallback]
    if cfg.use_sft_trainer:
        trainer = mtrain.get_sft_trainer(model, dset, tokenizer, train_args, peft_config, callbacks=callbacks, max_packed_seqlength=cfg.chunk_size)
    else:
        trainer = mtrain.get_trainer(model, dset, tokenizer, train_args, callbacks=callbacks)


    write_first_batches(trainer, batchsample_dir='_tmp')

    if num_custom_tokens:
        tokenization.save_embeddings(model, train_args.output_dir)

    safe_train(trainer, resume_ckpt)
    
    try:
        penult_state = trainer.state.log_history[-2]
        train_loss = penult_state.get('loss', penult_state.get('train_loss'))
        cfg.update(train_loss=train_loss, eval_loss=trainer.state.log_history[-1].get('eval_loss'))
    except IndexError as e:
        print(e)
    
    OmegaConf.save(cfg, os.path.join(train_args.output_dir, 'config.yaml'))


    if train_args.save_strategy == 'steps':
        mtrain.save_last_step(trainer)

    return train_args

def get_cli_args():
    parser = argparse.ArgumentParser(description='Finetune an LLM on your Discord chat export data.')
    parser.add_argument('-c','--config', default=None, type=str, required=False,
                        help='Path/to/config_file.yaml. If not set, will use file at ./config/train/train_config.yaml')
    
    parser.add_argument('-r','--resume', default=None, type=str, required=False,
                        help='Path/to/checkpoint-dir to resume training from. If set, --config will be ignored and the local config.yaml file will be used.')
    
    return parser.parse_args()

if __name__ == "__main__":
    load_dotenv()
    args = get_cli_args()
    train_args = main(args)