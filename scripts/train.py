import gc
import os

from omegaconf import OmegaConf
from dotenv import load_dotenv
import torch
from transformers import GPTQConfig,BitsAndBytesConfig
from peft import LoraConfig, LoftQConfig
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
    
    if cfg.flashattn_lib == 'unsloth': 
        if cfg.quant_method != 'bnb4':
            raise ValueError('for flashattn_lib=unsloth, only quant_method=bnb4 is supported')
        if cfg.lora_use_dora:
            raise ValueError('Unsloth does not support DoRA training. Set lora_use_dora: false to use unsloth')
        if cfg.lora_target_modules == 'all-linear':
            print('Unsloth does not support target_modules=all-linear, setting to default: qkvogud')
            cfg.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            

def main():
    # Decent example for HfArgumentParser
    # https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py
    
    cfg = OmegaConf.load(cpaths.ROOT_DIR/'config'/'train_config.yaml')
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
        'NousResearch/Nous-Hermes-2-Mistral-7B-DPO': 'mistral-7b-hermes2-dpo' # chatml
        # Add aliases for new models here
    }
    # https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B
    model_name = model_map.get(cfg.model_id, cfg.model_id.split('/')[-1])  
    
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

    if cfg.resume_from_checkpoint is not None:
        peft_config = LoraConfig.from_pretrained(cfg.resume_from_checkpoint)
        peft_config.inference_mode = False
    
        
    if cfg.dataset.name == 'chunkh': # append hours_between_session
        hbs = cfg.dataset.hours_between_sessions
        cfg.dataset.name += str(hbs) if isinstance(hbs, int) else ''.join(map(str,hbs))
    
    base_outdir = cpaths.RUNS_DIR/f'{model_name}/{cfg.dataset.name}'
    args = mtrain.create_args(
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
        num_train_epochs=cfg.num_epochs,
        warmup_ratio=cfg.warmup_ratio,
        warmup_steps=cfg.warmup_steps,
        learning_rate=cfg.learning_rate,
        save_strategy=('epoch' if isinstance(cfg.dataset.hours_between_sessions, int) else 'steps'), #-- TODO Think about better solution
        save_steps=cfg.save_steps,
        #eval_steps=cfg.eval_steps,
        logging_steps=(5 if cfg.use_sft_trainer else cfg.logging_steps), 
        
        optim=cfg.optimizer,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler, 
        neftune_noise_alpha=cfg.neftune_noise_alpha,
        group_by_length=(cfg.group_by_length and not cfg.use_sft_trainer),
        weight_decay=cfg.weight_decay,
        custom_scheduler=cfg.custom_scheduler,
        logging_first_step=True,
        #disable_tqdm = cfg.use_sft_trainer # honestly, even it's wrong, still nice to have
        #torch_compile=True,
    )
    
    model, tokenizer = mllm.model_tokenizer_from_config(peft_config, cfg)
    

    cfg.ctx_len = cfg.chunk_size
    cfg.has_custom_tokens=(num_custom_tokens is not None and num_custom_tokens > 0)
    cfg.dtype = 'bfloat16' if cfg.bf16 else 'float16'
    cfg.fprompt = None
    cfg.base_dir = args.output_dir.replace(str(cpaths.ROOT_DIR/'runs/full/'),'').strip('/')

    if cfg.instruct_model:
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
    elif cfg.tune_type in ['chatml', 'chat']:
        dset = dataset.author_role_dataset(data_file_path, tokenizer, cfg)
    elif cfg.tune_type == 'instruct':
        dset = dataset.instruct_dataset_timechunks(data_file_path, tokenizer, cfg, has_system=None)    
    else:
        dset = dataset.dataset_timechunk(data_file_path, tokenizer, cfg, text_only=False)
    
    
    
    if cfg.resume_from_checkpoint is not None:
        load_model_safetensors(model, os.path.join(cfg.resume_from_checkpoint, 'model.safetensors'))

    #assert not peft_config.inference_mode and not model.config.use_cache and model.training, "Peft is not training or cache enabled"
    #model = model.to(torch.bfloat16)#.to_bettertransformer()
    
    callbacks = [] # [GenerationCallback(20), FullSaveCallback]
    if cfg.use_sft_trainer:
        trainer = mtrain.get_sft_trainer(model, dset, tokenizer, args, peft_config, callbacks=callbacks, max_packed_seqlength=cfg.chunk_size)
    else:
        trainer = mtrain.get_trainer(model, dset, tokenizer, args, callbacks=callbacks)


    write_first_batches(trainer, batchsample_dir='_tmp')

    if num_custom_tokens:
        tokenization.save_embeddings(model, args.output_dir)

    safe_train(trainer, cfg.resume_from_checkpoint)
    
    try:
        cfg.update(train_loss=trainer.state.log_history[-2].get('train_loss'), eval_loss=trainer.state.log_history[-1].get('eval_loss'))
    except IndexError as e:
        print(e)
    
    OmegaConf.save(cfg, os.path.join(args.output_dir, 'config.yaml'))


    if args.save_strategy == 'steps':
        mtrain.save_last_step(trainer)

    return args


if __name__ == "__main__":
    load_dotenv()
    args = main()