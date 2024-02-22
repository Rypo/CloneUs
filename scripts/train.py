import gc
import os

from omegaconf import OmegaConf
from dotenv import load_dotenv
import torch
from transformers import GPTQConfig,BitsAndBytesConfig
from peft import LoraConfig
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

def main():
    # Decent example for HfArgumentParser
    # https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py
    
    cfg = OmegaConf.load(cpaths.ROOT_DIR/'config'/'train_config.yaml')
    
    custom_token_map = tokenization.author_special_tokens(cfg.custom_tokens, pad_vocab_to=cfg.pad_vocab_to) if cfg.custom_tokens else None

    model_map = {
        'NousResearch/Llama-2-7b-hf':'llama2-7b-i4', 
        'mistralai/Mistral-7B-v0.1':'mistral-7b-i4',
        'mistralai/Mistral-7B-Instruct-v0.1':'mistral-inst-v01-7b-i4',
        'mistralai/Mistral-7B-Instruct-v0.2':'mistral-inst-v02-7b-i4',
        'teknium/OpenHermes-2.5-Mistral-7B':'mistral-inst-OpenHermes2.5',
        'NousResearch/Llama-2-13b-hf':'llama2-13b-i4', 
        'TheBloke/Llama-2-13B-GPTQ':'llama2-13b-gptq',
        'TinyLlama/TinyLlama-1.1B-Chat-v1.0':'tinyllama1b-chat-v1',
        'NousResearch/Nous-Hermes-2-SOLAR-10.7B':'solar-10b-inst-hermes2',
        # Add aliases for new models here
    }
    
    model_name = model_map.get(cfg.model_id, cfg.model_id.split('/')[-1])  
    base_outdir = cpaths.RUNS_DIR/f'{model_name}/{cfg.dataset.name}'
    
    if 'gptq' in model_name:
        # https://huggingface.co/docs/optimum/llm_quantization/usage_guides/quantization
        quant_config = GPTQConfig(bits=4, use_exllama=False) # , use_cuda_fp16=True
    else: 
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    peft_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=list(cfg.lora_target_modules),
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode = False,
    )
    if custom_token_map: 
        # this should now be handled automatically as of peft 0.8.0
        # https://github.com/huggingface/peft/releases/tag/v0.8.0
        for modu in ["embed_tokens", "lm_head"]:
            if modu not in peft_config.target_modules:
                peft_config.target_modules.add(modu)
            #if modu not in peft_config.modules_to_save: peft_config.modules_to_save.append(modu)
        #peft_config.modules_to_save = ["embed_tokens", "lm_head"]
    
    if cfg.resume_from_checkpoint is not None:
        peft_config = LoraConfig.from_pretrained(cfg.resume_from_checkpoint)
        peft_config.inference_mode = False


    tokenizer = tokenization.get_tokenizer(cfg.model_id, padding_side=cfg.padding_side)
    
    num_custom_tokens = tokenizer.add_special_tokens(custom_token_map) if custom_token_map else None

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
        save_strategy=('epoch' if isinstance(cfg.dataset.hours_between_sessions, int) else 'steps'),
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
        #disable_tqdm = cfg.use_sft_trainer # honestly, even it's wrong, still nice to have
        #torch_compile=True,
    )

    verify_config(cfg)

        
    if cfg.dataset.name == 'chunkh': # append hours_between_session
        hbs = cfg.dataset.hours_between_sessions
        cfg.dataset.name += str(hbs) if isinstance(hbs, int) else ''.join(map(str,hbs))
        
    cfg.ctx_len = cfg.chunk_size
    cfg.has_custom_tokens=(custom_token_map is not None)
    cfg.dtype = 'bfloat16' if cfg.bf16 else 'float16'
    cfg.fprompt = None
    cfg.base_dir = args.output_dir.replace(str(cpaths.ROOT_DIR/'runs/full/'),'').strip('/')

    data_file_path = cpaths.ROOT_DIR/cfg.dataset.chatlog_csv

    if cfg.dataset.train_jsonl and cfg.dataset.eval_jsonl:
        dset = dataset.dataset_tc_files(tokenizer, maxlen=4096, train_jsonl=cfg.train_jsonl, eval_jsonl=cfg.eval_jsonl) 
    elif cfg.dataset.name == 'ungrouped_eos':
        dset = dataset.dataset_ungrouped(data_file_path, tokenizer, cfg, text_only=False)
    elif cfg.dataset.name == 'chunk_maxlen':
        dset = dataset.dataset_all_chunks(data_file_path, tokenizer, cfg)

    elif cfg.instruct_model:
        name_mapping = ', '.join(roles.format_author_tag(author, cfg.author_tag) for author in roles.author_display_names)
        cfg.prompt.name_mapping = name_mapping
        fprompt = cfg.prompt.template.format(name_mapping=name_mapping, task=cfg.prompt.task)
        cfg.fprompt = fprompt


        if cfg.custom_chat_template:
            dset = dataset.author_role_dataset(data_file_path, tokenizer, cfg, cfg.custom_chat_template)
        else:
            dset = dataset.instruct_dataset_timechunks(data_file_path, tokenizer, cfg, has_system=None)
            
    else:
        dset = dataset.dataset_timechunk(data_file_path, tokenizer, cfg, text_only=False)
    
    
    
    callbacks = [] # [GenerationCallback(20), FullSaveCallback]
    
    if cfg.flashattn_lib=='huggingface':
        model,peft_config = mllm.get_model(cfg.model_id, peft_config, quant_config, tokenizer, 
                                           custom_tokens_map=custom_token_map, attn_implementation=cfg.attn_implementation, lora_target_linear=cfg.lora_target_linear)
        
    elif cfg.flashattn_lib=='unsloth':
        
        print('before:',peft_config)
        model, tokenizer = mllm.get_unsloth(cfg.model_id, peft_config, max_seq_length=cfg.chunk_size)
        if cfg.padding_side and cfg.padding_side != tokenizer.padding_side:
            print(f'WARNING. Unsloth padding_side ({tokenizer.padding_side}) != config padding_side ({cfg.padding_side}). Overriding with padding_side={cfg.padding_side}.')
            tokenizer.padding_side = cfg.padding_side
        
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            print('WARNING. PAD = EOS. Overriding.')
            tokenizer.pad_token_id = tokenizer.unk_token_id

        # TODO: look into ~4-7gb higher vRAM usage after changing padding_side=right -> padding_side=left
        # https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side
        
    else:
        print('Defaulting to no flash attention')
        model,peft_config = mllm.get_model(cfg.model_id, peft_config, quant_config, tokenizer, 
                                           custom_tokens_map=custom_token_map, attn_implementation=None, lora_target_linear=cfg.lora_target_linear)
            

    if tokenizer.padding_side != 'left':
        print('WARNING. padding_side = right. You know what you doing?')

    if cfg.custom_chat_template:
        print('Using custom chat template')
        tokenizer.chat_template = cfg.custom_chat_template
    
    if cfg.resume_from_checkpoint is not None:
        load_model_safetensors(model, os.path.join(cfg.resume_from_checkpoint, 'model.safetensors'))

    #assert not peft_config.inference_mode and not model.config.use_cache and model.training, "Peft is not training or cache enabled"
    #model = model.to(torch.bfloat16)#.to_bettertransformer()
    if cfg.use_sft_trainer:
        trainer = mtrain.get_sft_trainer(model, dset, tokenizer, args, peft_config, callbacks=callbacks, max_packed_seqlength=cfg.chunk_size)
    else:
        trainer = mtrain.get_trainer(model, dset, tokenizer, args, callbacks=callbacks)#, collator_pad_multiple=8) # TODO: reenable?


    write_first_batches(trainer, batchsample_dir='_tmp')

    if custom_token_map:
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