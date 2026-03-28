import gc
import os
import json
import logging
import warnings
import argparse
from omegaconf import OmegaConf
from dotenv import load_dotenv

import torch
import unsloth # Unsloth should be imported before trl, transformers, peft to ensure all optimizations are applied.
from peft import LoraConfig
import transformers

import cloneus.training.model as mllm
import cloneus.training.trainer as mtrain

from cloneus.data import dataset, tokenization, useridx
from cloneus.types import cpaths


logger = logging.getLogger('scripts')

def safe_train(trainer:mtrain.Trainer, checkpoint_path=None):
    torch.cuda.empty_cache()
    gc.collect()

    try:
        trainer.train(checkpoint_path)
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
        if trainer.state.global_step:
            mtrain.create_resumable_save(trainer)

def write_first_batches(trainer:mtrain.Trainer, batchsample_dir='_tmp/sample_batches'):
    sample_path = (cpaths.ROOT_DIR/batchsample_dir)
    sample_path.mkdir(parents=True, exist_ok=True)
    
    with open(sample_path/'eval_batch.txt', 'w') as f:
        try:
            f.writelines(mtrain.get_batch(trainer, train=False))
        except Exception as e:
            f.write(str(e))
    
    with open(sample_path/'eval_batch_masked.txt', 'w') as f:
        try:
            f.writelines(mtrain.get_batch(trainer, train=False, masked=True))
        except Exception as e:
            f.write(str(e))

    with open(sample_path/'train_batch.txt', 'w') as f:
        try:
            f.writelines(mtrain.get_batch(trainer, train=True))
        except Exception as e:
            f.write(str(e))

    with open(sample_path/'train_batch_masked.txt', 'w') as f:
        try:
            f.writelines(mtrain.get_batch(trainer, train=True, masked=True))
        except Exception as e:
            f.write(str(e))

    with open(sample_path/'train_item_masked.txt', 'w') as f:
        try:
            tokenizer = trainer.processing_class
            masked_sample = tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[0]["labels"]]).replace(tokenizer.pad_token, " ")
            f.write(masked_sample)
        except Exception as e: # KeyError expected if if no labels
            f.write(str(e))

def model_id_alias(model_id:str, alias_filepath:str=None):
    '''If `alias_filepath` exists, read and map `model_id` to user defined alias, otherwise return portion of model_id after `/`
    
    Args:
        model_id: huggingface model id repo/model-id-format
        alias_filepath: path to model aliases json file, if None default ROOT/config/model_aliases.json
    '''
    model_map = {}
    if alias_filepath is None:
        alias_filepath = cpaths.ROOT_DIR/'config'/'model_aliases.json'
    try:
        with open(alias_filepath, 'r') as f:
            model_map = json.load(f)
    except FileNotFoundError:
        pass
    
    return model_map.get(model_id, model_id.split('/')[-1])

def verify_config(cfg):
    if 'fname' in cfg.author_tag:
        # Check that all users have assigned firstName if using "fname" in tag
        for dispname, fname in useridx.get_users('fname', by='dname').items():
            if fname is None:
                raise KeyError(f'users.json missing firstName for "{dispname}". Add firstName for all users or remove "fname" from `author_tag` in train_config.yaml')
    
    if cfg.tag_placement == 'replace_role':
        if any([cfg.tag_sep is not None, cfg.postfix is not None]):
            logger.warning("tag_placement == 'replace_role' does not use postfix/tag_sep -- postfix, tag_sep will be set to None")
            cfg.tag_sep = None 
            cfg.postfix = None 
    elif cfg.tag_placement == 'content_prefix':
        if cfg.postfix is not None:
            logger.warning("tag_placement == 'content_prefix' does not use postfix -- postfix will be set to None")
            cfg.postfix = None
        if cfg.tag_sep is None:
            logger.warning("tag_placement == 'content_prefix' uses tag_sep but found None. Setting to a single space ' '. To avoid this behavior, set to an empty string `tag_sep=''` ")
            cfg.tag_sep = ' '
    elif cfg.tag_placement == 'tag_only':
        if any([cfg.author_tag is None, cfg.tag_sep is None, cfg.postfix is None]):
            raise KeyError("For tag_placement == 'tag_only', each of {author_tag, tag_sep, postfix} must be explicitly set in config.")


    if cfg.flashattn_lib == 'unsloth': 
        if cfg.lora_use_dora:
            raise ValueError('Unsloth does not support DoRA training. Set lora_use_dora: false to use unsloth')
            
    if cfg.dataset.name != 'max_tokens':
        from cloneus.data.etl import has_time_column
        if not has_time_column(cfg.dataset.chatlog_csv):
            warnings.warn(f'Warning: Neither "Date" or "timestamp" found in csv columns. dataset.name {cfg.dataset.name} requires date column. Config will be updated: dataset.name="max_tokens"')
            cfg.dataset.name = 'max_tokens'

def main(args):
    # Decent example for HfArgumentParser
    # https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py
    config_filepath = args.config if args.config else cpaths.ROOT_DIR/'config'/'train'/'train_config.yaml'
    cfg = OmegaConf.load(config_filepath)
    
    if args.datapath:
        rel_dpath = args.datapath #os.path.relpath(args.datapath, cpaths.ROOT_DIR)
        cfg.dataset.chatlog_csv = rel_dpath
        print('Training with data:', rel_dpath)

    
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
        logger.info(f'Resuming training from: {resume_ckpt}')
        cfg.resume_from_checkpoint = os.path.relpath(resume_ckpt, cpaths.ROOT_DIR)
    
    verify_config(cfg)

    #https://huggingface.co/RLHFlow/LLaMA3-iterative-DPO-final
    model_name = model_id_alias(cfg.model_id)
    
    if resume_ckpt:
        peft_config = LoraConfig.from_pretrained(resume_ckpt)
        peft_config.inference_mode = False
    else:
        peft_config = mtrain.read_peft_config(cfg)
    
    # This does nothing currently.
    num_custom_tokens = tokenization.apply_special_tokens(tokenizer=None, custom_tokens=cfg.custom_tokens, pad_vocab_to=cfg.pad_vocab_to)

        
    if cfg.dataset.name == 'chunkh': # append hours_between_session
        hbs = cfg.dataset.hours_between_sessions
        cfg.dataset.name += str(hbs) if isinstance(hbs, int) else ''.join(map(str,hbs))
    
    base_outdir = cpaths.RUNS_DIR/f'{model_name}/{cfg.dataset.name}'
    train_args = mtrain.create_args(
        base_outdir,
        peft_config,
        cfg,
        n_custom_tokens=num_custom_tokens,
        report_to="wandb",
        save_only_model = False, # if True (default False), can't resume training from checkpoint. Doesn't store the optimizer, scheduler & rng state. Must use from_pretrained if True
        resume_from_checkpoint=resume_ckpt,
    )

    model, tokenizer = mllm.model_tokenizer_from_config(peft_config, cfg)

    # if (special_token_overrides := dict(filter(lambda kv: kv[1], cfg.special_tokens.items()))):
    #     num_new_vocab = tokenizer.add_special_tokens(special_token_overrides)
    #     assert num_new_vocab==0, 'Using non-vocab special tokens is not currently supported.'
    
    # Config Autofill
    cfg.model_capabilities = ['text']

    # TODO: better way to check for this
    is_processor = hasattr(tokenizer, 'tokenizer')
    if 'tool_call' in tokenizer.chat_template:
        cfg.model_capabilities += ['tools']
    if hasattr(tokenizer,'image_processor'):
        cfg.model_capabilities += ['image']
    if hasattr(tokenizer,'video_processor'):
        cfg.model_capabilities += ['video']

    cfg.special_tokens = tokenizer.tokenizer.special_tokens_map if is_processor else tokenizer.special_tokens_map
    cfg.model_architecture = model.config.architectures[0]
    cfg.ctx_len = cfg.chunk_size
    cfg.has_custom_tokens=(num_custom_tokens is not None and num_custom_tokens > 0)
    cfg.dtype = 'bfloat16' if cfg.bf16 else 'float16'
    cfg.prompt.name_mapping = None # filled during dataset creation
    cfg.fprompt = None # filled during dataset creation
    cfg.base_dir = train_args.output_dir.replace(str(cpaths.ROOT_DIR/'runs/full/'),'').strip('/')

    data_file_path = cpaths.ROOT_DIR/cfg.dataset.chatlog_csv

    df_chat = dataset.prepare_dataset_dataframe(data_file_path, cfg)
    cfg = dataset.fill_cfg_from_data(df_chat['formatted_author_tag'], cfg) # fill fprompt, name_mappings, name_mappings_json

    dataset_format  = ('tokens' if is_processor else 'text') 
            
    if cfg.dataset.train_jsonl and cfg.dataset.eval_jsonl:
        dset = dataset.jsonl_dataset(cfg.train_jsonl, cfg.eval_jsonl, tokenizer, cfg, dataset_format=dataset_format) 
    elif cfg.dataset.name == 'max_tokens':
        dset = dataset.max_tokens_dataset(df_chat, tokenizer, cfg, dataset_format=dataset_format)
    elif cfg.dataset.name == 'ungrouped':
        dset = dataset.ungrouped_dataset(df_chat, tokenizer, cfg, dataset_format=dataset_format)
    elif 'chunkh' in cfg.dataset.name:
        if cfg.tag_placement == 'content_prefix_ot':
            logger.info('Using Data Format: subsession_completions_dataset')
            # convert dataset to UA One Turn format 
            dset = dataset.subsession_completions_dataset(df_chat, tokenizer, cfg, tag_sep=cfg.tag_sep, ctx_template = '{role}{tag_sep}{content}', ctx_sep = '\n', dataset_format='text')
            
        else:
            dset = dataset.chat_sessions_dataset(df_chat, tokenizer, cfg, dataset_format=dataset_format)
    else:
        raise NotImplementedError(f'Unknown dataset format: {cfg.dataset.name!r}')
    
    trainer = mtrain.get_trainer(model, dset, tokenizer, train_args, peft_config=peft_config, callbacks=[]) # [GenerationCallback(20), FullSaveCallback]
    
    role_tag_template = cfg.get('mask_roletag_template')

    if role_tag_template and isinstance(trainer.data_collator, transformers.DataCollatorForLanguageModeling):
        logger.error('Trainer using vanilla Transformers DataCollatorForLM. Labels will be ignored and roletag masking will not be applied.')
    
    
    # trainer = unsloth.chat_templates.train_on_responses_only(
    #     trainer,
    #     instruction_part = "<|im_start|>system\n",
    #     response_part = "<|im_start|>assistant\n",
    # )
    # apply masking to packed dataset, unsloth should automatically disable packing if is_processor
    if (cfg.use_sft_trainer and trainer.args.packing and role_tag_template):
        assert not is_processor, 'Packing have been auto-disabled for multi-modal models, something is wrong, dataset may already be masked. Investigate.'
        trainer.train_dataset = dataset.apply_role_mask_tokens(trainer.train_dataset, trainer.processing_class, cfg, num_proc = 32)
        
        if trainer.eval_dataset:
            trainer.eval_dataset = dataset.apply_role_mask_tokens(trainer.eval_dataset, trainer.processing_class, cfg, num_proc = 16)


    write_first_batches(trainer, batchsample_dir='_tmp/sample_batches')


    safe_train(trainer, resume_ckpt)
    
    try:
        penult_state = trainer.state.log_history[-2]
        train_loss = penult_state.get('loss', penult_state.get('train_loss'))
        cfg.update(train_loss=train_loss, eval_loss=trainer.state.log_history[-1].get('eval_loss'))
    except IndexError as e:
        logger.error('Failed to add loss information to config', exc_info=e)
    
    OmegaConf.save(cfg, os.path.join(train_args.output_dir, 'config.yaml'))


    if train_args.save_strategy == 'steps':
        mtrain.save_last_step(trainer)

    return train_args

def get_cli_args():
    parser = argparse.ArgumentParser(description='Finetune an LLM on your chat export data. Send in the clones.')
    parser.add_argument('-c','--config', default=None, type=str, required=False,
                        help='Path/to/config_file.yaml. If not set, will use file at ./config/train/train_config.yaml')
    
    parser.add_argument('-d','--datapath', default=None, type=str, required=False,
                        help='Path/to/chat.csv. If set, will take precedence over dataset.chatlog_csv in config.yaml.')
    
    parser.add_argument('-r','--resume', default=None, type=str, required=False,
                        help='Path/to/checkpoint-dir to resume training from. If set, --config will be ignored and the local config.yaml file will be used.')
    
    return parser.parse_args()

if __name__ == "__main__":
    load_dotenv()
    args = get_cli_args()
    train_args = main(args)