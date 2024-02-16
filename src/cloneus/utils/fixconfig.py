import re
import json
from pathlib import Path


import matplotlib.pyplot as plt
import torch
import peft
import transformers
from omegaconf import OmegaConf

# from ..config import settings as rsettings
from cloneus.core import paths as cpaths
from cloneus.data import roles




def print_loss(tstate):
    end_train_loss=tstate['log_history'][-2].get('loss')
    end_eval_loss=tstate['log_history'][-1].get('eval_loss')
    print(f'{end_train_loss=}, {end_eval_loss=}')
    return end_train_loss, end_eval_loss

def plot_lr(tstate):
    fig,ax=plt.subplots(figsize=(5,3))
    ax.set(title='LR Plot',xlabel='logging step',ylabel='learning rate')
    ax.plot(list(filter(None,[l.get('learning_rate') for l in tstate['log_history']])))
    return print_loss(tstate)

def empty_cfg(cfg):
    for k,v in cfg.items():
        if not OmegaConf.is_config(v):
            cfg[k] = None
        elif OmegaConf.is_dict(v):
            cfg[k] = empty_cfg(v)
        elif OmegaConf.is_list(v):
            cfg[k] = []
    return cfg

def get_blankcfg(full=True):
    fcfg = OmegaConf.load(cpaths.ROOT_DIR/'config'/'train_config.yaml')
    full_config = empty_cfg(fcfg.copy())
    
    if full:
        return full_config
    
    partial_config = OmegaConf.create({'ctx_len':None, 
                                       **OmegaConf.masked_copy(fcfg, ['attn_implementation','tag_sep','postfix','author_tag','has_custom_tokens','dtype','prompt','notes',]), 
                                       'base_dir':None})
    return partial_config

def read_trainerstate(model_path, return_full=False):
    try:
        with open((Path(model_path)/'trainer_state.json'),'r') as f:
            tstate = json.load(f)
    except FileNotFoundError:
        with open((Path(model_path).parent/'trainer_state.json'),'r') as f:
            tstate = json.load(f)
    if return_full:
        return tstate

    return {k: tstate[k] for k in ['epoch','num_train_epochs','log_history']}

def parse_modelpath(model_path):
    # assumes a ../runs/full/{model_nick}/{dataset}/{composite_name} structure
    model_path = str(model_path)
    parts = model_path.split('/')
    model_nick = parts[3]
    dataset=parts[4]
    print(parts[3:])
    hours_between_sessions=-1

    if (hrs_bsess:=re.search(r'chunk(\d+)h', dataset)) is not None:
        hours_between_sessions = [int(h) for h in hrs_bsess.group(1)]
        if len(hours_between_sessions)==1:
            hours_between_sessions=hours_between_sessions[0]
    
    
    composite = parts[5]
    comparts = composite.split('-')
    
    if (custom_tok:=re.search(r'ctk(\d+)',model_path)) is not None:
        custom_tok=custom_tok.group(1)
    
    chunk_size = int(re.search(r'(\d+)',comparts[0]).group(1))
    scheduler = comparts[1]

    custom_scheduler = ('warmup_const_cosine' if 'warmup_const_cosine' in model_path else None)
    return {'model_nick':model_nick,'dataset':dataset,'chunk_size':chunk_size, 'hours_between_sessions':hours_between_sessions,
            'custom_tokens':custom_tok,'scheduler':scheduler,'custom_scheduler':custom_scheduler,}
            #'warmup_ratio':warmup_ratio,'warmup_steps':warmup_steps}

def create_configs(model_path, manual_kwargs=None):
    assert 'checkpoint' in str(model_path),'need a checkpoint to go off of'
    
    fconfig = get_blankcfg(True)
    config = get_blankcfg(False)
    
    tstate = read_trainerstate(model_path, False)
    
    train_loss, eval_loss = print_loss(tstate)
    
    ta = torch.load(Path(model_path)/'training_args.bin')
    mp_stats = parse_modelpath(model_path)
    
    #pconf=model.peft_config['default']
    pconf = peft.PeftConfig.from_pretrained(model_path)
    model_id = pconf.base_model_name_or_path
    model_architecture = 'LlamaForCausalLM' if 'llama' in pconf.base_model_name_or_path.lower() else 'MistralForCausalLM'

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    print(ta.output_dir.replace('runs/full/',''))

    base_dir = str(Path(model_path).resolve().relative_to(cpaths.RUNS_DIR)).split('/checkpoint')[0]
    mbase_path = cpaths.RUNS_DIR/base_dir

    fullconfig_path = mbase_path/'full_config.yaml'
    config_path = mbase_path/'config.yaml'
    
    if manual_kwargs is None:
        if fullconfig_path.exists():
            manual_kwargs = OmegaConf.load(fullconfig_path)
        elif config_path.exists():
            manual_kwargs = OmegaConf.load(config_path)
        else:
            manual_kwargs = {}

    if manual_kwargs.get('prompt') is None:
        manual_kwargs['prompt'] = config.prompt

    fconfig.update({
        # Manual Entry
        'flashattn_lib': manual_kwargs.get('flashattn_lib',None), 
        'attn_implementation': manual_kwargs.get('attn_implementation', None), 
        'use_sft_trainer': manual_kwargs.get('use_sft_trainer', None), 
        #'hours_between_sessions': manual_kwargs['hours_between_sessions'],
        'author_tag': manual_kwargs.get('author_tag','???'), 
        'tag_sep': manual_kwargs.get('tag_sep','???'), 
        'postfix': manual_kwargs.get('postfix','???'), 
        'prompt': manual_kwargs.get('prompt', config.prompt),
        'notes': manual_kwargs.get('notes', None), 
        # End Manual Entry
        'train_loss':train_loss,
        'eval_loss':eval_loss,
        'sample_output': manual_kwargs.get('sample_output',None),
        'model_id': model_id, #model.config._name_or_path, 
        'model_type': model_architecture, #model.config.architectures[0], 
        'instruct_model': (tokenizer.chat_template is not None), 
        'padding_side': tokenizer.padding_side, 
        'chunk_size': mp_stats['chunk_size'], 
        'batch_size': ta.train_batch_size, 
        'gradient_accumulation_steps': ta.gradient_accumulation_steps, 
        'group_by_length': ta.group_by_length, 
        'num_epochs': ta.num_train_epochs, 
        'learning_rate': ta.learning_rate, 
        'lr_scheduler': ta.lr_scheduler_type.value, 
        'custom_scheduler': mp_stats['custom_scheduler'], 
        'dataset': {'name': mp_stats['dataset'], 
                    'train_jsonl': None, 
                    'eval_jsonl': None, 
                    'hours_between_sessions': mp_stats['hours_between_sessions'], 
                    'min_session_length': manual_kwargs.get('min_session_length', None)}, 
        'lora_r': pconf.r, 
        'lora_alpha': pconf.lora_alpha, 
        'lora_dropout': pconf.lora_dropout, 
        'lora_target_linear': None, 
        'lora_fan_in_fan_out': pconf.fan_in_fan_out, 
        'lora_target_modules': list(pconf.target_modules), 
        'optimizer': ta.optim.value, 
        'weight_decay': ta.weight_decay, 
        'warmup_steps': ta.warmup_steps, 
        'warmup_ratio': ta.warmup_ratio, 
        'max_grad_norm': ta.max_grad_norm, 
        'neftune_noise_alpha': ta.neftune_noise_alpha, 
        'gradient_checkpointing': ta.gradient_checkpointing, 
        'bf16': ta.bf16, 
        'fp16': ta.fp16, 
        'tf32': ta.tf32, 
        'special_tokens': {
            'bos_token': tokenizer.bos_token, 
            'eos_token': tokenizer.eos_token, 
            'unk_token': tokenizer.unk_token, 
            'pad_token': tokenizer.pad_token
        }, 
        'pad_vocab_to': mp_stats['custom_tokens'], 
        'custom_tokens': mp_stats['custom_tokens'], 
        'resume_from_checkpoint': manual_kwargs.get('resume_from_checkpoint', None),
    })
    
    config.update({
        'ctx_len': fconfig.chunk_size,
        'author_tag': manual_kwargs.get('author_tag','???'), 
        'tag_sep': manual_kwargs.get('tag_sep','???'), 
        'postfix': manual_kwargs.get('postfix','???'), 
        'has_custom_tokens': fconfig.pad_vocab_to is not None,
        'attn_implementation': fconfig.attn_implementation,
        'dtype': 'bfloat16',
        'prompt': fconfig.prompt,
        'fprompt': (fconfig.prompt.template.format(name_mapping=fconfig.prompt.name_mapping, task=fconfig.prompt.task) if fconfig.prompt.template is not None else None),
        'notes': fconfig.notes,
        'base_dir':base_dir,
    })
    
    
    if (fc_path := mbase_path/'full_config.yaml').exists():
        fc_path.rename(mbase_path/'orig_full_config.yaml')
    
    if OmegaConf.missing_keys(fconfig):
        fullconfig_path = fullconfig_path.with_name('MISSING_full_config.yaml')
    

    
    if (c_path := mbase_path/'config.yaml').exists():
        c_path.rename(mbase_path/'orig_config.yaml')
    
    if OmegaConf.missing_keys(config):
        config_path = config_path.with_name('MISSING_config.yaml')
    
    OmegaConf.save(fconfig, fullconfig_path)
    print('wrote to:',fc_path)

    OmegaConf.save(config, config_path)
    print('wrote to:',config_path)
        
    return fconfig, config


def human_name(conf):
    model_hn = dict(zip(
        ['llama2-13b-i4','llama2-13b-gptq','llama2-7b-i4','mistral-7b-i4','mistral-inst-v01-7b-i4','mistral-inst-v02-7b-i4','mistral-inst-OpenHermes2.5'], 
        ['llama13', 'llama13g', 'llama7', 'mister7', 'mister7i1', 'mister7i2', 'mrhermesi']))
    chunk_hn = dict(zip([512, 1024, 2048, 4096, 8192, 16384],['vvshort', 'vshort', 'short', 'long', 'vlong', 'vvlong']))
    rank_hn = dict(zip([4, 8, 16, 32, 64, 128, 256],['vvsmall', 'vsmall', 'small', 'med', 'large', 'vlarge', 'vvlarge']))
    targ_hn = dict(zip(['qv', 'qvok', 'qvokudg', 'qvokudgLhEt'], ['few', 'some', 'most', 'all']))
    ar_hn = dict(zip([1/8, 1/4, 1/2, 1, 2, 4, 8], ['vvlight','vlight', 'light', 'balanced', 'heavy', 'vheavy', 'vvheavy']))