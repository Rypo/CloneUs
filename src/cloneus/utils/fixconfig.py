import re
import json
from pathlib import Path

import torch
import peft
import transformers
from omegaconf import OmegaConf, DictConfig


from cloneus.core import paths as cpaths



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

def read_training_args(model_path):
    return torch.load(Path(model_path)/'training_args.bin')

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


def dump_configs(fullcfg, cfg, base_dirpath:Path):
    prev_cfg_dir = (base_dirpath/'prev_configs')
    prev_cfg_dir.mkdir(exist_ok=True)
    
    fullcfg_path = base_dirpath/'full_config.yaml'
    cfg_path = base_dirpath/'config.yaml'
        
    if fullcfg_path.exists():
        fullcfg_path.rename(prev_cfg_dir/'orig_full_config.yaml')
    
    if OmegaConf.missing_keys(fullcfg):
        fullcfg_path = fullcfg_path.with_name('MISSING_full_config.yaml')
    
    if cfg_path.exists():
        cfg_path.rename(prev_cfg_dir/'orig_config.yaml')
    
    if OmegaConf.missing_keys(cfg):
        cfg_path = cfg_path.with_name('MISSING_config.yaml')
    

    OmegaConf.save(fullcfg, fullcfg_path)
    print('wrote to:', fullcfg_path)

    OmegaConf.save(cfg, cfg_path)
    print('wrote to:',cfg_path)

def create_configs(model_path, manual_kwargs=None):
    assert 'checkpoint' in str(model_path),'need a checkpoint to go off of'
    
    fconfig = get_blankcfg(True)
    config = get_blankcfg(False)
    
    tstate = read_trainerstate(model_path, False)
    
    train_loss= tstate.log_history[-2].get('loss', tstate.get('train_loss'))
    eval_loss = tstate.log_history[-1].get('eval_loss')
    
    ta = read_training_args(model_path)
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
    
    dump_configs(fconfig, config, mbase_path)
        
    return fconfig, config

def human_name(conf):
    model_hn = dict(zip(
        ['llama2-13b-i4','llama2-13b-gptq','llama2-7b-i4','mistral-7b-i4','mistral-inst-v01-7b-i4','mistral-inst-v02-7b-i4','mistral-inst-OpenHermes2.5'], 
        ['llama13', 'llama13g', 'llama7', 'mister7', 'mister7i1', 'mister7i2', 'mrhermesi']))
    chunk_hn = dict(zip([512, 1024, 2048, 4096, 8192, 16384],['vvshort', 'vshort', 'short', 'long', 'vlong', 'vvlong']))
    rank_hn = dict(zip([4, 8, 16, 32, 64, 128, 256],['vvsmall', 'vsmall', 'small', 'med', 'large', 'vlarge', 'vvlarge']))
    targ_hn = dict(zip(['qv', 'qvok', 'qvokudg', 'qvokudgLhEt'], ['few', 'some', 'most', 'all']))
    ar_hn = dict(zip([1/8, 1/4, 1/2, 1, 2, 4, 8], ['vvlight','vlight', 'light', 'balanced', 'heavy', 'vheavy', 'vvheavy']))


def to_dot_keys(cfg, base_name = '', dot_key_list=None):
    if dot_key_list is None:
        dot_key_list = set()
    
    for k,v in cfg.items():
        name = base_name+'.'+k if base_name else k
        if OmegaConf.is_dict(v):
            dot_key_list |= to_dot_keys(cfg[k], name, dot_key_list)
        else:
            dot_key_list.add(name)

    return dot_key_list

def find_all_missing(reference_cfg, cfg_paths:list[str|Path]):
    reference_keys = to_dot_keys(reference_cfg)
    miss_list = []
    for cfg_pth in cfg_paths:
        miss_keys = reference_keys - to_dot_keys(OmegaConf.load(cfg_pth))
        miss_list.append({'path':cfg_pth, 'missing':miss_keys})

    return miss_list


def get_toplevel_cfgpaths():
    return [p for p in cpaths.RUNS_DIR.rglob('config.yaml') if p.parent.name != 'prev_configs' and any(p.parent.glob('*checkpoint*'))]




def backup_and_update(keyfix_map: dict[str, callable[DictConfig]]):
    '''Fills missing keys in config.yaml, saves and backup to prev_configs. Does not work for dot keys yet. 
    
    The function it maps to must take cfg as the only argument amd return the value for that key.

    ex:
        {'tag_placement': determine_tag_placement, 
        'chat_template_format': determine_chat_template_format}
    
    Args:
        keyfix_map: dict (key:func) where keys are keys you want in all configs and func takes a cfg and returns a value for that key
    '''
    import datetime
    fix_keys = keyfix_map.keys() # ['tag_placement', 'chat_template_format']
    all_top_level_configs = get_toplevel_cfgpaths()

    for cfg_path in all_top_level_configs:
        prev_cfg_dir = (cfg_path.parent/'prev_configs')
        prev_cfg_dir.mkdir(exist_ok=True)
        date = datetime.datetime.now().strftime('%Y%m%d')
        cfg = OmegaConf.load(cfg_path)
        
        print('cfg_path:', cfg_path)
        backup_path = prev_cfg_dir/f'config_{date}.yaml'
        
        was_updated = False
        for key in fix_keys:
            if key not in cfg:
                cfg[key] = cfg.setdefault(key, keyfix_map[key](cfg))
                was_updated = True

        if was_updated:
            cfg_path.rename(backup_path)
            OmegaConf.save(cfg, cfg_path)
            
            print('Updated. backup_path:', backup_path)    
        else:
            print('ALL KEY PRESENT')
        
        print()


def determine_chat_template_format(cfg):
    current_value = OmegaConf.select(cfg, 'chat_template_format')
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
    base_chat_template = tokenizer.chat_template or ""
    custom_chat_template = cfg.custom_chat_template or ""
    
    base_chat_ml = '<|im_start|>' in base_chat_template
    custom_chat_ml = '<|im_start|>' in custom_chat_template
    print(f'{current_value=}, {base_chat_ml=}, {custom_chat_ml=}')
    
    if current_value:
        return current_value
        
    if custom_chat_ml and not base_chat_ml:
        chat_template_format = 'chatml'
    else:
        chat_template_format = None
    
    return chat_template_format
    
def determine_tag_placement(cfg):
    current_value = OmegaConf.select(cfg, 'tag_placement')
    has_custom_chat_template = cfg.custom_chat_template is not None

    if has_custom_chat_template:
        tag_placement = 'replace_role'
    elif cfg.base_tune_type == 'foundation':
        tag_placement = 'tag_only'
    elif cfg.base_tune_type in ['instruct','chat']:
        tag_placement = 'content_prefix'

    print(f'{tag_placement} \t {cfg.postfix!r} \t {cfg.tag_sep!r} \t {cfg.author_tag!r}')
    
    if current_value:
        return current_value

    return tag_placement