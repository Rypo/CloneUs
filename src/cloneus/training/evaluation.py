import os
import json
import random
import typing
import itertools
from pathlib import Path
from contextlib import contextmanager

import numpy as np
from tqdm.auto import tqdm
import torch

from cloneus.data import useridx
from cloneus import Cloneus



def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_test_questions(questions_filepath:str|Path, n:int|None = 1, best_only=False):
    with open(questions_filepath, 'r') as f:
        test_questions = f.read().splitlines()
    test_qs = [q.strip('*').replace('\\n','\n') for q in test_questions if (q.endswith('*') or not best_only)]
    if n is None:
        return test_qs
    random.shuffle(test_qs)
    return test_qs[:n]


def mock_msgcache(*amsgs):
    '''Format <initial>:<message>
    e.g. a:what's up everyone, b:nm you?,...'''
    mcache = []
    i_to_dname = useridx.get_users('dname', by='initial')
    for amsg in amsgs:
        a,msg = amsg.split(':',1)
        auth = i_to_dname[a]
        mcache.append((auth,msg))
    return mcache


def textborder(center_text, sym='-', out_n=64, cent_n=0):
    bpad=int(out_n>0)
    cpad=int(cent_n>0)
    top_border = (sym*out_n) + ('\n'*bpad)
    midleft_border = (sym*cent_n) + (' '*cpad)
    midright_border = (' '*cpad) + (sym*cent_n) + '\n'
    bot_border = (sym*out_n) + ('\n'*bpad)
    outstr = top_border + midleft_border + center_text + midright_border + bot_border
    return outstr

def write_samples(gen_samples:dict[str, dict[str, list[str]]], out_filepath:str|Path):
    with open(out_filepath, 'w') as f:
        for gc, in_outs in gen_samples.items():
            f.write(gc)
            for inp,outs in in_outs.items():    
                f.write(inp)
                f.writelines(outs)

def eval_ckpt(clo:Cloneus, checkpoint_path:Path, genconfig_modes:list[str], prompts:list[str], question_author:str, author_list:list[str]):
    '''Evaluate a single checkpoint in run'''
    
    clo.swap_model(checkpoint_path)
    checkpoint_name = checkpoint_path.name
    gc_inps_outs = {} # {gmode1 : {input_text: [out1, out2, ... ]}}
    
    # use defaults from old genconfig.get_config func for consistency (note the temp=1 instead of 0.8)
    compat_kw = {'ms':dict(do_sample=True, top_p=1, temperature=1, top_k=50),
                 'cs':dict(penalty_alpha=0.6, top_k=4, low_memory=False,)}
    
    for gmode in genconfig_modes:
        _delta= clo.set_genconfig(False, preset=gmode, **compat_kw.get(gmode, {})) #= genconfig.get_config(clo.tokenizer, gmode)
        gconf_line = textborder(gmode.upper() +': '+ json.dumps(clo.get_genconfig(True)), '-', 176, 0)
        inps_outs = gc_inps_outs.setdefault(gconf_line, {})
        
        for prompt in tqdm(prompts, leave=False):
            seed_everything(42)

            input_text, author_prompts, out_texts, _, _ = clo.batch_generate([(question_author, prompt)], author_list, '', return_tuple=True)
            outputs='\n'.join([atag+'⋙'+repr(text) for atag,text in zip(author_prompts,out_texts)])
            
            input_line = textborder('INPUT: '+ repr(input_text), '=', 88, 0)
            ckpt_output = f'CHECKPOINT: {checkpoint_name}\n{outputs}\n'
            
            inps_outs.setdefault(input_line, []).append(ckpt_output)
    return gc_inps_outs

def eval_run(checkpoint_paths:list[Path], prompts: list[str], genconfig_modes:list[str], question_author:str, author_list:list[str]):
    '''Evaluate a single run consisting of multiple checkpoints'''
    
    clo = Cloneus.from_pretrained(checkpoint_paths[0], load=True)
    
    gc_inps_outs = {}
    for ckpth in tqdm(checkpoint_paths): 
        ckpt_samples = eval_ckpt(clo, ckpth, genconfig_modes=genconfig_modes, prompts=prompts, question_author=question_author, author_list=author_list)
        
        for gm,in_outs in ckpt_samples.items():
            for inp,outs in in_outs.items():
                gc_inps_outs.setdefault(gm,{}).setdefault(inp, []).extend(outs)
    
    return gc_inps_outs
    

def sample_trained(runs_path, prompts: list[str], outfile='test_samples.log', genconfig_modes:list[str]=None, question_author:str = None, response_authors:list[str]|typing.Literal['rest','all']='rest'):
    if genconfig_modes is None:
        genconfig_modes = ['cs','ms']
    
    author_display_names = useridx.get_users('dname')

    if question_author is None:
        question_author = author_display_names[0]
    
    if isinstance(response_authors, list):
        author_list = response_authors
    elif response_authors == 'rest':
        author_list = [a for a in author_display_names if a!=question_author]
    elif response_authors == 'all':
        author_list = author_display_names
    
    runs_path = Path(runs_path)

    #if (runs_path/'config.yaml').exists():
        # sample eval from 1 run, multiple checkpoints
    
    config_paths = list(runs_path.rglob('*config.yaml*'))

    if config_paths:
        for config_path in config_paths:
            run_path = config_path.parent
            checkpoint_paths = sorted(run_path.glob('*checkpoint*'), key=lambda p: int(p.name.split('-')[1])) # checkpoint-123-awq -> sortby=int(123)
            
            gc_inps_outs = eval_run(checkpoint_paths=checkpoint_paths, prompts=prompts, genconfig_modes=genconfig_modes, question_author=question_author, author_list=author_list)
            write_samples(gc_inps_outs, run_path/outfile)
    elif any(runs_path.glob('*.safetensors')):
        run_path = runs_path
        print('running 1 checkpoint:',run_path)
        # sample eval from 1 checkpoint. Write to log file inside of the checkpoint dir
        clo = Cloneus.from_pretrained(run_path, load=False)#.load_model() # load taken care of in eval_ckpt
        gc_inps_outs = eval_ckpt(clo, checkpoint_path=run_path, genconfig_modes=genconfig_modes, prompts=prompts, question_author=question_author, author_list=author_list)
        write_samples(gc_inps_outs, run_path/outfile)
    else:
        raise FileNotFoundError('Unable to find any models to evaluate')


def eval_params(model_path, param_grid, prompts:list[str], outfile='test_params.log', question_author:str = None, response_authors:list[str]|typing.Literal['rest','all']='rest'):
    if (cpts:=list(Path(model_path).glob('*checkpoint*'))):
        model_path = cpts[0]
    
    clo = Cloneus.from_pretrained(model_path, load=True)
    
    author_display_names = useridx.get_users('dname')

    if question_author is None:
        question_author = author_display_names[0]
    
    if response_authors == 'rest':
        author_list = [a for a in author_display_names if a!=question_author]
    elif response_authors == 'all':
        author_list = author_display_names
    elif isinstance(response_authors, list):
        author_list = response_authors

    spgrid = dict(sorted(param_grid.items(), key=lambda kv: len(kv[1])))
    param_sets = [dict(zip(spgrid, pvals)) for pvals in itertools.product(*spgrid.values())]
    print(f'Evaluating {len(param_sets)} parameter combinations')
    
    outfile = clo.path_data.run_path/outfile
    with open(outfile, 'w') as f:
        for prompt in prompts:
            input_text, _, _, _, _ = clo.batch_generate([(question_author, prompt)], author_list, '', return_tuple=True)
            f.write(textborder('INPUT: '+ repr(input_text), '=', 88, 0))
            for pset in param_sets:
                f.write('\n'+ json.dumps(pset))
                delt=clo.set_genconfig(save_on_change=False, **pset)
                seed_everything(42)
            
                input_text, author_prompts, out_texts, _, _ = clo.batch_generate([(question_author, prompt)], author_list, '', return_tuple=True)
                outputs='\n'.join([repr(atag+'⋙'+text) for atag,text in zip(author_prompts,out_texts)])
                f.write('\n'+outputs+'\n')