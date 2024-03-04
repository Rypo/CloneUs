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
import transformers
from transformers import GenerationConfig

from cloneus.core import paths as cpaths
from cloneus.data import roles
from cloneus.inference import genconfig
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

def apply_template(author, text_content, author_tag, tag_sep, postfix):
    atag=author_tag.format(author=author, lauthor=author.lower(), fname=roles.author_to_fname[author])
    return f'{atag}{tag_sep}{text_content}{postfix}'

def mock_msgcache(*amsgs):
    '''Format <initial>:<message>
    e.g. a:what's up everyone, b:nm you?,...'''
    mcache = []
    for amsg in amsgs:
        a,msg = amsg.split(':',1)
        auth = roles.initial_to_author[a]
        mcache.append((auth,msg))
    return mcache

@torch.inference_mode()
def generate(model:transformers.PreTrainedModel, tokenizer:transformers.PreTrainedTokenizer, gen_config:GenerationConfig, seed_text='', stop_criteria=None):
    model.eval()
    input_text = seed_text
    inputs = tokenizer(input_text, return_tensors="pt", return_length=True)
    input_len = inputs.pop('length')[0].item()
    output = model.generate(**inputs.to(0), generation_config=gen_config, stopping_criteria=stop_criteria).detach()
    out_tokens = output[0,input_len:]
    output_len = out_tokens.shape[0]
    out_text = tokenizer.decode(out_tokens, skip_special_tokens=False)

    return input_text, out_text, input_len, output_len

#@torch.no_grad()
@torch.inference_mode()
def test_model(model, tokenizer, genconfs, seed_text='', do_print=True):
    """While LoRA is significantly smaller and faster to train, you may encounter latency issues during inference 
    due to separately loading the base model and the LoRA model. To eliminate latency, use the merge_and_unload() 
    function to merge the adapter weights with the base model which allows you to effectively use the newly merged model as a standalone model."""
    was_training = model.training
    model.eval()

    inputs = tokenizer(seed_text, return_tensors="pt", return_length=True)
    input_len = inputs.pop('length')[0].item()

    genconfs = [genconfs] if not isinstance(genconfs, list) else genconfs
   
    outputs = [model.generate(**inputs.to(0), generation_config=gconf, stopping_criteria=None).detach() for gconf in genconfs]
    
    # get input tokens again because generate() may have changed formatting
    
    in_text = tokenizer.decode(outputs[0][:input_len], skip_special_tokens=False)
    out_texts = tokenizer.batch_decode([out[0, input_len:] for out in outputs], skip_special_tokens=False)
    #in_text,out_text = text[:textlen], text[textlen:]
    
    if do_print:
        sep = '\n'+'-'*64+'\n'
        print(f"Input text:\n{in_text}\n[[[🤖]]]\n{sep.join(out_texts)}")
    
    #model.config.use_cache = False
    model.train(was_training)
    
    return in_text,out_texts


    
def textborder(center_text, sym='-', out_n=64, cent_n=0):
    bpad=int(out_n>0)
    cpad=int(cent_n>0)
    top_border = (sym*out_n) + ('\n'*bpad)
    midleft_border = (sym*cent_n) + (' '*cpad)
    midright_border = (' '*cpad) + (sym*cent_n) + '\n'
    bot_border = (sym*out_n) + ('\n'*bpad)
    outstr = top_border + midleft_border + center_text + midright_border + bot_border
    return outstr


def eval_model(model_path, questions_filepath, outfile='test_samples.log', gmodes:list[str]=None, question_author:str = None, response_authors:list[str]|typing.Literal['rest','all']='rest'):
    if (cpts:=list(Path(model_path).glob('*checkpoint*'))):
        model_path = cpts[0]

    clo = Cloneus(model_path)
    clo.load_model()
    checkpoints = sorted([p.name for p in clo.mdir_comps.basedir_path.glob('*checkpoint*')], key=lambda t: int(t.split('-')[1]))
    
    test_qs = get_test_questions(questions_filepath, None, True)
    
    if gmodes is None:
        gmodes = ['cs','ms']
        
    if question_author is None:
        question_author = roles.author_display_names[0]
    
    if response_authors == 'rest':
        author_list = [a for a in roles.author_display_names if a!=question_author]
    elif response_authors == 'all':
        author_list = roles.author_display_names
    elif isinstance(response_authors, list):
        author_list = response_authors
        
    
    outfile = clo.mdir_comps.basedir_path/outfile
    with open(outfile, 'w') as f:
        gc_inps_outs = {}
        for c in tqdm(checkpoints): 
            clo.switch_model(clo.mdir_comps.basedir_path, c)

            for gmode in gmodes:
                clo.gen_config = genconfig.get_config(clo.tokenizer, gmode)
                gconf_line = textborder(gmode.upper() +': '+ json.dumps(clo.get_genconf(True)), '-', 176, 0)
                inps_outs = gc_inps_outs.setdefault(gconf_line, {})
                
                for tq in tqdm(test_qs, leave=False):
                    seed_everything(42)

                    input_text, author_prompts, out_texts, _, _ = clo.batch_generate([(question_author, tq)], author_list, '')
                    outputs='\n'.join([atag+'⋙'+repr(text) for atag,text in zip(author_prompts,out_texts)])
                    
                    input_line = textborder('INPUT: '+ repr(input_text), '=', 88, 0)
                    ckpt_output = f'CHECKPOINT: {c}\n{outputs}\n'
                    
                    inps_outs.setdefault(input_line, []).append(ckpt_output)
        
        for gc, inpo in gc_inps_outs.items():
            f.write(gc)
            for k,v in inpo.items():    
                f.write(k)
                f.writelines(v)


def eval_params(model_path, pgrid, test_qs, outfile='test_params.log', question_author:str = None, response_authors:list[str]|typing.Literal['rest','all']='rest'):
    if (cpts:=list(Path(model_path).glob('*checkpoint*'))):
        model_path = cpts[0]
    
    clo = Cloneus(model_path)
    clo.load_model()
    
    if question_author is None:
        question_author = roles.author_display_names[0]
    
    if response_authors == 'rest':
        author_list = [a for a in roles.author_display_names if a!=question_author]
    elif response_authors == 'all':
        author_list = roles.author_display_names
    elif isinstance(response_authors, list):
        author_list = response_authors

    spgrid = dict(sorted(pgrid.items(), key=lambda kv: len(kv[1])))
    param_sets = [dict(zip(spgrid, pvals)) for pvals in itertools.product(*spgrid.values())]
    print(f'Evaluating {len(param_sets)} parameter combinations')
    
    outfile = clo.mdir_comps.basedir_path/outfile
    with open(outfile, 'w') as f:
        for tq in test_qs:
            input_text, _, _, _, _ = clo.batch_generate([(question_author, tq)], author_list, '')
            f.write(textborder('INPUT: '+ repr(input_text), '=', 88, 0))
            for pset in param_sets:
                f.write('\n'+ json.dumps(pset))
                delt=clo.set_genconf(**pset)
                seed_everything(42)
            
                input_text, author_prompts, out_texts, _, _ = clo.batch_generate([(question_author, tq)], author_list, '')
                outputs='\n'.join([repr(atag+'⋙'+text) for atag,text in zip(author_prompts,out_texts)])
                f.write('\n'+outputs+'\n')