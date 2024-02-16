import os
import json
import random
import typing
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


with open(cpaths.DATA_DIR/'testfiles/test_questions.txt', 'r') as f:
    TEST_QUESTIONS = f.read().splitlines()

with open(cpaths.DATA_DIR/'testfiles/sample_conversations/evalset_convo.json','r') as f:
    EVALSET_SAMPLE: list[tuple[str,str]] = [(i['user'],i['content']) for i in json.load(f)]

with open(cpaths.DATA_DIR/'testfiles/sample_conversations/testset_convo.json','r') as f:
    TESTSET_SAMPLE: list[tuple[str,str]] = [(i['user'],i['content']) for i in json.load(f)]


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_test_questions(n:int|None = 1, best_only=False):
    test_qs = [q.strip('*').replace('\\n','\n') for q in TEST_QUESTIONS if (q.endswith('*') or not best_only)]
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


def eval_model(model_path, test_qs=None, outfile='test_samples.log', gmodes:list[str]=None, question_author:str = None, response_authors='rest'):
    if (cpts:=list(Path(model_path).glob('*checkpoint*'))):
        model_path = cpts[0]

    clo = Cloneus(model_path)
    clo.load_model()
    checkpoints = sorted([p.name for p in clo.mdir_comps.basedir_path.glob('*checkpoint*')], key=lambda t: int(t.split('-')[1]))
    
    if test_qs is None:
        test_qs = get_test_questions(None, True)
    
    if gmodes is None:
        gmodes = ['cs','ms']
        
    if question_author is None:
        question_author = roles.author_display_names[0]
    
    if response_authors == 'rest':
        author_list = [a for a in roles.author_display_names if a!=question_author]
    
    #for c in checkpoints:
    #    rai.model.load_adapter(rai.mdir_comps.basedir_path/c, adapter_name=c)
        
    # TODO: fix order so checkpoints don't need to be repeatedly reloaded. 
    outfile = clo.mdir_comps.basedir_path/outfile
    with open(outfile, 'w') as f:
        for gmode in gmodes:
            clo.gen_config = genconfig.get_config(clo.tokenizer, gmode)
            f.write(textborder(gmode.upper() +': '+ str(clo.get_genconf(True)), '-', 176, 0))

            for tq in tqdm(test_qs):
                # TODO: can instruct models disregard postfix? Is it a splitting dependency? I dont remember.
                #ftq = mot.apply_template(question_author, tq, mot.tag_sep, mot.postfix) 
                
                
                for i,c in enumerate(checkpoints): 
                    seed_everything(42)
                    #print('CHECKPOINT:',c)
                    #rai.model.set_adapter(c)
                    clo.switch_model(clo.mdir_comps.basedir_path, c)
                    clo.gen_config = genconfig.get_config(clo.tokenizer, gmode)
                    
                    input_text, author_prompts, out_texts, _, _ = clo.batch_generate([(question_author, tq)], author_list, '')
                    outputs='\n'.join([atag+'⋙'+repr(text) for atag,text in zip(author_prompts,out_texts)])
                    if i==0:
                        f.write(textborder('INPUT: '+ repr(input_text), '=', 88, 0))
                    f.write(f'CHECKPOINT: {c}\n{outputs}\n')
                    
