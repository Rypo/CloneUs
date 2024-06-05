import argparse
from pathlib import Path
from dotenv import load_dotenv
from omegaconf import OmegaConf

import cloneus.training.evaluation as meval
from cloneus.types import cpaths


def get_cli_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('runpath', default=None, type=str, 
                        help='Path to directory with models to test. Can be a single run, a single checkpoint, or multiple runs')
    
    parser.add_argument('-c','--config', default=None, type=str, required=False,
                        help='path/to/eval_config.yaml. If None will use ./config/eval/eval_config.yaml')
    
    parser.add_argument('--sweep', action='store_true',
                        help='perform a grid search over parameters')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_cli_args()

    run_path = Path(args.runpath)
    config_path = args.config if args.config else cpaths.ROOT_DIR/'config'/'eval'/'eval_config.yaml'
    
    config = OmegaConf.load(config_path)
    print('loaded config from:', config_path)
    
    if args.sweep:
        cfg = config.sweep
        meval.eval_params(run_path, param_grid=cfg.param_grid, prompts=cfg.prompts, outfile=cfg.outfile, question_author = cfg.question_author, response_authors=cfg.response_authors)
    else:
        cfg = config.sample
        #test_qs = meval.get_test_questions(cpaths.DATA_DIR/'testfiles/test_questions.txt', None, True)
        meval.sample_trained(run_path, prompts=cfg.prompts, outfile=cfg.outfile, genconfig_modes=cfg.genconfig_modes, question_author = cfg.question_author, response_authors=cfg.response_authors)
