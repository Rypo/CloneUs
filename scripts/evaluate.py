import argparse
from pathlib import Path
from dotenv import load_dotenv

import cloneus.training.evaluation as meval
import cloneus.core.paths as cpaths


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--modelname', default='mistral-7b-i4', type=str, required=False,
                        help='model name for testing')
    
    parser.add_argument('-p','--mpath', default=None, type=str, required=False,
                        help='base dir path for model to test')
    
    parser.add_argument('-o','--outfile',default='test_samples.log', 
                        help='file name to dump results into')
    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    if args.mpath is None:
        if args.modelname == '*':
            run_path = Path(cpaths.RUNS_DIR)
            for model_dir in run_path.iterdir():
                print('MODEL_DIR:', model_dir)
                for dpath in model_dir.iterdir():
                    print('RUNNING_DATASET:', dpath)
                    for p in dpath.iterdir():
                        if any(p.glob('config.yaml*')):
                            print(p)
                            
                            try:
                                meval.eval_model(p, outfile=args.outfile, )
                            except Exception as e:
                                print(e)
        else:
            run_path = Path(cpaths.RUNS_DIR/args.modelname)
            for dpath in run_path.iterdir():
                print('RUNNING:', dpath)
                for p in dpath.iterdir():
                    if any(p.glob('config.yaml*')):
                        print(p)
                        
                        try:
                            meval.eval_model(p, outfile=args.outfile, )
                        except Exception as e:
                            print(e)
    else:
        run_path = Path(args.mpath)
        meval.eval_model(run_path, outfile=args.outfile, )
