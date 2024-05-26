import gc
import os
import datetime
import functools
import math
import torch
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    GenerationConfig
)
from safetensors.torch import load_model as load_model_safetensors, save_model as save_model_safetensors
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model

from trl import SFTTrainer
# from trl.trainer import ConstantLengthDataset
# import wandb.vendor.pynvml.pynvml
# wandb.vendor.pynvml.pynvml.nvmlDeviceGetName = lambda handle: "NVIDIA GeForce RTX 3090"

def _get_cosine_const_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_const_steps:int, num_training_steps: int, num_cycles: float):
    
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    elif current_step < num_warmup_steps+num_const_steps:
        return 1.0
    num_wuconst_steps = (num_warmup_steps+num_const_steps)
    progress = float(current_step - num_wuconst_steps) / float(max(1, num_training_steps - num_wuconst_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def get_const_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_const_steps:int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_const_steps (`int`):
            The number of steps for the constant phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = functools.partial(
        _get_cosine_const_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_const_steps=num_const_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

class CTrainer(Trainer):
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.args.lr_scheduler_type!='linear': # default value
            return super().create_scheduler(num_training_steps, optimizer)
        
        self.lr_scheduler = get_const_cosine_schedule_with_warmup(
            optimizer=self.optimizer if optimizer is None else optimizer,
            num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
            num_const_steps=num_training_steps//self.args.num_train_epochs, # 1 epoch # TODO: maybe just 1 epoch on the outslope 
            num_training_steps=num_training_steps,
            num_cycles=0.5,
        )
        self.args.lr_scheduler_type == 'warmup_const_cosine'
        return self.lr_scheduler
        

    
def formatfunc(sample):
    return sample['text']

def get_trainer(model, data, tokenizer, args, callbacks=None, collator_pad_multiple=None):
    trainer = CTrainer(
        model=model,
        train_dataset=data["train"],
        eval_dataset=data['validation'],
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=collator_pad_multiple),
        args=args,
        callbacks=callbacks,
    )

    return trainer

def get_sft_trainer(model, data, tokenizer, args, peft_config, callbacks=None, max_packed_seqlength=2048, neftune_noise_alpha: (int | None) = None):
    # https://huggingface.co/docs/trl/main/en/sft_trainer#packing-dataset-constantlengthdataset
    if args.group_by_length:
        print('WARNING: group_by_length cannot be used with packing, disabling')
        args.group_by_length = False
    trainer = SFTTrainer(
        model=model,
        args=args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        train_dataset=data['train'],
        eval_dataset=data['validation'],
        tokenizer=tokenizer,
        peft_config=peft_config,
        dataset_text_field="text",
        packing=True,
        #formatting_func=formatfunc,
        max_seq_length=max_packed_seqlength,
        callbacks=callbacks,
        neftune_noise_alpha=neftune_noise_alpha
    )
    return trainer


def save_last_step(trainer:Trainer|SFTTrainer):
    try:
        ckpt = f'checkpoint-{trainer.state.global_step}'
    except Exception as e:
        print(e)
        ckpt = f'checkpoint-final'
        
    checkpoint_path = os.path.join(trainer.args.output_dir, ckpt)
        
    trainer.model.save_pretrained(checkpoint_path, safe_serialization=True) # trainer.save_model(checkpoint_path)
    trainer.tokenizer.save_pretrained(checkpoint_path)
    trainer.state.save_to_json(os.path.join(checkpoint_path, 'trainer_state.json')) # trainer.save_state()
    torch.save(trainer.args, os.path.join(checkpoint_path,'training_args.bin'))
    #save_model_safetensors(trainer.model,  os.path.join(checkpoint_path, "model.safetensors"))


def create_resumable_save(trainer:Trainer|SFTTrainer):    
    # The underscore prefix implies this is not a recommended way to save, but it works for now.
    # TODO: look in to implications
    if trainer.state.global_step:
        trainer._save_checkpoint(trainer.model, trainer._trial, metrics=None)


def format_arg_names(args, base_outdir, chunk_size, peft_config, n_custom_tokens, attn_implementation, custom_scheduler):
    if peft_config.target_modules == 'all-linear':
        lora_modules_dstr = 'AllLinear'
    else:
        lora_modules_dstr = ''.join(
            [m[0] if m.endswith('proj') else ''.join([n[0] for n in m.split('_')]).title() 
            for m in peft_config.target_modules])
    
    lora_layers_dstr = f'_l{lay[0]}-{lay[-1]}' if (lay := peft_config.layers_to_transform) is not None else ''
    schd_abbrs = {'cosine':'cos', 'linear':'lin', 'constant':'const', 'constant_with_warmup':'constwu'}
    schedule_dstr = (schd_abbrs.get(args.lr_scheduler_type,args.lr_scheduler_type) if custom_scheduler is None else custom_scheduler)

    custom_token_dstr = f'ctk{n_custom_tokens}' if n_custom_tokens is not None else ''

    warmup_dstr = f'-wu{args.warmup_ratio or args.warmup_steps}' #if 'warmup' in schedule_dstr else ''

    tnow = datetime.datetime.now().strftime('%Y%m%dT%H%M%S') # YYYYmmddTHHMMSS

    args.output_dir = args.output_dir.format(
        tnow=tnow,
        base_outdir=base_outdir,
        chunksize=chunk_size, 
        ctk_pad=custom_token_dstr, 
        scheduler=schedule_dstr, 
        warmup=warmup_dstr, 
        lora_a=peft_config.lora_alpha, 
        lora_r=peft_config.r, 
        #lora_dropout=peft_config.lora_dropout, 
        lora_modules=lora_modules_dstr,
        lora_layers=lora_layers_dstr
    )
    
    
    dirparts = args.output_dir.split('/')
    model_shortname = dirparts[2]
    dirargs = dirparts[-1]

    optalias = {'paged_adamw_32bit':'padam32', 'paged_adamw_8bit':'padam8', 'adamw_hf':'adamw', 'adamw_bnb_8bit':'adamw8b'}
    
    args.run_name = args.run_name.format(
        modelname=model_shortname,
        dirargs=dirargs, 
        optim=optalias.get(args.optim, args.optim), 
        batchsize=args.per_device_train_batch_size, 
        max_gradnorm=args.max_grad_norm
    )+('-flashattn' if attn_implementation else '')

    return args


def create_args(base_outdir, peft_config: LoraConfig, cfg,  n_custom_tokens=None,  **kwargs):
    #batch_size, ga_steps = batchsize_gasteps #chunk_to_batchga(chunk_size, target_batch_size=8, chunklen_upperbound=4096)

    warmup_ratio = cfg.warmup_ratio #kwargs.pop('warmup_ratio', 0.0)
    warmup_steps = cfg.warmup_steps #kwargs.pop('warmup_steps', 0)
    if warmup_ratio is None:
        warmup_ratio = 0.0
    if warmup_steps is None:
        warmup_steps = 0
    
    # batch_size = kwargs.pop('batch_size', 4)
    batch_size=cfg.batch_size
    chunk_size = cfg.chunk_size
    attn_implementation=cfg.attn_implementation
    custom_scheduler=cfg.custom_scheduler
    
    args = TrainingArguments(
        num_train_epochs=cfg.num_epochs,#kwargs.pop('num_train_epochs', 3),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,#kwargs.pop('gradient_accumulation_steps', 1),
        gradient_checkpointing=True, # add # If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        #gradient_checkpointing_kwargs = dict(use_reentrant=True), # when = False, vRAM usage sky rockets. Not sure if bug or bad.
        eval_strategy='steps',
        eval_steps=kwargs.pop('eval_steps', None), # Will default to the same value as logging_steps
        save_strategy=kwargs.pop('save_strategy','epoch'),
        save_steps=cfg.save_steps,#kwargs.pop('save_steps',500),
        learning_rate=cfg.learning_rate,#kwargs.pop('learning_rate', 2e-4),
        bf16=cfg.bf16,#kwargs.pop('bf16', True),
        fp16=cfg.fp16,#kwargs.pop('fp16', False), # https://huggingface.co/docs/transformers/perf_train_gpu_one#mixed-precision-training
        tf32=cfg.tf32,#kwargs.pop('tf32', True), # add (IMPLICATIONS WITH BF16,gradaccum: https://github.com/huggingface/transformers/issues/14608#issuecomment-1004392537 )
        logging_steps=cfg.logging_steps,#kwargs.pop('logging_steps', 5),
        output_dir='{base_outdir}/{tnow}_cnk{chunksize}{ctk_pad}-{scheduler}{warmup}--r{lora_r}a{lora_a}_{lora_modules}{lora_layers}', # -sft256
        optim=cfg.optimizer,#kwargs.pop('optim', 'paged_adamw_32bit'),#'paged_adamw_8bit',# #"adamw_hf"
        max_grad_norm=cfg.max_grad_norm,#kwargs.pop('max_grad_norm', 0.3),
        warmup_ratio=warmup_ratio,#kwargs.pop('warmup_ratio', 0.00),
        warmup_steps=warmup_steps,#kwargs.pop('warmup_steps', 0),
        lr_scheduler_type=cfg.lr_scheduler, #kwargs.pop('lr_scheduler_type', 'linear'),
        weight_decay=cfg.weight_decay,#kwargs.pop('weight_decay',0),
        torch_compile=kwargs.pop('torch_compile',False),
        #dataloader_num_workers=kwargs.pop('dataloader_num_workers',0), # Doesn't work with tokenizer
        neftune_noise_alpha=cfg.neftune_noise_alpha,#kwargs.pop('neftune_noise_alpha', None),

        disable_tqdm=kwargs.pop('disable_tqdm', None),
        save_total_limit=kwargs.pop('save_total_limit', None),
        save_safetensors=True,
        logging_first_step=True,
        group_by_length=kwargs.pop('group_by_length', True), # Might have consequences, disable? -- yep, will consume absurd memory when combining largest items (unless truncated beforehand)
        
        run_name="{modelname}-{dirargs}-{optim}-b{batchsize}-mgn{max_gradnorm}",
        **kwargs,
    )
    args = format_arg_names(args, base_outdir, chunk_size, peft_config, n_custom_tokens, attn_implementation, custom_scheduler)

    # if kwargs:
    #     nargs=args.to_dict()
    #     nargs.update(kwargs)
    #     args = TrainingArguments(**nargs)

    return args

def get_batch(trainer:Trainer, train=False):
    dl = trainer.get_train_dataloader() if train else trainer.get_eval_dataloader()
    b0=next(iter(dl))
    return trainer.tokenizer.batch_decode(b0.input_ids, skip_special_tokens=False)

class FullSaveCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        #torch.save(kwargs["model"].state_dict(), os.path.join(checkpoint_path, "pytorch_model.bin"))
        save_model_safetensors(kwargs["model"],  os.path.join(checkpoint_path,"model.safetensors"))
        

class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))


# class GenerationCallback(TrainerCallback):
#     "A callback tests model every `predict_steps` steps and logs the results to `args.output_dir/logs/generations.log`"

#     def __init__(self, log_step_multiplier=10):
#         self.log_step_multiplier = log_step_multiplier
#         self.logfile = None

#     def _init_logfile(self, args):
#         logdir = os.path.join(args.output_dir, 'logs')
#         os.makedirs(logdir, exist_ok=True)
#         self.logfile = os.path.join(logdir, 'generations.log')

#     def on_log(self, args, state, control, **kwargs):
#         if state.global_step % int(state.logging_steps*self.log_step_multiplier) == 0:
#             if self.logfile is None:
#                 self._init_logfile(args)
            
            
#             model = kwargs.get('model')
#             tokenizer = kwargs.get('tokenizer')

#             shared_genargs = dict(
#                 max_new_tokens=128,
#                 renormalize_logits=True,
#                 repetition_penalty=1.1, # Setting this too high may prevent sequential same-author messages. 
#                 eos_token_id=model.config.eos_token_id,
#                 pad_token_id=model.config.pad_token_id,
#             )

#             cs_genconf = GenerationConfig(penalty_alpha=0.6, top_k=4, **shared_genargs) #contrastive
#             #bsms_genconf = GenerationConfig(do_sample=True, top_p=0.95, temperature=0.9, num_beams=4, early_stopping=True, **shared_genargs)
#             #dbsd_genconf = GenerationConfig(do_sample=False, num_beams=4, num_beam_groups=4, early_stopping=True, diversity_penalty=0.5, **shared_genargs)
#             ms_genconf = GenerationConfig(do_sample=True, top_p=1, temperature=1, **shared_genargs)
#             #gd_genconf = GenerationConfig(do_sample=False, num_beams=1, **shared_genargs)
            
#             header = f"{'-'*25} [Step: {state.global_step}] {'-'*25}"
#             seper = '-'*64
#             nlsep = '\n'+seper+'\n'
#             footer = '='*64
#             with torch.inference_mode():
#                 in_text, out_texts = test_model(model, tokenizer, [cs_genconf,ms_genconf,], do_print=False)

#             #fstring = '\n'.join([header, in_text, seper, sout_text, footer])
#             fstring = '\n'.join([header, in_text, seper]) + '\n' + nlsep.join(out_texts) + '\n' + footer
            
#             with open(self.logfile, 'a') as f:
#                 f.write(fstring+'\n')
            

        
