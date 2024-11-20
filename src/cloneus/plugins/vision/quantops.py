
import gc
import json
import copy
import typing
from os import PathLike
from pathlib import Path
import logging
import torch
torch.backends.cuda.matmul.allow_tf32 = True
import bitsandbytes as bnb

from bitsandbytes.nn.modules import Params4bit, QuantState
from bitsandbytes.functional import dequantize_4bit
from accelerate import init_empty_weights, cpu_offload
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model, set_module_tensor_to_device, compute_module_sizes
from diffusers import FluxTransformer2DModel, SD3Transformer2DModel, BitsAndBytesConfig, FluxPipeline
from diffusers.models.model_loading_utils import load_model_dict_into_meta



from huggingface_hub import snapshot_download, hf_hub_download

from transformers import T5EncoderModel, AutoModelForTextEncoding
from transformers.quantizers.quantizers_utils import get_module_from_name
from optimum import quanto

from cloneus.utils.common import timed,release_memory,read_state_dict
from . import loraops

logger = logging.getLogger(__name__)


def meta_transformer(model_scaffold:typing.Literal['flux_dev', 'sd35_medium', 'sd35_large'], dtype=torch.bfloat16):
    model_ids = {
        'flux_dev': 'black-forest-labs/FLUX.1-dev',
        'sd35_medium': 'stabilityai/stable-diffusion-3.5-medium',
        'sd35_large': 'stabilityai/stable-diffusion-3.5-large',
    }
    with init_empty_weights():
        if model_scaffold == 'flux_dev':
            config = FluxTransformer2DModel.load_config(model_ids[model_scaffold], subfolder="transformer")
            transformer = FluxTransformer2DModel.from_config(config).to(dtype).eval()
        else:
            config = SD3Transformer2DModel.load_config(model_ids[model_scaffold], subfolder="transformer")
            transformer = SD3Transformer2DModel.from_config(config).to(dtype).eval()
    return transformer

# ---------------------------------------------------------------------------
# BitsandBytes Quantization
# https://github.com/huggingface/diffusers/issues/9165#issue-2462431761
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _load_bnb4_transformer(weights_location, config, bnb_quant_config:BnbQuantizationConfig):
    with init_empty_weights():
        transformer = FluxTransformer2DModel.from_config(config, torch_dtype=torch.bfloat16).eval()
    
    transformer = load_and_quantize_model(transformer, weights_location=weights_location, bnb_quantization_config=bnb_quant_config, device_map = "auto").eval()
    return transformer

@torch.inference_mode()
def create_4bit_transformer(model_id_or_path:str, bnb_quant_config:BnbQuantizationConfig=None):
    if bnb_quant_config is None:
        bnb_quant_config = BnbQuantizationConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype='bf16', torch_dtype=torch.bfloat16, )
        
    if (model_path := Path(model_id_or_path)).exists():
        config = FluxTransformer2DModel.load_config(model_path/'config.json') # if it's local, it will complain about model_index.json
        return _load_bnb4_transformer(model_path, config, bnb_quant_config)
    
    weights_location = Path(snapshot_download(model_id_or_path, allow_patterns="transformer/*")).joinpath('transformer')
    
    config  = FluxTransformer2DModel.load_config(weights_location)
    transformer = _load_bnb4_transformer(weights_location, config, bnb_quant_config)

    return transformer


@torch.inference_mode()
def manual_4bit_model(meta_model:FluxTransformer2DModel|SD3Transformer2DModel, converted_state_dict:dict, quant_config:BnbQuantizationConfig|BitsAndBytesConfig=None, dtype=torch.bfloat16):
    if meta_model.device.type != 'meta':
        raise RuntimeError('`meta_model` not on meta device, use with init_empty_weights()')

    mismatch_keys = set(meta_model.state_dict()) ^ set(converted_state_dict)
    
    if mismatch_keys:
        raise KeyError(f'state dict is missing or extra keys: {mismatch_keys}')

    if quant_config is None:
        quant_config = BnbQuantizationConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=False, bnb_4bit_compute_dtype='bf16', torch_dtype=dtype, )
    
    _replace_with_bnb_linear(meta_model, quant_config)

    for param_name, param in converted_state_dict.items():
        param = param.to(dtype)
        
        if not check_quantized_param(meta_model, param_name):
            set_module_tensor_to_device(meta_model, param_name, device=0, value=param)
        else:
            create_quantized_param(meta_model, param, param_name, target_device=0)

    
    release_memory()
    # technically are writing to an internal frozen dict class but it's the only way to make it work out of the box with from_pretrained 
    meta_model.config["quantization_config"] = quant_config
    meta_model.config.quantization_config = quant_config
    # to save the 4bit quant:
    # transformer.save_pretrained('extras/quantized/flux/nf4/hyper')
    return meta_model

@torch.inference_mode()
def flux_hyper_transformer_4bit(model_id="black-forest-labs/FLUX.1-dev", scale=0.125, dtype=torch.bfloat16, bnb_qconfig=None):
    transformer = meta_transformer('flux_dev', dtype)
    
    
    # hyper_path = hf_hub_download("ByteDance/Hyper-SD", "Hyper-FLUX.1-dev-8steps-lora.safetensors")
    hyper_path = hf_hub_download("ByteDance/Hyper-SD", "Hyper-FLUX.1-dev-16steps-lora.safetensors")
    merged_state_dict = loraops.state_dict_merge_lora(hyper_path, model_id, scale)
    if bnb_qconfig is None:
        bnb_qconfig = BnbQuantizationConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=False, bnb_4bit_compute_dtype='bf16', torch_dtype=torch.bfloat16, )
    
    transformer = manual_4bit_model(transformer, merged_state_dict, bnb_qconfig, dtype)
    # to save the 4bit quant:
    # transformer.save_pretrained('extras/quantized/flux/nf4/hyper')
    
    return transformer
    

@torch.inference_mode()
def _replace_with_bnb_linear(
    model,
    bnb_qconfig:BnbQuantizationConfig,
    #method="nf4",
    has_been_replaced=False,
):
    """
    Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successfull or not.
    """
    # SRC: # https://github.com/huggingface/diffusers/issues/9165#issue-2462431761
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            with init_empty_weights():
                in_features = module.in_features
                out_features = module.out_features

                #if method == "llm_int8":
                if bnb_qconfig.load_in_8bit:
                    model._modules[name] = bnb.nn.Linear8bitLt(
                        in_features,
                        out_features,
                        module.bias is not None,
                        has_fp16_weights=False,
                        threshold=bnb_qconfig.llm_int8_threshold#6.0,
                    )
                    has_been_replaced = True
                else:
                    model._modules[name] = bnb.nn.Linear4bit(
                        in_features,
                        out_features,
                        module.bias is not None,
                        compute_dtype=bnb_qconfig.bnb_4bit_compute_dtype,#torch.bfloat16,
                        compress_statistics=bnb_qconfig.bnb_4bit_use_double_quant,#False,
                        quant_type=bnb_qconfig.bnb_4bit_quant_type,#"nf4",
                    )
                    has_been_replaced = True
                # Store the module class in case we need to transpose the weight later
                model._modules[name].source_cls = type(module)
                # Force requires grad to False to avoid unexpected errors
                model._modules[name].requires_grad_(False)

        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_bnb_linear(
                module,
                bnb_qconfig,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
    return model, has_been_replaced

@torch.inference_mode()
def check_quantized_param(model, param_name: str,) -> bool:
    module, tensor_name = get_module_from_name(model, param_name)
    # if isinstance(module, bnb.nn.Params4bit) or isinstance(module._parameters.get(tensor_name, None), bnb.nn.Params4bit):
    if isinstance(module._parameters.get(tensor_name, None), bnb.nn.Params4bit):
        # Add here check for loaded components' dtypes once serialization is implemented
        return True
    elif isinstance(module, bnb.nn.Linear4bit) and tensor_name == "bias":
        # bias could be loaded by regular set_module_tensor_to_device() from accelerate,
        # but it would wrongly use uninitialized weight there.
        return True
    else:
        return False

@torch.inference_mode()
def create_quantized_param(
    model,
    param_value: "torch.Tensor",
    param_name: str,
    target_device: "torch.device",
    state_dict=None,
    unexpected_keys=None,
    pre_quantized=False
):
    module, tensor_name = get_module_from_name(model, param_name)

    if tensor_name not in module._parameters:
        raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")

    old_value = getattr(module, tensor_name)

    if tensor_name == "bias":
        if param_value is None:
            new_value = old_value.to(target_device)
        else:
            new_value = param_value.to(target_device)

        new_value = torch.nn.Parameter(new_value, requires_grad=old_value.requires_grad)
        module._parameters[tensor_name] = new_value
        return

    if not isinstance(module._parameters[tensor_name], bnb.nn.Params4bit):
        raise ValueError("this function only loads `Linear4bit components`")
    if (
        old_value.device == torch.device("meta")
        and target_device not in ["meta", torch.device("meta")]
        and param_value is None
    ):
        raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {target_device}.")

    if pre_quantized:
        if (param_name + ".quant_state.bitsandbytes__fp4" not in state_dict) and (
            param_name + ".quant_state.bitsandbytes__nf4" not in state_dict
        ):
            raise ValueError(
                f"Supplied state dict for {param_name} does not contain `bitsandbytes__*` and possibly other `quantized_stats` components."
            )

        quantized_stats = {}
        for k, v in state_dict.items():
            # `startswith` to counter for edge cases where `param_name`
            # substring can be present in multiple places in the `state_dict`
            if param_name + "." in k and k.startswith(param_name):
                quantized_stats[k] = v
                if unexpected_keys is not None and k in unexpected_keys:
                    unexpected_keys.remove(k)

        new_value = bnb.nn.Params4bit.from_prequantized(
            data=param_value,
            quantized_stats=quantized_stats,
            requires_grad=False,
            device=target_device,
        )

    else:
        new_value = param_value.to("cpu")
        kwargs = old_value.__dict__
        new_value = bnb.nn.Params4bit(new_value, requires_grad=False, **kwargs).to(target_device)

    module._parameters[tensor_name] = new_value

@torch.inference_mode()
def bnb4_transformer(model_id:str = "black-forest-labs/FLUX.1-dev", dtype = torch.bfloat16, bnb_quant_config:BnbQuantizationConfig = None):
    model = meta_transformer('flux_dev', dtype)

    if bnb_quant_config is None:
        # keep defaults from orig implementation (no compress stats)
        bnb_quant_config = BnbQuantizationConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=False, 
            bnb_4bit_compute_dtype='bf16', torch_dtype=torch.bfloat16, )
    
    _replace_with_bnb_linear(model, bnb_quant_config)
    converted_state_dict = read_state_dict(model_id, subfolder='transformer')
    #assert len(converted_state_dict) > 5, 'failed to read state dict'
    
    for param_name, param in converted_state_dict.items():
        param = param.to(dtype)
        if not check_quantized_param(model, param_name):
            set_module_tensor_to_device(model, param_name, device=0, value=param)
        else:
            create_quantized_param(model, param, param_name, target_device=0)

    del converted_state_dict
    gc.collect()

    return model

@torch.inference_mode()
def load_prequantized_transformer(safetensor_filepath:str|Path, dtype = torch.bfloat16, bnb_quant_config:BnbQuantizationConfig = None):
    is_torch_e4m3fn_available = hasattr(torch, "float8_e4m3fn")
    #ckpt_path = hf_hub_download("sayakpaul/flux.1-dev-nf4", filename="diffusion_pytorch_model.safetensors")
    original_state_dict = read_state_dict(safetensor_filepath)

    if bnb_quant_config is None:
        # keep defaults from orig implementation (no compress stats)
        bnb_quant_config = BnbQuantizationConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=False, 
            bnb_4bit_compute_dtype='bf16', torch_dtype=torch.bfloat16, )
    
    model = meta_transformer('flux_dev', dtype)
    expected_state_dict_keys = list(model.state_dict().keys()) #("sayakpaul/flux.1-dev-nf4")


    _replace_with_bnb_linear(model, bnb_quant_config)

    for param_name, param in original_state_dict.items():
        if param_name not in expected_state_dict_keys:
            continue
        
        is_param_float8_e4m3fn = is_torch_e4m3fn_available and param.dtype == torch.float8_e4m3fn
        if torch.is_floating_point(param) and not is_param_float8_e4m3fn:
            param = param.to(dtype)
        
        if not check_quantized_param(model, param_name):
            set_module_tensor_to_device(model, param_name, device=0, value=param)
        else:
            create_quantized_param(
                model, param, param_name, target_device=0, state_dict=original_state_dict, pre_quantized=True
            )

    del original_state_dict
    gc.collect()

    return model



# ---------------------------------------------------------------------------
# Quanto Quantization (float8, int8, int4)
# ---------------------------------------------------------------------------

class QuantizedFluxTransformer2DModel(quanto.QuantizedDiffusersModel):
    base_class = FluxTransformer2DModel
    
class QuantizedModelForTextEncoding(quanto.QuantizedTransformersModel):
    auto_class = AutoModelForTextEncoding

@torch.inference_mode()
def load_or_quantize(quant_path_or_model_id:str|Path, qtype:typing.Literal['qint8','qfloat8','qint4'], module:typing.Literal["transformer","text_encoder_2"], device:torch.device=None):
    torch_dtype = torch.bfloat16
    transformer_exclude = None
    
    quant_device = 0 if torch_dtype == torch.bfloat16 else None # {'':0}
    device = device if device is not None else 'cpu'

    if qtype == 'qint4':
        #torch_dtype = torch.float32
        transformer_exclude = ["proj_out", "x_embedder", "norm_out", "context_embedder"]
    
    quant_path = None
    local_path = False
    if (model_path := Path(quant_path_or_model_id)).exists():
        local_path = True
        if (model_path/'quanto_qmap.json').exists():
            quant_path = model_path
        # else assume is a local unquantized directory (e.g. merged transformer)
    # else assume it a huggingface model_id


    if quant_path is not None:
        with timed(f'Load - {module}'):
            if module=='transformer':
                model = QuantizedFluxTransformer2DModel.from_pretrained(quant_path).to(device) # .to(device) is required to shed the wrapper
            elif module=='text_encoder_2':
                model = QuantizedModelForTextEncoding.from_pretrained(quant_path).to(device)
                # .to(device) is required even if device=None, otherwise will get
                # "TypeError: 'QuantizedModelForTextEncoding' object is not callable" since QuantizedTransformersModel doesn't implement __call__
        
        release_memory()
        return model

    
    path_kwargs = {'pretrained_model_name_or_path': model_path}
    if not local_path:
        path_kwargs['subfolder'] = module

    if module=='transformer':
        model:FluxTransformer2DModel = FluxTransformer2DModel.from_pretrained(**path_kwargs, torch_dtype=torch_dtype, device_map=quant_device)
        exclude = transformer_exclude

    elif module=='text_encoder_2':
        model = T5EncoderModel.from_pretrained(**path_kwargs, torch_dtype=torch_dtype, device_map=quant_device)
        exclude = None

        
    with timed(f'Quantize - {module}'):
        quanto.quantize(model, weights=qtype, exclude=exclude)
        quanto.freeze(model)
    
    release_memory()
    return model.to(device)




class FluxLoadHelper:
    def __init__(self, model_path, quant_basedir, text_enc_model_id, variant) -> None:
        self.model_path = model_path
        self.quant_basedir = quant_basedir
        self.text_enc_model_id = text_enc_model_id
        self.variant = variant

    @torch.inference_mode()
    def _load_transformer(self, qtype:str, device:torch.device, torch_dtype = torch.bfloat16):
        if qtype == 'bnb4':
            print('load transformer')
            
            if str(self.model_path).endswith('.safetensors'):
                # return bnbops.load_quantized(safetensor_filepath=self.model_path, bnb_quant_config=bnb_qconfig).to(device, dtype=torch_dtype)
                return load_prequantized_transformer(safetensor_filepath=self.model_path).to(device, dtype=torch_dtype)
                #transformer = bnbops.bnb4_transformer(self.model_path, torch_dtype).to(device, dtype=torch_dtype)
            else:
               
                bnb_qconfig = BnbQuantizationConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=False, bnb_4bit_compute_dtype='bf16', torch_dtype=torch.bfloat16, )
                                                    # bnb_4bit_use_double_quant=False, bnb_4bit_compute_dtype='fp32', torch_dtype=torch.float32, )

                #return  bnbops.create_4bit_transformer(self.model_path, bnb_qconfig).to(device, dtype=torch_dtype)
                return  bnb4_transformer(self.model_path, torch_dtype, bnb_qconfig).to(device, dtype=torch_dtype)
        
        if not (model_path := self.quant_basedir/qtype/self.variant/'transformer/').exists():
            model_path = self.model_path
        
        return load_or_quantize(model_path, qtype=qtype, module='transformer', device=device)
    
    @torch.inference_mode()
    def _load_text_encoder_2(self, te2_qtype:str, device:torch.device, torch_dtype = torch.bfloat16):
        from diffusers import DiffusionPipeline, BitsAndBytesConfig

        if te2_qtype=='bf16':
           return T5EncoderModel.from_pretrained(self.text_enc_model_id, subfolder="text_encoder_2", torch_dtype=torch_dtype, low_cpu_mem_usage=True, device_map='auto')
        
        if te2_qtype.startswith('q'):
            if not (model_path := self.quant_basedir/te2_qtype/'text_encoder_2/').exists():
                model_path = self.model_path
            return load_or_quantize(model_path, qtype=te2_qtype, module='text_encoder_2', device=device)

        if te2_qtype in ['bnb4','bnb8']:
            # leave text_encoder_2 empty in the pipe line and instead use a seperate pipeline for text encoding so that the transformer can still be offloaded
            bnb_quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type= "nf4", bnb_4bit_compute_dtype=torch_dtype, bnb_4bit_use_double_quant=False)
            
            text_encoder = T5EncoderModel.from_pretrained(self.text_enc_model_id, subfolder="text_encoder_2", quantization_config=bnb_quant_config, torch_dtype=torch_dtype, low_cpu_mem_usage=True, device_map='auto')
            self.text_pipe = DiffusionPipeline.from_pretrained(self.model_path, transformer=None, vae=None, text_encoder_2=text_encoder, torch_dtype=torch_dtype, low_cpu_mem_usage=True) 
            return None

    @torch.inference_mode()
    def load_quant(self, qtype:typing.Literal['bnb4','qint4','qint8','qfloat8'], te2_qtype:typing.Literal['bnb4','bnb8','bf16', 'qint4','qint8','qfloat8'], offload:bool=False,):
        '''Try to load quantized from disk otherwise, quantize on the fly
        
        Args:
            qtype: quantization type for transformer
            te2_quant: quantization type for T5 encoder. If bf16, no quantization.
            offload: enable model cpu off (incompatible with bnb)
            
        '''
        
        print(f'transformer quant: {qtype}, text_encoder_2 quant: {te2_qtype}')
        
        device = None if offload else 0 #'cuda'
        torch_dtype = torch.bfloat16 
        
        
        te_kwarg = {'text_encoder':None} if te2_qtype.startswith('bnb') else {}
        pipe: FluxPipeline = FluxPipeline.from_pretrained(self.text_enc_model_id, transformer=None, text_encoder_2=None, **te_kwarg, torch_dtype=torch_dtype)  # 
        # don't move to device yet, loading transformer could use a LOT of vram if quantizing
        pipe.transformer = self._load_transformer(qtype, device, torch_dtype)
        release_memory()
        if pipe.text_encoder_2 is None:
            pipe.text_encoder_2 = self._load_text_encoder_2(te2_qtype, device, torch_dtype).to(device)
        
        pipe.transformer = pipe.transformer.to(device)
        release_memory()
        
        # Do not call pipe.to(device) more than once. It will break things
        pipe = pipe.to(device, dtype=torch_dtype)
        
        
        print('PIPE:', pipe.dtype, pipe.device)
        print('TRANSFORMER:', pipe.transformer.dtype, pipe.transformer.device)
        
        if pipe.text_encoder_2 is None:
            print('text_pipe - TEXT ENC 2:', self.text_pipe.text_encoder_2.dtype, self.text_pipe.text_encoder_2.device)
        else:
            print('TEXT ENC 2:', pipe.text_encoder_2.dtype, pipe.text_encoder_2.device)
        
        release_memory()
        return pipe