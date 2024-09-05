
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
from accelerate import init_empty_weights
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model, set_module_tensor_to_device, compute_module_sizes
from diffusers import FluxTransformer2DModel
from diffusers.models.model_loading_utils import load_model_dict_into_meta
import safetensors.torch as sft
from huggingface_hub import snapshot_download

from transformers import T5EncoderModel, AutoModelForTextEncoding
from transformers.quantizers.quantizers_utils import get_module_from_name
from optimum import quanto

from cloneus.utils.common import timed,release_memory

logger = logging.getLogger(__name__)


def read_state_dict(model_id_or_path:str|Path, allow_empty:bool = False):
    if (model_path := Path(model_id_or_path)).exists():
        weights_loc = model_path if model_path.is_dir() else model_path.parent

    else:
        weights_loc = Path(snapshot_download(repo_id=model_id_or_path, allow_patterns="transformer/*")).joinpath('transformer')
    
    state_dict = {}
    for shard in sorted(Path(weights_loc).glob('*.safetensors')):
        state_dict.update(sft.load_file(shard))

    if not state_dict and not allow_empty:
        raise RuntimeError('Failed to read state dict data from given path')

    return state_dict

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
                        compute_dtype=bnb_qconfig.torch_dtype,#torch.bfloat16,
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
def bnb4_transformer(model_id:str = "black-forest-labs/flux.1-dev", dtype = torch.bfloat16, bnb_quant_config:BnbQuantizationConfig = None):
    with init_empty_weights():
        config = FluxTransformer2DModel.load_config("black-forest-labs/flux.1-dev", subfolder="transformer")
        model = FluxTransformer2DModel.from_config(config).to(torch.bfloat16).eval()

    if bnb_quant_config is None:
        # keep defaults from orig implementation (no compress stats)
        bnb_quant_config = BnbQuantizationConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=False, 
            bnb_4bit_compute_dtype='bf16', torch_dtype=torch.bfloat16, )
    
    _replace_with_bnb_linear(model, bnb_quant_config)
    converted_state_dict = read_state_dict(model_id)
    assert len(converted_state_dict) > 5, 'failed to read state dict'
    
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
def load_sd_bnb4_transformer(safetensor_filepath:str|Path):
    dtype = torch.bfloat16
    is_torch_e4m3fn_available = hasattr(torch, "float8_e4m3fn")
    #ckpt_path = hf_hub_download("sayakpaul/flux.1-dev-nf4", filename="diffusion_pytorch_model.safetensors")
    original_state_dict = read_state_dict(safetensor_filepath)

    with init_empty_weights():
        #config = FluxTransformer2DModel.load_config("sayakpaul/flux.1-dev-nf4")
        config = FluxTransformer2DModel.load_config("black-forest-labs/flux.1-dev", subfolder="transformer")
        model = FluxTransformer2DModel.from_config(config).to(dtype).eval()
        expected_state_dict_keys = list(model.state_dict().keys())

    bnb_qconfig = BnbQuantizationConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=False, 
                                        bnb_4bit_compute_dtype='bf16', torch_dtype=torch.bfloat16, )
    _replace_with_bnb_linear(model, bnb_qconfig)#nf4

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