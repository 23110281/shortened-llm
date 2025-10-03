import copy
import csv
import gc
import json
import random
import time

import numpy as np
import torch
from LLMPruner.peft import PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
)

# Import the torch_xla library for TPU support
try:
    import torch_xla.core.xla_model as xm
    _TPU_AVAILABLE = True
except ImportError:
    _TPU_AVAILABLE = False


def set_seed(random_seed=1234):
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def set_model_device_evalmode(
    model, device, fix_decapoda_config=False, use_bfloat=False
):
    # --- TPU/GPU/CPU device detection ---
    if device == 'auto':
        if _TPU_AVAILABLE:
            device = xm.xla_device()
            print("INFO: Auto-detected and using TPU device.")
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"INFO: Auto-detected and using {device} device.")

    # Move model to the selected device
    model = model.to(device)

    # Apply half precision only if using a CUDA device
    if "cuda" in str(device):
        model.half()

    if fix_decapoda_config:
        # unwind broken decapoda-research config
        model.config.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
    model.eval()

    if use_bfloat:
        model = model.bfloat16()

    gc.collect()
    # Clear CUDA cache only if a CUDA device is used
    if "cuda" in str(device):
        torch.cuda.empty_cache()

    return model, device # Return the determined device as well


def get_model(
    base_model=None,
    ckpt=None,
    lora_ckpt=None,
    tokenizer=None,
    model_type="pretrain",
    device="auto", # Changed default to 'auto' for flexibility
    fix_decapoda_config=False,
    use_bfloat=False,
):
    tokenizer = base_model if tokenizer is None else tokenizer
    if model_type == "pretrain":
        config = AutoConfig.from_pretrained(base_model)
        if "gptq" in base_model.lower():
            from auto_gptq import AutoGPTQForCausalLM

            model = AutoGPTQForCausalLM.from_quantized(
                base_model,
                use_safetensors=True,
                trust_remote_code=True,
                use_triton=False,
                quantize_config=None,
            )
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        elif (
            "LlamaForCausalLM" in config.__getattribute__("architectures")
            and "llama-3" not in base_model.lower()
        ):
            model = LlamaForCausalLM.from_pretrained(base_model, low_cpu_mem_usage=True)
            tokenizer = LlamaTokenizer.from_pretrained(tokenizer)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model, low_cpu