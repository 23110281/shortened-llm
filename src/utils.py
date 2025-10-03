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
    if device == 'auto':
        if _TPU_AVAILABLE:
            device = xm.xla_device()
            print("INFO: Auto-detected and using TPU device.")
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"INFO: Auto-detected and using {device} device.")

    model = model.to(device)

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
    if "cuda" in str(device):
        torch.cuda.empty_cache()

    return model, device


def get_model(
    base_model=None,
    ckpt=None,
    lora_ckpt=None,
    tokenizer=None,
    model_type="pretrain",
    device="auto",
    fix_decapoda_config=False,
    use_bfloat=False,
):
    tokenizer = base_model if tokenizer is None else tokenizer
    if model_type == "pretrain":
        config = AutoConfig.from_pretrained(base_model)