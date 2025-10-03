import copy
import csv
import gc
import json
import os
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
except Exception:
    xm = None
    _TPU_AVAILABLE = False


def safe_cuda_available() -> bool:
    """Return True if CUDA is actually usable.

    torch.cuda.is_available() can attempt a lazy init that raises when the
    NVIDIA driver isn't installed. Wrap in try/except to avoid crashing.
    """
    try:
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


def is_xla_device(device) -> bool:
    try:
        s = str(device).lower()
        return "xla" in s or "tpu" in s
    except Exception:
        return False


def resolve_device(device="auto"):
    """Resolve a device string/object to a safe device that won't trigger lazy CUDA init.

    Rules:
    - If device == 'auto': prefer XLA if available and xla_device() succeeds, else CUDA if usable, else CPU.
    - If device is a string that mentions 'cuda' but CUDA is not actually available, fall back to 'cpu' (with a warning).
    - Return either a torch.device-like object (for CPU/CUDA) or the XLA device object.
    """
    if device == "auto":
        if _TPU_AVAILABLE and xm is not None:
            try:
                dev = xm.xla_device()
                print("INFO: Auto-detected and using TPU/XLA device.")
                return dev
            except Exception as e:
                print("WARNING: xla_device() failed:", e)
                # continue to fallback
        # Fallback to CUDA if safe
        if safe_cuda_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # If user explicitly passed a device string or object
    s = str(device).lower()
    if "cuda" in s and not safe_cuda_available():
        print(
            "WARNING: Requested CUDA device but CUDA is not available on this system. Falling back to CPU."
        )
        return torch.device("cpu")
    return device


def set_seed(random_seed: int = 1234):
    torch.manual_seed(random_seed)
    if safe_cuda_available():
        try:
            torch.cuda.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    np.random.seed(random_seed)
    random.seed(random_seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def set_model_device_evalmode(
    model, device, fix_decapoda_config: bool = False, use_bfloat: bool = False
):
    """Move model to device and set evaluation mode safely for CUDA/CPU/XLA.

    Important behavior:
    - We only move the model to a resolved, safe device (resolve_device handles invalid CUDA)
    - For XLA/TPU we allow bfloat16 conversion when requested
    - For CUDA we apply .half() only if CUDA is actually available
    """
    device = resolve_device(device)

    # Move model to the selected device
    try:
        model = model.to(device)
    except Exception as e:
        # If moving to device fails (e.g., misconfigured XLA runtime), fall back to CPU
        print(f"WARNING: moving model to device {device} failed: {e}. Falling back to CPU.")
        device = torch.device("cpu")
        model = model.to(device)

    # Apply half precision only if using a CUDA device and CUDA is usable
    if isinstance(device, torch.device) and device.type == "cuda" and safe_cuda_available():
        try:
            model.half()
        except Exception:
            pass

    if fix_decapoda_config:
        try:
            model.config.pad_token_id = 0
            model.config.bos_token_id = 1
            model.config.eos_token_id = 2
        except Exception:
            pass

    model.eval()

    # If requested, convert to bfloat16 (prefer XLA/TPU)
    if use_bfloat:
        if is_xla_device(device):
            try:
                model = model.bfloat16()
            except Exception:
                pass
        else:
            # Try on non-XLA backends but don't crash if unsupported
            try:
                model = model.bfloat16()
            except Exception:
                pass

    gc.collect()

    # Clear CUDA cache only if using CUDA
    if isinstance(device, torch.device) and device.type == "cuda" and safe_cuda_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return model


def get_model(
    base_model=None,
    ckpt=None,
    lora_ckpt=None,
    tokenizer=None,
    model_type="pretrain",
    device="auto",
    fix_decapoda_config: bool = False,
    use_bfloat: bool = False,
):
    """Load model and tokenizer, with safe device auto-detection for TPU/GPU/CPU.

    Additional safeguards:
    - If base_model looks like a local path, we check for existence and config.json to provide a clear error message.
    - If user passes a device string that requests CUDA but CUDA is not available, we fall back to CPU.
    """
    # Resolve device; do this early to avoid lazy CUDA init surprises
    resolved_device = resolve_device(device)

    # If base_model is a local directory, verify it contains config.json (common user mistake)
    if isinstance(base_model, str) and os.path.exists(base_model):
        if not os.path.isdir(base_model):
            # it's a file path; don't try to validate further
            pass
        else:
            cfg_path = os.path.join(base_model, "config.json")
            if not os.path.exists(cfg_path):
                raise FileNotFoundError(
                    f"The directory '{base_model}' exists but does not contain a 'config.json'.
"
                    "If this is a HuggingFace repo ID, use 'namespace/repo_name'.
"
                    "If this is a local model folder, ensure it contains config.json (or pass the correct folder)."
                )

    tokenizer = base_model if tokenizer is None else tokenizer
    if model_type == "pretrain":
        try:
            config = AutoConfig.from_pretrained(base_model)
        except Exception as e:
            # Provide a more helpful error message
            raise RuntimeError(
                f"Failed to load model config for '{base_model}': {e}
"
                "If 'base_model' points to a local directory, ensure it exists and contains 'config.json'."
            )

        if "gptq" in str(base_model).lower():
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
            and "llama-3" not in str(base_model).lower()
        ):
            model = LlamaForCausalLM.from_pretrained(base_model, low_cpu_mem_usage=True)
            tokenizer = LlamaTokenizer.from_pretrained(tokenizer)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model, low_cpu_mem_usage=True
            )
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    elif model_type in ["pruneLLM", "tune_pruneLLM"]:
        pruned_dict = torch.load(ckpt, map_location="cpu")
        model = pruned_dict["model"]
        tokenizer = pruned_dict["tokenizer"]
        if model_type == "tune_pruneLLM":
            model = PeftModel.from_pretrained(
                model, lora_ckpt, torch_dtype=torch.float16, low_cpu_mem_usage=True
            )
    else:
        raise NotImplementedError

    description = "Model Type: {}
 Base: {} 
 Pruned: {}
 LORA: {}".format(
        model_type, base_model, ckpt, lora_ckpt
    )

    if fix_decapoda_config:
        try:
            tokenizer.pad_token_id = 0
        except Exception:
            pass

    # Finally move model to device and set eval/bfloat as requested
    model = set_model_device_evalmode(model, resolved_device, fix_decapoda_config, use_bfloat)

    return model, tokenizer, description


def copy_weight(model, model_orig, list_pruned_blocks):
    """Copy weights from model_orig into model safely.

    Strategy:
    - Keep model_pruned on CPU while copying to avoid accidental CUDA/XLA calls.
    - Copy all tensors on CPU (safer and avoids device-to-device copies which can trigger lazy init).
    - After copying, the caller will move the pruned model to the target device via set_model_device_evalmode.
    """
    connect_info = {}
    connect_info["model.embed_tokens.weight"] = "model.embed_tokens.weight"
    connect_info["model.norm.weight"] = "model.norm.weight"
    connect_info["lm_head.weight"] = "lm_head.weight"

    k = 0
    for k_orig in range(model_orig.config.__getattribute__("num_hidden_layers")):
        if k_orig in list_pruned_blocks:  # uncopied = pruned blocks
            continue
        connect_info[f"model.layers.{k}."] = f"model.layers.{k_orig}."
        print(f"original model.layers.{k_orig} --> pruned model.layers.{k}")
        k += 1

    print(f" ** excluded blocks {list_pruned_blocks}")

    # Ensure both models are on CPU for safe copying
    try:
        model = model.to(torch.device("cpu"))
    except Exception:
        pass
    try:
        model_orig = model_orig.to(torch.device("cpu"))
    except Exception:
        pass

    t0 = time.perf_counter()
    for k in list(model.state_dict().keys()):
        flag = 0
        k_orig = k
        for prefix_key in connect_info.keys():
            if k.startswith(prefix_key):
                flag = 1
                k_orig = k_orig.replace(prefix_key, connect_info[prefix_key])
                break
        if flag == 1:
            print(f"** forced COPY {k_orig} -> {k}")
            try:
                tgt = model.state_dict()[k]
                src = model_orig.state_dict()[k_orig]
                # Copy via CPU to avoid cross-device issues
                tgt.copy_(src.to("cpu"))
            except Exception as e:
                print(f"Warning: failed to copy {k_orig} -> {k}: {e}")
    print(f"copy time --- {(time.perf_counter()-t0):.1f} sec")

    return model


def get_block_pruned_network(
    model_orig,
    unimportance_order,
    num_pruned_blocks: int = 1,
    device: str = "auto",
    fix_decapoda_config: bool = False,
    use_bfloat: bool = False,
):
    # resolve device safely
    resolved_device = resolve_device(device)

    # Define the block-pruned architecture with random initialization
    config = copy.deepcopy(model_orig.config)
    print(f"# blocks before pruning: {config.num_hidden_layers}")
    config.__setattr__(
        "num_hidden_layers", (config.num_hidden_layers - num_pruned_blocks)
    )
    print(f"# blocks after pruning: {config.num_hidden_layers}")

    # Create pruned model on CPU to do safe weight copying
    model_pruned = AutoModelForCausalLM.from_config(config)

    # Copy the original model's weights to the pruned model (on CPU)
    model_pruned = copy_weight(
        model_pruned, model_orig, unimportance_order[:num_pruned_blocks]
    )

    # Now move pruned model to the requested device and set eval/bfloat
    model_pruned = set_model_device_evalmode(
        model_pruned, resolved_device, fix_decapoda_config, use_bfloat
    )
    return model_pruned


def convert_json2csv_zeroshot(json_path, csv_path):
    with open(json_path, "r") as file:
        data = json.load(file)

    select_key = {
        "boolq": "acc",
        "piqa": "acc",
        "hellaswag": "acc_norm",
        "winogrande": "acc",
        "arc_easy": "acc",
        "arc_challenge": "acc_norm",
        "openbookqa": "acc_norm",
    }

    list_task = []
    list_metric = []
    list_score = []

    ave_score = 0
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for name, key in select_key.items():
            list_task.append(name)
            list_metric.append(key)

            score = data["results"][name][key] * 100
            list_score.append(score)
            ave_score += score

        ave_score /= len(select_key)

        list_task.append("AVE")
        list_metric.append("n/a")
        list_score.append(ave_score)

        writer.writerow(list_task)
        writer.writerow(list_metric)
        writer.writerow(list_score)

    print(csv_path)
