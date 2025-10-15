"""
Model loading utilities.

- Downloads/loads the selected base model into a local cache.
- Applies optional LoRA adapter when present (prod profile).
- Returns a ModelBundle with tokenizer, model, and device info.
"""

import os
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import get_settings


class ModelBundle:
    """Container holding tokenizer, model, and the selected device."""
    def __init__(self, tokenizer, model, device: str):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device


def _select_device() -> str:
    """Return 'cuda' if available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _torch_dtype_for_device() -> torch.dtype:
    """Use float16 on GPU, float32 on CPU."""
    return torch.float16 if torch.cuda.is_available() else torch.float32


def load_model() -> ModelBundle:
    """
    Load the base model and optional LoRA adapter based on environment settings.

    Notes:
    - For prod 4-bit models, ensure bitsandbytes is installed and a capable GPU is used.
    - For dev profile, we default to a small model without an adapter.
    """
    cfg = get_settings()
    os.makedirs(cfg.model_cache, exist_ok=True)

    # Minimal, conservative loading kwargs
    load_kwargs = dict(torch_dtype=_torch_dtype_for_device())

    # Hugging Face token (optional). Use "token" arg for recent transformers versions.
    token = cfg.hf_token if cfg.hf_token else None

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model,
        cache_dir=cfg.model_cache,
        token=token,
    )

    # Base model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            cache_dir=cfg.model_cache,
            token=token,
            **load_kwargs,
        )
    except Exception as e:
        # Provide a clear error to the caller; most common issues:
        # - Insufficient VRAM (prod profile with large model)
        # - Missing bitsandbytes for 4-bit quantization
        # - Gated/private models without HF_TOKEN
        raise RuntimeError(
            f"Failed to load base model '{cfg.base_model}'. "
            f"Profile={cfg.model_profile}, Quantization={cfg.quantization}. Original error: {e}"
        ) from e

    # Optional LoRA adapter
    if cfg.adapter_path:
        try:
            adapter_files = os.listdir(cfg.adapter_path) if os.path.isdir(cfg.adapter_path) else []
            if any(name.startswith("adapter_model") for name in adapter_files):
                model = PeftModel.from_pretrained(model, cfg.adapter_path)
            else:
                print(
                    f"[model_loader] WARNING: Adapter path '{cfg.adapter_path}' missing adapter_model files. "
                    f"Continuing without LoRA."
                )
        except Exception as e:
            print(f"[model_loader] ERROR: Failed loading adapter from '{cfg.adapter_path}': {e}")

    # Finalize
    device = _select_device()
    model.eval()
    model.to(device)

    return ModelBundle(tokenizer=tokenizer, model=model, device=device)