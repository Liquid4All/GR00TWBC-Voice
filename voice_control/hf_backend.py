"""Local HuggingFace ``transformers`` text-generation backend.

Loads a HuggingFace causal-LM checkpoint in-process and generates text on
device. Default target is the LiquidAI LFM2 on-device reasoning model
(``tim_grpo230M_..._HF``), a small (~230M) hybrid conv/attention model designed
for edge deployment (Jetson Orin, etc.).

Heavy dependencies (``torch`` / ``transformers``) are imported lazily so the rest
of the package imports and runs without them; callers fall back to a heuristic if
this backend is unavailable. The loaded model/tokenizer are cached per model id
so a Jetson runtime only pays the load cost once.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, Optional, Tuple

from .config import ParserConfig

log = logging.getLogger(__name__)

# model_id -> (tokenizer, model). Cached across calls / pipeline instances.
_MODEL_CACHE: Dict[str, Tuple[object, object]] = {}
_CACHE_LOCK = threading.Lock()


def _resolve_dtype(name: str):
    import torch

    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "auto": "auto",
    }.get(str(name).lower(), "auto")


def _resolve_device(name: str) -> str:
    import torch

    name = str(name).lower()
    if name in ("cpu", "cuda", "mps"):
        return name
    # auto
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load(cfg: ParserConfig) -> Tuple[object, object, str]:
    """Return (tokenizer, model, device), loading + caching on first use."""

    model_id = cfg.hf_model_id
    device = _resolve_device(cfg.hf_device)
    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(model_id)
        if cached is not None:
            return cached[0], cached[1], device

        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("Loading HF model %r (device=%s, dtype=%s)", model_id, device, cfg.hf_dtype)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=_resolve_dtype(cfg.hf_dtype),
            trust_remote_code=True,
        )
        model.to(device)
        model.eval()
        _MODEL_CACHE[model_id] = (tokenizer, model)
        log.info("HF model %r loaded.", model_id)
        return tokenizer, model, device


def is_available() -> bool:
    """True if torch + transformers can be imported."""

    import importlib.util

    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("transformers") is not None
    )


def generate(cfg: ParserConfig, prompt: str, max_new_tokens: Optional[int] = None) -> str:
    """Generate a completion for ``prompt`` using the configured HF model."""

    import torch

    tokenizer, model, device = _load(cfg)

    # Prefer the model's chat template when it defines one (LFM2 ships one).
    # return_dict gives us an attention_mask too, which generate() needs to
    # avoid undefined behaviour when there's no pad token.
    messages = [{"role": "user", "content": prompt}]
    inputs = None
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    except Exception:
        inputs = None
    if inputs is None or not hasattr(inputs, "get") or inputs.get("input_ids") is None:
        # No usable chat template -> plain tokenization.
        inputs = tokenizer(prompt, return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[-1]

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    temperature = float(cfg.hf_temperature)
    do_sample = temperature > 0.0
    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens or cfg.hf_max_new_tokens),
        do_sample=do_sample,
        pad_token_id=pad_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)

    # generate() may return a tensor or a GenerateOutput; normalise to the
    # sequence tensor, then decode only the newly generated tokens.
    sequences = getattr(output, "sequences", output)
    new_tokens = sequences[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
