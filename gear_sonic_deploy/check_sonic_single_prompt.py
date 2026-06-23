#!/usr/bin/env python3
"""Run one SONIC tool-calling prompt through a local HF checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max-new-tokens", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sonic_tool_calling_eval import SYSTEM_PROMPT, parse_generated, trim_generation

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=dtype,
        local_files_only=True,
    ).to(device).eval()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": args.prompt},
    ]
    try:
        templated = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
    except TypeError:
        templated = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            tokenize=True,
            add_generation_prompt=True,
        )

    if isinstance(templated, torch.Tensor):
        input_ids = templated.to(device)
        attention_mask = None
    else:
        input_ids = templated["input_ids"].to(device)
        attention_mask = templated.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    generate_kwargs = {
        "input_ids": input_ids,
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": pad_token_id,
    }
    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask

    with torch.no_grad():
        output = model.generate(**generate_kwargs)

    raw = tokenizer.decode(output[0][input_ids.shape[-1] :], skip_special_tokens=False)
    trimmed = trim_generation(raw)
    calls, parse_info = parse_generated(trimmed)
    result = {
        "model": args.model,
        "prompt": args.prompt,
        "device": device,
        "raw_generated_text": raw,
        "generated_text": trimmed,
        "parse_info": parse_info,
        "calls": [{"name": call.name, "arguments": call.arguments} for call in calls],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
