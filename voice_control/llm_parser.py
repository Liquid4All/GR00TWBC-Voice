"""Optional local-LLM fallback parser.

Off by default. Only consulted when the deterministic parser returns ``clarify``
*and* ``parser.use_llm_fallback`` is true. The LLM is asked to emit a single
JSON tool call constrained to the exact closed schema; the output is then
validated with Pydantic, so a hallucinated or malformed payload is rejected.

The backend is a local ``llama.cpp`` server (``/completion`` HTTP endpoint) by
default. Everything runs on-device; no cloud services are used. Networking uses
only the Python standard library so this module imports without extra deps.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

from pydantic import ValidationError

from .config import ParserConfig
from .schemas import ClarifyCommand, ParseResult, validate_tool_call

log = logging.getLogger(__name__)


def call_llm_completion(
    cfg: ParserConfig,
    prompt: str,
    *,
    grammar: Optional[str] = None,
    n_predict: int = 256,
    temperature: float = 0.0,
    stop: Optional[List[str]] = None,
    timeout: float = 10.0,
) -> str:
    """Call the configured local LLM ``/completion`` endpoint and return text.

    Shared by the fallback parser and the duration estimator. Dispatches to the
    configured backend:

    * ``llama_cpp``      -- local llama.cpp HTTP ``/completion`` server (stdlib only).
    * ``hf_transformers``-- a HuggingFace checkpoint loaded in-process.

    Everything runs on-device; no cloud services.
    """

    if cfg.llm_backend in ("hf_transformers", "transformers", "hf"):
        from .hf_backend import generate as hf_generate

        # n_predict is a llama.cpp notion; the HF backend uses hf_max_new_tokens
        # so a reasoning model has room to think before emitting the answer.
        return hf_generate(cfg, prompt)

    if cfg.llm_backend != "llama_cpp":
        raise RuntimeError(f"Unsupported llm_backend: {cfg.llm_backend!r}")

    import urllib.request

    payload: dict = {"prompt": prompt, "temperature": temperature, "n_predict": n_predict}
    if grammar:
        payload["grammar"] = grammar
    if stop:
        payload["stop"] = stop
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        cfg.llm_endpoint, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - local only
        data = json.loads(resp.read().decode("utf-8"))
    # llama.cpp server returns {"content": "..."}
    return data.get("content", "")

ALLOWED_TOOLS = (
    "stop",
    "set_navigation",
    "set_crawl",
    "set_posture",
    "set_boxing_action",
    "get_up",
    "clarify",
)

SYSTEM_PROMPT = (
    "You translate a single spoken robot command into ONE JSON tool call for a "
    "humanoid kinematic motion planner.\n"
    "Return only valid JSON. Use only the allowed tools. If the request is "
    "unsupported or ambiguous, return clarify.\n"
    "Do not invent tools or fields.\n\n"
    "Allowed tools and schemas:\n"
    '  {"tool":"stop","reason":"user_request|safety|unknown"}\n'
    '  {"tool":"set_navigation","velocity_mps":float,"heading_deg":float,'
    '"style":"walking|running|happy|stealth|injured|drunken","duration_s":float|null}\n'
    '  {"tool":"set_crawl","velocity_mps":float,"heading_deg":float,'
    '"crawl_style":"elbow_knee|hand_crawl","duration_s":float|null}\n'
    '  {"tool":"set_posture","posture":"squat|kneel_one_leg|kneel_two_legs|stand",'
    '"pelvis_height_m":float|null,"duration_s":float|null}\n'
    '  {"tool":"set_boxing_action","action":"idle|stance|block|left_jab|right_jab|'
    'left_hook|right_hook|side_step","duration_s":float|null}\n'
    '  {"tool":"get_up"}\n'
    '  {"tool":"clarify","question":string,"original_text":string}\n\n'
    "Heading convention: 0=forward, -90=left, 90=right, 180=backward.\n"
)

# A GBNF grammar to hard-constrain llama.cpp output to JSON objects. Kept loose
# (any JSON object) because the precise tool union is validated by Pydantic.
JSON_GRAMMAR = r"""
root   ::= object
object ::= "{" ws (string ws ":" ws value (ws "," ws string ws ":" ws value)*)? ws "}"
value  ::= object | array | string | number | "true" | "false" | "null"
array  ::= "[" ws (value (ws "," ws value)*)? ws "]"
string ::= "\"" ([^"\\] | "\\" .)* "\""
number ::= "-"? [0-9]+ ("." [0-9]+)?
ws     ::= [ \t\n]*
"""


class LLMParser:
    """Local LLM fallback parser (llama.cpp ``/completion`` server)."""

    def __init__(self, cfg: ParserConfig) -> None:
        self.cfg = cfg

    def parse(self, raw_text: str, normalized_text: str) -> ParseResult:
        prompt = self._build_prompt(raw_text)
        try:
            completion = self._call_backend(prompt)
        except Exception as exc:  # network / backend errors -> clarify
            log.warning("LLM backend unavailable: %s", exc)
            return self._clarify(raw_text, normalized_text, f"LLM backend error: {exc}")

        payload = _extract_json(completion)
        if payload is None:
            return self._clarify(raw_text, normalized_text, "LLM returned no JSON.")

        try:
            command = validate_tool_call(payload)
        except ValidationError as exc:
            log.warning("Rejected malformed LLM tool call: %s", exc)
            return self._clarify(raw_text, normalized_text, "LLM output failed schema validation.")

        confidence = 0.0 if getattr(command, "tool", None) == "clarify" else 0.8
        return ParseResult(
            ok=confidence > 0.0,
            confidence=confidence,
            raw_text=raw_text,
            normalized_text=normalized_text,
            command=command,
            reason="llm_fallback",
        )

    # ------------------------------------------------------------------ #
    def _build_prompt(self, raw_text: str) -> str:
        return f"{SYSTEM_PROMPT}\nCommand: {raw_text!r}\nJSON:"

    def _call_backend(self, prompt: str) -> str:
        return call_llm_completion(
            self.cfg, prompt, grammar=JSON_GRAMMAR, n_predict=256, stop=["\n\n"]
        )

    def _clarify(self, raw_text: str, norm: str, reason: str) -> ParseResult:
        return ParseResult(
            ok=False,
            confidence=0.0,
            raw_text=raw_text,
            normalized_text=norm,
            command=ClarifyCommand(
                question="Could you rephrase that as a movement command?",
                original_text=raw_text,
            ),
            reason=reason,
        )


def _extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of an LLM completion."""

    if not text:
        return None
    # Fast path: whole string is JSON.
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None
