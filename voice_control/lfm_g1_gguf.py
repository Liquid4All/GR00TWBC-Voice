"""GGUF (llama.cpp) backend for the LFM G1 tool-calling parser.

Uses ``llama-cpp-python`` for generation and the HuggingFace tokenizer from
``parser.lfm_model_id`` (or ``parser.lfm_tokenizer_id``) only for chat templating.
The transformers + torch path remains in ``voice_control.lfm_g1``.

Example::

    python -m voice_control.lfm_g1_gguf --diagnose
    python -m voice_control.lfm_g1_gguf --text "walk forward" --dry-run
    python -m voice_control.cli --parser lfm_g1_gguf --text "walk forward" --dry-run
"""

from __future__ import annotations

import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import Config, ParserConfig
from .lfm_g1 import (
    G1ToolMapper,
    build_chat_messages,
    extract_g1_tool_calls,
    resolve_lfm_model_path,
    trim_generation,
)
from .parsers import CONF_NONE, CONF_STRONG, ClarifyCommand, ParseResult, register

log = logging.getLogger(__name__)


def resolve_gguf_path(raw_path: str) -> str:
    """Resolve GGUF file path; supports ``/GR00T-WBC/...`` and directories containing ``*.gguf``."""
    raw = raw_path.strip()
    if not raw:
        raise ValueError("lfm_gguf_path is empty")

    if raw.startswith("/GR00T-WBC"):
        path = Path.home() / raw.lstrip("/")
    else:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path

    if path.is_dir():
        matches = sorted(path.glob("*.gguf"))
        if not matches:
            raise FileNotFoundError(f"no .gguf file under {path}")
        return str(matches[0].resolve())

    if path.suffix.lower() != ".gguf":
        raise ValueError(f"expected a .gguf file, got {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path.resolve())


def _require_llama_cpp() -> Any:
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            f"llama-cpp-python is required for parser backend lfm_g1_gguf: {exc}. "
            f"Install with: {sys.executable} -m pip install llama-cpp-python"
        ) from exc
    return Llama


def _require_tokenizer(tokenizer_id: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            f"transformers is required for chat templating (tokenizer only): {exc}. "
            f"Install with: {sys.executable} -m pip install 'transformers>=5.0.0'"
        ) from exc
    path, local_only = resolve_lfm_model_path(tokenizer_id)
    return AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, local_files_only=local_only,
    )


def render_chat_prompt(
    tokenizer: Any,
    text: str,
    *,
    messages: Optional[List[Dict[str, str]]] = None,
) -> str:
    chat = build_chat_messages(text, history=messages)
    try:
        return tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(chat, add_generation_prompt=True)


class LFMG1GGUFParser:
    """LFM G1 parser using a local GGUF weights file via llama.cpp."""

    def __init__(self, cfg: ParserConfig) -> None:
        self.cfg = cfg
        self.mapper = G1ToolMapper(cfg)
        self._llm: Any = None
        self._tokenizer: Any = None

    def parse(self, text: str, *, boxing_active: bool = False) -> ParseResult:
        plan = self.parse_plan(text, boxing_active=boxing_active)
        return plan[0] if plan else self._clarify(text, "empty plan")

    def parse_plan(
        self,
        text: str,
        *,
        boxing_active: bool = False,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> List[ParseResult]:
        del boxing_active
        try:
            raw = self._generate(text, messages=messages)
            calls = extract_g1_tool_calls(trim_generation(raw))
        except Exception as exc:
            msg = str(exc) or repr(exc)
            log.warning("LFM G1 GGUF parse failed (%s): %s", type(exc).__name__, msg, exc_info=True)
            return [self._clarify(text, msg)]
        results: List[ParseResult] = []
        for name, args in calls:
            try:
                cmd = self.mapper.map(name, args)
            except Exception as exc:
                log.warning("G1 tool map failed for %s(%s): %s", name, args, exc)
                return [self._clarify(text, f"{name}: {exc}")]
            if cmd is None:
                continue
            results.append(ParseResult(
                ok=True, confidence=CONF_STRONG, raw_text=text, normalized_text=text.strip(),
                command=cmd, reason="lfm_g1_gguf",
            ))
        return results or [self._clarify(text, "no executable tool calls")]

    def _generate(self, text: str, *, messages: Optional[List[Dict[str, str]]] = None) -> str:
        self._ensure_model()
        prompt = render_chat_prompt(self._tokenizer, text, messages=messages)
        log.debug("GGUF prompt length: %d chars", len(prompt))
        out = self._llm.create_completion(
            prompt,
            max_tokens=self.cfg.lfm_max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            repeat_penalty=1.05,
        )
        raw = out["choices"][0]["text"]
        log.info("LFM GGUF raw output: %r", raw[:500])
        return raw

    def _ensure_model(self) -> None:
        if self._llm is not None:
            return
        Llama = _require_llama_cpp()
        gguf_path = resolve_gguf_path(self.cfg.lfm_gguf_path)
        tokenizer_id = (self.cfg.lfm_tokenizer_id or self.cfg.lfm_model_id).strip()
        log.info("Loading GGUF model %s (n_ctx=%d, n_gpu_layers=%d) ...",
                 gguf_path, self.cfg.lfm_gguf_n_ctx, self.cfg.lfm_gguf_n_gpu_layers)
        self._tokenizer = _require_tokenizer(tokenizer_id)
        self._llm = Llama(
            model_path=gguf_path,
            n_ctx=self.cfg.lfm_gguf_n_ctx,
            n_gpu_layers=self.cfg.lfm_gguf_n_gpu_layers,
            verbose=False,
        )

    def _clarify(self, text: str, reason: str) -> ParseResult:
        return ParseResult(
            ok=False, confidence=CONF_NONE, raw_text=text, normalized_text=text.strip().lower(),
            command=ClarifyCommand(
                question="Could not parse that as a robot command.", original_text=text,
            ),
            reason=reason,
        )


@register("lfm_g1_gguf")
def _build_lfm_g1_gguf(cfg: ParserConfig) -> LFMG1GGUFParser:
    return LFMG1GGUFParser(cfg)


def serve(cfg: Optional[ParserConfig] = None, host: str = "0.0.0.0", port: int = 8766) -> None:
    """Run GGUF inference HTTP server (default port 8766, separate from ``lfm_g1``)."""
    parser = LFMG1GGUFParser(cfg or ParserConfig())

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            log.info("lfm_gguf_server " + fmt, *args)

        def do_POST(self) -> None:
            if self.path not in ("/", "/parse"):
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                data = json.loads(self.rfile.read(length).decode() or "{}")
                raw = parser._generate(data.get("text", ""))
                body = json.dumps({"raw": raw}).encode()
            except Exception as exc:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(exc)}).encode())
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    log.info("LFM GGUF server on http://%s:%d/parse", host, port)
    HTTPServer((host, port), Handler).serve_forever()


def diagnose_gguf_env(cfg: Optional[ParserConfig] = None) -> int:
    """Check llama-cpp-python, GGUF path, and tokenizer load."""
    rc = 0
    local_cfg = cfg or ParserConfig()
    print(f"python: {sys.executable}")
    print(f"version: {sys.version}")

    try:
        import llama_cpp
        print(f"llama_cpp: {getattr(llama_cpp, '__version__', '?')} @ {llama_cpp.__file__}")
    except Exception as exc:
        print(f"llama_cpp: FAILED ({exc})")
        rc = 1

    try:
        gguf = resolve_gguf_path(local_cfg.lfm_gguf_path)
        print(f"gguf path: ok ({gguf})")
    except Exception as exc:
        print(f"gguf path: FAILED ({exc})")
        rc = 1

    tokenizer_id = (local_cfg.lfm_tokenizer_id or local_cfg.lfm_model_id).strip()
    try:
        tok = _require_tokenizer(tokenizer_id)
        sample = render_chat_prompt(tok, "walk forward")
        print(f"tokenizer: ok ({tokenizer_id}), sample prompt {len(sample)} chars")
    except Exception as exc:
        print(f"tokenizer: FAILED ({exc})")
        rc = 1

    if rc:
        return rc
    print("LFM GGUF env looks OK.")
    return 0


def run_text_once(config: Config, text: str) -> None:
    from .pipeline import VoicePipeline, run_text_once as _run

    pipeline = VoicePipeline(config)
    _run(pipeline, text)
    pipeline.close()


def main() -> None:
    import argparse

    from .config import Config

    p = argparse.ArgumentParser(description="LFM G1 GGUF parser / server / diagnostics")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--text", type=str, default=None, help="Parse one utterance and print planner output")
    p.add_argument("--dry-run", action="store_true", help="With --text, never send to robot")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--serve", action="store_true", help="Run HTTP parse server")
    p.add_argument("--diagnose", action="store_true", help="Print GGUF / tokenizer diagnostics")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    cfg.parser.backend = "lfm_g1_gguf"

    if args.diagnose:
        raise SystemExit(diagnose_gguf_env(cfg.parser))
    if args.text is not None:
        if args.dry_run:
            cfg.safety.dry_run = True
        run_text_once(cfg, args.text)
        return
    serve(cfg.parser, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
