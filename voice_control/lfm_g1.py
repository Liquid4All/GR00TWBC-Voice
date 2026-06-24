"""LiquidAI LFM2.5-250M-G1-FCv1 parser: G1 tool calls -> voice_control planner commands."""

from __future__ import annotations

import ast
import json
import logging
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen

from .config import ParserConfig
from .parsers import (
    CONF_NONE,
    CONF_STRONG,
    ClarifyCommand,
    CrawlStyle,
    NavStyle,
    ParseResult,
    Posture,
    SetBoxingActionCommand,
    SetCrawlCommand,
    SetNavigationCommand,
    SetPostureCommand,
    StopCommand,
    StopReason,
    register,
)
from .skills import BoxingAction

log = logging.getLogger(__name__)

_TOOL_CALL_START = "<|tool_call_start|>"
_TOOL_CALL_END = "<|" + "redacted_tool_call_end_kimi" + "|>"
_TOOL_CALL_RE = re.compile(
    re.escape(_TOOL_CALL_START) + r"\s*(\[.*?\])\s*" + re.escape(_TOOL_CALL_END),
    re.DOTALL,
)

# Matches gear_sonic_deploy/sonic_tool_calling_eval.py SYSTEM_PROMPT (SFT training format).
SYSTEM_PROMPT = """List of tools:

[{"type":"function","function":{"name":"select_motion_mode","description":"Select a planner motion set and mode matching the robot keyboard controller.","parameters":{"type":"object","properties":{"motion_set":{"type":"string","enum":["locomotion","squat_ground","boxing","styled_walking"]},"mode":{"type":"string","enum":["slow_walk","walk","run","happy","stealth","injured","squat","kneel_two_legs","kneel_one_leg","hand_crawl","elbow_crawl","idle_boxing","walk_boxing","left_jab","right_jab","random_punches","left_hook","right_hook","careful","object_carrying","crouch","happy_dance","zombie","point","scared"]}},"required":["motion_set","mode"]}}},{"type":"function","function":{"name":"planner_move","description":"Execute body-relative planner movement at a heading and speed for a fixed duration.","parameters":{"type":"object","properties":{"velocity_mps":{"type":"number"},"heading_deg":{"type":"number"},"yaw_rate_dps":{"type":"number"},"duration_s":{"type":"number"}},"required":["velocity_mps","heading_deg","yaw_rate_dps","duration_s"]}}},{"type":"function","function":{"name":"rotate_in_place","description":"Turn the robot in place by a relative angle.","parameters":{"type":"object","properties":{"angle_deg":{"type":"number"},"yaw_rate_dps":{"type":"number"},"duration_s":{"type":"number"}},"required":["angle_deg","yaw_rate_dps","duration_s"]}}},{"type":"function","function":{"name":"set_body_height","description":"Set body height for squat and ground modes.","parameters":{"type":"object","properties":{"height_m":{"type":"number","minimum":0.2,"maximum":0.8},"duration_s":{"type":"number"}},"required":["height_m","duration_s"]}}},{"type":"function","function":{"name":"hold_pose","description":"Hold the current posture or planner state.","parameters":{"type":"object","properties":{"duration_s":{"type":"number"}},"required":["duration_s"]}}},{"type":"function","function":{"name":"stop","description":"Stop all motion.","parameters":{"type":"object","properties":{"reason":{"type":"string","enum":["user_request","safety","sequence_complete"]}},"required":["reason"]}}}]

Instructions:
You convert voice commands into ordered humanoid robot planner tool calls.
Emit only <|tool_call_start|>[...]<|tool_call_end|>.
Use body-relative coordinates: +vx forward, -vx backward, +vy left, -vy right.
Positive yaw/angle turns left; negative yaw/angle turns right.
Split multi-stage commands into sequential calls.
Use velocity and duration rather than distance in planner_move calls.
Use heading_deg for body-relative translation direction: 0 forward, 90 left, 180 backward, 270 right.
Do not emit prose, explanations, markdown, or tool observations."""

_NAV_MODES = frozenset({"slow_walk", "walk", "run", "happy", "stealth", "injured", "careful", "zombie"})
_STYLED = {"happy": NavStyle.HAPPY, "stealth": NavStyle.STEALTH, "injured": NavStyle.INJURED}
_BOXING_MODES = {
    "idle_boxing": BoxingAction.IDLE, "walk_boxing": BoxingAction.SIDE_STEP,
    "left_jab": BoxingAction.LEFT_JAB, "right_jab": BoxingAction.RIGHT_JAB,
    "left_hook": BoxingAction.LEFT_HOOK, "right_hook": BoxingAction.RIGHT_HOOK,
}
_POSTURE_MODES = {
    "squat": Posture.SQUAT, "kneel_two_legs": Posture.KNEEL_TWO_LEGS,
    "kneel_one_leg": Posture.KNEEL_ONE_LEG, "crouch": Posture.SQUAT,
}
_CRAWL_MODES = {"hand_crawl": CrawlStyle.HAND_CRAWL, "elbow_crawl": CrawlStyle.ELBOW_KNEE}
_STOP_REASON = {
    "user_request": StopReason.USER_REQUEST, "safety": StopReason.SAFETY,
    "sequence_complete": StopReason.UNKNOWN, "user_stop": StopReason.SAFETY,
    "segment_complete": StopReason.UNKNOWN,
}


def resolve_lfm_model_path(model_id: str) -> tuple[str, bool]:
    """Return (path_or_hub_id, local_files_only). Supports ~/GR00T-WBC/models/lfm_g1."""
    raw = model_id.strip()
    if raw.startswith("/GR00T-WBC"):
        path = Path.home() / raw.lstrip("/")
    else:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
    if path.is_dir() and (path / "config.json").is_file():
        return str(path.resolve()), True
    return raw, False


def _parse_version(version: str) -> Tuple[int, ...]:
    parts: List[int] = []
    for token in version.split("+")[0].split("."):
        try:
            parts.append(int(token))
        except ValueError:
            break
    return tuple(parts)


def _require_local_torch() -> "Any":
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            f"torch failed to import in {sys.executable}: {exc}. "
            "On Jetson, PyPI torch often does not work — use NVIDIA's JetPack wheel."
        ) from exc
    try:
        torch.tensor([1.0])
    except Exception as exc:
        raise RuntimeError(f"torch imported but is broken: {exc}") from exc
    if _parse_version(torch.__version__) < (2, 4, 0):
        raise RuntimeError(
            f"torch {torch.__version__} is too old (need >=2.4). "
            f"Your pip upgrade may not have worked on aarch64/Jetson. "
            f"Check: {sys.executable} -m pip show torch"
        )
    from transformers.utils import is_torch_available

    if not is_torch_available():
        raise RuntimeError(
            f"transformers still disabled PyTorch (torch {torch.__version__}). "
            "Reinstall in this order in a NEW shell: "
            f"(1) {sys.executable} -m pip uninstall -y transformers torch; "
            f"(2) {sys.executable} -m pip install 'torch>=2.4.0'; "
            f"(3) {sys.executable} -m pip install 'transformers>=5.0.0'. "
            "On Jetson use NVIDIA's PyTorch index, or set parser.lfm_remote_url."
        )
    return torch


def _require_transformers_v5() -> None:
    import transformers

    if _parse_version(transformers.__version__) < (5, 0):
        raise RuntimeError(
            f"LFM models need transformers>=5.0 (you have {transformers.__version__}). "
            f"Run: {sys.executable} -m pip install 'transformers>=5.0.0' 'tokenizers>=0.21.0'"
        )


def _parse_remote_response(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("raw", "output", "text", "content"):
            if key in payload:
                return str(payload[key])
    raise RuntimeError(f"unexpected remote LFM response: {payload!r}")


def _ast_calls(source: str) -> List[Tuple[str, Dict[str, Any]]]:
    tree = ast.parse(source, mode="eval")
    if not isinstance(tree.body, ast.List):
        raise ValueError("expected a list of tool calls")
    out: List[Tuple[str, Dict[str, Any]]] = []
    for elt in tree.body.elts:
        if not isinstance(elt, ast.Call) or not isinstance(elt.func, ast.Name):
            raise ValueError(f"invalid tool call node: {ast.dump(elt)}")
        args = {kw.arg: ast.literal_eval(kw.value) for kw in elt.keywords if kw.arg}
        out.append((elt.func.id, args))
    return out


def trim_generation(text: str) -> str:
    """Cut generation at the first tool-call end tag (sonic_tool_calling_eval)."""
    if _TOOL_CALL_END in text:
        return text[: text.find(_TOOL_CALL_END) + len(_TOOL_CALL_END)].strip()
    return text.strip()


def build_chat_messages(
    user: str,
    *,
    system: str = SYSTEM_PROMPT,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build chat messages like sonic_tool_calling_eval run_inference."""
    if history:
        return [{"role": "system", "content": system}, *history]
    return [{"role": "system", "content": system}, {"role": "user", "content": user.strip()}]


def extract_g1_tool_calls(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    ends = (_TOOL_CALL_END, "<|tool_call_end|>")
    for end in ends:
        pat = re.compile(
            re.escape(_TOOL_CALL_START) + r"\s*(\[.*?\])\s*" + re.escape(end),
            re.DOTALL,
        )
        match = pat.search(text)
        if match:
            return _ast_calls(match.group(1))
    start = text.find(_TOOL_CALL_START)
    if start >= 0:
        rest = text[start + len(_TOOL_CALL_START):]
        lb = rest.find("[")
        if lb >= 0:
            depth = 0
            for i, ch in enumerate(rest[lb:], start=lb):
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        return _ast_calls(rest[lb : i + 1])
    preview = text[:800] + ("..." if len(text) > 800 else "")
    raise ValueError(f"no tool call block in model output: {preview!r}")


class G1ToolMapper:
    def __init__(self, cfg: ParserConfig) -> None:
        self.cfg = cfg
        self.motion_set = "locomotion"
        self.mode = "walk"

    def map(self, name: str, args: Dict[str, Any]) -> Optional[Any]:
        if name == "select_motion_mode":
            self.motion_set = str(args.get("motion_set", self.motion_set))
            self.mode = str(args.get("mode", self.mode))
            return None
        if name == "stop":
            reason = _STOP_REASON.get(str(args.get("reason", "user_request")), StopReason.USER_REQUEST)
            return StopCommand(reason=reason)
        if name == "reset_motion_momentum":
            return SetNavigationCommand(
                velocity_mps=0.0, heading_deg=0.0, style=self._nav_style(),
                duration_s=float(args.get("duration_s") or 0.2),
            )
        if name == "hold_pose":
            return SetNavigationCommand(
                velocity_mps=0.0, heading_deg=0.0, style=self._nav_style(),
                duration_s=float(args["duration_s"]),
            )
        if name == "set_body_height":
            return SetPostureCommand(
                posture=Posture.SQUAT, pelvis_height_m=float(args["height_m"]),
                duration_s=float(args["duration_s"]),
            )
        if name == "rotate_in_place":
            return SetNavigationCommand(
                velocity_mps=0.0, heading_deg=float(args["angle_deg"]), style=self._nav_style(),
                duration_s=float(args["duration_s"]),
            )
        if name == "planner_move":
            return self._planner_move(args)
        raise ValueError(f"unknown G1 tool {name!r}")

    def _nav_style(self) -> NavStyle:
        if self.mode in _STYLED:
            return _STYLED[self.mode]
        if self.mode == "run":
            return NavStyle.RUNNING
        return NavStyle.WALKING

    def _planner_move(self, args: Dict[str, Any]) -> Any:
        velocity = float(args["velocity_mps"])
        heading = float(args["heading_deg"])
        duration = float(args["duration_s"])
        mode = self.mode
        if mode in _BOXING_MODES and velocity <= 1e-6:
            return SetBoxingActionCommand(action=_BOXING_MODES[mode], duration_s=duration)
        if mode in _POSTURE_MODES:
            return SetPostureCommand(posture=_POSTURE_MODES[mode], duration_s=duration)
        if mode in _CRAWL_MODES:
            return SetCrawlCommand(
                velocity_mps=velocity, heading_deg=heading,
                crawl_style=_CRAWL_MODES[mode], duration_s=duration,
            )
        return SetNavigationCommand(
            velocity_mps=velocity, heading_deg=heading,
            style=self._nav_style(), duration_s=duration,
        )


class LFMG1Parser:
    def __init__(self, cfg: ParserConfig) -> None:
        self.cfg = cfg
        self.mapper = G1ToolMapper(cfg)
        self._model = None
        self._tokenizer = None

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
        del boxing_active  # G1 FC model owns mode selection via select_motion_mode
        try:
            raw = self._generate(text, messages=messages)
            calls = extract_g1_tool_calls(trim_generation(raw))
        except Exception as exc:
            msg = str(exc) or repr(exc)
            log.warning("LFM G1 parse failed (%s): %s", type(exc).__name__, msg, exc_info=True)
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
                command=cmd, reason="lfm_g1",
            ))
        return results or [self._clarify(text, "no executable tool calls")]

    def _generate(self, text: str, *, messages: Optional[List[Dict[str, str]]] = None) -> str:
        if self.cfg.lfm_remote_url:
            return self._generate_remote(text)
        return self._generate_local(text, messages=messages)

    def _resolve_device(self, torch: Any) -> str:
        want = (self.cfg.lfm_device or "cpu").lower()
        if want != "auto":
            return want
        if torch.cuda.is_available():
            try:
                torch.zeros(1, device="cuda")
                return "auto"
            except Exception as exc:
                log.warning("CUDA reported available but unusable (%s); using CPU", exc)
        else:
            log.warning("CUDA unavailable; using CPU for LFM (set parser.lfm_device: cpu to silence)")
        return "cpu"

    def _generate_local(self, text: str, *, messages: Optional[List[Dict[str, str]]] = None) -> str:
        import torch

        self._ensure_model()
        device = self._model.device
        chat = build_chat_messages(text, history=messages)
        try:
            templated = self._tokenizer.apply_chat_template(
                chat,
                return_tensors="pt",
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            )
        except TypeError:
            templated = self._tokenizer.apply_chat_template(
                chat,
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

        pad_token_id = self._tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self._tokenizer.eos_token_id if self._tokenizer.eos_token_id is not None else 0

        generate_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "do_sample": False,
            "max_new_tokens": self.cfg.lfm_max_new_tokens,
            "pad_token_id": pad_token_id,
        }
        if attention_mask is not None:
            generate_kwargs["attention_mask"] = attention_mask

        with torch.no_grad():
            output = self._model.generate(**generate_kwargs)

        raw = self._tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=False)
        log.info("LFM raw output: %r", raw[:500])
        return raw

    def _generate_remote(self, text: str) -> str:
        url = self.cfg.lfm_remote_url
        if not url:
            raise RuntimeError("lfm_remote_url is empty")
        log.info("LFM remote parse via %s", url)
        req = Request(
            url,
            data=json.dumps({"text": text}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode())
        except URLError as exc:
            raise RuntimeError(f"lfm remote failed ({url}): {exc}") from exc
        return _parse_remote_response(payload)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if self.cfg.lfm_remote_url:
            return

        torch = _require_local_torch()
        _require_transformers_v5()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                f"transformers model import failed after torch {torch.__version__}: {exc}. "
                f"Run: python -m voice_control.lfm_g1 --diagnose"
            ) from exc
        device = self._resolve_device(torch)
        model_path, local_only = resolve_lfm_model_path(self.cfg.lfm_model_id)
        log.info(
            "Loading LFM G1 model %s (torch %s, device %s, local_only=%s) ...",
            model_path, torch.__version__, device, local_only,
        )
        load_kw: Dict[str, Any] = {"trust_remote_code": True, "local_files_only": local_only}
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, **load_kw)
        kwargs: Dict[str, Any] = {"trust_remote_code": True, "local_files_only": local_only}
        if device == "cpu":
            kwargs["device_map"] = "cpu"
            kwargs["dtype"] = torch.float32
        else:
            kwargs["device_map"] = device
            kwargs["dtype"] = torch.bfloat16
        self._model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self._model.eval()

    def _clarify(self, text: str, reason: str) -> ParseResult:
        return ParseResult(
            ok=False, confidence=CONF_NONE, raw_text=text, normalized_text=text.strip().lower(),
            command=ClarifyCommand(
                question="Could not parse that as a robot command.", original_text=text,
            ),
            reason=reason,
        )


@register("lfm_g1")
def _build_lfm_g1(cfg: ParserConfig) -> LFMG1Parser:
    return LFMG1Parser(cfg)


def serve(cfg: Optional[ParserConfig] = None, host: str = "0.0.0.0", port: int = 8765) -> None:
    """Run local LFM inference HTTP server (for robot parser.lfm_remote_url)."""
    local_cfg = cfg or ParserConfig()
    local_cfg.lfm_remote_url = None
    parser = LFMG1Parser(local_cfg)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            log.info("lfm_server " + fmt, *args)

        def do_POST(self) -> None:
            if self.path not in ("/", "/parse"):
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                data = json.loads(self.rfile.read(length).decode() or "{}")
                raw = parser._generate_local(data.get("text", ""))
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

    log.info("LFM server on http://%s:%d/parse", host, port)
    HTTPServer((host, port), Handler).serve_forever()


def diagnose_lfm_env() -> int:
    """Print torch/transformers state for debugging Jetson installs."""
    rc = 0
    print(f"python: {sys.executable}")
    print(f"version: {sys.version}")
    has_int_digits = hasattr(sys, "get_int_max_str_digits")
    print(f"sys.get_int_max_str_digits: {has_int_digits}")
    if "rc" in sys.version.lower() or not has_int_digits:
        print(
            "WARNING: Python looks like an old 3.11 pre-release. torch 2.12 + transformers 5 "
            "need a final Python 3.11.9+ or 3.12 — recreate .venv_voice with a proper interpreter."
        )
        rc = 1
    for pkg in ("torch", "transformers", "tokenizers", "accelerate"):
        try:
            mod = __import__(pkg)
            print(f"{pkg}: {getattr(mod, '__version__', '?')} @ {getattr(mod, '__file__', '?')}")
        except Exception as exc:
            print(f"{pkg}: FAILED ({exc})")
            rc = 1
    try:
        import torch
        t = torch.tensor([1.0])
        print(f"torch tensor ok: {t.item()}")
    except Exception as exc:
        print(f"torch tensor FAILED: {exc}")
        return 1
    try:
        from transformers.utils import is_torch_available
        print(f"transformers is_torch_available: {is_torch_available()}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("AutoModelForCausalLM: ok")
    except Exception as exc:
        print(f"transformers model import FAILED: {exc}")
        cause = exc.__cause__
        while cause is not None:
            print(f"  caused by: {type(cause).__name__}: {cause}")
            cause = cause.__cause__
        if not has_int_digits:
            print(
                "Fix: install Python 3.11.9+ or 3.12, recreate the venv, reinstall packages, "
                "or use parser.lfm_remote_url with the server on a laptop."
            )
        return 1
    if rc:
        return rc
    print("LFM local env looks OK.")
    return 0


def main() -> None:
    import argparse
    from .config import Config

    p = argparse.ArgumentParser(description="LFM G1 server / diagnostics")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--serve", action="store_true", help="Run HTTP parse server")
    p.add_argument("--diagnose", action="store_true", help="Print torch/transformers diagnostics")
    args = p.parse_args()
    if args.diagnose:
        raise SystemExit(diagnose_lfm_env())
    cfg = Config.from_yaml(args.config).parser if args.config else ParserConfig()
    serve(cfg, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
