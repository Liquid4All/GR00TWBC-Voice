"""LiquidAI LFM2.5-250M-G1-FCv1 parser: G1 tool calls -> voice_control planner commands."""

from __future__ import annotations

import ast
import json
import logging
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
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
    validate_tool_call,
)
from .skills import BoxingAction

log = logging.getLogger(__name__)

_TOOL_CALL_START = "<|tool_call_start|>"
_TOOL_CALL_END = "<|" + "redacted_tool_call_end_kimi" + "|>"
_TOOL_CALL_RE = re.compile(
    re.escape(_TOOL_CALL_START) + r"\s*(\[.*?\])\s*" + re.escape(_TOOL_CALL_END),
    re.DOTALL,
)

G1_SYSTEM_PROMPT = (
    'List of tools:\n\n[{"type":"function","function":{"name":"select_motion_mode","description":'
    '"Select a planner motion set and mode matching the robot keyboard controller.",'
    '"parameters":{"type":"object","properties":{"motion_set":{"type":"string","enum":'
    '["locomotion","squat_ground","boxing","styled_walking"]},"mode":{"type":"string","enum":'
    '["slow_walk","walk","run","happy","stealth","injured","squat","kneel_two_legs","kneel_one_leg",'
    '"hand_crawl","elbow_crawl","idle_boxing","walk_boxing","left_jab","right_jab","random_punches",'
    '"left_hook","right_hook","careful","object_carrying","crouch","happy_dance","zombie","point","scared"]}},'
    '"required":["motion_set","mode"]}}},{"type":"function","function":{"name":"planner_move",'
    '"description":"Execute body-relative planner movement at a heading and speed for a fixed duration.",'
    '"parameters":{"type":"object","properties":{"velocity_mps":{"type":"number"},"heading_deg":{"type":"number"},'
    '"yaw_rate_dps":{"type":"number"},"duration_s":{"type":"number"}},"required":["velocity_mps","heading_deg",'
    '"yaw_rate_dps","duration_s"]}}},{"type":"function","function":{"name":"rotate_in_place",'
    '"description":"Turn the robot in place by a relative angle.","parameters":{"type":"object","properties":'
    '{"angle_deg":{"type":"number"},"yaw_rate_dps":{"type":"number"},"duration_s":{"type":"number"}},'
    '"required":["angle_deg","yaw_rate_dps","duration_s"]}}},{"type":"function","function":{"name":"set_body_height",'
    '"description":"Set body height for squat and ground modes.","parameters":{"type":"object","properties":'
    '{"height_m":{"type":"number","minimum":0.2,"maximum":0.8},"duration_s":{"type":"number"}},'
    '"required":["height_m","duration_s"]}}},{"type":"function","function":{"name":"hold_pose",'
    '"description":"Hold the current posture or planner state.","parameters":{"type":"object","properties":'
    '{"duration_s":{"type":"number"}},"required":["duration_s"]}}},{"type":"function","function":{"name":'
    '"reset_motion_momentum","description":"Immediately reset movement momentum without exiting control.",'
    '"parameters":{"type":"object","properties":{"reason":{"type":"string","enum":["segment_complete",'
    '"user_stop","safety"]}},"required":["reason"]}}},{"type":"function","function":{"name":"stop",'
    '"description":"Stop all motion.","parameters":{"type":"object","properties":{"reason":{"type":"string","enum":'
    '["user_request","safety","sequence_complete"]}},"required":["reason"]}}}]\n\nInstructions:\n'
    "You convert voice commands into ordered humanoid robot planner tool calls.\n"
    "Emit only " + _TOOL_CALL_START + "[...]" + _TOOL_CALL_END + ".\n"
    "Use body-relative coordinates: +vx forward, -vx backward, +vy left, -vy right.\n"
    "Positive yaw/angle turns left; negative yaw/angle turns right.\n"
    "Split multi-stage commands into sequential calls.\n"
    "Use velocity and duration rather than distance in planner_move calls.\n"
    "Use heading_deg for body-relative translation direction: 0 forward, 90 left, 180 backward, 270 right.\n"
    "Use reset_motion_momentum after each timed movement or turn segment.\n"
    "Always include duration_s on timed actions; infer it from the utterance when possible.\n"
    "Do not emit prose, explanations, markdown, or tool observations."
)

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


def _parse_version(version: str) -> Tuple[int, ...]:
    parts: List[int] = []
    for token in version.split("+")[0].split("."):
        try:
            parts.append(int(token))
        except ValueError:
            break
    return tuple(parts)


def _require_local_torch() -> None:
    try:
        import torch
        from transformers.utils import is_torch_available
    except Exception as exc:
        raise RuntimeError(
            f"lfm_g1 needs torch>=2.4 in {sys.executable}: {exc}. "
            f"Run: {sys.executable} -m pip install 'torch>=2.4.0' 'transformers>=5.0.0'"
        ) from exc
    if not is_torch_available():
        raise RuntimeError(
            f"transformers disabled PyTorch (installed torch {torch.__version__}, need >=2.4). "
            f"Upgrade: {sys.executable} -m pip install -U 'torch>=2.4.0'. "
            "On Jetson use NVIDIA's PyTorch wheel for your JetPack, or set parser.lfm_remote_url "
            "and run `python -m voice_control.lfm_g1 --serve` on a laptop."
        )
    if _parse_version(torch.__version__) < (2, 4, 0):
        raise RuntimeError(
            f"torch {torch.__version__} is too old for transformers 5 (need >=2.4). "
            f"Upgrade: {sys.executable} -m pip install -U 'torch>=2.4.0'"
        )


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


def extract_g1_tool_calls(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    match = _TOOL_CALL_RE.search(text)
    if not match:
        raise ValueError(f"no {_TOOL_CALL_START}...{_TOOL_CALL_END} block found")
    return _ast_calls(match.group(1))


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
                velocity_mps=max(0.0, velocity), heading_deg=heading,
                crawl_style=_CRAWL_MODES[mode], duration_s=duration,
            )
        return SetNavigationCommand(
            velocity_mps=max(0.0, velocity), heading_deg=heading,
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

    def parse_plan(self, text: str, *, boxing_active: bool = False) -> List[ParseResult]:
        del boxing_active  # G1 FC model owns mode selection via select_motion_mode
        try:
            raw = self._generate(text)
            calls = extract_g1_tool_calls(raw)
        except Exception as exc:
            log.warning("LFM G1 parse failed: %s", exc, exc_info=log.isEnabledFor(logging.DEBUG))
            return [self._clarify(text, str(exc))]
        results: List[ParseResult] = []
        for name, args in calls:
            filled = self._defaults(name, args)
            try:
                cmd = self.mapper.map(name, filled)
            except Exception as exc:
                log.warning("G1 tool map failed for %s(%s): %s", name, filled, exc)
                return [self._clarify(text, f"{name}: {exc}")]
            if cmd is None:
                continue
            try:
                cmd = validate_tool_call(cmd.model_dump(mode="json"))
            except Exception as exc:
                return [self._clarify(text, f"schema: {exc}")]
            results.append(ParseResult(
                ok=True, confidence=CONF_STRONG, raw_text=text, normalized_text=text.strip().lower(),
                command=cmd, reason="lfm_g1",
            ))
        return results or [self._clarify(text, "no executable tool calls")]

    def _defaults(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        c = self.cfg
        base: Dict[str, Any] = {
            "planner_move": {
                "velocity_mps": c.lfm_default_velocity_mps, "heading_deg": 0.0,
                "yaw_rate_dps": c.lfm_default_yaw_rate_dps, "duration_s": c.lfm_default_duration_s,
            },
            "rotate_in_place": {
                "angle_deg": 0.0, "yaw_rate_dps": c.lfm_default_yaw_rate_dps,
                "duration_s": c.lfm_default_duration_s,
            },
            "set_body_height": {"height_m": c.lfm_default_height_m, "duration_s": c.lfm_default_duration_s},
            "hold_pose": {"duration_s": c.lfm_default_duration_s},
            "reset_motion_momentum": {"reason": "segment_complete", "duration_s": 0.2},
            "stop": {"reason": "user_request"},
            "select_motion_mode": {"motion_set": "locomotion", "mode": "walk"},
        }.get(name, {})
        out = dict(base)
        out.update(args)
        for key in ("duration_s", "velocity_mps", "heading_deg", "yaw_rate_dps", "angle_deg", "height_m"):
            if key in out and out[key] is None:
                out[key] = base.get(key, c.lfm_default_duration_s if key == "duration_s" else 0.0)
        return out

    def _user_prompt(self, text: str) -> str:
        c = self.cfg
        return (
            f"Defaults if omitted: velocity_mps={c.lfm_default_velocity_mps}, "
            f"yaw_rate_dps={c.lfm_default_yaw_rate_dps}, duration_s={c.lfm_default_duration_s} "
            f"(infer duration from speech), height_m={c.lfm_default_height_m}. "
            f"Command: {text.strip()}"
        )

    def _generate(self, text: str) -> str:
        if self.cfg.lfm_remote_url:
            return self._generate_remote(text)
        return self._generate_local(text)

    def _generate_local(self, text: str) -> str:
        self._ensure_model()
        messages = [
            {"role": "system", "content": G1_SYSTEM_PROMPT},
            {"role": "user", "content": self._user_prompt(text)},
        ]
        inputs = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        ).to(self._model.device)
        out = self._model.generate(
            inputs,
            max_new_tokens=self.cfg.lfm_max_new_tokens,
            do_sample=self.cfg.lfm_temperature > 0,
            temperature=max(self.cfg.lfm_temperature, 1e-5),
            top_k=self.cfg.lfm_top_k,
            repetition_penalty=1.05,
        )
        text_out = self._tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=False)
        log.debug("LFM raw output: %r", text_out)
        return text_out

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

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            hint = ""
            if "get_int_max_str_digits" in str(exc) or "GenerationMixin" in str(exc):
                hint = (
                    " Likely torch/Python mismatch on Jetson. "
                    "Use parser.lfm_remote_url or upgrade Python + Jetson torch."
                )
            raise RuntimeError(
                f"lfm_g1 import failed in {sys.executable}: {exc}.{hint} "
                f"Install: {sys.executable} -m pip install 'torch>=2.4.0' 'transformers>=5.0.0'"
            ) from exc
        _require_local_torch()
        _require_transformers_v5()
        log.info("Loading LFM G1 model %s ...", self.cfg.lfm_model_id)
        load_kw: Dict[str, Any] = {"trust_remote_code": True}
        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.lfm_model_id, **load_kw)
        kwargs: Dict[str, Any] = {"device_map": self.cfg.lfm_device, "trust_remote_code": True}
        if self.cfg.lfm_device != "cpu":
            kwargs["torch_dtype"] = torch.bfloat16
        self._model = AutoModelForCausalLM.from_pretrained(self.cfg.lfm_model_id, **kwargs)
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


def main() -> None:
    import argparse
    from .config import Config

    p = argparse.ArgumentParser(description="Local LFM G1 parse server for lfm_remote_url clients")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    cfg = Config.from_yaml(args.config).parser if args.config else ParserConfig()
    serve(cfg, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
