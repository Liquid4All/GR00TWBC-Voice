#!/usr/bin/env python3
"""Unseen eval harness for Sonic robotic tool-calling SFT checkpoints."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

START_TAG = "<|tool_call_start|>"
END_TAG = "<|tool_call_end|>"
DEFAULT_TRAIN_DATASET = (
    "/home/tim@liquid.ai/liquid_lfm/datasets/"
    "sonic_tool_calling_refined_100k_v4_return50_no_reset/train.parquet"
)
DEFAULT_FT_MODEL = "/home/shared_tw/checkpoints/tim_sft_230m_sonic100k_heading_lr1e-6_bs2_ga4_1145773_HF"
DEFAULT_BASE_MODEL = "/home/shared_tw/checkpoints/LFM2.5-230M-HF"
DEFAULT_OUT_DIR = "/home/tim@liquid.ai/liquid_lfm/eval_outputs/sonic_tool_calling_unseen"

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

TOOL_SCHEMAS = {
    "select_motion_mode": {
        "required": {"motion_set": str, "mode": str},
        "enums": {
            "motion_set": {"locomotion", "squat_ground", "boxing", "styled_walking"},
            "mode": {
                "slow_walk",
                "walk",
                "run",
                "happy",
                "stealth",
                "injured",
                "squat",
                "kneel_two_legs",
                "kneel_one_leg",
                "hand_crawl",
                "elbow_crawl",
                "idle_boxing",
                "walk_boxing",
                "left_jab",
                "right_jab",
                "random_punches",
                "left_hook",
                "right_hook",
                "careful",
                "object_carrying",
                "crouch",
                "happy_dance",
                "zombie",
                "point",
                "scared",
            },
        },
    },
    "planner_move": {
        "required": {
            "velocity_mps": (int, float),
            "heading_deg": (int, float),
            "yaw_rate_dps": (int, float),
            "duration_s": (int, float),
        },
        "enums": {},
    },
    "rotate_in_place": {
        "required": {
            "angle_deg": (int, float),
            "yaw_rate_dps": (int, float),
            "duration_s": (int, float),
        },
        "enums": {},
    },
    "set_body_height": {
        "required": {"height_m": (int, float), "duration_s": (int, float)},
        "enums": {},
    },
    "hold_pose": {"required": {"duration_s": (int, float)}, "enums": {}},
    "stop": {
        "required": {"reason": str},
        "enums": {"reason": {"user_request", "safety", "sequence_complete"}},
    },
}

MODE_TO_SET = {
    "slow_walk": "locomotion",
    "walk": "locomotion",
    "run": "locomotion",
    "happy": "locomotion",
    "stealth": "locomotion",
    "injured": "locomotion",
    "hand_crawl": "squat_ground",
    "elbow_crawl": "squat_ground",
    "careful": "styled_walking",
    "object_carrying": "styled_walking",
    "crouch": "styled_walking",
    "happy_dance": "styled_walking",
    "zombie": "styled_walking",
    "scared": "styled_walking",
}

MODE_SPEEDS = {
    "slow_walk": 0.35,
    "walk": 1.0,
    "run": 2.0,
    "happy": 0.45,
    "stealth": 0.45,
    "injured": 0.3,
    "hand_crawl": 0.25,
    "elbow_crawl": 0.25,
    "careful": 0.45,
    "object_carrying": 0.45,
    "crouch": 0.35,
    "zombie": 0.35,
    "scared": 0.6,
}

HEADING_PHRASES = [
    ("forward", 0),
    ("ahead", 0),
    ("to the left", 90),
    ("left", 90),
    ("backward", 180),
    ("back", 180),
    ("to the right", 270),
    ("right", 270),
    ("diagonally forward left", 45),
    ("forward-left", 45),
    ("diagonally backward left", 135),
    ("back-left", 135),
    ("diagonally backward right", 225),
    ("back-right", 225),
    ("diagonally forward right", 315),
    ("forward-right", 315),
]


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def jsonl_read(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def fmt_num(value: float) -> int | float:
    rounded = round(value + 1e-9, 2)
    if abs(rounded - round(rounded)) < 1e-9:
        return int(round(rounded))
    return rounded


def call(name: str, **arguments: Any) -> dict[str, Any]:
    return {"name": name, "arguments": dict(arguments)}


def select_mode(mode: str) -> dict[str, Any]:
    return call("select_motion_mode", motion_set=MODE_TO_SET[mode], mode=mode)


def move_call(mode: str, heading: int, distance_m: float) -> dict[str, Any]:
    speed = MODE_SPEEDS[mode]
    return call(
        "planner_move",
        velocity_mps=fmt_num(speed),
        heading_deg=heading,
        yaw_rate_dps=0,
        duration_s=fmt_num(distance_m / speed),
    )


def rotate_call(angle_deg: float) -> dict[str, Any]:
    return call(
        "rotate_in_place",
        angle_deg=fmt_num(angle_deg),
        yaw_rate_dps=90,
        duration_s=fmt_num(abs(angle_deg) / 90),
    )


def signed_turn_delta(current_heading: float, target_heading: float) -> float:
    return (target_heading - current_heading + 180.0) % 360.0 - 180.0


def expected_to_string(expected: list[dict[str, Any]]) -> str:
    parts = []
    for item in expected:
        args = ", ".join(f'{key}={format_arg(value)}' for key, value in item["arguments"].items())
        parts.append(f'{item["name"]}({args})')
    return f"{START_TAG}[{', '.join(parts)}]{END_TAG}"


def format_arg(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    return str(value)


def movement_example(
    idx: int,
    user: str,
    mode: str,
    heading: int,
    distance_m: float,
    category: str,
    extra_expected: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    expected = [select_mode(mode), move_call(mode, heading, distance_m)]
    if extra_expected:
        expected.extend(extra_expected)
    return make_row(idx, category, user, expected)


def make_row(idx: int, category: str, user: str, expected: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": f"sonic_eval_unseen_{idx:06d}",
        "category": category,
        "system": SYSTEM_PROMPT,
        "user": user,
        "expected": expected,
        "expected_text": expected_to_string(expected),
    }


def make_contextual_row(
    idx: int,
    category: str,
    messages: list[dict[str, str]],
    expected: list[dict[str, Any]],
) -> dict[str, Any]:
    final_user = next(item["content"] for item in reversed(messages) if item["role"] == "user")
    row = make_row(idx, category, final_user, expected)
    row["messages"] = messages
    return row


def assistant_text(expected: list[dict[str, Any]]) -> str:
    return expected_to_string(expected)


def generate_candidate_rows(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    idx = 0
    modes = ["slow_walk", "walk", "run", "happy", "stealth", "injured", "hand_crawl", "elbow_crawl", "careful", "object_carrying", "crouch", "zombie"]
    distances = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    verbs = {
        "slow_walk": ["amble", "slow walk", "move slowly"],
        "walk": ["walk", "go", "move"],
        "run": ["run", "jog", "hurry"],
        "happy": ["move happily", "happy walk", "go with a happy gait"],
        "stealth": ["sneak", "creep quietly", "move in stealth mode"],
        "injured": ["limp", "move like you are injured", "injured walk"],
        "hand_crawl": ["hand crawl", "crawl on your hands", "crawl low using your hands"],
        "elbow_crawl": ["elbow crawl", "crawl on your elbows", "crawl very low"],
        "careful": ["walk carefully", "carefully move", "take careful steps"],
        "object_carrying": ["move as if carrying something", "carry-object walk", "walk while carrying"],
        "crouch": ["crouch walk", "move crouched", "stay crouched and move"],
        "zombie": ["zombie walk", "stagger like a zombie", "move with a zombie gait"],
    }
    for _ in range(900):
        mode = rng.choice(modes)
        phrase, heading = rng.choice(HEADING_PHRASES)
        distance = rng.choice(distances)
        verb = rng.choice(verbs[mode])
        templates = [
            f"{verb.capitalize()} {phrase} for {distance:g} meters.",
            f"I want you to {verb} {distance:g} meters {phrase}.",
            f"Please {verb} {phrase}; make it {distance:g} meters.",
            f"{phrase.capitalize()} for {distance:g} m using {mode.replace('_', ' ')}.",
        ]
        idx += 1
        rows.append(movement_example(idx, rng.choice(templates), mode, heading, distance, "synthetic_movement"))

    turn_specs = [
        ("turn left", 90),
        ("rotate left", 90),
        ("spin left", 180),
        ("turn right", -90),
        ("rotate right", -90),
        ("face back to the right", -180),
    ]
    for text, angle in turn_specs * 35:
        idx += 1
        expected = [
            call("rotate_in_place", angle_deg=angle, yaw_rate_dps=90, duration_s=fmt_num(abs(angle) / 90)),
        ]
        rows.append(make_row(idx, "synthetic_turn", f"{text.capitalize()} by {abs(angle)} degrees.", expected))

    height_specs = [
        ("Lower to 0.4 meters, pause, then stand tall again.", 0.4),
        ("Drop your body height to 0.5 m and come back up.", 0.5),
        ("Squat down to 0.6 meters before returning to normal height.", 0.6),
        ("Set your body to 0.3 meters, then stand.", 0.3),
    ]
    for text, height in height_specs * 30:
        idx += 1
        expected = [
            call("select_motion_mode", motion_set="squat_ground", mode="squat"),
            call("set_body_height", height_m=height, duration_s=1),
            call("select_motion_mode", motion_set="squat_ground", mode="squat"),
            call("set_body_height", height_m=0.8, duration_s=1),
        ]
        rows.append(make_row(idx, "synthetic_height", text, expected))

    stop_texts = [
        "Freeze immediately.",
        "Stop moving now.",
        "Halt all motion.",
        "Emergency stop.",
        "Cut motion for safety.",
    ]
    for text in stop_texts * 35:
        idx += 1
        stop_reason = "safety" if "Emergency" in text or "safety" in text else "user_request"
        expected = [call("stop", reason=stop_reason)]
        rows.append(make_row(idx, "synthetic_stop_safety", text, expected))

    return rows


def hand_authored_rows(start_idx: int) -> list[dict[str, Any]]:
    specs = [
        ("Take two careful steps backward for about 2 meters, then hold still for 3 seconds.", [select_mode("careful"), move_call("careful", 180, 2.0), call("hold_pose", duration_s=3)], "hand_composition"),
        ("Scoot forward-right in a crouch for three quarters of a meter.", [select_mode("crouch"), move_call("crouch", 315, 0.75)], "hand_paraphrase"),
        ("Back-left crawl on elbows for 1.25 meters, then stop.", [select_mode("elbow_crawl"), move_call("elbow_crawl", 135, 1.25), call("stop", reason="sequence_complete")], "hand_composition"),
        ("Jog left one meter, then rotate a quarter turn to the right.", [select_mode("run"), move_call("run", 90, 1.0), call("rotate_in_place", angle_deg=-90, yaw_rate_dps=90, duration_s=1)], "hand_composition"),
        ("Sneak forward for 0.5 meters, pause for 2 seconds, then sneak backward 0.5 meters.", [select_mode("stealth"), move_call("stealth", 0, 0.5), call("hold_pose", duration_s=2), select_mode("stealth"), move_call("stealth", 180, 0.5)], "hand_composition"),
        ("Please kneel on both legs and then return to standing height.", [call("select_motion_mode", motion_set="squat_ground", mode="kneel_two_legs"), call("select_motion_mode", motion_set="squat_ground", mode="squat"), call("set_body_height", height_m=0.8, duration_s=1)], "hand_posture"),
        ("Make one right hook, then freeze.", [call("select_motion_mode", motion_set="boxing", mode="right_hook"), call("stop", reason="user_request")], "hand_boxing_stop"),
        ("Walk like you are carrying a fragile box to the right for 1.5 meters.", [select_mode("object_carrying"), move_call("object_carrying", 270, 1.5)], "hand_paraphrase"),
        ("Move like a zombie diagonally backward right for 2 meters.", [select_mode("zombie"), move_call("zombie", 225, 2.0)], "hand_paraphrase"),
        ("Lower to 0.2 meters and hold for four seconds.", [call("select_motion_mode", motion_set="squat_ground", mode="squat"), call("set_body_height", height_m=0.2, duration_s=1), call("hold_pose", duration_s=4)], "hand_height_hold"),
    ]
    modifiers = [
        "",
        " Keep the response minimal.",
        " Use only the planner calls.",
        " Do it in that order.",
        " No extra commentary.",
        " Treat this as a fresh command.",
        " Maintain the same body-relative frame.",
        " Keep the movement segmented.",
        " Follow with exact tool syntax.",
        " Avoid any explanation.",
        " Make the call list concise.",
        " Preserve the requested order.",
        " Keep the robot safe.",
        " Use the trained tool format.",
        " Return only executable calls.",
    ]
    rows = []
    for i in range(len(specs) * len(modifiers)):
        text, expected, category = specs[i % len(specs)]
        rows.append(make_row(start_idx + i + 1, category, text + modifiers[i // len(specs)], expected))
    rng = random.Random(9917)
    modes = ["stealth", "run", "careful", "object_carrying", "hand_crawl", "elbow_crawl", "crouch", "zombie"]
    for _ in range(300):
        mode_a = rng.choice(modes)
        mode_b = rng.choice([mode for mode in modes if mode != mode_a])
        phrase_a, heading_a = rng.choice(HEADING_PHRASES)
        phrase_b, heading_b = rng.choice(HEADING_PHRASES)
        dist_a = rng.choice([0.5, 0.75, 1.25, 1.75, 2.25])
        dist_b = rng.choice([0.5, 1.0, 1.5, 2.0])
        hold_s = rng.choice([1, 2, 3, 4])
        text = (
            f"Operator check: use {mode_a.replace('_', ' ')} {phrase_a} for {dist_a:g} meters, "
            f"hold for {hold_s} seconds, then {mode_b.replace('_', ' ')} {phrase_b} for {dist_b:g} meters."
        )
        expected = [
            select_mode(mode_a),
            move_call(mode_a, heading_a, dist_a),
            call("hold_pose", duration_s=hold_s),
            select_mode(mode_b),
            move_call(mode_b, heading_b, dist_b),
        ]
        rows.append(make_row(start_idx + len(rows) + 1, "hand_generated_composition", text, expected))
    return rows


def return_to_origin_expected(
    route_legs: list[tuple[float, float]],
    current_heading: float,
    *,
    starts_crouched: bool = False,
) -> list[dict[str, Any]]:
    expected: list[dict[str, Any]] = []
    if starts_crouched:
        expected.extend(
            [
                call("select_motion_mode", motion_set="squat_ground", mode="squat"),
                call("set_body_height", height_m=0.8, duration_s=1),
            ]
        )
    expected.append(select_mode("walk"))
    for heading, distance in reversed(route_legs):
        target_heading = (heading + 180.0) % 360.0
        turn_angle = signed_turn_delta(current_heading, target_heading)
        if abs(turn_angle) > 1e-9:
            expected.append(rotate_call(turn_angle))
            current_heading = target_heading
        expected.append(move_call("walk", 0, distance))
    expected.append(call("stop", reason="sequence_complete"))
    return expected


def return_to_origin_rows(start_idx: int) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []

    first = [
        select_mode("walk"),
        move_call("walk", 0, 1.25),
        rotate_call(90),
    ]
    scenarios.append(
        {
            "category": "return_origin_single_leg",
            "messages": [
                {"role": "user", "content": "Walk forward 1.25 meters, then turn left 90 degrees."},
                {"role": "assistant", "content": assistant_text(first)},
                {"role": "user", "content": "Return to your initial position and stop."},
            ],
            "expected": return_to_origin_expected([(0, 1.25)], 90),
        }
    )

    first = [
        select_mode("walk"),
        move_call("walk", 0, 1.0),
        rotate_call(90),
        move_call("walk", 0, 0.75),
    ]
    scenarios.append(
        {
            "category": "return_origin_corner",
            "messages": [
                {
                    "role": "user",
                    "content": "Walk forward 1 meter, then turn left 90 degrees and walk forward 0.75 meters.",
                },
                {"role": "assistant", "content": assistant_text(first)},
                {"role": "user", "content": "Retrace the path back to the origin and stop."},
            ],
            "expected": return_to_origin_expected([(0, 1.0), (90, 0.75)], 90),
        }
    )

    turn_one = [select_mode("walk"), move_call("walk", 0, 1.5)]
    turn_two = [
        rotate_call(-45),
        move_call("walk", 0, 1.0),
        call("select_motion_mode", motion_set="squat_ground", mode="squat"),
        call("set_body_height", height_m=0.5, duration_s=1),
    ]
    scenarios.append(
        {
            "category": "return_origin_multiturn_inspection",
            "messages": [
                {"role": "user", "content": "Walk forward 1.5 meters."},
                {"role": "assistant", "content": assistant_text(turn_one)},
                {
                    "role": "user",
                    "content": "Turn right 45 degrees, walk forward 1 meter, then crouch down to 0.5 meters for inspection.",
                },
                {"role": "assistant", "content": assistant_text(turn_two)},
                {"role": "user", "content": "Return the way you came and stop."},
            ],
            "expected": return_to_origin_expected([(0, 1.5), (315, 1.0)], 315, starts_crouched=True),
        }
    )

    turn_one = [
        select_mode("walk"),
        rotate_call(30),
        move_call("walk", 0, 0.75),
    ]
    turn_two = [
        rotate_call(60),
        move_call("walk", 0, 1.25),
        call("select_motion_mode", motion_set="squat_ground", mode="squat"),
        call("set_body_height", height_m=0.6, duration_s=1),
    ]
    scenarios.append(
        {
            "category": "return_origin_angled_inspection",
            "messages": [
                {"role": "user", "content": "Turn left 30 degrees, then walk forward 0.75 meters."},
                {"role": "assistant", "content": assistant_text(turn_one)},
                {
                    "role": "user",
                    "content": "Turn left 60 degrees, walk forward 1.25 meters, and lower to 0.6 meters.",
                },
                {"role": "assistant", "content": assistant_text(turn_two)},
                {"role": "user", "content": "Go back to the starting point along the same route and stop."},
            ],
            "expected": return_to_origin_expected([(30, 0.75), (90, 1.25)], 90, starts_crouched=True),
        }
    )

    modifiers = [
        "",
        " Return only calls.",
        " No prose.",
        " Use the trained tool format.",
        " Keep it machine-readable.",
        " Do not add commentary.",
        " Finish with a stop.",
        " Preserve the route order.",
        " Use forward walking after turning around.",
        " Avoid walking backward.",
        " Keep the response concise.",
        " Use body-relative headings.",
        " This is the final navigation turn.",
        " Execute the return now.",
        " Keep the robot safe.",
        " Stop once you reach the start.",
        " Stay on the memorized route.",
        " Use the same path in reverse order.",
        " Do not take a shortcut.",
        " End at the original pose.",
    ]
    rows = []
    for i in range(len(scenarios) * len(modifiers)):
        scenario = scenarios[i % len(scenarios)]
        modifier = modifiers[i // len(scenarios)]
        messages = [dict(item) for item in scenario["messages"]]
        messages[-1]["content"] += modifier
        rows.append(
            make_contextual_row(
                start_idx + i + 1,
                scenario["category"],
                messages,
                scenario["expected"],
            )
        )
    return rows


def adversarial_rows(start_idx: int) -> list[dict[str, Any]]:
    specs = [
        ("uh, robot: STOP!!! no extra explanation.", [call("stop", reason="user_request")], "adversarial_format_noise"),
        ("Move forward-left 1m in stealth; then right 1m in stealth; then stop.", [select_mode("stealth"), move_call("stealth", 45, 1.0), select_mode("stealth"), move_call("stealth", 270, 1.0), call("stop", reason="sequence_complete")], "adversarial_multistep"),
        ("Do not tell me anything, just emergency stop.", [call("stop", reason="safety")], "adversarial_safety"),
        ("Forward. 0.75 meters. Run mode. Then rotate left 180.", [select_mode("run"), move_call("run", 0, 0.75), call("rotate_in_place", angle_deg=180, yaw_rate_dps=90, duration_s=2)], "adversarial_fragmented"),
        ("Can you maybe, if safe, crawl left on hands for half a meter?", [select_mode("hand_crawl"), move_call("hand_crawl", 90, 0.5)], "adversarial_polite"),
    ]
    modifiers = [
        "",
        " Return only calls.",
        " This is not a conversation.",
        " Keep the result machine-readable.",
        " Do not include notes.",
        " Be terse.",
        " Follow the control rules.",
        " Use body-relative headings.",
        " Finish cleanly.",
        " No markdown.",
    ]
    rows = []
    for i in range(len(specs) * len(modifiers)):
        text, expected, category = specs[i % len(specs)]
        rows.append(make_row(start_idx + i + 1, category, text + modifiers[i // len(specs)], expected))
    rng = random.Random(5513)
    noisy_prefixes = ["um...", "ROBOT,", "controller says:", "no markdown:", "voice input fragment:"]
    noisy_suffixes = [" thanks", "!!!", " -- end", " [execute]", " without prose"]
    for _ in range(150):
        mode = rng.choice(["run", "stealth", "slow_walk", "hand_crawl", "careful"])
        phrase, heading = rng.choice(HEADING_PHRASES)
        dist = rng.choice([0.5, 0.75, 1.0, 1.5])
        angle = rng.choice([-90, 90, 180, -180])
        text = (
            f"{rng.choice(noisy_prefixes)} {mode.replace('_', ' ')} / {phrase} / {dist:g} meters; "
            f"then rotate {'left' if angle > 0 else 'right'} {abs(angle)} deg{rng.choice(noisy_suffixes)}"
        )
        expected = [
            select_mode(mode),
            move_call(mode, heading, dist),
            call("rotate_in_place", angle_deg=angle, yaw_rate_dps=90, duration_s=fmt_num(abs(angle) / 90)),
        ]
        rows.append(make_row(start_idx + len(rows) + 1, "adversarial_noisy_multistep", text, expected))
    return rows


def load_training_sets(dataset_name: str) -> tuple[set[str], set[str]]:
    from datasets import load_dataset

    path = Path(dataset_name)
    if path.exists():
        if path.is_dir():
            parquet = path / "train.parquet"
            jsonl = path / "train.jsonl"
            if parquet.exists():
                ds = load_dataset("parquet", data_files=str(parquet), split="train")
            elif jsonl.exists():
                ds = load_dataset("json", data_files=str(jsonl), split="train")
            else:
                raise FileNotFoundError(f"no train.parquet or train.jsonl found under {path}")
        elif path.suffix == ".parquet":
            ds = load_dataset("parquet", data_files=str(path), split="train")
        elif path.suffix in {".jsonl", ".json"}:
            ds = load_dataset("json", data_files=str(path), split="train")
        else:
            raise ValueError(f"unsupported local training dataset file: {path}")
    else:
        ds = load_dataset(dataset_name, split="train")
    users = set()
    assistants = set()
    for row in ds:
        conv = row.get("conversations") or row.get("messages")
        user_text = "\n".join(msg.get("content", "") for msg in conv if msg.get("role") == "user")
        assistant_text = "\n".join(msg.get("content", "") for msg in conv if msg.get("role") == "assistant")
        users.add(normalize_text(user_text))
        assistants.add(normalize_text(assistant_text))
    return users, assistants


def generate_samples(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    train_users, train_assistants = load_training_sets(args.train_dataset)
    target_counts = {"synthetic": args.n_synthetic, "hand": args.n_hand, "adversarial": args.n_adversarial}
    candidates = generate_candidate_rows(args.seed)
    selected: list[dict[str, Any]] = []
    report = Counter()

    def row_user_texts(row: dict[str, Any]) -> list[str]:
        messages = row.get("messages")
        if messages:
            return [msg["content"] for msg in messages if msg.get("role") == "user"]
        return [row["user"]]

    def row_user_signature(row: dict[str, Any]) -> str:
        return "\n".join(normalize_text(text) for text in row_user_texts(row))

    def accept(row: dict[str, Any]) -> bool:
        if any(normalize_text(text) in train_users for text in row_user_texts(row)):
            report["rejected_user_match"] += 1
            return False
        if normalize_text(row["expected_text"]) in train_assistants:
            report["rejected_assistant_match"] += 1
            return False
        if row_user_signature(row) in {row_user_signature(item) for item in selected}:
            report["rejected_eval_duplicate_user"] += 1
            return False
        return True

    for row in candidates:
        if len([r for r in selected if r["category"].startswith("synthetic")]) >= target_counts["synthetic"]:
            break
        if accept(row):
            selected.append(row)

    base_idx = len(selected)
    for row in hand_authored_rows(base_idx):
        if len([r for r in selected if r["category"].startswith("hand")]) >= target_counts["hand"]:
            break
        if accept(row):
            selected.append(row)

    base_idx = len(selected)
    for row in return_to_origin_rows(base_idx):
        if len([r for r in selected if r["category"].startswith("return_origin")]) >= args.n_return:
            break
        if accept(row):
            selected.append(row)

    base_idx = len(selected)
    for row in adversarial_rows(base_idx):
        if len([r for r in selected if r["category"].startswith("adversarial")]) >= target_counts["adversarial"]:
            break
        if accept(row):
            selected.append(row)

    for i, row in enumerate(selected, start=1):
        row["id"] = f"sonic_eval_unseen_{i:06d}"

    samples_path = out_dir / "eval_samples_unseen.jsonl"
    report_path = out_dir / "decontamination_report.json"
    jsonl_write(samples_path, selected)
    report.update(
        {
            "selected_total": len(selected),
            "selected_synthetic": len([r for r in selected if r["category"].startswith("synthetic")]),
            "selected_hand": len([r for r in selected if r["category"].startswith("hand")]),
            "selected_return": len([r for r in selected if r["category"].startswith("return_origin")]),
            "selected_adversarial": len([r for r in selected if r["category"].startswith("adversarial")]),
            "training_unique_users": len(train_users),
            "training_unique_assistants": len(train_assistants),
        }
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {samples_path}")
    print(f"Wrote {report_path}")
    print(json.dumps(dict(report), indent=2, sort_keys=True))


def parse_generated(text: str) -> tuple[list[ToolCall], dict[str, Any]]:
    stripped = text.strip()
    info = {
        "format_valid": False,
        "parse_valid": False,
        "error": None,
        "prose_violation": False,
    }
    if stripped.count(START_TAG) != 1 or stripped.count(END_TAG) != 1:
        info["error"] = "missing_or_repeated_tool_tags"
        return [], info
    start = stripped.find(START_TAG)
    end = stripped.find(END_TAG)
    if start != 0 or end + len(END_TAG) != len(stripped):
        info["prose_violation"] = True
    if end < start:
        info["error"] = "end_tag_before_start_tag"
        return [], info
    body = stripped[start + len(START_TAG) : end]
    info["format_valid"] = not info["prose_violation"]
    try:
        expr = ast.parse(body, mode="eval").body
        if not isinstance(expr, ast.List):
            raise ValueError("tool body is not a list")
        calls = []
        for item in expr.elts:
            if not isinstance(item, ast.Call) or not isinstance(item.func, ast.Name):
                raise ValueError("list item is not a function call")
            args: dict[str, Any] = {}
            if item.args:
                raise ValueError("positional arguments are not allowed")
            for keyword in item.keywords:
                if keyword.arg is None:
                    raise ValueError("star keywords are not allowed")
                args[keyword.arg] = ast.literal_eval(keyword.value)
            calls.append(ToolCall(item.func.id, args))
        info["parse_valid"] = True
        return calls, info
    except Exception as exc:
        info["error"] = f"parse_error: {exc}"
        return [], info


def schema_valid(calls: list[ToolCall]) -> tuple[bool, list[str]]:
    errors = []
    for idx, item in enumerate(calls):
        schema = TOOL_SCHEMAS.get(item.name)
        if schema is None:
            errors.append(f"{idx}: unknown tool {item.name}")
            continue
        required = schema["required"]
        if set(item.arguments) != set(required):
            errors.append(f"{idx}: expected args {sorted(required)}, got {sorted(item.arguments)}")
            continue
        for key, typ in required.items():
            if not isinstance(item.arguments[key], typ):
                errors.append(f"{idx}: arg {key} has wrong type")
        for key, allowed in schema["enums"].items():
            if item.arguments.get(key) not in allowed:
                errors.append(f"{idx}: arg {key} invalid enum {item.arguments.get(key)}")
        if item.name == "set_body_height":
            height = float(item.arguments["height_m"])
            if height < 0.2 or height > 0.8:
                errors.append(f"{idx}: height_m out of range")
    return not errors, errors


def almost_equal_arg(key: str, actual: Any, expected: Any) -> bool:
    if isinstance(expected, str):
        return actual == expected
    try:
        a = float(actual)
        e = float(expected)
    except Exception:
        return actual == expected
    if key == "heading_deg":
        diff = abs((a - e + 180) % 360 - 180)
        return diff <= 1.0
    if key == "duration_s":
        return abs(a - e) <= max(0.05, 0.02 * max(abs(e), 1e-9))
    return abs(a - e) <= 0.01


def compare_calls(actual: list[ToolCall], expected: list[dict[str, Any]]) -> dict[str, Any]:
    exact = len(actual) == len(expected)
    semantic = len(actual) == len(expected)
    mode_ok = True
    movement_ok = True
    stop_ok = True
    mismatches = []
    for idx, expected_item in enumerate(expected):
        if idx >= len(actual):
            mismatches.append(f"{idx}: missing {expected_item['name']}")
            exact = False
            semantic = False
            continue
        got = actual[idx]
        if got.name != expected_item["name"]:
            mismatches.append(f"{idx}: expected tool {expected_item['name']}, got {got.name}")
            exact = False
            semantic = False
            if expected_item["name"] == "select_motion_mode":
                mode_ok = False
            if expected_item["name"] in {"planner_move", "rotate_in_place"}:
                movement_ok = False
            if expected_item["name"] == "stop":
                stop_ok = False
            continue
        if got.arguments != expected_item["arguments"]:
            exact = False
        for key, expected_value in expected_item["arguments"].items():
            if not almost_equal_arg(key, got.arguments.get(key), expected_value):
                mismatches.append(f"{idx}: {got.name}.{key} expected {expected_value}, got {got.arguments.get(key)}")
                semantic = False
                if got.name == "select_motion_mode":
                    mode_ok = False
                if got.name in {"planner_move", "rotate_in_place"}:
                    movement_ok = False
                if got.name == "stop":
                    stop_ok = False
    if len(actual) > len(expected):
        mismatches.append(f"extra calls: {len(actual) - len(expected)}")
        exact = False
        semantic = False
    return {
        "exact_sequence_match": exact,
        "semantic_match": semantic,
        "mode_accuracy": mode_ok,
        "movement_accuracy": movement_ok,
        "stop_safety_accuracy": stop_ok,
        "overcall": len(actual) > len(expected),
        "undercall": len(actual) < len(expected),
        "mismatches": mismatches,
    }


def trim_generation(text: str) -> str:
    if END_TAG in text:
        return text[: text.find(END_TAG) + len(END_TAG)].strip()
    return text.strip()


def run_inference(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    samples = jsonl_read(Path(args.samples))
    if args.limit:
        samples = samples[: args.limit]
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model_kwargs = {"trust_remote_code": True, "dtype": dtype}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).to(device).eval()
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    predictions = []
    for idx, sample in enumerate(samples, start=1):
        if sample.get("messages"):
            messages = [{"role": "system", "content": sample["system"]}, *sample["messages"]]
        else:
            messages = [
                {"role": "system", "content": sample["system"]},
                {"role": "user", "content": sample["user"]},
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
        generated_ids = output[0][input_ids.shape[-1] :]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        predictions.append(
            {
                "id": sample["id"],
                "category": sample["category"],
                "model": args.model,
                "user": sample["user"],
                "generated_text": trim_generation(generated_text),
                "raw_generated_text": generated_text,
            }
        )
        if idx % 25 == 0 or idx == len(samples):
            print(f"generated {idx}/{len(samples)}", flush=True)
    jsonl_write(Path(args.output), predictions)
    print(f"Wrote {args.output}")


def score_predictions(args: argparse.Namespace) -> None:
    samples = {row["id"]: row for row in jsonl_read(Path(args.samples))}
    predictions = jsonl_read(Path(args.predictions))
    scored_rows = []
    aggregate = Counter()
    by_category: dict[str, Counter] = defaultdict(Counter)
    failure_examples = []
    for pred in predictions:
        sample = samples[pred["id"]]
        calls, parse_info = parse_generated(pred["generated_text"])
        schema_ok, schema_errors = schema_valid(calls) if parse_info["parse_valid"] else (False, ["not_parseable"])
        comparison = compare_calls(calls, sample["expected"]) if schema_ok else {
            "exact_sequence_match": False,
            "semantic_match": False,
            "mode_accuracy": False,
            "movement_accuracy": False,
            "stop_safety_accuracy": False,
            "overcall": False,
            "undercall": False,
            "mismatches": schema_errors,
        }
        metrics = {
            "format_valid": parse_info["format_valid"],
            "parse_valid": parse_info["parse_valid"],
            "schema_valid": schema_ok,
            **{key: value for key, value in comparison.items() if isinstance(value, bool)},
        }
        category = sample["category"]
        aggregate["total"] += 1
        by_category[category]["total"] += 1
        for key, value in metrics.items():
            if value:
                aggregate[key] += 1
                by_category[category][key] += 1
        scored = {
            **pred,
            "expected_text": sample["expected_text"],
            "metrics": metrics,
            "parse_error": parse_info["error"],
            "schema_errors": schema_errors,
            "mismatches": comparison["mismatches"],
        }
        scored_rows.append(scored)
        if not metrics["semantic_match"] and len(failure_examples) < args.max_error_examples:
            failure_examples.append(scored)

    metric_keys = [
        "format_valid",
        "parse_valid",
        "schema_valid",
        "exact_sequence_match",
        "semantic_match",
        "mode_accuracy",
        "movement_accuracy",
        "stop_safety_accuracy",
        "overcall",
        "undercall",
    ]

    def rates(counter: Counter) -> dict[str, float | int]:
        total = counter["total"]
        out: dict[str, float | int] = {"total": total}
        for key in metric_keys:
            out[f"{key}_rate"] = counter[key] / total if total else 0.0
        return out

    result = {
        "predictions": str(args.predictions),
        "samples": str(args.samples),
        "aggregate": rates(aggregate),
        "by_category": {category: rates(counter) for category, counter in sorted(by_category.items())},
        "acceptance": {
            "format_valid_rate_gte_0.99": rates(aggregate)["format_valid_rate"] >= 0.99,
            "schema_valid_rate_gte_0.99": rates(aggregate)["schema_valid_rate"] >= 0.99,
            "semantic_match_rate_gte_0.95": rates(aggregate)["semantic_match_rate"] >= 0.95,
            "exact_sequence_match_rate_gte_0.90": rates(aggregate)["exact_sequence_match_rate"] >= 0.90,
            "all_categories_semantic_gte_0.90": all(rates(counter)["semantic_match_rate"] >= 0.90 for counter in by_category.values()),
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.scored_output:
        jsonl_write(Path(args.scored_output), scored_rows)
    if args.error_report:
        write_error_report(Path(args.error_report), result, failure_examples)
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    print(f"Wrote {out_path}")


def write_error_report(path: Path, metrics: dict[str, Any], failures: list[dict[str, Any]]) -> None:
    lines = ["# Sonic Tool-Calling Eval Error Report", "", "## Aggregate", ""]
    for key, value in metrics["aggregate"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Failure Examples", ""])
    for item in failures:
        lines.extend(
            [
                f"### {item['id']} - {item['category']}",
                "",
                f"User: `{item['user']}`",
                "",
                "Expected:",
                "```text",
                item["expected_text"],
                "```",
                "Generated:",
                "```text",
                item["generated_text"],
                "```",
                f"Mismatches: `{item['mismatches']}`",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def compare_metrics(args: argparse.Namespace) -> None:
    ft = json.loads(Path(args.finetuned_metrics).read_text(encoding="utf-8"))
    base = json.loads(Path(args.base_metrics).read_text(encoding="utf-8"))
    ft_sem = ft["aggregate"]["semantic_match_rate"]
    base_sem = base["aggregate"]["semantic_match_rate"]
    result = {
        "finetuned_metrics": args.finetuned_metrics,
        "base_metrics": args.base_metrics,
        "finetuned_semantic_match_rate": ft_sem,
        "base_semantic_match_rate": base_sem,
        "semantic_match_delta": ft_sem - base_sem,
        "passes_delta_gate_0.30": (ft_sem - base_sem) >= 0.30,
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-samples")
    gen.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    gen.add_argument("--train-dataset", default=DEFAULT_TRAIN_DATASET)
    gen.add_argument("--seed", type=int, default=20260622)
    gen.add_argument("--n-synthetic", type=int, default=320)
    gen.add_argument("--n-hand", type=int, default=150)
    gen.add_argument("--n-return", type=int, default=80)
    gen.add_argument("--n-adversarial", type=int, default=50)
    gen.set_defaults(func=generate_samples)

    infer = sub.add_parser("infer")
    infer.add_argument("--model", required=True)
    infer.add_argument("--samples", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--limit", type=int, default=0)
    infer.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    infer.add_argument("--attn-implementation", default=None)
    infer.add_argument("--max-new-tokens", type=int, default=512)
    infer.set_defaults(func=run_inference)

    score = sub.add_parser("score")
    score.add_argument("--samples", required=True)
    score.add_argument("--predictions", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--scored-output", default=None)
    score.add_argument("--error-report", default=None)
    score.add_argument("--max-error-examples", type=int, default=50)
    score.set_defaults(func=score_predictions)

    cmp_parser = sub.add_parser("compare")
    cmp_parser.add_argument("--finetuned-metrics", required=True)
    cmp_parser.add_argument("--base-metrics", required=True)
    cmp_parser.add_argument("--output", required=True)
    cmp_parser.set_defaults(func=compare_metrics)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
