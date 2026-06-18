# SONIC Voice Control

Offline, on-device voice command interface for the **SONIC kinematic motion
planner**. It turns spoken operator commands into *safe, validated, high-level*
planner tool calls and feeds them into the existing SONIC planner command path —
**never** joint commands, never the policy, never the model weights.

```
microphone → wake word / VAD → ASR → deterministic parser (+ optional tiny LLM)
   → validated PlannerCommand(s) → confidence/execute gate → existing SONIC planner path (ZMQ)
   → planner generates short-horizon motion → SONIC tracker follows it
```

Everything runs offline. No cloud services.

---

## Why this design

The SONIC planner is **velocity-conditioned** and **high-level**: it accepts a
locomotion *mode*, *target velocity*, *movement direction*, *facing direction*,
and skill-specific targets (height for postures), then generates a short-horizon
motion that the low-level tracker follows. The deployed robot consumes these
over **ZMQ** (`command` + `planner` topics; see
`gear_sonic_deploy/.../input_interface/zmq_manager.hpp` and the wire builders in
`gear_sonic/utils/teleop/zmq/zmq_planner_sender.py`). This package plugs into
that exact path, the same way keyboard/gamepad/VR network input does.

Key invariants preserved:

* Only **high-level** commands are published (mode / velocity / direction /
  height). No joint targets, no motion frames.
* The planner is treated as **velocity-conditioned**: holding a command
  re-publishes the *same* target velocity (no integration/ramping).
* Replans at up to **10 Hz** (`planner_dt = 0.1s`) and immediately on change.
* The policy, TensorRT inference, motor writer, and learned weights are **not**
  touched.

---

## Architecture

| Module | Responsibility |
|--------|----------------|
| `schemas.py` | Closed Pydantic tool-call schema (the only things ever published). |
| `skills.py` | Synonyms, `LocomotionMode` mapping, heading↔direction convention, skill inventory. |
| `parser.py` | **Deterministic** parser (normalisation, numbers, units, styles, headings). |
| `llm_parser.py` | Optional local LLM fallback (llama.cpp), grammar + schema constrained. |
| `safety.py` | Confidence gate + dry-run/execute gate (value clamps removed; pass-through). |
| `publisher.py` | Publisher abstraction + Stub/ZMQ/ExistingRepo/ROS2 adapters + 10 Hz hold/watchdog. |
| `audio.py` | Mic capture + VAD (optional `sounddevice`/`webrtcvad`). |
| `wake.py` | `no_wake_debug` / `push_to_talk` / `wake_word` / `vad_only`. |
| `asr_vosk.py` | Default ASR: Vosk with grammar-constrained vocabulary. |
| `asr_whisper_cpp.py` | Optional ASR: whisper.cpp subprocess wrapper. |
| `feedback.py` | Console + optional offline TTS confirmations. |
| `cli.py` | Orchestration + CLI entry point. |
| `config.py` | Config dataclasses + YAML loader (`configs/voice_control.yaml`). |

The deterministic parser works **without** any model. The LLM is only consulted
when deterministic parsing is inconclusive (`clarify`) and
`parser.use_llm_fallback: true`.

---

## Install

Core (parser/safety/publisher/CLI) needs only **Pydantic**:

```bash
pip install pydantic
# optional but recommended for YAML configs:
pip install pyyaml
```

Optional extras (only needed for the features you use):

```bash
pip install pyzmq                 # real planner path (ExistingRepoPublisher/ZmqPublisher)
pip install vosk sounddevice webrtcvad   # default mic ASR
pip install openwakeword         # or pvporcupine, for wake-word mode
pip install pyttsx3              # offline TTS feedback
pip install "transformers>=4.55" torch   # LLM duration estimator (hf_transformers backend)
```

All of the above are imported lazily; the package and its tests run without them.

---

## Quick start: dry-run text tests (no audio needed)

```bash
python -m voice_control.cli --text "walk forward slowly" --dry-run
python -m voice_control.cli --text "stop" --dry-run
python -m voice_control.cli --text "left jab" --dry-run
python -m voice_control.cli --text "crawl forward" --dry-run
python -m voice_control.cli --text "squat lower" --dry-run
python -m voice_control.cli --interactive-text --dry-run
```

`--text` bypasses the microphone and ASR entirely and exercises
parser → safety → publisher. In dry-run the **StubPublisher** is always used, so
nothing is ever sent to a robot.

---

## Push-to-talk

```bash
python -m voice_control.cli --config configs/voice_control.yaml --dry-run
```

With `wake.mode: push_to_talk` (default), press **Enter** to capture one
utterance, then it is transcribed and parsed.

---

## Configure Vosk (default ASR)

1. Download a small English model from <https://alphacephei.com/vosk/models>
   (e.g. `vosk-model-small-en-us-0.15`).
2. Unpack it and point the config at it:

   ```yaml
   asr:
     vosk_model_path: "models/vosk-model-small-en-us"
   ```
3. Run with the Vosk backend (default):

   ```bash
   python -m voice_control.cli --config configs/voice_control.yaml --backend vosk --dry-run
   ```

Vosk runs in **grammar mode**: the command vocabulary + synonyms are passed as a
restricted grammar for higher accuracy/lower latency on the closed command set.

## Configure whisper.cpp (optional ASR)

1. Build [whisper.cpp](https://github.com/ggerganov/whisper.cpp) and download a
   model (`ggml-tiny.en.bin` or `ggml-base.en.bin`). Nothing is auto-downloaded.
2. Point the config at the binary and model:

   ```yaml
   asr:
     whisper_cpp_bin: "/opt/whisper.cpp/main"
     whisper_model_path: "/opt/whisper.cpp/models/ggml-tiny.en.bin"
   ```
3. Run:

   ```bash
   python -m voice_control.cli --config configs/voice_control.yaml --backend whisper_cpp --dry-run
   ```

## Optional local LLM fallback

Off by default. To enable a tiny local instruction model (e.g.
`Qwen2.5-0.5B-Instruct` or `SmolLM2-1.7B-Instruct`, quantized) via a local
`llama.cpp` server:

```bash
# start llama.cpp server locally, e.g.:
./llama-server -m qwen2.5-0.5b-instruct-q4_k_m.gguf --port 8080
```

```yaml
parser:
  use_llm_fallback: true
  llm_backend: "llama_cpp"
  llm_endpoint: "http://127.0.0.1:8080/completion"
```

The LLM is grammar-constrained to emit JSON and prompted: *"Return only valid
JSON. Use only the allowed tools. If unsupported or ambiguous, return clarify."*
Its output is **validated with Pydantic** before anything is published — malformed
or hallucinated tool calls are rejected and become `clarify`.

---

## Switching the publisher from stub to the real planner

The discovered, real planner path is the **ZMQ `command`/`planner` topics**
consumed by the C++ `ZMQManager`. Wire to it with the `existing_repo` (or `zmq`)
backend:

```yaml
publisher:
  backend: "existing_repo"
  zmq_endpoint: "tcp://127.0.0.1:5556"   # ZMQManager planner-input port (NOT 5557 = debug out)
safety:
  dry_run: false
  execute: true                          # BOTH gates required for real commands
```

```bash
python -m voice_control.cli --config configs/voice_control.yaml --execute
```

`ExistingRepoPublisher` uses `build_command_message` / `build_planner_message`
from `gear_sonic.utils.teleop.zmq.zmq_planner_sender` over a `zmq.PUB` socket. It
sends a `command` message (`start=True, planner=True`) then streams `planner`
messages (mode/movement/facing/speed/height) at up to 10 Hz — exactly like
`PlannerStreamer` in `gear_sonic/scripts/pico_manager_thread_server.py`.

If `pyzmq` or the repo builders are missing, the publisher raises a precise error
explaining the single integration point and how to fix it.

A `Ros2Publisher` is also provided for environments that bridge planner commands
over a ROS2 topic, **but note the deployed SONIC stack uses ZMQ, not ROS2**, for
planner commands.

---

## Composition (sequential skills)

A single utterance can chain multiple skills using natural connectors
(`and then`, `then`, `and`, `after that`, `followed by`, `next`, `,`). The
utterance is split into ordered sub-commands, each parsed and published in turn.

```
voice "walk forward at velocity 5 meters per second and then turn around and kneel on one leg"
```

produces a 3-step plan:

1. `set_navigation` v=5.0, heading 0 (walk forward)
2. `set_navigation` v=0.0, heading 180 (turn around in place)
3. `set_posture` kneel_one_leg

In **execute** mode each motion step is held for its `duration_s`, re-published
at `planner_dt`, before advancing to the next step; the final step is left
active. In dry-run the full plan (with durations) is printed without dwelling.

### LLM-decided step durations

Set `parser.use_llm_duration: true` to have a **local** LLM dynamically decide
how long each step should run before advancing. The estimator is given the parsed
tool call (action + arguments) *and* the original phrase, so it can reason about
distances that aren't planner fields — e.g. "walk 5 m at 3 m/s" gets a longer
duration than "walk 3 m at 3 m/s". It returns a single `{"duration_s": ...}`
(grammar-constrained, validated, and bounded by `llm_duration_min_s` /
`llm_duration_max_s`).

If the LLM is unavailable or returns garbage, a deterministic heuristic is used
so the pipeline still works fully offline:

* navigation/crawl with a stated distance → `distance / speed + 1 s` settle,
* in-place turn (speed 0) → ~1 s per 60° of heading,
* posture → ~3 s, boxing → ~1.5 s, get_up → ~4.5 s,
* otherwise → `publisher.segment_dwell_s` (default 3 s).

The chosen duration is written back onto each step's `duration_s`, so it shows up
in the printed plan and drives the execute-mode dwell. No cloud services.

#### LLM backends

`parser.llm_backend` selects how the local model is run:

* `hf_transformers` (default) — loads a HuggingFace checkpoint **in-process** via
  `transformers` + `torch`. Defaults to the LiquidAI LFM2 on-device reasoning
  model (`tim_grpo230M_..._HF`, ~230M params, hybrid conv/attention, built for
  Jetson/edge). Configure with `hf_model_id`, `hf_device` (`auto|cpu|cuda|mps`),
  `hf_dtype`, `hf_max_new_tokens`, `hf_temperature`. The model/tokenizer are
  loaded once and cached. Output parsing is robust to reasoning traces — it
  accepts strict JSON, a `duration_s: N` fragment, a `\boxed{N}` answer, or a
  bare trailing number.
* `llama_cpp` — POSTs to a local llama.cpp `/completion` server at `llm_endpoint`
  (grammar-constrained JSON).

If the selected backend can't be loaded/reached, the deterministic heuristic
above is used so the pipeline always works offline.

Install the HF backend deps on the target device:

```bash
pip install "transformers>=4.55" torch
```

---

## Safety

> **Note:** value clamps have been removed by request. Navigation/crawl speeds,
> pelvis heights, and headings are passed through to the planner **unchanged**.
> The non-clamping gates below are still in place.

* **Dry-run is the default.** `safety.dry_run: true` and `safety.execute: false`
  ship by default. Real commands require `--execute` **and** `safety.execute:
  true` — either alone keeps you in dry-run.
* Stop words (`stop`, `halt`, `freeze`, `cancel`, `emergency stop`, …) are honoured
  immediately, with the highest priority, even without a wake word.
* Velocity / height / heading are **not clamped** — spoken values are forwarded
  verbatim (the planner ONNX still bounds outputs internally).
* Headings are canonicalised to `[-180, 180]` at parse time (representation only).
* Low-confidence or ambiguous commands **do not move** — they return `clarify`.
* **Command timeout watchdog**: if no fresh command arrives within
  `command_timeout_s`, a safety stop is published (the C++ side also resets to
  IDLE after its own 1 s planner timeout).
* Never produces joint commands; never bypasses planner/policy safety.

Every raw transcript, normalised text, parsed tool call, and final planner
command is logged (`logging.log_file`, default `logs/voice_control.log`).

---

## Supported utterances (examples)

| Say | Result |
|-----|--------|
| "walk forward" | `set_navigation` v≈0.6, heading 0, walking |
| "move forward slowly" | `set_navigation` v=0.3 |
| "walk forward fast" | `set_navigation` v=1.0 |
| "walk forward at 5 meters per second" | `set_navigation` v=5.0 (no clamp) |
| "run forward" | `set_navigation` v=1.5, running |
| "sprint forward" | `set_navigation` v=2.0, running |
| "turn around" | in-place facing change, heading 180 |
| "X and then Y" | sequential plan (see Composition) |
| "go backward" | heading 180, v=0.4 |
| "move left" / "move right" | heading −90 / +90 |
| "turn left" | in-place facing change (v=0), heading −90 |
| "stop" / "emergency stop" | `stop` |
| "squat" / "squat lower" / "squat higher" | `set_posture` squat, height 0.5 / 0.35 / 0.65 |
| "kneel" / "kneel on one knee" | `set_posture` kneel_two_legs / kneel_one_leg |
| "stand up" | `set_posture` stand |
| "get up" | `get_up` |
| "crawl forward" | `set_crawl` v=0.25, elbow_knee |
| "hand crawl forward" | `set_crawl` v=0.20, hand_crawl |
| "left jab" / "right hook" / "block" / "boxing stance" | `set_boxing_action` |
| "dance like a monkey" | `clarify` (unsupported — no text-motion/GEM path here) |

Natural-language / text-to-motion ("dance like a monkey") is intentionally **not**
exposed; this interface only drives the closed kinematic-planner skill set.

---

## Adding a new skill

1. Add the value to the relevant enum in `schemas.py` (e.g. a new `NavStyle`).
2. Map it to a planner `LocomotionMode` in `skills.py`
   (`STYLE_TO_MODE` / `POSTURE_TO_MODE` / `BOXING_ACTION_TO_MODE` / …).
3. Add synonyms/trigger words in `skills.py` and detection logic in `parser.py`.
4. Add a unit test in `tests/test_voice_parser.py`.
5. (Optional) mention the new phrase in the LLM prompt in `llm_parser.py`.

The publisher mapping (`tool_call_to_planner_fields`) will pick up the new command
automatically as long as it maps to a `LocomotionMode`.

---

## Troubleshooting microphones on Jetson

* List input devices: `python -c "import sounddevice as sd; print(sd.query_devices())"`
  then set `audio.device` to the desired index.
* Install PortAudio if `sounddevice` import fails:
  `sudo apt install libportaudio2 libportaudiocpp0 portaudio19-dev`.
* USB mics sometimes enumerate at 44.1/48 kHz only; keep `audio.sample_rate:
  16000` (both Vosk and whisper.cpp want 16 kHz) and let the driver resample, or
  pick a 16 kHz-capable device.
* If VAD cuts speech too aggressively, lower `audio.vad_aggressiveness` (0–3) or
  increase `audio.phrase_timeout_s`.
* Permission errors: ensure your user is in the `audio` group
  (`sudo usermod -aG audio $USER`) and re-login.
* Keep ASR off the control host if CPU is tight — run voice control as a separate
  process that only emits high-level ZMQ commands (it already runs in its own
  process and threads, so it does not affect planner/control timing).

---

## Tests

```bash
pip install pytest
pytest tests/test_voice_parser.py tests/test_voice_safety.py \
       tests/test_voice_schema.py tests/test_voice_publisher_stub.py \
       tests/test_voice_duration.py -q
```
