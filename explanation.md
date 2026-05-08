# Swarm — Project Explanation & Simulator Guide

This document explains what every part of the `Swarm/` repo does, how a natural-language command actually becomes drone motion, and exactly how to run it on your laptop to see the language commands working. There are two run paths:

1. **Mock-only mode** — runs immediately on your Mac, no drone software needed. Best for verifying that natural-language commands are interpreted and routed correctly.
2. **Full SITL mode** — uses PX4's Software-In-The-Loop simulator (with Gazebo) so you can see a virtual drone actually fly in response to your commands.

---

## 1. The big picture

The system is a **two-tier text-to-flight controller**:

```
"fly to the northeast field at 30 meters"
        │
        ▼
┌───────────────────────────────┐
│  TIER 1 — Interpretation       │   seconds-latency, runs once per command
│  Vision-Language Model (VLM)   │
│   + KnowNo ambiguity gating    │
│  → DroneGoal{action, alt, …}   │
└────────────┬───────────────────┘
             │
             ▼
┌───────────────────────────────┐
│  TIER 2 — Execution            │   millisecond-latency, runs continuously
│  YOLOv8n + monocular depth     │
│  → ObstacleMap                 │
│  Waypoint planner + reactive   │
│   potential-field avoidance    │
│  → MAVLink velocity setpoints  │
│  → PX4 (real or SITL)          │
└───────────────────────────────┘
```

**Why split it this way?** A full Vision-Language-Action model (e.g. OpenVLA-7B) is too slow for the inner control loop — only ~3 FPS even on an AGX Orin. So the project pushes the slow, smart model up to Tier 1 (intent only), and uses fast classical CV + control in Tier 2 (reaction).

---

## 2. Module-by-module walk-through

Every Python module corresponds to one box in the diagram above.

### `drone/agent.py` — the orchestrator
The `DroneAgent` class wires every subsystem together. On `start()` it:

1. Builds a `MavlinkBridge` and connects to either SITL (`udp://:14540`) or a serial port.
2. Constructs a `CommandInterpreter` (Tier 1).
3. Starts a `PerceptionLoop` background thread (Tier 2 sensing).
4. Constructs a `WaypointPlanner` (Tier 2 acting).

`handle_command(text)` is the public entry point: text → interpreter → goal → planner → flight. If the interpreter is uncertain, the function instead returns a `CLARIFY: …` string with a follow-up question rather than acting.

`run_interactive()` is a simple REPL that reads commands from stdin — this is what `--interactive` uses.

### `drone/command_interpreter.py` — Tier 1 (the “brain”)
Three pieces:

- **`VLMBackend`** — wraps a HuggingFace VLM (`PaliGemma2-3B`, `SmolVLM-2B`, `Qwen2.5-VL-3B`, `moondream2`). The system prompt forces the model to emit a JSON object containing four candidate plans `A–D` plus an `E` = "ask for clarification", along with a probability for each.
- **`KnowNoGate`** — a conformal-prediction-style filter. It builds a "prediction set" of options whose probability ≥ τ (default 0.6). If the set is exactly one option (and not `E`), it executes; otherwise it asks the operator. This is the project's principled way of saying "don't guess on ambiguous commands."
- **`CommandInterpreter`** — glues those together and packages the result as `InterpretResult` (either a `DroneGoal` or a `clarification_question`).

**Crucially for testing**: if the VLM fails to load (no GPU, no internet, no `transformers` install), `VLMBackend._mock_interpret()` kicks in. It uses keyword heuristics so you can validate the entire pipeline without ever loading a real model:
- Words like `return`, `home`, `rtl` → high-confidence `return` action.
- Words like `hover`, `stay`, `hold` → high-confidence `hover` action.
- Words like `unclear`, `which`, `where exactly` → forces `E=ask` to win (triggers clarification).
- Anything else → moderate confidence on `fly_to` (often below τ = 0.6, so the gate asks for clarification — good for demoing the ambiguity behavior).

### `drone/mavlink_bridge.py` — the "hands"
A thin async wrapper over [MAVSDK-Python](https://mavsdk.mavlink.io/main/en/python/). Exposes `arm`, `takeoff`, `land`, `return_to_launch`, `fly_to_position` (GPS waypoint), `send_velocity_ned` (used by the reactive planner), plus telemetry properties (`position`, `velocity`, `is_armed`, `flight_mode`).

If `mavsdk` isn't installed it falls into a **mock mode** that just logs every action — useful for unit-testing the upstream logic.

### `perception/perception_loop.py` — Tier 2 sensing
Runs a background thread that, at `fps` Hz:
1. Pulls a frame from a camera backend (`MockCamera`, `WebcamCamera`, or `RealSenseCamera`).
2. Runs YOLOv8n (`ObstacleDetector`) for bounding boxes — classes filtered to "people, vehicles, animals, trees, buildings."
3. Computes depth — either from RealSense stereo, or `MonocularDepthEstimator` (MiDaS-Small) on a webcam.
4. Calls `fuse_depth_with_detections` to assign a metric distance + bearing to each box.
5. Publishes a single `ObstacleMap` (frame + depth + detections + nearest obstacle) on an `asyncio.Queue` that the planner reads.

Each backend gracefully degrades: missing `pyrealsense2` → `MockCamera`; missing `ultralytics` → mock detections that occasionally fake a person at 8 m. So this also runs end-to-end on a Mac with nothing installed.

### `planner/waypoint_planner.py` — Tier 2 acting
Two layers:

- **`GlobalPlanner`** — turns a `DroneGoal` into an ordered list of `Waypoint(lat, lon, alt)`. Stage 1 is intentionally simple: it scans `goal.region_description` for direction keywords (`north`, `northeast`, `forward`, `left`, …) and projects 100 m in that direction from current GPS. Special actions: `survey_area` → lawnmower pattern, `orbit` → 8-point circle, `hover/land/return` → handled directly.
- **`LocalPlanner`** — a textbook **potential-field** controller. Each control tick (10 Hz) it sums an attractive velocity toward the next waypoint and a repulsive velocity away from the nearest obstacle (only active when the obstacle is inside `safety_radius_m`). The output is clamped to `max_speed_m_s` and sent as an NED velocity to PX4. This is the inner part of what EGO-Planner does — Stage 3 will replace it with the full ESDF-based version.

`WaypointPlanner.execute_goal()` arms, takes off, switches to OFFBOARD, then loops over waypoints and runs `_navigate_to_waypoint`, which is the actual reactive-flight loop.

### `configs/`
- `sim_default.yaml` — everything mocked (`model_id: "mock"`, `camera_backend: "mock"`, `mavsdk` connects to SITL on `udp://:14540`). This is the file you want for laptop testing.
- `hardware_orin_nano.yaml` — production profile (PaliGemma2-3B on CUDA, RealSense, real serial port, INT4 quantization).

### `scripts/`
- `launch_sitl.sh` — wraps `make px4_sitl gazebo-classic_iris` inside a PX4 checkout. You run this in a separate terminal.
- `send_command.py` — CLI for sending one command to a running agent. With `--direct --sim` it spawns the agent **inline** and runs a single command end-to-end, which is the easiest way to test.

### `tests/`
Pytest suite that exercises the interpreter, planner, and perception loop with all mocks. Running them is the very first sanity check that everything is wired up correctly.

---

## 3. The end-to-end code path of one command

Tracing `"fly to the northeast field at 30 meters"`:

1. `scripts/send_command.py` calls `DroneAgent.handle_command(text)`.
2. `CommandInterpreter.interpret(text, image=latest_camera_frame)` runs the VLM (or mock). The output is a JSON of candidates + probabilities like `{"A": 0.55, "B": 0.20, …}`.
3. `KnowNoGate.decide(probs)` builds the prediction set. With τ = 0.6 and no option above 0.6, the set is empty → **`needs_clarification = True`**, and the user gets back: `CLARIFY: I'm not sure what you mean. Did you want me to: A) Fly toward described destination …`.
4. The user answers (or sends a less ambiguous command). When one option clears τ, the interpreter returns a `DroneGoal(action="fly_to", region_description="northeast field", target_alt_m=30)`.
5. `WaypointPlanner.execute_goal(goal)` calls `GlobalPlanner.generate_waypoints(...)`. Since `"northeast"` is in the description, it offsets current GPS by `(70.7 m N, 70.7 m E)`. One `Waypoint` is produced.
6. The agent arms, takes off to 30 m, switches to OFFBOARD.
7. `_navigate_to_waypoint` runs at 10 Hz: pulls the latest `ObstacleMap`, runs `LocalPlanner.compute_velocity` (attraction toward waypoint + repulsion if any obstacle < 5 m), sends `send_velocity_ned(vN, vE, 0)` to PX4.
8. When `haversine_m(current, waypoint) < 5 m`, the loop exits, the drone loiters 5 s, and the goal completes.

The "natural language → flight" mapping is therefore: **VLM JSON → KnowNo gate → DroneGoal → keyword-resolved GPS waypoint → reactive velocity loop → PX4**.

---

## 4. Running it on your machine

You're on macOS, so I'll give you the fast path first (which is what you actually want for "do the natural-language commands work?") and then the heavier full-flight path.

### 4.1. Setup (do this once)

```bash
cd /Users/jackcook/Desktop/Personal_projects/Swarm

python3.11 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Notes:
- Python 3.11 is what `pyproject.toml` targets.
- On Apple Silicon, `torch` will install the MPS-capable wheel automatically — fine for the mock path. You don't need CUDA.
- If `mavsdk` fails to install with a binary error on macOS, you can skip it for the mock-only path; `MavlinkBridge` will simply use mock mode.

Sanity check the wiring with the unit tests:

```bash
pytest -v
```

All tests should pass without any drone software, camera, or GPU — they exhaustively exercise the interpreter, gate, planner, and perception loop using mocks.

### 4.2. Fast path: interactive REPL with no PX4

This is the right first thing to do. Use `configs/mock_default.yaml`, which sets:
- `interpreter.model_id: "mock"` → keyword-heuristic VLM (no GPU/network).
- `mavlink.mock: true`            → MAVSDK skips its real connection attempt and just logs every action.
- `perception.camera_backend: "mock"` → synthetic frames + mock detections.
- `interpreter.knowno_tau: 0.50`   → low enough that generic `fly_to` commands execute.

Start the persistent agent in interactive mode:

```bash
python -m drone.agent --sim --config configs/mock_default.yaml
```

You'll get a `Command >` prompt. Type whatever you want:

```
Command > return home
  → Executing: Return to launch point
    [MOCK] return_to_launch()

Command > hover at 25 meters
  → Executing: Hover at current position
    [MOCK] arm()

Command > fly northeast and check the field
  → Executing: Fly toward described destination
    [MOCK] takeoff(altitude=30.0)

Command > unclear go to the thing somewhere?
  → CLARIFY: I'm not sure what you mean. Did you want me to:
      A) Hover in place
      B) Fly north 100m
      C) Survey nearby area
      D) Return to launch
    Or something else entirely?

Command > quit
```

Every line that begins with `[MOCK]` is what the bridge would have sent to the real PX4 (`arm`, `takeoff(altitude=...)`, `return_to_launch`, `goto_location(lat, lon, alt)`, `set_velocity_ned(vN, vE, vD)`). So you can confirm the natural-language → flight-action path is correct without needing a virtual drone.

If you'd rather fire one-shot commands instead of staying in the REPL:

```bash
python scripts/send_command.py --direct --sim --config configs/mock_default.yaml "return to home base"
python scripts/send_command.py --direct --sim --config configs/mock_default.yaml "hover here at 25 meters"
python scripts/send_command.py --direct --sim --config configs/mock_default.yaml "fly northeast and check the field"
python scripts/send_command.py --direct --sim --config configs/mock_default.yaml "unclear go to the thing somewhere?"
```

What you should see:
- `"return to home base"` → `Executing: Return to launch point` (singleton at p=0.80, gate accepts).
- `"hover here at 25 meters"` → `Executing: Hover at current position` (singleton at p=0.70).
- `"fly northeast and check the field"` → likely `CLARIFY: …` because the mock assigns `A=0.55` which is below τ=0.6. This is the intended demo of the KnowNo gate.
- `"unclear go to the thing somewhere?"` → `CLARIFY: …` (E=0.6 dominates, asking for help).

If you instead use `configs/sim_default.yaml`, the agent will block on `MavlinkBridge.connect()` waiting for a real PX4 SITL on `udp://:14540` and never reach the prompt. That config is for the SITL path in §4.4. For typed-command testing without a virtual drone, always use `configs/mock_default.yaml`.

You can also tune the ambiguity threshold to see how the KnowNo gate changes behavior. Edit `configs/mock_default.yaml`:

```yaml
interpreter:
  knowno_tau: 0.60    # default in sim_default — generic fly_to (p=0.55) gets clarified
  knowno_tau: 0.50    # current mock_default — generic fly_to executes
```

### 4.3. Want to use a real VLM in the mock-flight path?

Even without SITL, you can swap the mock interpreter for an actual small VLM and watch it produce structured plans. Edit `configs/sim_default.yaml`:

```yaml
interpreter:
  model_id: "vikhyatk/moondream2"   # tiny + fast; runs on CPU/MPS
  device: "mps"                     # or "cpu"
  quantize: false
```

Then re-run the same commands. The VLM will be downloaded on first use (a few hundred MB) and `_interpret_sync` will produce real probabilities. The flight side is still mocked; only the interpretation tier becomes "real."

### 4.4. Full path: see the drone actually fly in PX4 SITL (with Hawkeye 3D viewer)

If you want to *watch* the drone respond visually on macOS, the working stack as of 2026 is:

- **Backend**: PX4 SITL with the **SIH** (Simulation-In-Hardware) target. Physics runs inside the PX4 process, no external sim engine. Officially supported on Apple Silicon.
- **Visualizer**: **Hawkeye** — PX4's lightweight 3D viewer, single Homebrew install. It's the modern replacement for jMAVSim (which was removed from PX4 in late 2025) and avoids Gazebo's macOS install pain.

Three pieces, three terminals.

#### One-time install (~30 min on first build)

```bash
# 1. Hawkeye visualizer (Homebrew, ~10 s)
brew tap PX4/px4
brew install PX4/px4/hawkeye

# 2. PX4-Autopilot source (~1 GB clone, several min)
git clone https://github.com/PX4/PX4-Autopilot.git --recursive ~/PX4-Autopilot
cd ~/PX4-Autopilot

# 3. macOS dev environment (ARM cross-compiler, build deps; ~10–20 min)
bash ./Tools/setup/macos.sh

# 4. macOS docs say to manually link the ARM compiler after install
brew link --overwrite --force arm-gcc-bin@13

# 5. Increase open-files limit (PX4 docs recommend this)
echo 'ulimit -S -n 2048' >> ~/.zshrc
ulimit -S -n 2048

# 6. Set PX4_ROOT so launch_sitl.sh finds it
export PX4_ROOT=$HOME/PX4-Autopilot
echo 'export PX4_ROOT=$HOME/PX4-Autopilot' >> ~/.zshrc
```

#### Run, in three terminals

**Terminal 1 — PX4 SITL with SIH backend:**

```bash
cd /Users/jackcook/Desktop/Personal_projects/Swarm
./scripts/launch_sitl.sh
```

That wraps `make px4_sitl_sih sihsim_quadx` in `~/PX4-Autopilot`. First build is ~5–10 min; subsequent runs are instant. Wait until you see `INFO  [commander] Ready for takeoff!`. PX4 is now:
- Listening for MAVSDK on UDP **14540** (what our agent connects to).
- Streaming `HIL_STATE_QUATERNION` to UDP **19410** (what Hawkeye renders).

**Terminal 2 — Hawkeye 3D viewer:**

```bash
hawkeye
```

A window opens with a 3D view of the quadrotor sitting on the ground. It auto-detects the vehicle model from MAVLink and starts rendering immediately. Useful keybinds: `C` cycles cameras (chase / FPV / free), `WASD+QE` flies the free camera, `G` toggles ground track, `Tab` cycles vehicles in multi-drone setups.

**Terminal 3 — the drone agent connected to SITL:**

```bash
cd /Users/jackcook/Desktop/Personal_projects/Swarm
source .venv/bin/activate
python -m drone.agent --sim --config configs/sim_default.yaml
```

You'll see `Connected to drone.` once MAVSDK handshakes with PX4, then `DroneAgent ready. Waiting for commands.` and a `Command >` prompt. Now type:

```
Command > hover at 20 meters
Command > fly northeast and check the field
Command > orbit here
Command > return home
```

Watch Hawkeye: the quad arms, takes off, flies the resolved GPS offsets, and reacts to mock obstacles via the potential-field controller. Terminal 3 logs lines like `OBSTACLE at 2.3 m bearing 5.0° → repulsion (-1.99, -0.17)` whenever the mock detector spawns a synthetic obstacle inside the 5 m safety radius.

#### Switching to a real (small) VLM in SITL

Same edit as in §4.3 — change `interpreter.model_id` in `configs/sim_default.yaml`. Everything else (PX4 SITL, Hawkeye, planner, perception) is unchanged.

#### Map view alternative — QGroundControl

If you also want a top-down map of the flight, install QGroundControl from <https://qgroundcontrol.com/downloads> and launch it. It auto-connects to PX4 on UDP 14550 and shows the drone marker on a real-world map alongside Hawkeye's 3D view.

### 4.5. Running the unit tests against the real planner/perception

```bash
pytest tests/test_perception_and_planner.py -v
pytest tests/test_command_interpreter.py -v
```

These cover:
- `KnowNoGate` thresholding (singleton, ambiguous, ask-option-wins, exactly-at-threshold).
- The mock VLM's keyword routing for `return`, `hover`, ambiguous, generic.
- Coordinate helpers (`haversine_m`, `offset_gps`).
- `GlobalPlanner` waypoint generation for `hover`, `fly_to`, `orbit`, `survey`, unknown direction.
- `LocalPlanner` attraction-only, repulsion-on-obstacle, max-speed clamp.
- `PerceptionLoop` start/stop, queue publishing, frame shape.

If you change interpreter logic or the planner, run these first.

---

## 5. What is and isn't real yet (Stage 1 honesty)

It's worth being explicit about which parts are production-quality and which are scaffolding:

| Component | Status |
|---|---|
| Two-tier orchestration (`DroneAgent`) | Real, working glue. |
| KnowNo gate | Real, but τ is hard-coded — calibration with a labeled dev set is `NotImplementedError`. |
| VLM backend | Real wrapper, but the prompt assumes the model can output exactly `{options, probs}` JSON; a small open-weight VLM may need few-shot prompting to comply reliably. The mock backend is what's used in tests. |
| MAVLink bridge | Real MAVSDK wrapper, falls back to mock. |
| Perception (YOLO + depth) | Real wrappers around `ultralytics`/MiDaS/RealSense, with mock fallbacks. The depth fusion is correct; monocular MiDaS depth is uncalibrated (scaled to 0–20 m). |
| Global planner | Direction-keyword heuristics only — it does **not** ground "the barn" to a GPS coordinate. Visual grounding (NanoOWL) is Stage 2. |
| Local planner | A simplified potential field. The full EGO-Planner is Stage 3. |
| Swarm / multi-drone | Folder doesn't exist yet despite the README's module map. Stage 3. |
| Comms (`comms/`) | Empty `__init__.py` — placeholder for ZMQ + SwarmRaft work. |

So when you "see the natural-language commands work" in simulation, what you're really verifying is:

- The VLM (or its mock) emits structured options.
- KnowNo correctly chooses execute-vs-ask.
- The chosen option is parsed into a `DroneGoal` and projected into a GPS waypoint by direction keywords.
- The waypoint is flown via PX4 with reactive velocity setpoints.

That is the full Stage 1 promise, and everything required to demo it is in this repo plus PX4 SITL.

---

## 6. Quick troubleshooting

- **`mavsdk` import error on macOS** → safe to ignore for the mock path; or install `mavsdk` from a working wheel and re-run.
- **Agent prints `[MOCK] MavlinkBridge.connect()` even though SITL is up** → `mavsdk` failed to import. Reinstall it in the active venv.
- **Every command returns `CLARIFY:` in mock mode** → expected for generic commands; the mock assigns A=0.55 and τ=0.6 rejects it. Lower `knowno_tau` to 0.5 or use direct commands like `return`, `hover`, `land`.
- **`No GPS position available`** in SITL → wait until PX4 logs `Ready for takeoff!` before sending commands; `bridge.position` is populated by the telemetry stream.
- **PX4 build is slow first time** → normal; subsequent `make px4_sitl gazebo-classic_iris` runs are incremental.
- **Want only the interpreter, no flight at all?** Run `pytest tests/test_command_interpreter.py -v` — pure-Python, instant.
