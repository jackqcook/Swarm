# 🚁 Swarm — Text-to-Goal Drone Control

A modular, computationally practical system for controlling drones (and swarms of drones) using vague natural language commands. Built around a **two-tier architecture**: a small Vision-Language Model (VLM) for high-level intent interpretation, and classical perception + planning for millisecond-latency obstacle avoidance.

---

## Architecture Overview

```
User Command ("fly over there and check that field")
        │
        ▼
┌─────────────────────────────┐
│     TIER 1: Interpretation  │  ~1–4 seconds
│  Small VLM (PaliGemma2-3B)  │
│  + KnowNo Ambiguity Gating  │
│  → Structured Goal/Waypoint │
└─────────────┬───────────────┘
              │  goal: {lat, lon, alt, task}
              ▼
┌─────────────────────────────┐
│     TIER 2: Execution       │  ~10–30 Hz perception
│  YOLOv8n  + MonoDepth       │  ~1 ms reactive avoidance
│  EGO-Planner local planning │
│  PX4 autopilot (250–1kHz)   │
└─────────────────────────────┘
```

---

## Module Map

```
Swarm/
├── drone/              # Single-drone agent: state, command interface, MAVLink
├── swarm/              # Multi-drone coordination: leader election, task distribution
├── perception/         # YOLOv8n obstacle detection, monocular depth estimation
├── planner/            # Waypoint generation, EGO-style local replanning
├── comms/              # Mesh networking, trajectory broadcast, SwarmRaft consensus
├── tests/              # Unit + integration tests, SITL simulation harness
├── configs/            # Hardware profiles (Orin Nano, Orin NX, VOXL 2), mission params
├── scripts/            # Setup, calibration, SITL launch helpers
└── docs/               # Architecture decisions, hardware guide, wiring diagrams
```

---

## Stage Roadmap

| Stage | Description | Status |
|-------|-------------|--------|
| 1 | Single drone, GPS outdoors, Pixhawk + Jetson Orin Nano, text→waypoint | 🔨 Active |
| 2 | GPS-denied (ORB-SLAM3 VIO), NanoOWL open-vocab grounding | ⏳ Planned |
| 3 | Swarm: EGO-Swarm decentralized planner, mesh comms, task distribution | ⏳ Planned |
| 4 | Upgrade compute path (Orin NX 16GB, then AGX Orin if needed) | ⏳ Planned |

---

## Quick Start (Simulation)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch SITL (requires PX4 installed separately)
./scripts/launch_sitl.sh

# 3. Run the drone agent in simulation mode
python -m drone.agent --sim --config configs/sim_default.yaml

# 4. Send a command
python scripts/send_command.py "fly to the northeast corner and hover at 30 meters"
```

---

## Hardware Target (Stage 1)

- **Flight controller:** Holybro Pixhawk 6C or Cube Orange+
- **Companion computer:** NVIDIA Jetson Orin Nano 8GB (Super Mode, JetPack 6.2+)
- **Carrier board:** Holybro Pixhawk Jetson Baseboard
- **Depth/Stereo:** Intel RealSense D435i
- **GPS:** u-blox M9N or M10
- **Comms (FC↔companion):** MAVLink over UART / uXRCE-DDS

---

## Key Design Decisions

1. **No full VLA in the inner loop.** Full VLAs (OpenVLA-7B) run at 3 FPS on AGX Orin — far too slow for reactive flight. The VLM only emits waypoints; classical planners handle obstacle avoidance.
2. **Clarification-first ambiguity handling.** KnowNo conformal prediction decides ask-vs-act. The drone won't guess on ambiguous commands.
3. **Decentralized swarm.** EGO-Swarm style: each drone runs its own planner and broadcasts trajectories. No central coordinator in the safety-of-flight path.
4. **PX4 as the hard-real-time layer.** PX4 runs at 250–1000 Hz on NuttX. The companion computer talks to it over MAVLink/MAVSDK.

---

## Dependencies

See `requirements.txt` for the full list. Key packages:
- `mavsdk` — Python bindings for MAVSDK (MAVLink drone control)
- `ultralytics` — YOLOv8
- `transformers` — VLM inference (PaliGemma2, SmolVLM)
- `numpy`, `scipy`, `opencv-python`
- `pyzmq` — inter-process and inter-drone messaging
- `pyyaml` — config management
