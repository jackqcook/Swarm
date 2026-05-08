# Swarm — Commands to Run the Simulation

All shell commands, in order. Three sections:
- **A. One-time setup** — only the first time on this machine.
- **B. Run the visual SITL simulation** — three terminals, every time.
- **C. Run the mock simulation (no PX4)** — one terminal, fast sanity check.

---

## A. One-time setup

Run these once, in order. Each block is independent — if one fails, fix it before moving on.

### A1. Swarm Python venv (agent dependencies)

```bash
cd /Users/jackcook/Desktop/Personal_projects/Swarm
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
deactivate
```

### A2. Hawkeye 3D viewer

```bash
brew tap PX4/px4
brew install PX4/px4/hawkeye
```

### A3. PX4-Autopilot source + macOS toolchain

```bash
git clone https://github.com/PX4/PX4-Autopilot.git --recursive ~/PX4-Autopilot
cd ~/PX4-Autopilot
bash ./Tools/setup/macos.sh
brew link --overwrite --force arm-gcc-bin@13
```

### A4. Dedicated PX4 Python venv (build dependencies)

Keeps PX4's build deps isolated from the Swarm venv.

```bash
cd ~/PX4-Autopilot
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r Tools/setup/requirements.txt
deactivate
```

### A5. Shell config (open-files limit + PX4_ROOT)

```bash
echo 'ulimit -S -n 2048' >> ~/.zshrc
echo 'export PX4_ROOT=$HOME/PX4-Autopilot' >> ~/.zshrc
ulimit -S -n 2048
export PX4_ROOT=$HOME/PX4-Autopilot
```

### A6. Make the launch script executable

```bash
chmod +x /Users/jackcook/Desktop/Personal_projects/Swarm/scripts/launch_sitl.sh
```

---

## B. Run the visual simulation (three terminals)

Open three separate terminal tabs/windows. Run them in this order — each one waits for the previous one to be ready.

### Terminal 1 — PX4 SITL (the simulator backend)

```bash
cd /Users/jackcook/Desktop/Personal_projects/Swarm
./scripts/launch_sitl.sh
```

**Wait for**: `INFO [commander] Ready for takeoff!`
**Leave running.** First build is 5–10 min; subsequent runs are instant.

### Terminal 2 — Hawkeye (the 3D viewer)

```bash
hawkeye
```

A 3D window opens showing the quadrotor on the ground.
**Leave running.** Useful keys: `C` cycles camera modes, `WASD+QE` flies the free camera, `G` toggles ground track.

### Terminal 3 — the drone agent (where you type commands)

```bash
cd /Users/jackcook/Desktop/Personal_projects/Swarm
source .venv/bin/activate
python -m drone.agent --sim --config configs/sim_default.yaml
```

**Wait for**: `Connected to drone.` then `DroneAgent ready. Waiting for commands.`

You'll get a `Command >` prompt. Type natural-language commands:

```
Command > hover at 20 meters
Command > fly northeast and check the field
Command > orbit here
Command > return home
Command > quit
```

Watch the drone respond in Hawkeye's window.

---

## C. Mock simulation (no PX4, fast sanity check)

Use this to verify the natural-language pipeline without installing PX4 / running the simulator. One terminal, instant startup.

```bash
cd /Users/jackcook/Desktop/Personal_projects/Swarm
source .venv/bin/activate
python -m drone.agent --sim --config configs/mock_default.yaml
```

Then at the `Command >` prompt:

```
Command > return home
Command > hover at 25 meters
Command > fly northeast and check the field
Command > unclear go to the thing somewhere?
Command > quit
```

Lines starting with `[MOCK]` are the MAVLink actions the agent would have sent to the real PX4.

### One-shot variant (no REPL)

```bash
python scripts/send_command.py --direct --sim --config configs/mock_default.yaml "fly northeast and check the field"
```

---

## D. Run the unit tests

Sanity check that nothing's broken after edits:

```bash
cd /Users/jackcook/Desktop/Personal_projects/Swarm
source .venv/bin/activate
pytest -q
```

Expected: 25 tests pass.

---

## Tearing down

- **Terminal 3 (agent)**: type `quit` or hit Ctrl+C.
- **Terminal 2 (Hawkeye)**: close the window or Ctrl+C.
- **Terminal 1 (PX4)**: Ctrl+C (sends `shutdown` to the PX4 console).
