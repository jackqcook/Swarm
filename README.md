# Swarm PPO

This project trains a shared-policy multi-agent PPO controller in a 3D environment with:

- velocity and acceleration 2-norm limits
- randomly sampled training objectives
- fixed evaluation objectives
- Gaussian objective rewards with per-objective amplitudes and variances
- ordinal task completion penalties
- obstacle penalties
- training curves and a 3D rollout animation

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python train_swarm_ppo.py --episodes 250 --steps-per-episode 150
```

If you want a coordination-friendly shaping variant while keeping the rest of the setup the same:

```bash
.venv/bin/python train_swarm_ppo.py --reward-aggregation max_agents --completion-bonus-scale 4.0
```

Outputs are written to `results/<timestamp>/`:

- `losses.png`
- `metrics.png`
- `evaluation_summary.json`
- `trained_rollout.gif` when Pillow export is available

## Notes

- The environment defaults to your requested reward form: sum of active objective Gaussians, plus a small per-step time penalty and obstacle / out-of-order penalties.
- If you want stronger specialization between agents, the next reward change worth testing is swapping from a sum over all agents to a per-objective max-over-agents contribution.
