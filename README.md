# Swarm PPO

This project trains a shared-policy multi-agent PPO controller in a 3D environment with:

- velocity and acceleration 2-norm limits
- randomly sampled training objectives
- randomized training obstacles plus fixed and generalized evaluation
- fixed evaluation objectives
- completion-dominant rewards with small Gaussian shaping
- explicit per-agent target assignment for local progress shaping
- explicit assignment features in each agent observation
- team-level objective completion in the shortest possible timeframe
- obstacle penalties
- imitation warm start from an assignment-based expert controller
- staged PPO refinement with a phase-1 generalization gate
- BC-retention regularization to reduce post-cloning policy drift
- mixed-difficulty curriculum sampling and top-k checkpoint retention
- patience-based early stopping when no better checkpoint appears
- PPO minibatching, observation normalization, and bounded tanh-squashed actions
- fixed-map and randomized-map evaluation, training curves, and a 3D rollout animation

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python train_swarm_ppo.py --episodes 250 --steps-per-episode 150 --rollouts-per-update 6
```

For a longer run with the full curriculum:

```bash
.venv/bin/python train_swarm_ppo.py --episodes 600 --curriculum-episodes 200 --eval-episodes 8 --top-k-checkpoints 5
```

Outputs are written to `results/<timestamp>/`:

- `losses.png`
- `metrics.png`
- `evaluation_summary.json`
- `trained_rollout.gif` when Pillow export is available

## Notes

- The training reward treats objectives as unordered. Any agent can complete any remaining objective, and the reward is now dominated by completion bonuses and a strong time penalty.
- Training now begins with a short behavior-cloning warm start from a simple assignment-based expert, then fine-tunes with PPO.
- PPO fine-tuning is split into a short conservative phase and a longer refinement phase, and phase 2 only proceeds if generalized evaluation improves over the BC baseline.
- The trainer now keeps the top-k checkpoints under `results/<timestamp>/checkpoints/` so the best intermediate policies are not lost when later updates drift.
- Longer runs now stop early when the checkpoint leaderboard stagnates, which is meant to prevent late-stage degradation from overwriting a good training window.
- The script evaluates both a fixed canonical scenario and randomized unseen scenarios each training iteration.
- See [understanding.md](/Users/jackcook/Desktop/Personal_projects/Swarm/understanding.md) for a summary of the changes and the reasoning behind them.
