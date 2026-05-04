# Understanding The Improvements

This file explains the changes made to the original swarm PPO prototype and why they matter.

## Current Objective

The environment no longer uses ordered task completion. The learning problem is now:

- there are multiple active objectives in 3D space
- any agent can complete any objective
- the team should complete all objectives as quickly as possible
- obstacles should be avoided while doing so

## Core Problem In The First Version

The original reward was dominated by local Gaussian value around objectives. That created a bad incentive:

- agents could earn reward by hovering near high-value objectives
- completing an objective could remove future shaping reward
- ordering penalties made exploration brittle

That is why the learned behavior tended to stall or behave conservatively instead of just clearing the map efficiently.

## Reward Function Changes

The reward now focuses on **unordered team coverage**.

- `objective_gaussian_scale`: keeps a small local shaping signal around every active objective
- `progress_reward_scale`: rewards reduction in global remaining-coverage cost
- `local_progress_reward_scale`: rewards each assigned agent for progress toward its current assigned objective
- `completion_bonus_scale`: gives a large one-time reward for completing any active task
- `sequence_completion_bonus`: gives a large terminal reward for finishing all objectives
- `step_penalty`: encourages faster completion
- `idle_penalty`: discourages low-speed, low-progress stalling
- obstacle and inter-agent collision penalties remain active

This makes the intended behavior much clearer: spread or reposition as needed, reduce total remaining work, and clear all objectives quickly.

## Coordination Changes

The shaping reward can now aggregate agents using:

- `closest_agent`
- `max_agents`
- `sum_agents`

The default is `closest_agent`, which is appropriate for unordered coverage because it rewards whichever agent is best placed to cover each remaining objective without paying every agent for clustering.

Each step also computes an explicit agent-to-objective assignment. That assignment is used to create local progress shaping for the shared policy, which improves credit assignment compared with a purely global team reward.

The observation now also includes the agent's current assigned objective as an explicit feature. Without that, the reward was shaping behavior toward an internal assignment that the policy had to infer indirectly, which made the shared-policy problem unnecessarily hard.

## PPO / Optimization Changes

The trainer was upgraded in several ways:

- training begins with behavior cloning from a simple expert that greedily accelerates each agent toward its current assignment
- PPO fine-tuning is split into a short conservative phase followed by a longer refinement phase
- phase 2 can be gated on generalized improvement over the BC baseline
- a KL-style retention term can keep PPO from drifting too far from the cloned policy
- multiple rollouts are now collected per PPO update
- PPO now trains with minibatches rather than one monolithic batch
- time-limit truncations are now bootstrapped with the critic instead of being forced to zero
- actor actions use a tanh-squashed Gaussian distribution, which is a better match for bounded accelerations
- observation and critic-state normalization were added
- reward scaling was added to reduce optimization instability

These changes reduce gradient variance and make the learning problem less brittle.

The imitation warm start is especially important here because the environment is easy for a simple controller but expensive for PPO to rediscover from scratch. PPO is now used to refine and generalize a competent baseline instead of learning basic coverage behavior from nothing.

## Curriculum And Generalization

Training now uses a curriculum:

- early episodes sample agents and objectives from a smaller spatial region
- obstacle layouts are randomized during training
- difficulty can either ramp monotonically or be sampled from a mixed-difficulty band under the current curriculum ceiling

Evaluation is explicit:

- **fixed evaluation** uses the operator-defined objectives and obstacles
- **generalization evaluation** uses randomized objectives and randomized obstacle layouts

This helps check whether the learned policy is memorizing one map or actually learning transferable coverage behavior.

## Diagnostics Improvements

The saved plots now include more useful signals:

- optimization curves
- train reward, fixed-eval reward, and generalization reward
- fixed and generalized success/completion
- remaining team coverage cost
- completed objective count
- first-completion step
- completion AUC, which measures how quickly objectives are accumulated during an episode
- obstacle-event diagnostics
- curriculum progression
- episode length

The checkpoint selection logic was also improved. The saved “best” model is no longer chosen only by success rate. It now prefers:

1. higher success rate
2. higher generalized success rate
3. higher completion fraction
4. higher generalized completion fraction
5. faster completion AUC
6. faster generalized completion AUC
7. lower remaining coverage cost
8. lower generalized remaining coverage cost

The trainer also keeps a top-k leaderboard of checkpoint files instead of only a single best model. That matters because PPO can discover a strong policy and then partially degrade it later, especially once the curriculum gets harder.

## Why These Changes Should Help

The new setup addresses the original failure mode directly:

- hovering is less attractive
- progress is explicitly rewarded
- completing any remaining task is decisively better than lingering
- full-map completion has a clear terminal incentive
- generalization is trained and measured, not just assumed

## Recommended Next Steps

If performance is still weak after a longer run, the next upgrades worth trying are:

- adaptive early stopping or checkpoint rollback when generalized metrics degrade
- stronger regularization against post-BC drift
- longer training with more seeds rather than relying on a single run
- recurrent or attention-based policies for richer coordination
- randomization of dynamics noise, sensor noise, and control delay for stronger transfer
