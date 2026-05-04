# Swarm PPO: Architecture, Rewards, Policies, and Training Design

This document explains what the current system is doing at a technical level: the environment, the policy architecture, the critic, the reward function, the role of expert behavior cloning, and how PPO is used to refine the controller.

It is written to describe the current implementation in `train_swarm_ppo.py`, not a hypothetical future system.

## High-Level Objective

The problem is a cooperative multi-agent control task in 3D:

- a team of agents moves through continuous 3D space
- there are multiple objectives scattered through the environment
- any agent can complete any remaining objective
- obstacles should be avoided
- the team should complete all objectives as quickly as possible

The important design choice is that this is **unordered cooperative coverage**, not ordered task execution. The system is not trying to learn "objective 1, then objective 2, then objective 3." It is trying to learn "clear the whole map efficiently."

## System Overview

The overall training stack has four main pieces:

1. `Swarm3DEnv`
   This defines the dynamics, observations, rewards, termination rules, obstacle handling, and completion logic.

2. `Actor`
   This is the shared policy used by every agent. It maps each agent's observation to a continuous acceleration action in 3D.

3. `Critic`
   This is a centralized value function. It sees the full global state and estimates the team value for PPO.

4. PPO + behavior cloning warm start
   Training begins with a supervised imitation phase from a simple expert controller, then transitions into PPO fine-tuning.

This is a standard **centralized-training / decentralized-execution** pattern:

- during execution, each agent uses only its own observation
- during training, the critic uses a centralized state for lower-variance value estimation

## Environment Design

### Continuous Dynamics

Each agent exists in a bounded 3D world and has:

- position `(x, y, z)`
- velocity `(vx, vy, vz)`

The action is a 3D acceleration command. The environment clips:

- acceleration norm to `max_accel`
- velocity norm to `max_speed`

The state update is simple Newtonian integration:

- `v_{t+1} = clip(v_t + a_t * dt)`
- `p_{t+1} = p_t + v_{t+1} * dt`

If an agent crosses the world bounds, its position is clipped back into the domain and its velocity is zeroed.

### Objectives

Each objective has:

- a 3D position
- an amplitude
- a Gaussian width parameter `sigma`
- a binary completed flag

An objective is completed when **any** agent comes within `objective_radius`.

That completion is one-time. After that, the objective is removed from the active task set.

### Obstacles

Obstacles are spheres with:

- a 3D center
- a radius
- a penalty
- an `active` flag

Training uses randomized obstacles; fixed evaluation uses a fixed obstacle set. This separation is intentional:

- fixed evaluation measures progress on a canonical benchmark map
- randomized evaluation measures generalization

## Observation Model

The actor is a **shared policy**, meaning every agent uses the same neural network weights. For that to work, each agent observation must encode enough local context for the policy to specialize behavior by situation rather than by hard-coded agent identity.

Each per-agent observation contains:

- the agent's own normalized position
- the agent's own normalized velocity
- a one-hot agent identity vector
- relative positions of all other agents
- per-objective features for every objective:
  - relative objective position
  - objective completed flag
  - objective amplitude
  - objective sigma
- the global completed-objective mask
- explicit assignment features for this agent:
  - one-hot assigned-objective identity
  - relative vector to the assigned objective
  - assigned-objective distance magnitude
- obstacle features:
  - relative obstacle center
  - obstacle radius
  - obstacle active flag

### Why the Assignment Features Matter

This was one of the most important fixes.

The reward includes a local progress term based on an internal agent-to-objective assignment. Before the fix, the policy was being rewarded for moving toward a target that it did not directly observe as "its assigned target." That creates an avoidable credit-assignment gap.

By explicitly exposing the current assignment in the observation, the policy can align its action with the reward signal much more directly.

## Global State for the Critic

The critic uses a centralized state containing:

- all agent positions
- all agent velocities
- all objective positions
- all objective amplitudes
- all objective sigmas
- all objective completion flags
- all obstacle features

This state is more informative than any single local observation, which is exactly what we want for centralized training. The critic is not required to obey decentralized information constraints.

## Current Policy Architecture

### Actor

The actor is a feedforward MLP:

- linear
- `tanh`
- linear
- `tanh`
- linear

The output head predicts the **mean** of a Gaussian policy in 3 action dimensions.

There is also a learned global `log_std` parameter for the Gaussian. The actor therefore defines a diagonal Gaussian distribution in action space.

### Bounded Action Parameterization

The raw Gaussian sample is not used directly. Instead:

- sample or take the mean in unconstrained space
- apply `tanh`
- scale by `action_limit`

This gives a smooth bounded action in `[-max_accel, max_accel]` per dimension while preserving stochastic exploration.

This is better than naïvely clipping raw Gaussian actions because:

- the policy distribution is consistent with the action bounds
- gradients behave more smoothly
- exploration is naturally shaped around the valid control domain

The PPO log-probability includes the correct `tanh` squash correction term.

### Critic

The critic is also a feedforward MLP:

- linear
- `tanh`
- linear
- `tanh`
- linear to scalar value

It estimates the expected return from the centralized team state.

## Why a Shared Policy

The system currently uses one actor shared across all agents rather than separate policies per agent.

That design has several advantages:

- parameter efficiency
- faster learning from pooled experience
- natural permutation-style reuse across agents
- better scaling as agent count grows

The cost is that the policy must infer role specialization from context. That is why the observation contains:

- agent identity
- relative teammate positions
- assigned-objective features

Those features let a shared policy produce different behavior for different agents without giving each agent separate weights.

## Reward Design

The reward is deliberately **completion-dominant**, with shaping terms that help optimization but do not redefine the task.

### 1. Gaussian Objective Shaping

There is a small local shaping term around active objectives:

- for each active objective, compute Gaussian response from agent distances
- aggregate that response according to `reward_aggregation`
- multiply by `objective_gaussian_scale`

Supported aggregations:

- `closest_agent`
- `max_agents`
- `sum_agents`

The default is `closest_agent`.

#### Why this exists

This term gives dense reward near objectives, which helps exploration.

#### Why it is intentionally small

If this term dominates, agents can learn to hover near rewarding regions instead of finishing the task. That was a core failure mode in earlier versions.

### 2. Global Coverage Progress Reward

The environment computes a team-level coverage cost:

- for each remaining objective, find the nearest agent
- sum those nearest-agent distances across objectives

Reward uses the reduction in that cost:

- `progress = previous_cost - new_cost`
- reward contribution is `progress_reward_scale * progress`

This shaping term rewards actions that reduce total remaining team work.

#### Why this matters

It transforms the sparse objective-completion problem into a denser optimization signal tied to the real cooperative objective: minimizing remaining coverage burden.

### 3. Local Assigned-Progress Reward

The environment computes an explicit agent-to-objective assignment over remaining objectives. It then tracks whether each assigned agent got closer to its current assigned objective.

Reward contribution:

- `local_progress_reward_scale * assigned_distance_reduction`

This is a local credit-assignment aid layered on top of the global progress objective.

#### Why this matters

Purely global rewards are often too ambiguous in multi-agent control. This term helps the shared policy learn coordinated division of labor.

### 4. Completion Bonus

When any agent completes an objective, the team receives a one-time reward:

- `completion_bonus_scale * objective_amplitude`

This makes actual objective completion decisively more valuable than just moving near objectives.

### 5. Sequence Completion Bonus

When all objectives are complete, the team gets an additional terminal bonus:

- `sequence_completion_bonus`

This strongly reinforces full-map completion.

### 6. Step Penalty

Every step incurs a fixed negative reward:

- `-step_penalty`

This creates urgency and discourages slow dithering.

### 7. Idle Penalty

If the team makes too little progress while moving too slowly, it gets an additional penalty:

- progress below `idle_progress_threshold`
- average speed below `idle_speed_threshold`

This specifically attacks stagnation and hovering.

### 8. Obstacle and Collision Penalties

The system penalizes:

- entering obstacles
- pairwise inter-agent proximity collisions

Obstacle penalties can be one-time per agent-obstacle pair or repeated, though the current default is one-time.

## Assignment Logic

The current environment computes a hard assignment between agents and remaining objectives using a minimum-distance matching procedure.

Conceptually:

- build pairwise agent-objective distances
- search over feasible assignments
- choose the mapping with minimum total assignment cost

This assignment is used for:

- local progress shaping
- explicit observation features
- diagnostic cost calculation
- expert-controller generation

It is not a separate learned planner. It is an environment-side coordination scaffold used to make the learning problem easier and more structured.

## Expert Policy and Behavior Cloning

### What the Expert Does

The expert is simple and hand-designed:

- for each assigned agent, point toward its assigned objective
- choose a desired velocity in that direction
- convert the gap between desired velocity and current velocity into an acceleration command

This is not globally optimal, but it is competent.

### Why Behavior Cloning Is Used

This was the second major improvement.

The environment is solvable by a simple controller, but PPO from scratch had difficulty discovering that behavior robustly. In practice, that meant the optimization budget was being spent on basic navigation and task discovery rather than refinement.

Behavior cloning solves that by first teaching the actor a reasonable control prior from expert demonstrations.

The current training flow is:

1. collect expert rollouts
2. normalize observations using those rollouts
3. train the actor with supervised regression to expert actions
4. evaluate the BC-initialized policy
5. run a short conservative PPO refinement phase
6. evaluate generalized improvement after that short phase
7. only continue into the longer PPO phase if the short phase actually improves generalized performance enough

This changes the role of RL:

- before: PPO had to invent competence from scratch
- now: PPO starts from a competent baseline and tries to improve or generalize it

### What the Cloning Loss Is

The current cloning objective is simple mean-squared error between:

- actor squashed mean action
- expert action

This is enough because the expert is deterministic and the action space is continuous.

## PPO Training Loop

After behavior cloning, PPO takes over.

### Data Collection

Each update collects multiple rollouts:

- reset the environment with curriculum-controlled difficulty
- run the current shared policy
- store normalized observations, centralized states, actions, log-probs, values, rewards, and done signals

### Reward Scaling

The raw environment reward is scaled before GAE and PPO updates.

This reduces optimization instability without changing the behavior ordering induced by the reward.

### GAE

Generalized Advantage Estimation is used to compute:

- advantages
- returns

The implementation bootstraps time-limit truncations through the critic rather than treating them as hard zero-value terminals unless the episode ended in true terminal success.

That is the correct choice for finite-horizon truncation.

### PPO Objective

PPO uses:

- clipped policy-ratio surrogate loss
- value-function MSE loss
- entropy regularization
- optional KL-style retention regularization against the cloned reference actor
- minibatch SGD over multiple epochs

This is standard PPO, but applied to pooled per-agent samples from a shared policy.

### Centralized Value, Decentralized Policy

When preparing batches:

- actor data is flattened over time and agents
- critic states are repeated per agent so each policy sample has an aligned value target

This means the actor learns from decentralized per-agent observations, while value estimation remains centralized.

## Curriculum Design

Training uses a difficulty curriculum:

- early episodes spawn agents and objectives in a smaller region
- later episodes expand the spatial spread
- obstacle complexity also ramps with difficulty
- the trainer can either use a pure difficulty ramp or sample from a mixed-difficulty band below the current curriculum ceiling

This is meant to make the optimization landscape smoother:

- first learn short-range coordination
- then extend to wider coverage and more obstacle variety

The curriculum is useful, but it also creates a risk: later harder phases can destabilize a policy that looked strong at easier difficulties. That is part of what we are now monitoring in evaluation.

## Evaluation Design

The system evaluates two distinct regimes:

### Fixed Evaluation

This uses a canonical fixed map:

- fixed objectives
- fixed amplitudes
- fixed obstacles

This is the easiest way to measure whether the policy learned the intended task at all.

### Generalized Evaluation

This uses randomized unseen maps:

- randomized objectives
- randomized obstacles

This is the real test of transfer.

### Why Both Matter

A policy that only performs well on the fixed map may be exploiting benchmark regularities rather than learning robust cooperative coverage.

A policy that has modest but real generalized performance is often more valuable than a policy with slightly higher fixed-map scores but no transfer.

## Metrics That Matter

The code tracks several diagnostics. The most important ones are:

- `success_rate`
  Fraction of evaluation episodes that complete all objectives.

- `completed_fraction`
  Average fraction of objectives completed.

- `remaining_assignment_cost`
  Proxy for how much coverage work remains at the end.

- `first_completion_step`
  How quickly the team starts making real progress.

- `completion_auc`
  How quickly objectives accumulate over the episode, not just whether they eventually complete.

### Best Checkpoint Selection

The current "best model" selection prefers:

1. fixed success rate
2. generalized success rate
3. fixed completion fraction
4. generalized completion fraction
5. fixed completion AUC
6. generalized completion AUC
7. lower fixed remaining cost
8. lower generalized remaining cost

This is important because a naïve checkpoint rule can preserve the wrong model, especially when PPO temporarily improves and later degrades.

The trainer also keeps the top-k checkpoints on disk, rather than only one best snapshot. That matters because the most transferable policy may appear mid-run and should still be available even if later training drifts.

## What the Current Model Is Good At

With the latest changes, the system can now:

- learn nontrivial cooperative coverage behavior
- solve the fixed map at strong intermediate checkpoints
- show meaningful transfer to randomized environments

That is a major improvement over the earlier regime where PPO often failed to discover even basic completion behavior.

## What the Current Model Is Still Weak At

The main weakness is **stability after competence**.

The policy can become good, but later PPO training can partially unlearn that competence. So the current problem is no longer "can it learn anything?" but "can it preserve and generalize what it learns?"

This usually points to one or more of:

- PPO updates that are too aggressive after warm start
- curriculum pressure that becomes destabilizing at later difficulty
- insufficient regularization against drifting away from the expert prior
- a gap between fixed-map optimization and generalization

## Conceptual Summary

At a sophisticated level, the current system is a hybrid of three ideas:

### 1. Structured coordination through environment-side assignment

The environment injects useful inductive structure by computing agent-objective assignments and using them for shaping and observations.

### 2. Centralized training with decentralized execution

The actor remains local and shared, while the critic uses global information for more stable policy optimization.

### 3. Imitation-to-RL refinement

Instead of forcing RL to discover basic competence from scratch, imitation provides an initial manifold of sensible behavior and PPO refines it.

This is a pragmatic design. It is not "pure RL" in the minimalist sense, but it is much closer to how difficult continuous-control multi-agent systems are made to work in practice.

## Practical Interpretation

If you want a concise mental model of the whole stack, it is this:

- a hand-designed assignment routine tells the system which objectives each agent should currently care about
- a shared actor sees its local world plus that assignment context and outputs bounded 3D accelerations
- a centralized critic estimates team value from the full scene
- the reward strongly favors actual completion and speed, while shaping terms help navigation and coordination
- behavior cloning teaches a competent baseline policy
- PPO then tries to improve and generalize that baseline

That is the current design philosophy of the project.
