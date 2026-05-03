from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mpl-cache").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from torch.distributions import Normal


@dataclass
class Obstacle:
    center: Tuple[float, float, float]
    radius: float
    penalty: float


@dataclass
class EnvConfig:
    num_agents: int = 4
    num_objectives: int = 4
    world_size: float = 12.0
    dt: float = 0.2
    max_speed: float = 2.0
    max_accel: float = 1.0
    objective_radius: float = 0.7
    objective_sigmas: Tuple[float, ...] = (1.8, 1.4, 1.6, 1.2)
    out_of_order_penalties: Tuple[float, ...] = (6.0, 8.0, 10.0, 12.0)
    completion_bonus_scale: float = 0.0
    obstacle_penalty_once: bool = False
    step_penalty: float = 0.08
    collision_radius: float = 0.35
    collision_penalty: float = 0.5
    max_steps: int = 150
    reward_aggregation: str = "sum_agents"
    fixed_objectives: Tuple[Tuple[float, float, float], ...] = (
        (-7.0, -4.0, -2.0),
        (-1.5, 3.5, 2.0),
        (3.0, -2.5, 4.5),
        (7.5, 5.0, -1.0),
    )
    fixed_amplitudes: Tuple[float, ...] = (18.0, 22.0, 16.0, 28.0)
    obstacles: Tuple[Obstacle, ...] = field(
        default_factory=lambda: (
            Obstacle(center=(-2.0, -1.0, 0.0), radius=1.6, penalty=10.0),
            Obstacle(center=(2.5, 2.0, 1.5), radius=1.8, penalty=12.0),
            Obstacle(center=(5.0, -4.0, 2.0), radius=1.4, penalty=8.0),
        )
    )


class Swarm3DEnv:
    def __init__(self, config: EnvConfig):
        self.cfg = config
        if len(self.cfg.objective_sigmas) != self.cfg.num_objectives:
            raise ValueError("objective_sigmas must match num_objectives")
        if len(self.cfg.out_of_order_penalties) != self.cfg.num_objectives:
            raise ValueError("out_of_order_penalties must match num_objectives")
        if len(self.cfg.fixed_objectives) != self.cfg.num_objectives:
            raise ValueError("fixed_objectives must match num_objectives")
        if len(self.cfg.fixed_amplitudes) != self.cfg.num_objectives:
            raise ValueError("fixed_amplitudes must match num_objectives")
        self.rng = np.random.default_rng()
        self.reset(training=True)

    @property
    def obs_dim(self) -> int:
        return (
            6
            + (self.cfg.num_agents - 1) * 3
            + self.cfg.num_objectives * 5
            + self.cfg.num_objectives
            + len(self.cfg.obstacles) * 4
        )

    @property
    def state_dim(self) -> int:
        return (
            self.cfg.num_agents * 6
            + self.cfg.num_objectives * 4
            + self.cfg.num_objectives
            + len(self.cfg.obstacles) * 4
        )

    def _sample_positions(self, count: int) -> np.ndarray:
        low = -0.8 * self.cfg.world_size
        high = 0.8 * self.cfg.world_size
        return self.rng.uniform(low=low, high=high, size=(count, 3))

    def reset(self, training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        self.training = training
        self.step_idx = 0
        self.agent_positions = self._sample_positions(self.cfg.num_agents)
        self.agent_velocities = np.zeros((self.cfg.num_agents, 3), dtype=np.float32)
        self.completed_objectives = np.zeros(self.cfg.num_objectives, dtype=bool)
        self.next_objective_idx = 0
        self.out_of_order_hits = 0
        self.total_hits = 0
        self.obstacle_hits = np.zeros((self.cfg.num_agents, len(self.cfg.obstacles)), dtype=bool)

        if training:
            self.objectives = self._sample_positions(self.cfg.num_objectives)
            self.amplitudes = self.rng.uniform(1.0, 30.0, size=self.cfg.num_objectives)
        else:
            self.objectives = np.array(self.cfg.fixed_objectives, dtype=np.float32)
            self.amplitudes = np.array(self.cfg.fixed_amplitudes, dtype=np.float32)

        self.sigmas = np.array(self.cfg.objective_sigmas, dtype=np.float32)
        return self._get_obs(), self._get_state()

    def _get_obs(self) -> np.ndarray:
        obs = []
        next_one_hot = np.zeros(self.cfg.num_objectives, dtype=np.float32)
        if self.next_objective_idx < self.cfg.num_objectives:
            next_one_hot[self.next_objective_idx] = 1.0

        for agent_idx in range(self.cfg.num_agents):
            own_pos = self.agent_positions[agent_idx] / self.cfg.world_size
            own_vel = self.agent_velocities[agent_idx] / max(self.cfg.max_speed, 1e-6)

            rel_agents = []
            for other_idx in range(self.cfg.num_agents):
                if other_idx == agent_idx:
                    continue
                rel = (self.agent_positions[other_idx] - self.agent_positions[agent_idx]) / self.cfg.world_size
                rel_agents.append(rel)

            obj_features = []
            for obj_idx in range(self.cfg.num_objectives):
                rel_obj = (self.objectives[obj_idx] - self.agent_positions[agent_idx]) / self.cfg.world_size
                obj_features.extend(
                    [
                        rel_obj[0],
                        rel_obj[1],
                        rel_obj[2],
                        float(self.completed_objectives[obj_idx]),
                        self.amplitudes[obj_idx] / 30.0,
                    ]
                )

            obstacle_features = []
            for obstacle in self.cfg.obstacles:
                center = np.array(obstacle.center, dtype=np.float32)
                rel_center = (center - self.agent_positions[agent_idx]) / self.cfg.world_size
                obstacle_features.extend(
                    [
                        rel_center[0],
                        rel_center[1],
                        rel_center[2],
                        obstacle.radius / self.cfg.world_size,
                    ]
                )

            agent_obs = np.concatenate(
                [
                    own_pos,
                    own_vel,
                    np.concatenate(rel_agents).astype(np.float32) if rel_agents else np.array([], dtype=np.float32),
                    np.array(obj_features, dtype=np.float32),
                    next_one_hot,
                    np.array(obstacle_features, dtype=np.float32),
                ]
            )
            obs.append(agent_obs.astype(np.float32))
        return np.stack(obs, axis=0)

    def _get_state(self) -> np.ndarray:
        next_one_hot = np.zeros(self.cfg.num_objectives, dtype=np.float32)
        if self.next_objective_idx < self.cfg.num_objectives:
            next_one_hot[self.next_objective_idx] = 1.0

        obstacle_features = []
        for obstacle in self.cfg.obstacles:
            obstacle_features.extend(
                [
                    obstacle.center[0] / self.cfg.world_size,
                    obstacle.center[1] / self.cfg.world_size,
                    obstacle.center[2] / self.cfg.world_size,
                    obstacle.radius / self.cfg.world_size,
                ]
            )

        state = np.concatenate(
            [
                self.agent_positions.reshape(-1) / self.cfg.world_size,
                self.agent_velocities.reshape(-1) / max(self.cfg.max_speed, 1e-6),
                np.column_stack(
                    [
                        self.objectives / self.cfg.world_size,
                        self.amplitudes.reshape(-1, 1) / 30.0,
                    ]
                ).reshape(-1),
                next_one_hot,
                np.array(obstacle_features, dtype=np.float32),
            ]
        )
        return state.astype(np.float32)

    @staticmethod
    def _clip_norm(vectors: np.ndarray, max_norm: float) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        scales = np.where(norms > max_norm, max_norm / np.maximum(norms, 1e-8), 1.0)
        return vectors * scales

    def _gaussian_rewards(self) -> float:
        total_reward = 0.0
        for obj_idx in range(self.cfg.num_objectives):
            if self.completed_objectives[obj_idx]:
                continue
            distances = np.linalg.norm(self.agent_positions - self.objectives[obj_idx], axis=1)
            responses = self.amplitudes[obj_idx] * np.exp(-0.5 * (distances / self.sigmas[obj_idx]) ** 2)
            if self.cfg.reward_aggregation == "max_agents":
                total_reward += float(np.max(responses))
            else:
                total_reward += float(np.sum(responses))
        return total_reward

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, bool, Dict]:
        self.step_idx += 1
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.cfg.num_agents, 3):
            raise ValueError(f"Expected actions {(self.cfg.num_agents, 3)}, got {actions.shape}")

        accelerations = self._clip_norm(actions, self.cfg.max_accel)
        self.agent_velocities = self._clip_norm(
            self.agent_velocities + accelerations * self.cfg.dt,
            self.cfg.max_speed,
        )
        self.agent_positions = self.agent_positions + self.agent_velocities * self.cfg.dt

        lower_bound = -self.cfg.world_size
        upper_bound = self.cfg.world_size
        hit_bounds = (self.agent_positions < lower_bound) | (self.agent_positions > upper_bound)
        if np.any(hit_bounds):
            self.agent_positions = np.clip(self.agent_positions, lower_bound, upper_bound)
            self.agent_velocities[hit_bounds.any(axis=1)] *= 0.0

        reward = self._gaussian_rewards()
        reward -= self.cfg.step_penalty

        obstacle_events = 0
        for obs_idx, obstacle in enumerate(self.cfg.obstacles):
            center = np.array(obstacle.center, dtype=np.float32)
            distances = np.linalg.norm(self.agent_positions - center, axis=1)
            inside = distances <= obstacle.radius
            if self.cfg.obstacle_penalty_once:
                new_hits = inside & ~self.obstacle_hits[:, obs_idx]
                obstacle_events += int(np.sum(new_hits))
                reward -= obstacle.penalty * float(np.sum(new_hits))
                self.obstacle_hits[:, obs_idx] |= inside
            else:
                obstacle_events += int(np.sum(inside))
                reward -= obstacle.penalty * float(np.sum(inside))

        pairwise_penalty = 0.0
        for i in range(self.cfg.num_agents):
            for j in range(i + 1, self.cfg.num_agents):
                if np.linalg.norm(self.agent_positions[i] - self.agent_positions[j]) < self.cfg.collision_radius:
                    pairwise_penalty += self.cfg.collision_penalty
        reward -= pairwise_penalty

        completed_this_step: List[int] = []
        out_of_order_this_step: List[int] = []
        for obj_idx in range(self.cfg.num_objectives):
            if self.completed_objectives[obj_idx]:
                continue

            any_close = np.any(
                np.linalg.norm(self.agent_positions - self.objectives[obj_idx], axis=1) <= self.cfg.objective_radius
            )
            if not any_close:
                continue

            self.total_hits += 1
            if obj_idx == self.next_objective_idx:
                self.completed_objectives[obj_idx] = True
                self.next_objective_idx += 1
                completed_this_step.append(obj_idx)
                reward += self.cfg.completion_bonus_scale * self.amplitudes[obj_idx]
            else:
                self.out_of_order_hits += 1
                out_of_order_this_step.append(obj_idx)
                reward -= self.cfg.out_of_order_penalties[obj_idx]

        done = bool(np.all(self.completed_objectives) or self.step_idx >= self.cfg.max_steps)
        info = {
            "completed_count": int(np.sum(self.completed_objectives)),
            "completed_fraction": float(np.mean(self.completed_objectives)),
            "success": bool(np.all(self.completed_objectives)),
            "completed_this_step": completed_this_step,
            "out_of_order_this_step": out_of_order_this_step,
            "order_accuracy": 1.0
            if self.total_hits == 0
            else 1.0 - (self.out_of_order_hits / float(self.total_hits)),
            "obstacle_events": obstacle_events,
            "pairwise_penalty": pairwise_penalty,
        }
        return self._get_obs(), self._get_state(), float(reward), done, info


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.net(obs)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    train_epochs: int = 10
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


class SharedPPO:
    def __init__(self, obs_dim: int, state_dim: int, action_dim: int, cfg: PPOConfig, device: str):
        self.cfg = cfg
        self.device = torch.device(device)
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mean, log_std = self.actor(obs_tensor)
            dist = Normal(mean, log_std.exp())
            action = mean if deterministic else dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action.cpu().numpy(), log_prob.cpu().numpy()

    def value(self, state: np.ndarray) -> float:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            value = self.critic(state_tensor)[0]
        return float(value.item())

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=self.device)
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        actor_loss_value = 0.0
        critic_loss_value = 0.0
        entropy_value = 0.0

        for _ in range(self.cfg.train_epochs):
            mean, log_std = self.actor(obs)
            dist = Normal(mean, log_std.exp())
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            ratios = torch.exp(new_log_probs - old_log_probs)
            clipped_ratios = torch.clamp(ratios, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio)
            actor_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

            values = self.critic(states)
            critic_loss = ((returns - values) ** 2).mean()
            loss = actor_loss + self.cfg.value_coef * critic_loss - self.cfg.entropy_coef * entropy

            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
            self.actor_optim.step()
            self.critic_optim.step()

            actor_loss_value = float(actor_loss.item())
            critic_loss_value = float(critic_loss.item())
            entropy_value = float(entropy.item())

        return {
            "actor_loss": actor_loss_value,
            "critic_loss": critic_loss_value,
            "entropy": entropy_value,
        }


def compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        next_val = next_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def collect_episode(env: Swarm3DEnv, agent: SharedPPO, deterministic: bool = False) -> Dict[str, np.ndarray]:
    obs, state = env.reset(training=not deterministic)
    done = False
    trajectory = {
        "obs": [],
        "actions": [],
        "log_probs": [],
        "states": [],
        "rewards": [],
        "dones": [],
    }
    final_info = {}
    rollout_positions = [env.agent_positions.copy()]

    while not done:
        actions, log_probs = agent.select_action(obs, deterministic=deterministic)
        value = agent.value(state)
        next_obs, next_state, reward, done, info = env.step(actions)

        trajectory["obs"].append(obs.copy())
        trajectory["actions"].append(actions.copy())
        trajectory["log_probs"].append(log_probs.copy())
        trajectory["states"].append(np.concatenate([state, np.array([value], dtype=np.float32)]))
        trajectory["rewards"].append(reward)
        trajectory["dones"].append(float(done))

        obs = next_obs
        state = next_state
        final_info = info
        rollout_positions.append(env.agent_positions.copy())

    trajectory["rollout_positions"] = np.stack(rollout_positions, axis=0)
    trajectory["final_info"] = final_info
    return trajectory


def prepare_batch(trajectory: Dict[str, np.ndarray], ppo_cfg: PPOConfig, next_value: float = 0.0) -> Dict[str, np.ndarray]:
    obs = np.asarray(trajectory["obs"], dtype=np.float32)
    actions = np.asarray(trajectory["actions"], dtype=np.float32)
    log_probs = np.asarray(trajectory["log_probs"], dtype=np.float32)
    states_and_values = np.asarray(trajectory["states"], dtype=np.float32)
    states = states_and_values[:, :-1]
    values = states_and_values[:, -1]
    rewards = np.asarray(trajectory["rewards"], dtype=np.float32)
    dones = np.asarray(trajectory["dones"], dtype=np.float32)

    advantages, returns = compute_gae(
        rewards,
        dones,
        values,
        next_value=next_value,
        gamma=ppo_cfg.gamma,
        gae_lambda=ppo_cfg.gae_lambda,
    )

    time_steps, num_agents, obs_dim = obs.shape
    _, _, action_dim = actions.shape
    flat_states = np.repeat(states, repeats=num_agents, axis=0)
    flat_advantages = np.repeat(advantages, repeats=num_agents, axis=0)
    flat_returns = np.repeat(returns, repeats=num_agents, axis=0)

    batch = {
        "obs": obs.reshape(time_steps * num_agents, obs_dim),
        "actions": actions.reshape(time_steps * num_agents, action_dim),
        "log_probs": log_probs.reshape(time_steps * num_agents),
        "states": flat_states,
        "advantages": flat_advantages,
        "returns": flat_returns,
    }
    return batch


def evaluate(agent: SharedPPO, env_cfg: EnvConfig, episodes: int) -> Dict[str, float]:
    eval_env = Swarm3DEnv(env_cfg)
    rewards = []
    success = []
    completed_fraction = []
    order_accuracy = []

    for _ in range(episodes):
        trajectory = collect_episode(eval_env, agent, deterministic=True)
        rewards.append(float(np.sum(trajectory["rewards"])))
        info = trajectory["final_info"]
        success.append(float(info["success"]))
        completed_fraction.append(float(info["completed_fraction"]))
        order_accuracy.append(float(info["order_accuracy"]))

    return {
        "eval_reward_mean": float(np.mean(rewards)),
        "eval_reward_std": float(np.std(rewards)),
        "success_rate": float(np.mean(success)),
        "completed_fraction": float(np.mean(completed_fraction)),
        "order_accuracy": float(np.mean(order_accuracy)),
    }


def plot_training_curves(history: Dict[str, List[float]], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["actor_loss"], label="Actor Loss")
    axes[0].plot(history["critic_loss"], label="Critic Loss")
    axes[0].plot(history["entropy"], label="Entropy")
    axes[0].set_title("Training Losses")
    axes[0].set_xlabel("Episode")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_reward"], label="Train Reward")
    axes[1].plot(history["eval_reward"], label="Eval Reward")
    axes[1].set_title("Reward")
    axes[1].set_xlabel("Episode")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "losses.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(history["eval_success"], label="Success Rate")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Evaluation Success")
    axes[0].set_xlabel("Episode")
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["eval_completion"], label="Completion Fraction", color="tab:green")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Objective Completion")
    axes[1].set_xlabel("Episode")
    axes[1].grid(alpha=0.3)

    axes[2].plot(history["eval_order_accuracy"], label="Order Accuracy", color="tab:red")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_title("Ordinal Accuracy")
    axes[2].set_xlabel("Episode")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "metrics.png", dpi=180)
    plt.close(fig)


def create_animation(
    rollout_positions: np.ndarray,
    env_cfg: EnvConfig,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    lim = env_cfg.world_size

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Trained Swarm Rollout")

    objectives = np.array(env_cfg.fixed_objectives)
    ax.scatter(objectives[:, 0], objectives[:, 1], objectives[:, 2], c="gold", s=80, label="Objectives")

    for obstacle in env_cfg.obstacles:
        center = np.array(obstacle.center)
        u = np.linspace(0, 2 * np.pi, 16)
        v = np.linspace(0, np.pi, 12)
        x = obstacle.radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = obstacle.radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = obstacle.radius * np.outer(np.ones_like(u), np.cos(v)) + center[2]
        ax.plot_wireframe(x, y, z, color="gray", alpha=0.2, linewidth=0.5)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    scatters = [
        ax.plot([], [], [], "o", color=colors[i % len(colors)], markersize=7, label=f"Agent {i}")[0]
        for i in range(env_cfg.num_agents)
    ]
    trails = [ax.plot([], [], [], color=colors[i % len(colors)], alpha=0.5, linewidth=1.5)[0] for i in range(env_cfg.num_agents)]
    ax.legend(loc="upper left")

    def update(frame_idx: int):
        frame = rollout_positions[frame_idx]
        for agent_idx in range(env_cfg.num_agents):
            scatters[agent_idx].set_data([frame[agent_idx, 0]], [frame[agent_idx, 1]])
            scatters[agent_idx].set_3d_properties([frame[agent_idx, 2]])

            trail = rollout_positions[: frame_idx + 1, agent_idx]
            trails[agent_idx].set_data(trail[:, 0], trail[:, 1])
            trails[agent_idx].set_3d_properties(trail[:, 2])
        return scatters + trails

    ani = animation.FuncAnimation(fig, update, frames=len(rollout_positions), interval=80, blit=False)
    try:
        ani.save(out_path, writer="pillow", fps=12)
    finally:
        plt.close(fig)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_env_config(args: argparse.Namespace) -> EnvConfig:
    sigma_cycle = [1.8, 1.4, 1.6, 1.2, 1.5, 1.7]
    penalty_cycle = [6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
    amplitude_cycle = [18.0, 22.0, 16.0, 28.0, 20.0, 24.0]
    fixed_objective_cycle = [
        (-7.0, -4.0, -2.0),
        (-1.5, 3.5, 2.0),
        (3.0, -2.5, 4.5),
        (7.5, 5.0, -1.0),
        (0.0, -7.0, 3.0),
        (6.5, 0.0, -4.0),
    ]
    objective_sigmas = tuple(sigma_cycle[i % len(sigma_cycle)] for i in range(args.num_objectives))
    out_of_order_penalties = tuple(penalty_cycle[i % len(penalty_cycle)] for i in range(args.num_objectives))
    fixed_amplitudes = tuple(amplitude_cycle[i % len(amplitude_cycle)] for i in range(args.num_objectives))
    fixed_objectives = tuple(fixed_objective_cycle[i % len(fixed_objective_cycle)] for i in range(args.num_objectives))

    return EnvConfig(
        num_agents=args.num_agents,
        num_objectives=args.num_objectives,
        max_steps=args.steps_per_episode,
        max_speed=args.max_speed,
        max_accel=args.max_accel,
        objective_radius=args.objective_radius,
        objective_sigmas=objective_sigmas,
        out_of_order_penalties=out_of_order_penalties,
        fixed_objectives=fixed_objectives,
        fixed_amplitudes=fixed_amplitudes,
        step_penalty=args.step_penalty,
        reward_aggregation=args.reward_aggregation,
        completion_bonus_scale=args.completion_bonus_scale,
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    env_cfg = build_env_config(args)
    train_env = Swarm3DEnv(env_cfg)

    ppo_cfg = PPOConfig(
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        train_epochs=args.ppo_epochs,
        entropy_coef=args.entropy_coef,
    )
    agent = SharedPPO(train_env.obs_dim, train_env.state_dim, 3, ppo_cfg, device=args.device)

    output_dir = Path("results") / time.strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    history: Dict[str, List[float]] = {
        "actor_loss": [],
        "critic_loss": [],
        "entropy": [],
        "train_reward": [],
        "eval_reward": [],
        "eval_success": [],
        "eval_completion": [],
        "eval_order_accuracy": [],
    }

    best_success = -math.inf
    best_rollout = None
    best_eval_metrics = {}

    for episode in range(args.episodes):
        trajectory = collect_episode(train_env, agent, deterministic=False)
        batch = prepare_batch(trajectory, ppo_cfg)
        update_stats = agent.update(batch)
        train_reward = float(np.sum(trajectory["rewards"]))
        eval_metrics = evaluate(agent, env_cfg, args.eval_episodes)

        history["actor_loss"].append(update_stats["actor_loss"])
        history["critic_loss"].append(update_stats["critic_loss"])
        history["entropy"].append(update_stats["entropy"])
        history["train_reward"].append(train_reward)
        history["eval_reward"].append(eval_metrics["eval_reward_mean"])
        history["eval_success"].append(eval_metrics["success_rate"])
        history["eval_completion"].append(eval_metrics["completed_fraction"])
        history["eval_order_accuracy"].append(eval_metrics["order_accuracy"])

        if eval_metrics["success_rate"] > best_success:
            best_success = eval_metrics["success_rate"]
            best_eval_metrics = eval_metrics
            rollout_env = Swarm3DEnv(env_cfg)
            best_rollout = collect_episode(rollout_env, agent, deterministic=True)
            torch.save(
                {
                    "actor": agent.actor.state_dict(),
                    "critic": agent.critic.state_dict(),
                    "env_config": asdict(env_cfg),
                    "ppo_config": asdict(ppo_cfg),
                },
                output_dir / "best_model.pt",
            )

        if (episode + 1) % max(1, args.log_every) == 0:
            print(
                f"episode={episode + 1} "
                f"train_reward={train_reward:.2f} "
                f"eval_reward={eval_metrics['eval_reward_mean']:.2f} "
                f"success={eval_metrics['success_rate']:.2f} "
                f"completion={eval_metrics['completed_fraction']:.2f} "
                f"order_acc={eval_metrics['order_accuracy']:.2f}"
            )

    plot_training_curves(history, output_dir)

    summary = {
        "args": vars(args),
        "env_config": asdict(env_cfg),
        "ppo_config": asdict(ppo_cfg),
        "best_eval_metrics": best_eval_metrics,
        "final_eval_metrics": {
            "eval_reward_mean": history["eval_reward"][-1],
            "success_rate": history["eval_success"][-1],
            "completed_fraction": history["eval_completion"][-1],
            "order_accuracy": history["eval_order_accuracy"][-1],
        },
    }
    with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if best_rollout is not None:
        try:
            create_animation(best_rollout["rollout_positions"], env_cfg, output_dir / "trained_rollout.gif")
        except Exception as exc:
            print(f"Animation export failed: {exc}")

    print(f"Results written to: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a shared-policy PPO swarm in 3D.")
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--steps-per-episode", type=int, default=150)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--num-objectives", type=int, default=4)
    parser.add_argument("--max-speed", type=float, default=2.0)
    parser.add_argument("--max-accel", type=float, default=1.0)
    parser.add_argument("--objective-radius", type=float, default=0.7)
    parser.add_argument("--step-penalty", type=float, default=0.08)
    parser.add_argument("--reward-aggregation", choices=["sum_agents", "max_agents"], default="sum_agents")
    parser.add_argument("--completion-bonus-scale", type=float, default=0.0)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
