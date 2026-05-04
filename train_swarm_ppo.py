from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from itertools import combinations, permutations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    active: bool = True


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
    reward_aggregation: str = "closest_agent"
    objective_gaussian_scale: float = 0.08
    progress_reward_scale: float = 3.0
    local_progress_reward_scale: float = 2.0
    completion_bonus_scale: float = 28.0
    sequence_completion_bonus: float = 140.0
    step_penalty: float = 0.45
    idle_speed_threshold: float = 0.15
    idle_progress_threshold: float = 0.02
    idle_penalty: float = 0.25
    obstacle_penalty_once: bool = True
    collision_radius: float = 0.35
    collision_penalty: float = 0.5
    max_steps: int = 150
    train_obstacle_count_range: Tuple[int, int] = (1, 3)
    fixed_objectives: Tuple[Tuple[float, float, float], ...] = (
        (-7.0, -4.0, -2.0),
        (-1.5, 3.5, 2.0),
        (3.0, -2.5, 4.5),
        (7.5, 5.0, -1.0),
    )
    fixed_amplitudes: Tuple[float, ...] = (18.0, 22.0, 16.0, 28.0)
    fixed_obstacles: Tuple[Obstacle, ...] = field(
        default_factory=lambda: (
            Obstacle(center=(-2.0, -1.0, 0.0), radius=1.6, penalty=10.0),
            Obstacle(center=(2.5, 2.0, 1.5), radius=1.8, penalty=12.0),
            Obstacle(center=(5.0, -4.0, 2.0), radius=1.4, penalty=8.0),
        )
    )


class Swarm3DEnv:
    def __init__(self, config: EnvConfig):
        self.cfg = config
        self.max_obstacles = len(self.cfg.fixed_obstacles)
        if len(self.cfg.objective_sigmas) != self.cfg.num_objectives:
            raise ValueError("objective_sigmas must match num_objectives")
        if len(self.cfg.fixed_objectives) != self.cfg.num_objectives:
            raise ValueError("fixed_objectives must match num_objectives")
        if len(self.cfg.fixed_amplitudes) != self.cfg.num_objectives:
            raise ValueError("fixed_amplitudes must match num_objectives")
        self.rng = np.random.default_rng()
        self.reset(training=True, difficulty=0.0)

    @property
    def obs_dim(self) -> int:
        return (
            6
            + self.cfg.num_agents
            + (self.cfg.num_agents - 1) * 3
            + self.cfg.num_objectives * 6
            + self.cfg.num_objectives
            + self.cfg.num_objectives
            + 4
            + self.max_obstacles * 5
        )

    @property
    def state_dim(self) -> int:
        return (
            self.cfg.num_agents * 6
            + self.cfg.num_objectives * 5
            + self.cfg.num_objectives
            + self.max_obstacles * 5
        )

    def _sample_positions(self, count: int, span_scale: float) -> np.ndarray:
        low = -span_scale * self.cfg.world_size
        high = span_scale * self.cfg.world_size
        return self.rng.uniform(low=low, high=high, size=(count, 3)).astype(np.float32)

    def _sample_training_obstacles(self, difficulty: float) -> List[Obstacle]:
        min_count, max_count = self.cfg.train_obstacle_count_range
        available = min(self.max_obstacles, max(min_count, int(round(min_count + difficulty * (max_count - min_count)))))
        active_count = int(self.rng.integers(min_count, available + 1))
        obstacles: List[Obstacle] = []

        for idx in range(self.max_obstacles):
            if idx < active_count:
                center = tuple(self._sample_positions(1, span_scale=0.7)[0].tolist())
                radius = float(self.rng.uniform(0.8, 1.3 + 0.9 * difficulty))
                penalty = float(self.rng.uniform(6.0, 16.0))
                obstacles.append(Obstacle(center=center, radius=radius, penalty=penalty, active=True))
            else:
                obstacles.append(Obstacle(center=(0.0, 0.0, 0.0), radius=0.0, penalty=0.0, active=False))
        return obstacles

    def reset(self, training: bool = True, difficulty: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        self.training = training
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))
        self.step_idx = 0
        self.completed_objectives = np.zeros(self.cfg.num_objectives, dtype=bool)
        self.cumulative_obstacle_events = 0
        self.obstacle_hits = np.zeros((self.cfg.num_agents, self.max_obstacles), dtype=bool)

        spawn_span = 0.35 + 0.45 * self.difficulty if training else 0.8
        self.agent_positions = self._sample_positions(self.cfg.num_agents, spawn_span)
        self.agent_velocities = np.zeros((self.cfg.num_agents, 3), dtype=np.float32)

        if training:
            objective_span = 0.35 + 0.45 * self.difficulty
            self.objectives = self._sample_positions(self.cfg.num_objectives, objective_span)
            self.amplitudes = self.rng.uniform(1.0, 30.0, size=self.cfg.num_objectives).astype(np.float32)
            self.obstacles = self._sample_training_obstacles(self.difficulty)
        else:
            self.objectives = np.array(self.cfg.fixed_objectives, dtype=np.float32)
            self.amplitudes = np.array(self.cfg.fixed_amplitudes, dtype=np.float32)
            self.obstacles = [Obstacle(**asdict(obstacle)) for obstacle in self.cfg.fixed_obstacles]

        self.sigmas = np.array(self.cfg.objective_sigmas, dtype=np.float32)
        self.current_assignment = self._assign_agents_to_objectives()
        self.prev_assignment_cost = self._coverage_cost()
        self.prev_local_distances = self._assigned_agent_distances(self.current_assignment)
        return self._get_obs(), self._get_state()

    def _active_obstacles(self) -> List[Obstacle]:
        return self.obstacles

    def _coverage_cost(self) -> float:
        remaining = np.where(~self.completed_objectives)[0]
        if len(remaining) == 0:
            return 0.0
        remaining_positions = self.objectives[remaining]
        pairwise = np.linalg.norm(self.agent_positions[:, None, :] - remaining_positions[None, :, :], axis=2)
        return float(np.sum(np.min(pairwise, axis=0)))

    def _assign_agents_to_objectives(self) -> Dict[int, int]:
        remaining = np.where(~self.completed_objectives)[0].tolist()
        if not remaining:
            return {}

        pairwise = np.linalg.norm(
            self.agent_positions[:, None, :] - self.objectives[remaining][None, :, :],
            axis=2,
        )
        agent_count = self.cfg.num_agents
        objective_count = len(remaining)
        match_count = min(agent_count, objective_count)

        best_cost = math.inf
        best_mapping: Dict[int, int] = {}

        if objective_count >= agent_count:
            for objective_indices in combinations(range(objective_count), match_count):
                for perm in permutations(objective_indices, agent_count):
                    cost = 0.0
                    mapping = {}
                    for agent_idx, obj_local_idx in enumerate(perm):
                        cost += float(pairwise[agent_idx, obj_local_idx])
                        mapping[agent_idx] = remaining[obj_local_idx]
                    if cost < best_cost:
                        best_cost = cost
                        best_mapping = mapping
        else:
            for agent_indices in combinations(range(agent_count), match_count):
                for perm in permutations(range(objective_count), match_count):
                    cost = 0.0
                    mapping = {}
                    for offset, agent_idx in enumerate(agent_indices):
                        obj_local_idx = perm[offset]
                        cost += float(pairwise[agent_idx, obj_local_idx])
                        mapping[agent_idx] = remaining[obj_local_idx]
                    if cost < best_cost:
                        best_cost = cost
                        best_mapping = mapping

        return best_mapping

    def _assigned_agent_distances(self, assignment: Dict[int, int]) -> Dict[int, float]:
        distances: Dict[int, float] = {}
        for agent_idx, objective_idx in assignment.items():
            distances[agent_idx] = float(
                np.linalg.norm(self.agent_positions[agent_idx] - self.objectives[objective_idx])
            )
        return distances

    def _active_objective_reward(self) -> float:
        total_reward = 0.0
        for obj_idx in range(self.cfg.num_objectives):
            if self.completed_objectives[obj_idx]:
                continue
            distances = np.linalg.norm(self.agent_positions - self.objectives[obj_idx], axis=1)
            responses = self.amplitudes[obj_idx] * np.exp(-0.5 * (distances / self.sigmas[obj_idx]) ** 2)
            if self.cfg.reward_aggregation == "sum_agents":
                total_reward += float(np.sum(responses))
            elif self.cfg.reward_aggregation == "max_agents":
                total_reward += float(np.max(responses))
            else:
                total_reward += float(responses[int(np.argmin(distances))])
        return total_reward

    def _get_obs(self) -> np.ndarray:
        obs = []

        for agent_idx in range(self.cfg.num_agents):
            own_pos = self.agent_positions[agent_idx] / self.cfg.world_size
            own_vel = self.agent_velocities[agent_idx] / max(self.cfg.max_speed, 1e-6)
            agent_identity = np.zeros(self.cfg.num_agents, dtype=np.float32)
            agent_identity[agent_idx] = 1.0

            rel_agents = []
            for other_idx in range(self.cfg.num_agents):
                if other_idx == agent_idx:
                    continue
                rel_agents.append((self.agent_positions[other_idx] - self.agent_positions[agent_idx]) / self.cfg.world_size)

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
                        self.sigmas[obj_idx] / max(self.cfg.world_size, 1e-6),
                    ]
                )

            assigned_features = np.zeros(self.cfg.num_objectives + 4, dtype=np.float32)
            assigned_objective = self.current_assignment.get(agent_idx)
            if assigned_objective is not None:
                rel_assigned = (
                    self.objectives[assigned_objective] - self.agent_positions[agent_idx]
                ) / self.cfg.world_size
                assigned_features[assigned_objective] = 1.0
                assigned_features[self.cfg.num_objectives : self.cfg.num_objectives + 3] = rel_assigned
                assigned_features[-1] = float(np.linalg.norm(rel_assigned))

            obstacle_features = []
            for obstacle in self._active_obstacles():
                center = np.array(obstacle.center, dtype=np.float32)
                rel_center = (center - self.agent_positions[agent_idx]) / self.cfg.world_size
                obstacle_features.extend(
                    [
                        rel_center[0],
                        rel_center[1],
                        rel_center[2],
                        obstacle.radius / max(self.cfg.world_size, 1e-6),
                        float(obstacle.active),
                    ]
                )

            obs.append(
                np.concatenate(
                    [
                        own_pos,
                        own_vel,
                        agent_identity,
                        np.concatenate(rel_agents).astype(np.float32) if rel_agents else np.array([], dtype=np.float32),
                        np.array(obj_features, dtype=np.float32),
                        self.completed_objectives.astype(np.float32),
                        assigned_features,
                        np.array(obstacle_features, dtype=np.float32),
                    ]
                ).astype(np.float32)
            )
        return np.stack(obs, axis=0)

    def _get_state(self) -> np.ndarray:
        objective_features = np.column_stack(
            [
                self.objectives / self.cfg.world_size,
                self.amplitudes.reshape(-1, 1) / 30.0,
                self.sigmas.reshape(-1, 1) / max(self.cfg.world_size, 1e-6),
            ]
        ).reshape(-1)

        obstacle_features = []
        for obstacle in self._active_obstacles():
            obstacle_features.extend(
                [
                    obstacle.center[0] / self.cfg.world_size,
                    obstacle.center[1] / self.cfg.world_size,
                    obstacle.center[2] / self.cfg.world_size,
                    obstacle.radius / max(self.cfg.world_size, 1e-6),
                    float(obstacle.active),
                ]
            )

        return np.concatenate(
            [
                self.agent_positions.reshape(-1) / self.cfg.world_size,
                self.agent_velocities.reshape(-1) / max(self.cfg.max_speed, 1e-6),
                objective_features,
                self.completed_objectives.astype(np.float32),
                np.array(obstacle_features, dtype=np.float32),
            ]
        ).astype(np.float32)

    @staticmethod
    def _clip_norm(vectors: np.ndarray, max_norm: float) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        scales = np.where(norms > max_norm, max_norm / np.maximum(norms, 1e-8), 1.0)
        return vectors * scales

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, bool, Dict]:
        self.step_idx += 1
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.cfg.num_agents, 3):
            raise ValueError(f"Expected actions {(self.cfg.num_agents, 3)}, got {actions.shape}")

        prev_cost = self.prev_assignment_cost
        prev_local_distances = dict(self.prev_local_distances)
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

        reward = 0.0
        reward += self.cfg.objective_gaussian_scale * self._active_objective_reward()
        new_cost = self._coverage_cost()
        progress = prev_cost - new_cost
        reward += self.cfg.progress_reward_scale * progress
        reward -= self.cfg.step_penalty

        assignment = self._assign_agents_to_objectives()
        local_distances = self._assigned_agent_distances(assignment)
        local_progress_total = 0.0
        for agent_idx, objective_idx in assignment.items():
            old_distance = prev_local_distances.get(agent_idx, local_distances[agent_idx])
            local_progress_total += old_distance - local_distances[agent_idx]
        reward += self.cfg.local_progress_reward_scale * local_progress_total

        avg_speed = float(np.mean(np.linalg.norm(self.agent_velocities, axis=1)))
        if progress < self.cfg.idle_progress_threshold and avg_speed < self.cfg.idle_speed_threshold:
            reward -= self.cfg.idle_penalty

        obstacle_events = 0
        for obs_idx, obstacle in enumerate(self._active_obstacles()):
            if not obstacle.active:
                continue
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
        for obj_idx in range(self.cfg.num_objectives):
            if self.completed_objectives[obj_idx]:
                continue
            any_close = np.any(
                np.linalg.norm(self.agent_positions - self.objectives[obj_idx], axis=1) <= self.cfg.objective_radius
            )
            if not any_close:
                continue

            self.completed_objectives[obj_idx] = True
            completed_this_step.append(obj_idx)
            reward += self.cfg.completion_bonus_scale * self.amplitudes[obj_idx]

        if np.all(self.completed_objectives):
            reward += self.cfg.sequence_completion_bonus

        self.current_assignment = self._assign_agents_to_objectives()
        self.prev_local_distances = self._assigned_agent_distances(self.current_assignment)
        self.prev_assignment_cost = self._coverage_cost()
        self.cumulative_obstacle_events += obstacle_events

        success = bool(np.all(self.completed_objectives))
        time_limit = bool(self.step_idx >= self.cfg.max_steps and not success)
        done = bool(success or time_limit)
        info = {
            "completed_count": int(np.sum(self.completed_objectives)),
            "completed_fraction": float(np.mean(self.completed_objectives)),
            "success": success,
            "terminal_success": success,
            "time_limit": time_limit,
            "completed_this_step": completed_this_step,
            "obstacle_events": obstacle_events,
            "cumulative_obstacle_events": self.cumulative_obstacle_events,
            "pairwise_penalty": pairwise_penalty,
            "progress": float(progress),
            "local_progress": float(local_progress_total),
            "avg_speed": avg_speed,
            "remaining_assignment_cost": float(self.prev_assignment_cost),
            "remaining_objectives": int(np.sum(~self.completed_objectives)),
        }
        return self._get_obs(), self._get_state(), float(reward), done, info


class RunningNorm:
    def __init__(self, shape: int, eps: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        array = np.asarray(x, dtype=np.float64)
        if array.ndim == 1:
            array = array[None, :]
        batch_mean = np.mean(array, axis=0)
        batch_var = np.var(array, axis=0)
        batch_count = array.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m_2 / total_count

        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-6)
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / np.sqrt(self.var + 1e-8)).astype(np.float32)


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
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.7))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.net(obs)
        log_std = self.log_std.expand_as(mean).clamp(-5.0, 1.5)
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
    minibatch_size: int = 512
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    reward_scale: float = 0.05
    bc_kl_coef: float = 0.0


class SharedPPO:
    def __init__(self, obs_dim: int, state_dim: int, action_dim: int, cfg: PPOConfig, action_limit: float, device: str):
        self.cfg = cfg
        self.device = torch.device(device)
        self.action_limit = float(action_limit)
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)
        self.obs_norm = RunningNorm(obs_dim)
        self.state_norm = RunningNorm(state_dim)
        self.reference_actor: Optional[Actor] = None

    def normalize_obs(self, obs: np.ndarray, update: bool) -> np.ndarray:
        if update:
            self.obs_norm.update(obs)
        return self.obs_norm.normalize(obs)

    def normalize_state(self, state: np.ndarray, update: bool) -> np.ndarray:
        if update:
            self.state_norm.update(state)
        return self.state_norm.normalize(state)

    def _distribution(self, obs: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        mean, log_std = self.actor(obs)
        std = log_std.exp()
        return Normal(mean, std), mean

    def _squash(self, raw_action: torch.Tensor) -> torch.Tensor:
        return torch.tanh(raw_action) * self.action_limit

    def _inverse_squash(self, action: torch.Tensor) -> torch.Tensor:
        scaled = torch.clamp(action / self.action_limit, -0.999999, 0.999999)
        return 0.5 * (torch.log1p(scaled) - torch.log1p(-scaled))

    def _log_prob_from_action(self, dist: Normal, action: torch.Tensor) -> torch.Tensor:
        raw_action = self._inverse_squash(action)
        squashed = torch.tanh(raw_action)
        correction = torch.log(1.0 - squashed.pow(2) + 1e-6)
        return (dist.log_prob(raw_action) - correction).sum(dim=-1)

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            dist, mean = self._distribution(obs_tensor)
            raw_action = mean if deterministic else dist.rsample()
            action = self._squash(raw_action)
            log_prob = self._log_prob_from_action(dist, action)
        return action.cpu().numpy(), log_prob.cpu().numpy()

    def value(self, state: np.ndarray) -> float:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            value = self.critic(state_tensor)[0]
        return float(value.item())

    def set_schedule(self, actor_lr: float, critic_lr: float, clip_ratio: float, entropy_coef: float, bc_kl_coef: float) -> None:
        self.cfg.actor_lr = float(actor_lr)
        self.cfg.critic_lr = float(critic_lr)
        self.cfg.clip_ratio = float(clip_ratio)
        self.cfg.entropy_coef = float(entropy_coef)
        self.cfg.bc_kl_coef = float(bc_kl_coef)
        for param_group in self.actor_optim.param_groups:
            param_group["lr"] = self.cfg.actor_lr
        for param_group in self.critic_optim.param_groups:
            param_group["lr"] = self.cfg.critic_lr

    def snapshot_reference_actor(self) -> None:
        reference_actor = Actor(self.actor.net[0].in_features, self.actor.net[-1].out_features).to(self.device)
        reference_actor.load_state_dict(copy.deepcopy(self.actor.state_dict()))
        reference_actor.eval()
        for param in reference_actor.parameters():
            param.requires_grad_(False)
        self.reference_actor = reference_actor

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=self.device)
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        num_samples = obs.shape[0]
        actor_loss_value = 0.0
        critic_loss_value = 0.0
        entropy_value = 0.0
        retention_kl_value = 0.0

        for _ in range(self.cfg.train_epochs):
            indices = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, self.cfg.minibatch_size):
                batch_idx = indices[start : start + self.cfg.minibatch_size]

                obs_mb = obs[batch_idx]
                actions_mb = actions[batch_idx]
                old_log_probs_mb = old_log_probs[batch_idx]
                states_mb = states[batch_idx]
                returns_mb = returns[batch_idx]
                advantages_mb = advantages[batch_idx]

                dist, _ = self._distribution(obs_mb)
                new_log_probs = self._log_prob_from_action(dist, actions_mb)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratios = torch.exp(new_log_probs - old_log_probs_mb)
                clipped_ratios = torch.clamp(ratios, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio)
                actor_loss = -torch.min(ratios * advantages_mb, clipped_ratios * advantages_mb).mean()

                values = self.critic(states_mb)
                critic_loss = ((returns_mb - values) ** 2).mean()
                retention_kl = torch.zeros((), device=self.device)
                if self.reference_actor is not None and self.cfg.bc_kl_coef > 0.0:
                    with torch.no_grad():
                        reference_dist, _ = self._distribution_from_actor(self.reference_actor, obs_mb)
                    current_dist, _ = self._distribution(obs_mb)
                    retention_kl = torch.distributions.kl_divergence(reference_dist, current_dist).sum(dim=-1).mean()

                loss = (
                    actor_loss
                    + self.cfg.value_coef * critic_loss
                    - self.cfg.entropy_coef * entropy
                    + self.cfg.bc_kl_coef * retention_kl
                )

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
                retention_kl_value = float(retention_kl.item())

        return {
            "actor_loss": actor_loss_value,
            "critic_loss": critic_loss_value,
            "entropy": entropy_value,
            "retention_kl": retention_kl_value,
        }

    def _distribution_from_actor(self, actor: Actor, obs: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        mean, log_std = actor(obs)
        std = log_std.exp()
        return Normal(mean, std), mean


def expert_actions(env: Swarm3DEnv) -> np.ndarray:
    actions = np.zeros((env.cfg.num_agents, 3), dtype=np.float32)
    for agent_idx, objective_idx in env.current_assignment.items():
        target = env.objectives[objective_idx]
        direction = target - env.agent_positions[agent_idx]
        distance = float(np.linalg.norm(direction))
        if distance <= 1e-6:
            continue
        desired_velocity = direction / max(distance, 1e-6) * env.cfg.max_speed
        acceleration = (desired_velocity - env.agent_velocities[agent_idx]) / max(env.cfg.dt, 1e-6)
        actions[agent_idx] = acceleration.astype(np.float32)
    return actions


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
        bootstrap_value = next_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * bootstrap_value * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def collect_episode(
    env: Swarm3DEnv,
    agent: SharedPPO,
    deterministic: bool = False,
    update_norm: bool = True,
    randomized_env: bool = True,
    difficulty: float = 1.0,
) -> Dict[str, np.ndarray]:
    obs, state = env.reset(training=randomized_env, difficulty=difficulty)
    norm_obs = agent.normalize_obs(obs, update=update_norm)
    norm_state = agent.normalize_state(state, update=update_norm)

    trajectory: Dict[str, List] = {
        "obs": [],
        "states": [],
        "actions": [],
        "log_probs": [],
        "values": [],
        "rewards": [],
        "dones": [],
        "completed_counts": [],
    }
    rollout_positions = [env.agent_positions.copy()]
    final_info: Dict = {}
    first_completion_step = -1
    objective_completion_steps = np.full(env.cfg.num_objectives, -1, dtype=np.int32)

    done = False
    while not done:
        actions, log_probs = agent.select_action(norm_obs, deterministic=deterministic)
        value = agent.value(norm_state)
        next_obs, next_state, reward, done, info = env.step(actions)

        trajectory["obs"].append(norm_obs.copy())
        trajectory["states"].append(norm_state.copy())
        trajectory["actions"].append(actions.copy())
        trajectory["log_probs"].append(log_probs.copy())
        trajectory["values"].append(value)
        trajectory["rewards"].append(reward * agent.cfg.reward_scale)
        trajectory["dones"].append(float(done and info["terminal_success"]))
        trajectory["completed_counts"].append(info["completed_count"])

        if info["completed_this_step"]:
            if first_completion_step < 0:
                first_completion_step = len(trajectory["rewards"])
            for objective_idx in info["completed_this_step"]:
                if objective_completion_steps[objective_idx] < 0:
                    objective_completion_steps[objective_idx] = len(trajectory["rewards"])

        norm_obs = agent.normalize_obs(next_obs, update=update_norm)
        norm_state = agent.normalize_state(next_state, update=update_norm)
        final_info = info
        rollout_positions.append(env.agent_positions.copy())

    next_value = 0.0 if final_info.get("terminal_success", False) else agent.value(norm_state)
    trajectory["next_value"] = next_value
    trajectory["rollout_positions"] = np.stack(rollout_positions, axis=0)
    trajectory["final_info"] = final_info
    trajectory["episode_return"] = float(np.sum(trajectory["rewards"]) / agent.cfg.reward_scale)
    trajectory["episode_length"] = len(trajectory["rewards"])
    trajectory["first_completion_step"] = first_completion_step
    trajectory["objective_completion_steps"] = objective_completion_steps
    return {k: np.asarray(v) if isinstance(v, list) else v for k, v in trajectory.items()}


def prepare_batch(trajectories: List[Dict[str, np.ndarray]], ppo_cfg: PPOConfig) -> Dict[str, np.ndarray]:
    obs_batches = []
    action_batches = []
    log_prob_batches = []
    state_batches = []
    return_batches = []
    advantage_batches = []

    for trajectory in trajectories:
        rewards = np.asarray(trajectory["rewards"], dtype=np.float32)
        dones = np.asarray(trajectory["dones"], dtype=np.float32)
        values = np.asarray(trajectory["values"], dtype=np.float32)
        advantages, returns = compute_gae(
            rewards=rewards,
            dones=dones,
            values=values,
            next_value=float(trajectory["next_value"]),
            gamma=ppo_cfg.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
        )

        obs = np.asarray(trajectory["obs"], dtype=np.float32)
        states = np.asarray(trajectory["states"], dtype=np.float32)
        actions = np.asarray(trajectory["actions"], dtype=np.float32)
        log_probs = np.asarray(trajectory["log_probs"], dtype=np.float32)
        time_steps, num_agents, obs_dim = obs.shape
        _, _, action_dim = actions.shape

        obs_batches.append(obs.reshape(time_steps * num_agents, obs_dim))
        action_batches.append(actions.reshape(time_steps * num_agents, action_dim))
        log_prob_batches.append(log_probs.reshape(time_steps * num_agents))
        state_batches.append(np.repeat(states, repeats=num_agents, axis=0))
        return_batches.append(np.repeat(returns, repeats=num_agents, axis=0))
        advantage_batches.append(np.repeat(advantages, repeats=num_agents, axis=0))

    return {
        "obs": np.concatenate(obs_batches, axis=0),
        "actions": np.concatenate(action_batches, axis=0),
        "log_probs": np.concatenate(log_prob_batches, axis=0),
        "states": np.concatenate(state_batches, axis=0),
        "returns": np.concatenate(return_batches, axis=0),
        "advantages": np.concatenate(advantage_batches, axis=0),
    }


def collect_expert_dataset(
    env: Swarm3DEnv,
    agent: SharedPPO,
    episodes: int,
    difficulty: float,
) -> Tuple[np.ndarray, np.ndarray]:
    obs_samples: List[np.ndarray] = []
    action_samples: List[np.ndarray] = []

    for _ in range(episodes):
        obs, _ = env.reset(training=True, difficulty=difficulty)
        done = False
        while not done:
            norm_obs = agent.normalize_obs(obs, update=True)
            actions = expert_actions(env)
            obs_samples.append(norm_obs.reshape(-1, env.obs_dim))
            action_samples.append(actions.reshape(-1, 3))
            obs, _, _, done, _ = env.step(actions)

    return np.concatenate(obs_samples, axis=0), np.concatenate(action_samples, axis=0)


def behavior_clone_actor(
    agent: SharedPPO,
    obs: np.ndarray,
    actions: np.ndarray,
    epochs: int,
    batch_size: int,
) -> None:
    if epochs <= 0 or obs.size == 0:
        return

    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
    action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=agent.device)
    num_samples = obs_tensor.shape[0]

    for _ in range(epochs):
        indices = torch.randperm(num_samples, device=agent.device)
        for start in range(0, num_samples, batch_size):
            batch_idx = indices[start : start + batch_size]
            mean, _ = agent.actor(obs_tensor[batch_idx])
            predicted_actions = agent._squash(mean)
            loss = ((predicted_actions - action_tensor[batch_idx]) ** 2).mean()

            agent.actor_optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.actor.parameters(), agent.cfg.max_grad_norm)
            agent.actor_optim.step()


def curriculum_difficulty(episode_idx: int, args: argparse.Namespace) -> float:
    ramp = min(1.0, episode_idx / max(1, args.curriculum_episodes - 1))
    if args.curriculum_strategy == "ramp":
        return ramp
    if args.curriculum_strategy == "mixed":
        mixed_floor = float(np.clip(args.curriculum_mixed_floor, 0.0, 1.0))
        return float(np.random.uniform(low=mixed_floor, high=max(mixed_floor, ramp)))
    raise ValueError(f"Unsupported curriculum strategy: {args.curriculum_strategy}")


def scheduled_value(episode_idx: int, total_episodes: int, start: float, end: float) -> float:
    if total_episodes <= 1:
        return float(end)
    alpha = float(np.clip(episode_idx / max(1, total_episodes - 1), 0.0, 1.0))
    return float(start + alpha * (end - start))


def save_checkpoint(
    path: Path,
    agent: SharedPPO,
    env_cfg: EnvConfig,
    ppo_cfg: PPOConfig,
    metadata: Dict,
) -> None:
    torch.save(
        {
            "actor": agent.actor.state_dict(),
            "critic": agent.critic.state_dict(),
            "env_config": asdict(env_cfg),
            "ppo_config": asdict(ppo_cfg),
            "obs_norm_mean": agent.obs_norm.mean.tolist(),
            "obs_norm_var": agent.obs_norm.var.tolist(),
            "state_norm_mean": agent.state_norm.mean.tolist(),
            "state_norm_var": agent.state_norm.var.tolist(),
            "metadata": metadata,
        },
        path,
    )


def evaluate(agent: SharedPPO, env_cfg: EnvConfig, episodes: int, generalized: bool) -> Dict[str, float]:
    eval_env = Swarm3DEnv(env_cfg)
    rewards = []
    success = []
    completed_fraction = []
    obstacle_events = []
    remaining_costs = []
    completion_steps = []
    completed_counts = []
    first_completion_steps = []
    success_steps = []
    completion_auc = []

    for _ in range(episodes):
        trajectory = collect_episode(
            eval_env,
            agent,
            deterministic=True,
            update_norm=False,
            randomized_env=generalized,
            difficulty=1.0,
        )
        info = trajectory["final_info"]
        rewards.append(float(trajectory["episode_return"]))
        success.append(float(info["success"]))
        completed_fraction.append(float(info["completed_fraction"]))
        obstacle_events.append(float(info["cumulative_obstacle_events"]))
        remaining_costs.append(float(info["remaining_assignment_cost"]))
        completion_steps.append(float(trajectory["episode_length"]))
        completed_counts.append(float(info["completed_count"]))
        first_completion_steps.append(
            float(trajectory["first_completion_step"]) if int(trajectory["first_completion_step"]) >= 0 else float(env_cfg.max_steps)
        )
        success_steps.append(float(trajectory["episode_length"]) if info["success"] else float(env_cfg.max_steps))
        cumulative_completed = np.asarray(trajectory["completed_counts"], dtype=np.float32)
        completion_auc.append(float(np.mean(cumulative_completed / max(env_cfg.num_objectives, 1))))

    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "success_rate": float(np.mean(success)),
        "completed_fraction": float(np.mean(completed_fraction)),
        "completed_count": float(np.mean(completed_counts)),
        "obstacle_events": float(np.mean(obstacle_events)),
        "remaining_assignment_cost": float(np.mean(remaining_costs)),
        "episode_length": float(np.mean(completion_steps)),
        "first_completion_step": float(np.mean(first_completion_steps)),
        "success_step": float(np.mean(success_steps)),
        "completion_auc": float(np.mean(completion_auc)),
    }


def plot_training_curves(history: Dict[str, List[float]], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].plot(history["actor_loss"], label="Actor Loss")
    axes[0, 0].plot(history["critic_loss"], label="Critic Loss")
    axes[0, 0].plot(history["entropy"], label="Entropy")
    axes[0, 0].plot(history["retention_kl"], label="Retention KL")
    axes[0, 0].set_title("Optimization")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(history["train_reward"], label="Train Reward")
    axes[0, 1].plot(history["fixed_reward"], label="Fixed Eval Reward")
    axes[0, 1].plot(history["random_reward"], label="Randomized Eval Reward")
    axes[0, 1].set_title("Reward")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(history["fixed_success"], label="Fixed Success")
    axes[1, 0].plot(history["random_success"], label="Randomized Success")
    axes[1, 0].plot(history["fixed_completion"], label="Fixed Completion")
    axes[1, 0].plot(history["random_completion"], label="Randomized Completion")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].set_title("Task Achievement")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(history["fixed_cost"], label="Fixed Remaining Coverage Cost")
    axes[1, 1].plot(history["random_cost"], label="Random Remaining Coverage Cost")
    axes[1, 1].plot(history["fixed_obstacles"], label="Fixed Obstacle Events")
    axes[1, 1].plot(history["random_obstacles"], label="Random Obstacle Events")
    axes[1, 1].set_title("Safety / Progress Diagnostics")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "losses.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].plot(history["fixed_completion_auc"], label="Fixed Completion AUC", color="tab:red")
    axes[0].plot(history["random_completion_auc"], label="Random Completion AUC", color="tab:pink")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Completion Speed")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["curriculum"], label="Curriculum Difficulty", color="tab:green")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Training Curriculum")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(history["train_episode_length"], label="Train Episode Length", color="tab:blue")
    axes[2].plot(history["fixed_first_completion"], label="Fixed First Completion", color="tab:orange")
    axes[2].plot(history["random_first_completion"], label="Randomized First Completion", color="tab:purple")
    axes[2].set_title("Completion Timing")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "metrics.png", dpi=180)
    plt.close(fig)


def create_animation(rollout_positions: np.ndarray, env_cfg: EnvConfig, out_path: Path) -> None:
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

    for obstacle in env_cfg.fixed_obstacles:
        center = np.array(obstacle.center)
        u = np.linspace(0, 2 * np.pi, 16)
        v = np.linspace(0, np.pi, 12)
        x = obstacle.radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = obstacle.radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = obstacle.radius * np.outer(np.ones_like(u), np.cos(v)) + center[2]
        ax.plot_wireframe(x, y, z, color="gray", alpha=0.2, linewidth=0.5)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    scatters = [ax.plot([], [], [], "o", color=colors[i % len(colors)], markersize=7)[0] for i in range(env_cfg.num_agents)]
    trails = [ax.plot([], [], [], color=colors[i % len(colors)], alpha=0.5, linewidth=1.5)[0] for i in range(env_cfg.num_agents)]

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
    fixed_amplitudes = tuple(amplitude_cycle[i % len(amplitude_cycle)] for i in range(args.num_objectives))
    fixed_objectives = tuple(fixed_objective_cycle[i % len(fixed_objective_cycle)] for i in range(args.num_objectives))
    fixed_obstacles = (
        Obstacle(center=(-2.0, -1.0, 0.0), radius=1.6, penalty=10.0),
        Obstacle(center=(2.5, 2.0, 1.5), radius=1.8, penalty=12.0),
        Obstacle(center=(5.0, -4.0, 2.0), radius=1.4, penalty=8.0),
    )

    return EnvConfig(
        num_agents=args.num_agents,
        num_objectives=args.num_objectives,
        max_steps=args.steps_per_episode,
        max_speed=args.max_speed,
        max_accel=args.max_accel,
        objective_radius=args.objective_radius,
        objective_sigmas=objective_sigmas,
        reward_aggregation=args.reward_aggregation,
        objective_gaussian_scale=args.objective_gaussian_scale,
        progress_reward_scale=args.progress_reward_scale,
        local_progress_reward_scale=args.local_progress_reward_scale,
        completion_bonus_scale=args.completion_bonus_scale,
        sequence_completion_bonus=args.sequence_completion_bonus,
        step_penalty=args.step_penalty,
        idle_penalty=args.idle_penalty,
        fixed_objectives=fixed_objectives,
        fixed_amplitudes=fixed_amplitudes,
        fixed_obstacles=fixed_obstacles,
    )


def best_metric_tuple(
    fixed_metrics: Dict[str, float],
    general_metrics: Dict[str, float],
) -> Tuple[float, float, float, float, float, float, float, float]:
    return (
        fixed_metrics["success_rate"],
        general_metrics["success_rate"],
        fixed_metrics["completed_fraction"],
        general_metrics["completed_fraction"],
        fixed_metrics["completion_auc"],
        general_metrics["completion_auc"],
        -fixed_metrics["remaining_assignment_cost"],
        -general_metrics["remaining_assignment_cost"],
    )


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    env_cfg = build_env_config(args)
    train_env = Swarm3DEnv(env_cfg)
    ppo_cfg = PPOConfig(
        clip_ratio=args.clip_ratio,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        train_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        entropy_coef=args.entropy_coef,
        reward_scale=args.reward_scale,
        bc_kl_coef=args.bc_kl_coef,
    )
    agent = SharedPPO(train_env.obs_dim, train_env.state_dim, 3, ppo_cfg, action_limit=env_cfg.max_accel, device=args.device)

    if args.bc_episodes > 0 and args.bc_epochs > 0:
        expert_obs, expert_actions_batch = collect_expert_dataset(
            train_env,
            agent,
            episodes=args.bc_episodes,
            difficulty=min(1.0, args.bc_difficulty),
        )
        behavior_clone_actor(
            agent,
            obs=expert_obs,
            actions=expert_actions_batch,
            epochs=args.bc_epochs,
            batch_size=args.bc_batch_size,
        )
        if args.bc_snapshot_reference:
            agent.snapshot_reference_actor()

    pretrain_fixed_metrics = evaluate(agent, env_cfg, args.eval_episodes, generalized=False)
    pretrain_random_metrics = evaluate(agent, env_cfg, args.eval_episodes, generalized=True)

    output_dir = Path("results") / time.strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    history: Dict[str, List[float]] = {
        "actor_loss": [],
        "critic_loss": [],
        "entropy": [],
        "retention_kl": [],
        "train_reward": [],
        "train_episode_length": [],
        "fixed_reward": [],
        "random_reward": [],
        "fixed_success": [],
        "random_success": [],
        "fixed_completion": [],
        "random_completion": [],
        "fixed_completed_count": [],
        "random_completed_count": [],
        "fixed_cost": [],
        "random_cost": [],
        "fixed_obstacles": [],
        "random_obstacles": [],
        "fixed_length": [],
        "random_length": [],
        "fixed_first_completion": [],
        "random_first_completion": [],
        "fixed_success_step": [],
        "random_success_step": [],
        "fixed_completion_auc": [],
        "random_completion_auc": [],
        "curriculum": [],
    }

    best_key = (-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf)
    best_rollout = None
    best_fixed_metrics = {}
    best_general_metrics = {}
    top_checkpoints: List[Dict] = []
    ppo_phase_start = max(0, args.ppo_phase1_episodes)
    phase2_enabled = args.ppo_phase2_episodes > 0
    phase2_allowed = not args.require_phase1_generalization_improvement
    phase1_gate_checked = False
    early_stop_reason = ""
    best_episode = -1
    episodes_since_best = 0

    for episode in range(args.episodes):
        if phase2_enabled and not phase2_allowed and episode >= ppo_phase_start:
            early_stop_reason = "phase1_generalization_gate_not_met"
            break

        if episode < ppo_phase_start:
            current_phase = "phase1"
            total_phase_episodes = max(1, args.ppo_phase1_episodes)
            phase_episode_idx = episode
            phase_epochs = args.ppo_phase1_epochs
            clip_ratio = scheduled_value(phase_episode_idx, total_phase_episodes, args.clip_ratio, args.phase1_clip_ratio)
            entropy_coef = scheduled_value(phase_episode_idx, total_phase_episodes, args.entropy_coef, args.phase1_entropy_coef)
            bc_kl_coef = scheduled_value(phase_episode_idx, total_phase_episodes, args.bc_kl_coef, args.phase1_bc_kl_coef)
        else:
            current_phase = "phase2"
            total_phase_episodes = max(1, args.ppo_phase2_episodes if phase2_enabled else args.episodes - ppo_phase_start)
            phase_episode_idx = max(0, episode - ppo_phase_start)
            phase_epochs = args.ppo_phase2_epochs if phase2_enabled else args.ppo_phase1_epochs
            clip_ratio = scheduled_value(phase_episode_idx, total_phase_episodes, args.phase1_clip_ratio, args.phase2_clip_ratio)
            entropy_coef = scheduled_value(phase_episode_idx, total_phase_episodes, args.phase1_entropy_coef, args.phase2_entropy_coef)
            bc_kl_coef = scheduled_value(phase_episode_idx, total_phase_episodes, args.phase1_bc_kl_coef, args.phase2_bc_kl_coef)

        agent.cfg.train_epochs = int(phase_epochs)
        agent.set_schedule(
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            clip_ratio=clip_ratio,
            entropy_coef=entropy_coef,
            bc_kl_coef=bc_kl_coef,
        )

        difficulty = curriculum_difficulty(episode, args)
        trajectories = [
            collect_episode(train_env, agent, deterministic=False, update_norm=True, randomized_env=True, difficulty=difficulty)
            for _ in range(args.rollouts_per_update)
        ]
        batch = prepare_batch(trajectories, ppo_cfg)
        update_stats = agent.update(batch)

        train_reward = float(np.mean([traj["episode_return"] for traj in trajectories]))
        train_length = float(np.mean([traj["episode_length"] for traj in trajectories]))
        fixed_metrics = evaluate(agent, env_cfg, args.eval_episodes, generalized=False)
        random_metrics = evaluate(agent, env_cfg, args.eval_episodes, generalized=True)

        history["actor_loss"].append(update_stats["actor_loss"])
        history["critic_loss"].append(update_stats["critic_loss"])
        history["entropy"].append(update_stats["entropy"])
        history["retention_kl"].append(update_stats["retention_kl"])
        history["train_reward"].append(train_reward)
        history["train_episode_length"].append(train_length)
        history["fixed_reward"].append(fixed_metrics["reward_mean"])
        history["random_reward"].append(random_metrics["reward_mean"])
        history["fixed_success"].append(fixed_metrics["success_rate"])
        history["random_success"].append(random_metrics["success_rate"])
        history["fixed_completion"].append(fixed_metrics["completed_fraction"])
        history["random_completion"].append(random_metrics["completed_fraction"])
        history["fixed_completed_count"].append(fixed_metrics["completed_count"])
        history["random_completed_count"].append(random_metrics["completed_count"])
        history["fixed_cost"].append(fixed_metrics["remaining_assignment_cost"])
        history["random_cost"].append(random_metrics["remaining_assignment_cost"])
        history["fixed_obstacles"].append(fixed_metrics["obstacle_events"])
        history["random_obstacles"].append(random_metrics["obstacle_events"])
        history["fixed_length"].append(fixed_metrics["episode_length"])
        history["random_length"].append(random_metrics["episode_length"])
        history["fixed_first_completion"].append(fixed_metrics["first_completion_step"])
        history["random_first_completion"].append(random_metrics["first_completion_step"])
        history["fixed_success_step"].append(fixed_metrics["success_step"])
        history["random_success_step"].append(random_metrics["success_step"])
        history["fixed_completion_auc"].append(fixed_metrics["completion_auc"])
        history["random_completion_auc"].append(random_metrics["completion_auc"])
        history["curriculum"].append(difficulty)

        current_key = best_metric_tuple(fixed_metrics, random_metrics)
        if current_key > best_key:
            best_key = current_key
            best_episode = episode + 1
            episodes_since_best = 0
            best_fixed_metrics = fixed_metrics
            best_general_metrics = random_metrics
            rollout_env = Swarm3DEnv(env_cfg)
            best_rollout = collect_episode(rollout_env, agent, deterministic=True, update_norm=False, randomized_env=False, difficulty=1.0)
            save_checkpoint(
                output_dir / "best_model.pt",
                agent,
                env_cfg,
                ppo_cfg,
                {
                    "episode": episode + 1,
                    "phase": current_phase,
                    "fixed_metrics": fixed_metrics,
                    "generalization_metrics": random_metrics,
                    "curriculum_difficulty": difficulty,
                },
            )
        else:
            episodes_since_best += 1

        checkpoint_record = {
            "episode": episode + 1,
            "phase": current_phase,
            "curriculum_difficulty": difficulty,
            "fixed_metrics": fixed_metrics,
            "generalization_metrics": random_metrics,
            "key": list(current_key),
            "path": str(checkpoints_dir / f"episode_{episode + 1:04d}.pt"),
        }
        leaderboard = top_checkpoints + [checkpoint_record]
        leaderboard = sorted(
            leaderboard,
            key=lambda item: tuple(item["key"]),
            reverse=True,
        )
        retained = leaderboard[: max(1, args.top_k_checkpoints)]
        retained_episodes = {item["episode"] for item in retained}
        if (episode + 1) in retained_episodes:
            save_checkpoint(Path(checkpoint_record["path"]), agent, env_cfg, ppo_cfg, checkpoint_record)
        top_checkpoints = retained
        retained_paths = {Path(item["path"]).resolve() for item in top_checkpoints}
        for existing_checkpoint in checkpoints_dir.glob("episode_*.pt"):
            if existing_checkpoint.resolve() not in retained_paths:
                existing_checkpoint.unlink(missing_ok=True)

        if args.early_stop_patience > 0 and episodes_since_best >= args.early_stop_patience:
            early_stop_reason = "no_checkpoint_improvement"
            break

        if (
            not phase1_gate_checked
            and phase2_enabled
            and args.require_phase1_generalization_improvement
            and (episode + 1) >= ppo_phase_start
        ):
            phase1_gate_checked = True
            random_success_gain = random_metrics["success_rate"] - pretrain_random_metrics["success_rate"]
            random_completion_gain = random_metrics["completed_fraction"] - pretrain_random_metrics["completed_fraction"]
            phase2_allowed = bool(
                random_success_gain >= args.phase1_min_general_success_gain
                or random_completion_gain >= args.phase1_min_general_completion_gain
            )
            if not phase2_allowed:
                early_stop_reason = "phase1_generalization_gate_not_met"

        if (episode + 1) % max(1, args.log_every) == 0:
            print(
                f"episode={episode + 1} "
                f"phase={current_phase} "
                f"difficulty={difficulty:.2f} "
                f"train_reward={train_reward:.2f} "
                f"clip={clip_ratio:.3f} "
                f"entropy={entropy_coef:.4f} "
                f"bc_kl={bc_kl_coef:.4f} "
                f"fixed_success={fixed_metrics['success_rate']:.2f} "
                f"fixed_completion={fixed_metrics['completed_fraction']:.2f} "
                f"fixed_auc={fixed_metrics['completion_auc']:.2f} "
                f"random_success={random_metrics['success_rate']:.2f} "
                f"random_completion={random_metrics['completed_fraction']:.2f}"
            )

    plot_training_curves(history, output_dir)

    summary = {
        "args": vars(args),
        "env_config": asdict(env_cfg),
        "ppo_config": asdict(ppo_cfg),
        "best_fixed_metrics": best_fixed_metrics,
        "best_generalization_metrics": best_general_metrics,
        "best_episode": best_episode,
        "pretrain_fixed_metrics": pretrain_fixed_metrics,
        "pretrain_generalization_metrics": pretrain_random_metrics,
        "top_checkpoints": top_checkpoints,
        "final_fixed_metrics": {
            "reward_mean": history["fixed_reward"][-1],
            "success_rate": history["fixed_success"][-1],
            "completed_fraction": history["fixed_completion"][-1],
            "completed_count": history["fixed_completed_count"][-1],
            "remaining_assignment_cost": history["fixed_cost"][-1],
            "first_completion_step": history["fixed_first_completion"][-1],
            "completion_auc": history["fixed_completion_auc"][-1],
        },
        "final_generalization_metrics": {
            "reward_mean": history["random_reward"][-1],
            "success_rate": history["random_success"][-1],
            "completed_fraction": history["random_completion"][-1],
            "completed_count": history["random_completed_count"][-1],
            "remaining_assignment_cost": history["random_cost"][-1],
            "first_completion_step": history["random_first_completion"][-1],
            "completion_auc": history["random_completion_auc"][-1],
        },
        "training_schedule": {
            "curriculum_strategy": args.curriculum_strategy,
            "curriculum_mixed_floor": args.curriculum_mixed_floor,
            "require_phase1_generalization_improvement": args.require_phase1_generalization_improvement,
            "phase1_min_general_success_gain": args.phase1_min_general_success_gain,
            "phase1_min_general_completion_gain": args.phase1_min_general_completion_gain,
            "ppo_phase1_episodes": args.ppo_phase1_episodes,
            "ppo_phase2_episodes": args.ppo_phase2_episodes,
            "ppo_phase1_epochs": args.ppo_phase1_epochs,
            "ppo_phase2_epochs": args.ppo_phase2_epochs,
            "phase1_clip_ratio": args.phase1_clip_ratio,
            "phase2_clip_ratio": args.phase2_clip_ratio,
            "phase1_entropy_coef": args.phase1_entropy_coef,
            "phase2_entropy_coef": args.phase2_entropy_coef,
            "phase1_bc_kl_coef": args.phase1_bc_kl_coef,
            "phase2_bc_kl_coef": args.phase2_bc_kl_coef,
        },
        "phase2_allowed": phase2_allowed,
        "early_stop_reason": early_stop_reason,
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
    parser.add_argument("--reward-aggregation", choices=["closest_agent", "max_agents", "sum_agents"], default="closest_agent")
    parser.add_argument("--objective-gaussian-scale", type=float, default=0.08)
    parser.add_argument("--progress-reward-scale", type=float, default=3.0)
    parser.add_argument("--local-progress-reward-scale", type=float, default=2.0)
    parser.add_argument("--completion-bonus-scale", type=float, default=28.0)
    parser.add_argument("--sequence-completion-bonus", type=float, default=140.0)
    parser.add_argument("--step-penalty", type=float, default=0.45)
    parser.add_argument("--idle-penalty", type=float, default=0.25)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--reward-scale", type=float, default=0.05)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--bc-kl-coef", type=float, default=0.02)
    parser.add_argument("--rollouts-per-update", type=int, default=6)
    parser.add_argument("--curriculum-episodes", type=int, default=120)
    parser.add_argument("--curriculum-strategy", choices=["ramp", "mixed"], default="mixed")
    parser.add_argument("--curriculum-mixed-floor", type=float, default=0.2)
    parser.add_argument("--eval-episodes", type=int, default=6)
    parser.add_argument("--bc-episodes", type=int, default=64)
    parser.add_argument("--bc-epochs", type=int, default=12)
    parser.add_argument("--bc-batch-size", type=int, default=1024)
    parser.add_argument("--bc-difficulty", type=float, default=1.0)
    parser.add_argument("--bc-snapshot-reference", action="store_true", default=True)
    parser.add_argument("--no-bc-snapshot-reference", action="store_false", dest="bc_snapshot_reference")
    parser.add_argument("--ppo-phase1-episodes", type=int, default=30)
    parser.add_argument("--ppo-phase2-episodes", type=int, default=220)
    parser.add_argument("--ppo-phase1-epochs", type=int, default=4)
    parser.add_argument("--ppo-phase2-epochs", type=int, default=8)
    parser.add_argument("--phase1-clip-ratio", type=float, default=0.12)
    parser.add_argument("--phase2-clip-ratio", type=float, default=0.18)
    parser.add_argument("--phase1-entropy-coef", type=float, default=0.003)
    parser.add_argument("--phase2-entropy-coef", type=float, default=0.001)
    parser.add_argument("--phase1-bc-kl-coef", type=float, default=0.05)
    parser.add_argument("--phase2-bc-kl-coef", type=float, default=0.01)
    parser.add_argument("--require-phase1-generalization-improvement", action="store_true", default=True)
    parser.add_argument("--no-require-phase1-generalization-improvement", action="store_false", dest="require_phase1_generalization_improvement")
    parser.add_argument("--phase1-min-general-success-gain", type=float, default=0.05)
    parser.add_argument("--phase1-min-general-completion-gain", type=float, default=0.05)
    parser.add_argument("--top-k-checkpoints", type=int, default=5)
    parser.add_argument("--early-stop-patience", type=int, default=80)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
