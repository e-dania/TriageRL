"""Simulated emergency department as a Gymnasium environment."""

from __future__ import annotations

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from .patients import (TREATMENT_MEAN_MIN, Patient, _vitals_for_severity,
                       arrival_rate, generate_patient)

TARGET_WAIT_MIN = {1: 0, 2: 10, 3: 30, 4: 60, 5: 120}
TREAT_REWARD = {1: 14.0, 2: 8.0, 3: 3.0, 4: 1.5, 5: 1.0}
HOLD_COST = {1: 0.25, 2: 0.12, 3: 0.02, 4: 0.008, 5: 0.004}
CRITICAL_DELAY_PENALTY = 2.0

DETERIORATION_HAZARD_PER_MIN = {2: 0.0008, 3: 0.0004, 4: 0.0002, 5: 0.0001}
HAZARD_WAIT_CAP_H = 3.0
DETERIORATION_PENALTY = 3.0


class EDTriageEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_beds: int = 5,
        num_staff: int = 7,
        queue_slots: int = 10,
        step_minutes: float = 5.0,
        episode_hours: float = 24.0,
        base_arrival_rate: float = 0.16,
        max_queue: int = 60,
        seed: int | None = None,
    ):
        super().__init__()
        self.num_beds = num_beds
        self.num_staff = num_staff
        self.K = queue_slots
        self.step_minutes = step_minutes
        self.episode_minutes = episode_hours * 60.0
        self.base_arrival_rate = base_arrival_rate
        self.max_queue = max_queue

        self.obs_dim = 4 + 8 * self.K
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.K + 1)

        self._rng = np.random.default_rng(seed)
        self._reset_state()

    def _reset_state(self):
        self.now = 0.0
        self.queue: list[Patient] = []
        self.in_treatment: list[tuple[Patient, float]] = []
        self.treated: list[Patient] = []
        self.episode_stats = {
            "arrivals": 0,
            "treated": 0,
            "invalid_actions": 0,
            "queue_length_sum": 0.0,
            "steps": 0,
            "deteriorations": 0,
        }

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_state()
        for _ in range(int(self._rng.integers(2, 6))):
            self.queue.append(generate_patient(self._rng, self.now))
            self.episode_stats["arrivals"] += 1
        return self._observation(), self._info()

    def step(self, action: int):
        assert self.action_space.contains(action)
        reward = 0.0
        slots = self._visible_slots()

        free_beds = self.num_beds - len(self.in_treatment)
        free_staff = self.num_staff - len(self.in_treatment)
        can_treat = free_beds > 0 and free_staff > 0

        if action < self.K:
            if action < len(slots) and can_treat:
                patient = slots[action]
                self.queue.remove(patient)
                patient.wait_time = self.now - patient.arrival_time
                end = self.now + patient.treatment_duration
                self.in_treatment.append((patient, end))
                target = max(TARGET_WAIT_MIN[patient.severity], 1.0)
                overshoot = max(0.0, patient.wait_time - target) / target
                reward += TREAT_REWARD[patient.severity] / (1.0 + 0.5 * overshoot)
            else:
                reward -= 1.0
                self.episode_stats["invalid_actions"] += 1
        else:
            if can_treat and len(self.queue) > 0:
                reward -= 0.5

        self.now += self.step_minutes

        still = []
        for patient, end in self.in_treatment:
            if end <= self.now:
                self.treated.append(patient)
                self.episode_stats["treated"] += 1
            else:
                still.append((patient, end))
        self.in_treatment = still

        lam = arrival_rate(self.now, self.base_arrival_rate) * self.step_minutes
        for _ in range(int(self._rng.poisson(lam))):
            if len(self.queue) < self.max_queue:
                self.queue.append(generate_patient(self._rng, self.now))
                self.episode_stats["arrivals"] += 1

        # Patient deterioration
        for p in self.queue:
            if p.severity <= 1:
                continue
            wait_h = (self.now - p.arrival_time) / 60.0
            hazard = (DETERIORATION_HAZARD_PER_MIN[p.severity]
                      * self.step_minutes
                      * min(wait_h, HAZARD_WAIT_CAP_H))
            if self._rng.random() < hazard:
                p.severity -= 1
                (p.heart_rate, p.systolic_bp, p.diastolic_bp,
                 p.spo2, p.temperature) = _vitals_for_severity(
                    self._rng, p.severity)
                p.treatment_duration = float(np.clip(
                    self._rng.exponential(TREATMENT_MEAN_MIN[p.severity]),
                    p.treatment_duration,
                    6 * TREATMENT_MEAN_MIN[p.severity]))
                self.episode_stats["deteriorations"] += 1
                reward -= DETERIORATION_PENALTY

        for p in self.queue:
            wait = self.now - p.arrival_time
            reward -= HOLD_COST[p.severity] * self.step_minutes
            target = max(TARGET_WAIT_MIN[p.severity], 5.0)
            if p.severity <= 2 and wait > 2 * target:
                reward -= CRITICAL_DELAY_PENALTY

        self.episode_stats["queue_length_sum"] += len(self.queue)
        self.episode_stats["steps"] += 1

        terminated = False
        truncated = self.now >= self.episode_minutes
        return self._observation(), reward, terminated, truncated, self._info()

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.K + 1, dtype=bool)
        free_beds = self.num_beds - len(self.in_treatment)
        free_staff = self.num_staff - len(self.in_treatment)
        if free_beds > 0 and free_staff > 0:
            mask[: len(self._visible_slots())] = True
        mask[self.K] = True
        return mask

    def _visible_slots(self) -> list[Patient]:
        ordered = sorted(self.queue, key=lambda p: p.arrival_time)
        return ordered[: self.K]

    def _observation(self) -> np.ndarray:
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        free_beds = self.num_beds - len(self.in_treatment)
        obs[0] = min(len(self.queue) / self.max_queue, 1.0)
        obs[1] = free_beds / self.num_beds
        obs[2] = (self.num_staff - len(self.in_treatment)) / self.num_staff
        obs[3] = (self.now % 1440) / 1440.0

        for i, p in enumerate(self._visible_slots()):
            base = 4 + i * 8
            wait = self.now - p.arrival_time
            obs[base + 0] = 1.0
            obs[base + 1] = (5 - p.severity) / 4.0
            obs[base + 2] = min(wait / 240.0, 1.0)
            obs[base + 3] = np.clip((p.heart_rate - 30) / 160.0, 0, 1)
            obs[base + 4] = np.clip((p.systolic_bp - 60) / 170.0, 0, 1)
            obs[base + 5] = np.clip((p.spo2 - 70) / 30.0, 0, 1)
            obs[base + 6] = np.clip((p.temperature - 34) / 7.5, 0, 1)
            obs[base + 7] = np.clip(p.age / 95.0, 0, 1)
        return obs

    def _info(self) -> dict:
        return {
            "time": self.now,
            "queue_length": len(self.queue),
            "in_treatment": len(self.in_treatment),
            "treated": len(self.treated),
            "stats": dict(self.episode_stats),
        }

    def episode_metrics(self) -> dict:
        waits = [p.wait_time for p in self.treated]
        crit = [p.wait_time for p in self.treated if p.severity <= 2]
        steps = max(self.episode_stats["steps"], 1)
        return {
            "throughput": len(self.treated),
            "arrivals": self.episode_stats["arrivals"],
            "avg_wait_min": float(np.mean(waits)) if waits else 0.0,
            "p90_wait_min": float(np.percentile(waits, 90)) if waits else 0.0,
            "avg_critical_wait_min": float(np.mean(crit)) if crit else 0.0,
            "avg_queue_length": self.episode_stats["queue_length_sum"] / steps,
            "invalid_actions": self.episode_stats["invalid_actions"],
            "still_waiting": len(self.queue),
            "deteriorations": self.episode_stats["deteriorations"],
        }
