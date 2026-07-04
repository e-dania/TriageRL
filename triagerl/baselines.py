"""Rule-based baseline triage policies (Section 1.3, Objective 3).

Each policy maps the environment's visible queue slots to an action index,
using exactly the same action interface as the DQN agent so comparisons are
fair.
"""

from __future__ import annotations

import numpy as np

from .ed_env import EDTriageEnv


def fifo_policy(env: EDTriageEnv) -> int:
    """First-in, first-out: always treat the longest-waiting patient."""
    slots = env._visible_slots()
    return 0 if slots else env.K


def severity_policy(env: EDTriageEnv) -> int:
    """ESI-style strict severity priority; ties broken by waiting time.

    This is the standard rule-based triage baseline the proposal compares
    against.
    """
    slots = env._visible_slots()
    if not slots:
        return env.K
    # Slots are ordered oldest-first, so the first index with the minimum
    # severity number is the longest-waiting most-critical patient.
    best = min(range(len(slots)), key=lambda i: slots[i].severity)
    return best


def random_policy(env: EDTriageEnv, rng: np.random.Generator | None = None) -> int:
    """Uniformly random choice among occupied slots (lower bound)."""
    rng = rng or np.random.default_rng()
    slots = env._visible_slots()
    if not slots:
        return env.K
    return int(rng.integers(0, len(slots)))


BASELINES = {
    "FIFO": fifo_policy,
    "SeverityRule": severity_policy,
    "Random": random_policy,
}
