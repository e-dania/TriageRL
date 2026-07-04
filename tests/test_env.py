"""Unit + property tests: ED simulation environment.

Testing strategies: Gymnasium API conformance, invariant checking over long
random rollouts, action-mask correctness, deterioration mechanics, and
deterministic reproducibility (same seed => same trajectory).
"""

import numpy as np
import pytest

from triagerl.ed_env import EDTriageEnv


@pytest.fixture
def env():
    return EDTriageEnv(seed=123)


class TestGymnasiumConformance:
    def test_official_checker(self, env):
        from gymnasium.utils.env_checker import check_env
        check_env(env, skip_render_check=True)

    def test_observation_shape_and_bounds(self, env):
        obs, _ = env.reset(seed=1)
        assert obs.shape == (env.obs_dim,)
        assert obs.dtype == np.float32
        assert env.observation_space.contains(obs)


class TestInvariants:
    """Invariants must hold at every step of a random rollout."""

    def test_500_step_rollout(self):
        env = EDTriageEnv(seed=5)
        obs, _ = env.reset(seed=5)
        for _ in range(500):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs)
            assert len(env.in_treatment) <= env.num_beds
            assert len(env.queue) <= env.max_queue
            assert all(1 <= p.severity <= 5 for p in env.queue)
            if term or trunc:
                obs, _ = env.reset()

    def test_reproducibility(self):
        """Identical seeds must produce identical trajectories."""
        rewards = []
        for _ in range(2):
            env = EDTriageEnv(seed=99)
            env.reset(seed=99)
            total, done = 0.0, False
            while not done:
                _, r, term, trunc, _ = env.step(0)
                total += r
                done = term or trunc
            rewards.append(total)
        assert rewards[0] == rewards[1]


class TestActionMasking:
    def test_mask_shape(self, env):
        env.reset(seed=2)
        mask = env.action_masks()
        assert mask.shape == (env.K + 1,)
        assert mask[env.K]  # defer always valid

    def test_masked_actions_never_invalid(self):
        env = EDTriageEnv(seed=7)
        env.reset(seed=7)
        rng = np.random.default_rng(0)
        done = False
        while not done:
            valid = np.flatnonzero(env.action_masks())
            _, _, term, trunc, _ = env.step(int(rng.choice(valid)))
            done = term or trunc
        assert env.episode_metrics()["invalid_actions"] == 0

    def test_no_capacity_masks_all_slots(self):
        env = EDTriageEnv(seed=3, num_beds=1, num_staff=1)
        env.reset(seed=3)
        # Fill the single bed
        env.step(0)
        if len(env.in_treatment) == 1:
            mask = env.action_masks()
            assert not mask[: env.K].any()
            assert mask[env.K]


class TestDeterioration:
    def test_deterioration_occurs_under_congestion(self):
        """With heavy load and slow service, some patients must worsen."""
        env = EDTriageEnv(seed=11, num_beds=1, num_staff=1,
                          base_arrival_rate=0.3)
        env.reset(seed=11)
        done = False
        while not done:
            _, _, term, trunc, _ = env.step(env.K)  # always defer
            done = term or trunc
        assert env.episode_metrics()["deteriorations"] > 0

    def test_severity_one_never_worsens(self):
        env = EDTriageEnv(seed=13, base_arrival_rate=0.3)
        env.reset(seed=13)
        for _ in range(200):
            _, _, term, trunc, _ = env.step(env.K)
            assert all(p.severity >= 1 for p in env.queue)
            if term or trunc:
                break


class TestMetrics:
    def test_episode_metrics_keys(self, env):
        env.reset(seed=4)
        done = False
        while not done:
            _, _, term, trunc, _ = env.step(0)
            done = term or trunc
        m = env.episode_metrics()
        for key in ("throughput", "avg_wait_min", "avg_critical_wait_min",
                    "deteriorations", "avg_queue_length"):
            assert key in m

    def test_throughput_conservation(self, env):
        """treated + waiting + in_treatment == arrivals at episode end."""
        env.reset(seed=6)
        done = False
        while not done:
            _, _, term, trunc, _ = env.step(0)
            done = term or trunc
        total = (len(env.treated) + len(env.queue) + len(env.in_treatment))
        assert total == env.episode_stats["arrivals"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
