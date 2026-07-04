"""Train the DQN prioritization agent (Section 3.6).

Usage:
    python -m triagerl.train --timesteps 300000 --out models/dqn_triage

Designed to run on CPU in under an hour for the default problem size;
increase --timesteps on Colab for better policies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from .ed_env import EDTriageEnv


def make_env(seed: int | None = None) -> Monitor:
    return Monitor(EDTriageEnv(seed=seed))


def train(timesteps: int, out: str, seed: int = 42, log_dir: str = "logs"):
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    env = make_env(seed)
    eval_env = make_env(seed + 1)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=2_000,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=log_dir,
        verbose=1,
        seed=seed,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(out_path.parent),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    model.learn(total_timesteps=timesteps, callback=eval_cb, progress_bar=True)
    model.save(str(out_path))
    print(f"Saved model to {out_path}.zip")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TriageRL DQN agent")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--out", type=str, default="models/dqn_triage")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.timesteps, args.out, args.seed)
