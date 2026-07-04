"""Train a Maskable PPO prioritization agent (action-masked alternative
to the DQN in train.py).

Action masking means the agent can only select occupied queue slots while
capacity is available — invalid actions are impossible rather than merely
penalized, which removes ~1/3 of wasted decisions observed with DQN.

Usage:
    python -m triagerl.train_ppo --timesteps 300000 --out models/ppo_triage
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

from .ed_env import EDTriageEnv


def mask_fn(env: EDTriageEnv):
    return env.action_masks()


def make_env(seed: int | None = None) -> Monitor:
    return Monitor(ActionMasker(EDTriageEnv(seed=seed), mask_fn))


def train(timesteps: int, out: str, seed: int = 42, log_dir: str = "logs"):
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    env = make_env(seed)
    eval_env = make_env(seed + 1)

    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=log_dir,
        verbose=1,
        seed=seed,
    )

    eval_cb = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(out_path.parent / "ppo_best"),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    model.learn(total_timesteps=timesteps, callback=eval_cb, progress_bar=True)
    model.save(str(out_path))
    print(f"Saved model to {out_path}.zip")
    print(f"Best checkpoint in {out_path.parent / 'ppo_best'}/best_model.zip")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TriageRL MaskablePPO agent")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--out", type=str, default="models/ppo_triage")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.timesteps, args.out, args.seed)
