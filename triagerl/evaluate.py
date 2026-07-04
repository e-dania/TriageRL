"""Evaluate trained agents (DQN, MaskablePPO) against rule-based baselines.

Statistical analysis:
    - 95% confidence intervals (t-distribution) for every metric
    - paired Wilcoxon signed-rank tests: each trained agent vs each baseline,
      paired by episode seed (all policies see identical arrival streams)

Usage:
    python -m triagerl.evaluate --model models/dqn_triage --episodes 30
    python -m triagerl.evaluate --ppo models/ppo_triage --episodes 30
    python -m triagerl.evaluate --model models/dqn_triage --ppo models/ppo_triage
    python -m triagerl.evaluate --episodes 30          # baselines only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .baselines import BASELINES
from .ed_env import EDTriageEnv

KEY_METRICS = [
    "avg_wait_min",
    "p90_wait_min",
    "avg_critical_wait_min",
    "throughput",
    "avg_queue_length",
    "deteriorations",
    "prioritization_accuracy",
]

AGENT_NAMES = ("DQN", "MaskablePPO")


def _urgency_rank(env: EDTriageEnv):
    slots = env._visible_slots()
    if not slots:
        return None
    return min(range(len(slots)), key=lambda i: (slots[i].severity, i))


def run_episodes(policy_fn, episodes: int, seed0: int = 1000) -> pd.DataFrame:
    rows = []
    for ep in range(episodes):
        env = EDTriageEnv(seed=seed0 + ep)
        env.reset(seed=seed0 + ep)
        done = False
        correct, decisions = 0, 0
        while not done:
            best = _urgency_rank(env)
            action = policy_fn(env)
            if action < env.K and best is not None:
                decisions += 1
                slots = env._visible_slots()
                if action < len(slots) and slots[action].severity == slots[best].severity:
                    correct += 1
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        m = env.episode_metrics()
        m["prioritization_accuracy"] = correct / decisions if decisions else 0.0
        m["episode"] = ep
        rows.append(m)
    return pd.DataFrame(rows)


def dqn_policy_factory(model_path: str):
    from stable_baselines3 import DQN

    model = DQN.load(model_path)

    def policy(env: EDTriageEnv) -> int:
        action, _ = model.predict(env._observation(), deterministic=True)
        return int(action)

    return policy


def ppo_policy_factory(model_path: str):
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(model_path)

    def policy(env: EDTriageEnv) -> int:
        action, _ = model.predict(
            env._observation(),
            action_masks=env.action_masks(),
            deterministic=True,
        )
        return int(action)

    return policy


def mean_ci(series: pd.Series, confidence: float = 0.95):
    n = len(series)
    mean = float(series.mean())
    if n < 2:
        return mean, 0.0
    sem = stats.sem(series)
    half = float(sem * stats.t.ppf((1 + confidence) / 2, n - 1))
    return mean, half


def build_stats_table(episode_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for policy, df in episode_dfs.items():
        row = {"policy": policy}
        for metric in KEY_METRICS:
            mean, half = mean_ci(df[metric])
            row[f"{metric}_mean"] = round(mean, 2)
            row[f"{metric}_ci95"] = round(half, 2)
        rows.append(row)
    return pd.DataFrame(rows).set_index("policy")


def build_significance_table(episode_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Paired Wilcoxon tests: every trained agent vs every baseline."""
    rows = []
    treatments = [n for n in AGENT_NAMES if n in episode_dfs]
    baselines = [n for n in episode_dfs if n not in AGENT_NAMES]
    for treatment in treatments:
        t_df = episode_dfs[treatment].sort_values("episode")
        for baseline in baselines:
            b_df = episode_dfs[baseline].sort_values("episode")
            for metric in KEY_METRICS:
                x = t_df[metric].to_numpy()
                y = b_df[metric].to_numpy()
                if np.allclose(x - y, 0):
                    stat, p = np.nan, 1.0
                else:
                    stat, p = stats.wilcoxon(x, y)
                base_mean = float(np.mean(y))
                change = ((float(np.mean(x)) - base_mean) / base_mean * 100
                          if base_mean else np.nan)
                rows.append({
                    "comparison": f"{treatment} vs {baseline}",
                    "metric": metric,
                    "treatment_mean": round(float(np.mean(x)), 2),
                    "baseline_mean": round(base_mean, 2),
                    "pct_change": round(change, 1),
                    "wilcoxon_stat": None if np.isnan(stat) else round(float(stat), 1),
                    "p_value": round(float(p), 5),
                    "significant_p05": bool(p < 0.05),
                })
    return pd.DataFrame(rows)


def main(model: str | None, episodes: int, out_dir: str,
         ppo_model: str | None = None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    policies = dict(BASELINES)
    if model:
        policies["DQN"] = dqn_policy_factory(model)
    if ppo_model:
        policies["MaskablePPO"] = ppo_policy_factory(ppo_model)

    episode_dfs = {}
    for name, fn in policies.items():
        print(f"Evaluating {name} over {episodes} episodes...")
        df = run_episodes(fn, episodes)
        df.to_csv(out / f"episodes_{name}.csv", index=False)
        episode_dfs[name] = df

    summary = pd.DataFrame(
        {name: df.mean(numeric_only=True) for name, df in episode_dfs.items()}
    ).T.drop(columns=["episode"])
    summary.index.name = "policy"
    summary.to_csv(out / "summary.csv")

    stats_table = build_stats_table(episode_dfs)
    stats_table.to_csv(out / "summary_ci.csv")

    print("\n=== Mean +/- 95% CI ===")
    for metric in KEY_METRICS:
        print(f"\n{metric}:")
        for policy in stats_table.index:
            m = stats_table.loc[policy, f"{metric}_mean"]
            h = stats_table.loc[policy, f"{metric}_ci95"]
            print(f"  {policy:14s} {m:8.2f} +/- {h:.2f}")

    sig = build_significance_table(episode_dfs)
    if not sig.empty:
        sig.to_csv(out / "significance.csv", index=False)
        print("\n=== Paired Wilcoxon signed-rank tests ===")
        print(sig.to_string(index=False))

    _plot(episode_dfs, out)
    print(f"\nResults written to {out}/")
    return summary


def _plot(episode_dfs: dict[str, pd.DataFrame], out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("avg_wait_min", "Average wait (min)"),
        ("avg_critical_wait_min", "Avg critical wait, sev 1-2 (min)"),
        ("p90_wait_min", "P90 wait (min)"),
        ("throughput", "Throughput (patients/episode)"),
        ("avg_queue_length", "Average queue length"),
        ("deteriorations", "Deteriorations (per episode)"),
    ]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    names = list(episode_dfs)
    for ax, (col, title) in zip(axes.flat, panels):
        means, halves = [], []
        for name in names:
            m, h = mean_ci(episode_dfs[name][col])
            means.append(m)
            halves.append(h)
        ax.bar(names, means, yerr=halves, capsize=5, color=colors[: len(names)])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        for i, v in enumerate(means):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("TriageRL: policy comparison (error bars = 95% CI)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TriageRL policies")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained DQN .zip")
    parser.add_argument("--ppo", type=str, default=None,
                        help="Path to trained MaskablePPO .zip")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()
    main(args.model, args.episodes, args.out, ppo_model=args.ppo)
