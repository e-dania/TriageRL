"""Performance benchmark — run on each machine/environment to produce the
hardware/software comparison evidence for testing documentation.

Usage:
    python scripts/benchmark.py                  
    python scripts/benchmark.py --model models/ppo_triage  
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from triagerl.ed_env import EDTriageEnv  # noqa: E402


def bench_environment(episodes: int = 5) -> dict:
    env = EDTriageEnv(seed=0)
    steps = 0
    t0 = time.perf_counter()
    for ep in range(episodes):
        env.reset(seed=ep)
        done = False
        while not done:
            valid = np.flatnonzero(env.action_masks())
            _, _, term, trunc, _ = env.step(int(valid[0]))
            steps += 1
            done = term or trunc
    dt = time.perf_counter() - t0
    return {"env_steps_per_sec": steps / dt, "episodes": episodes,
            "total_steps": steps, "wall_seconds": dt}


def bench_inference(model_path: str, n: int = 500) -> dict:
    try:
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(model_path)
        masked = True
    except Exception:
        from stable_baselines3 import DQN
        model = DQN.load(model_path)
        masked = False

    env = EDTriageEnv(seed=0)
    env.reset(seed=0)
    obs = env._observation()
    mask = env.action_masks()

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        if masked:
            model.predict(obs, action_masks=mask, deterministic=True)
        else:
            model.predict(obs, deterministic=True)
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "inference_mean_ms": statistics.mean(times),
        "inference_p95_ms": sorted(times)[int(0.95 * n)],
        "samples": n,
    }


def bench_api(n: int = 50) -> dict:
    import os
    import tempfile
    os.environ.setdefault("TRIAGERL_DB",
                          os.path.join(tempfile.gettempdir(), "bench.db"))
    from fastapi.testclient import TestClient

    from api.main import app
    client = TestClient(app)

    body = {"age": 45, "gender": "M", "symptom": "chest_pain",
            "heart_rate": 88, "systolic_bp": 125, "diastolic_bp": 82,
            "temperature": 37.1, "spo2": 97}

    reg_times, pri_times = [], []
    for _ in range(n):
        t0 = time.perf_counter()
        client.post("/patients", json=body)
        reg_times.append((time.perf_counter() - t0) * 1000)
    for _ in range(n):
        t0 = time.perf_counter()
        client.post("/prioritize")
        pri_times.append((time.perf_counter() - t0) * 1000)
    return {
        "register_mean_ms": statistics.mean(reg_times),
        "prioritize_mean_ms": statistics.mean(pri_times),
        "requests_each": n,
    }


def main(model: str | None):
    print("=" * 60)
    print("TriageRL performance benchmark")
    print("=" * 60)
    print(f"Python   : {platform.python_version()}")
    print(f"Platform : {platform.platform()}")
    print(f"Machine  : {platform.machine()}, "
          f"CPU count: {__import__('os').cpu_count()}")
    try:
        import torch
        dev = "CUDA " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        print(f"PyTorch  : {torch.__version__} ({dev})")
    except ImportError:
        print("PyTorch  : not installed")
    print("-" * 60)

    r = bench_environment()
    print(f"Simulation throughput : {r['env_steps_per_sec']:,.0f} steps/sec "
          f"({r['total_steps']} steps in {r['wall_seconds']:.2f}s)")

    if model:
        r = bench_inference(model)
        print(f"Policy inference      : {r['inference_mean_ms']:.2f} ms mean, "
              f"{r['inference_p95_ms']:.2f} ms p95  (n={r['samples']})")

    r = bench_api()
    print(f"API /patients         : {r['register_mean_ms']:.1f} ms mean")
    print(f"API /prioritize       : {r['prioritize_mean_ms']:.1f} ms mean")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Path to a trained model .zip for inference bench")
    args = parser.parse_args()
    main(args.model)
