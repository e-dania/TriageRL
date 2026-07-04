# TriageRL — Reinforcement Learning for Emergency Department Triage

An adaptive patient-prioritization system for resource-constrained emergency
departments. A **MaskablePPO** reinforcement learning agent, trained in a
simulated ED with stochastic arrivals, limited beds/staff, and patient
deterioration, decides which waiting patient should be treated next — and
outperforms traditional rule-based triage on every efficiency metric.

**Demo video:** [Click Here](https://drive.google.com/file/d/1xKa9FgESG11B0YHXsIxKq-Z9JYoKZOs_/view?usp=sharing)
**Live deployment:** [Click Here](https://triagerl.onrender.com/)

BSc. Software Engineering Capstone — Emmanuel Dania (Supervisor: Thadee Gatera  & Junior Turatsinze)

---

## Key results

Evaluated over 30 paired episodes (identical patient arrival streams per
policy), Wilcoxon signed-rank tests, in the deterioration-enabled environment:

| Metric | Severity rule (ESI-style) | **TriageRL (MaskablePPO)** | Change | p-value |
|---|---|---|---|---|
| Average wait | 179.5 min | **116.7 min** | **-35.0%** | < 0.0001 |
| P90 wait | 410.8 min | **265.3 min** | **-35.4%** | < 0.0001 |
| Critical wait (sev 1-2) | 124.8 min | **106.7 min** | -14.5% | 0.07 |
| Throughput | 119.0 /day | **128.0 /day** | **+7.6%** | 0.021 |
| Deteriorations | 15.1 /day | **13.5 /day** | -10.6% | n.s. |

The proposal targeted a >=15% wait reduction; the delivered system achieves
**35%**, with statistically significant gains in throughput and queue length,
while matching strict severity-based triage on critical-patient waiting.

## Architecture

```
+-------------+   REST    +--------------+   predict   +------------------+
|  React SPA  | --------> |  FastAPI     | ----------> | MaskablePPO      |
|  (Vite)     | <-------- |  backend     | <---------- | agent (SB3)      |
+-------------+           |  + SQLite    |             +------------------+
                          +--------------+                    ^ trained in
                                                       +------------------+
                                                       | Gymnasium ED     |
                                                       | simulation       |
                                                       +------------------+
```

- **Simulation** (`triagerl/ed_env.py`): 24h episodes, 5-min steps,
  time-varying Poisson arrivals, 5 beds / 7 staff, severity-dependent
  treatment times, and probabilistic patient deterioration under long waits.
- **Severity assessment** (`triagerl/patients.py`): transparent NEWS2-style
  vital-sign scoring mapped to ESI-like levels 1-5.
- **Agents** (`triagerl/train.py`, `triagerl/train_ppo.py`): DQN baseline and
  action-masked PPO (final model), Stable-Baselines3 / sb3-contrib.
- **Evaluation** (`triagerl/evaluate.py`): 5 policies x paired episodes, 95%
  CIs, Wilcoxon significance tests, comparison plots.
- **Backend** (`api/`): patient registration with automatic severity
  assessment and specialist recommendation, queue, RL-driven prioritization
  with rule-based fallback, decision audit log, metrics.
- **Frontend** (`frontend/`): live queue board, stat cards, decision log,
  registration form, demo mode.

## Project structure

```
triagerl/
|-- triagerl/          # core ML package (env, agents, evaluation)
|-- api/               # FastAPI backend + SQLite data layer
|-- frontend/          # React dashboard (Vite); dist/ is pre-built
|-- tests/             # pytest suite: unit, integration, edge cases
|-- scripts/           # benchmark.py (performance evidence)
|-- train_colab.ipynb  # Colab training notebook
|-- Dockerfile         # multi-stage build (Node -> Python), single container
`-- requirements.txt
```

## Installation & running (step by step)

Prerequisites: Python 3.10+ (https://python.org). Node.js is **not**
required — a pre-built dashboard is included.

```bash
# 1. Clone
git clone https://github.com/e-dania/TriageRL.git
cd TriageRL

# 2. Virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Dependencies (~5 min; PyTorch is large)
pip install -r requirements.txt

# 4. Run (the trained model in models/ is picked up automatically)
uvicorn api.main:app
```

Open **http://localhost:8000** — dashboard. API docs: http://localhost:8000/docs.

Quick tour: click **Demo mode** on the queue board to stream synthetic
arrivals and watch the agent prioritize automatically, or register patients
manually and click **Who's next?**.

### Docker

```bash
docker compose up --build     # serves everything on :8000
```

## Training (Google Colab)

Open `train_colab.ipynb` in Colab, upload the project zip, and run top to
bottom (~30-60 min per agent). Download the trained model into `models/` —
the API prefers `models/ppo_triage.zip`, falling back to
`models/dqn_triage.zip`, then to the severity rule.

Full evaluation with statistics:

```bash
python -m triagerl.evaluate --model models/dqn_triage --ppo models/ppo_triage --episodes 30
```

Outputs `results/summary_ci.csv`, `results/significance.csv`,
`results/comparison.png`.

## Testing

Four strategies, all runnable in one command:

```bash
pytest tests/ -v
```

| Strategy | Where | What it covers |
|---|---|---|
| Unit / boundary-value | `tests/test_severity.py` | Every vital-sign threshold in severity scoring; equivalence classes; full input-space sweep |
| Property / invariant | `tests/test_env.py` | Gymnasium API conformance; 500-step invariant rollouts; seed reproducibility; mask correctness; deterioration mechanics |
| Integration / edge-case | `tests/test_api.py` | Full patient lifecycle; critical-vs-mild prioritization with different data values; 10+ invalid-input rejections; empty-queue and extreme-but-valid boundaries |
| Statistical / simulation | `triagerl/evaluate.py` | 5 policies x 30 paired episodes; 95% CIs; Wilcoxon signed-rank tests |

### Performance across environments

```bash
python scripts/benchmark.py --model models/ppo_triage
```

| Environment | Simulation (steps/s) | Inference mean (ms) | API /prioritize (ms) |
|---|---|---|---|
| Laptop — Windows 11, AMD64, 8 cores, Python 3.11 | 1,492 | 1.45 (p95 2.05) | 26.9 |
| Google Colab — Linux, 2 vCPU, Python 3.12 | 2,578 | 0.79 (p95 0.72) | 18.7 |
| Linux container — 2 vCPU (Docker-equivalent) | 2,996 | — | 12.6 |
| Render deployment — free tier | — | — | verified live via /docs |

All 41 tests pass in every environment (Windows laptop, Colab Linux, container).

## Deployment plan (Render)

1. Push this repo to GitHub (public).
2. Ensure `models/ppo_triage.zip` is committed (the deployed app needs it).
3. https://render.com -> New -> **Web Service** -> connect the repo.
4. Environment: **Docker** (Render auto-detects the Dockerfile). Instance: Free.
5. Deploy. Render builds the frontend and API into one container and exposes
   it at `https://<your-app>.onrender.com`.
6. Verify: open the URL, enable demo mode, confirm decisions show
   `PPO agent`, and check `/model-info` and `/docs`.

Note: the free tier sleeps after inactivity — the first request after idle
takes about a minute.

## Analysis (vs. proposal objectives)

1. **Reduce waiting times >=15%** — EXCEEDED: -35.0% average wait vs
   rule-based triage (p < 0.0001), -35.4% at P90.
2. **Improve critical-patient prioritization** — ACHIEVED vs FIFO (-22.9%,
   p < 0.01); statistically on par with the strict severity rule (which
   optimizes only this metric) while beating it everywhere else.
3. **Increase throughput** — ACHIEVED: +7.6% vs severity rule (p = 0.021).

Notable findings: (a) action masking was worth more than reward tuning — a
2.5x reward-weight increase *degraded* DQN performance, while masked PPO with
unchanged rewards dominated; (b) under deterioration dynamics, severity-only
triage actively manufactures critical patients by starving mild ones (most
deteriorations of any policy); (c) FIFO minimizes deteriorations but at the
cost of the worst critical waits — the agent learns a middle path.

## Recommendations & future work

- Multi-hospital transfer: retrain per-site with local arrival patterns and
  resource levels (all configurable in `ed_env.py`).
- Extend the action space to resource assignment (which bed/clinician), not
  only ordering.
- Human-in-the-loop trial: measure clinician agreement with agent decisions.
- Multi-seed training and larger-scale sensitivity sweeps for robustness.

## License / attribution

Built with Gymnasium, Stable-Baselines3, sb3-contrib, FastAPI, React, and
Vite. Simulation only — not for clinical use.
