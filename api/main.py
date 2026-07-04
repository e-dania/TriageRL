"""TriageRL FastAPI backend (Section 3.4.3, Backend API).

Run:  uvicorn api.main:app --reload

Endpoints:
    POST /patients          register a patient (auto severity + specialist)
    GET  /queue             waiting patients, most urgent first
    POST /prioritize        decide who to treat next (DQN if model exists,
                            severity rule otherwise)
    POST /patients/{id}/treated   mark treatment complete
    GET  /metrics           operational summary
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from triagerl.patients import SYMPTOM_SPECIALISTS, assess_severity

from . import db

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "dqn_triage.zip"
PPO_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "ppo_triage.zip"

NUM_BEDS = 5
NUM_STAFF = 7
QUEUE_SLOTS = 10
MAX_QUEUE = 60

app = FastAPI(title="TriageRL API", version="0.1.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_model_kind = None


def _load_model():
    """Lazy-load a trained agent. Prefers MaskablePPO, falls back to DQN.

    Returns (model, kind) where kind is "ppo", "dqn", or (None, None).
    """
    global _model, _model_kind
    if _model is None:
        if PPO_MODEL_PATH.exists():
            from sb3_contrib import MaskablePPO
            _model = MaskablePPO.load(str(PPO_MODEL_PATH))
            _model_kind = "ppo"
        elif MODEL_PATH.exists():
            from stable_baselines3 import DQN
            _model = DQN.load(str(MODEL_PATH))
            _model_kind = "dqn"
    return _model, _model_kind


class PatientIn(BaseModel):
    name: str | None = None
    age: int = Field(ge=0, le=120)
    gender: str = Field(pattern="^(M|F)$")
    symptom: str
    heart_rate: float = Field(ge=20, le=250)
    systolic_bp: float = Field(ge=40, le=300)
    diastolic_bp: float = Field(ge=20, le=200)
    temperature: float = Field(ge=30, le=45)
    spo2: float = Field(ge=50, le=100)


@app.post("/patients")
def register_patient(p: PatientIn):
    if p.symptom not in SYMPTOM_SPECIALISTS:
        raise HTTPException(400, f"Unknown symptom. Valid: {list(SYMPTOM_SPECIALISTS)}")
    severity = assess_severity(p.heart_rate, p.systolic_bp, p.spo2,
                               p.temperature, p.age)
    specialist = SYMPTOM_SPECIALISTS[p.symptom]
    data = p.model_dump() | {"severity": severity, "specialist": specialist}
    pid = db.insert_patient(data)
    return {"patient_id": pid, "severity": severity, "specialist": specialist}


@app.get("/queue")
def get_queue():
    patients = db.waiting_patients()
    now = time.time()
    for pt in patients:
        pt["wait_minutes"] = round((now - pt["arrival_ts"]) / 60.0, 1)
    patients.sort(key=lambda x: (x["severity"], -x["wait_minutes"]))
    return {"count": len(patients), "patients": patients}


def _observation(patients: list[dict]) -> np.ndarray:
    """Encode the live queue exactly as EDTriageEnv encodes its state."""
    obs = np.zeros(4 + 8 * QUEUE_SLOTS, dtype=np.float32)
    in_tx = db.metrics()["in_treatment"]
    now = time.time()
    obs[0] = min(len(patients) / MAX_QUEUE, 1.0)
    obs[1] = max(NUM_BEDS - in_tx, 0) / NUM_BEDS
    obs[2] = max(NUM_STAFF - in_tx, 0) / NUM_STAFF
    lt = time.localtime(now)
    obs[3] = (lt.tm_hour * 60 + lt.tm_min) / 1440.0

    ordered = sorted(patients, key=lambda x: x["arrival_ts"])[:QUEUE_SLOTS]
    for i, pt in enumerate(ordered):
        base = 4 + i * 8
        wait = (now - pt["arrival_ts"]) / 60.0
        obs[base + 0] = 1.0
        obs[base + 1] = (5 - pt["severity"]) / 4.0
        obs[base + 2] = min(wait / 240.0, 1.0)
        obs[base + 3] = np.clip((pt["heart_rate"] - 30) / 160.0, 0, 1)
        obs[base + 4] = np.clip((pt["systolic_bp"] - 60) / 170.0, 0, 1)
        obs[base + 5] = np.clip((pt["spo2"] - 70) / 30.0, 0, 1)
        obs[base + 6] = np.clip((pt["temperature"] - 34) / 7.5, 0, 1)
        obs[base + 7] = np.clip(pt["age"] / 95.0, 0, 1)
    return obs


@app.post("/prioritize")
def prioritize():
    patients = db.waiting_patients()
    if not patients:
        raise HTTPException(404, "No patients waiting")

    ordered = sorted(patients, key=lambda x: x["arrival_ts"])[:QUEUE_SLOTS]
    model, kind = _load_model()

    if model is not None:
        obs = _observation(patients)
        if kind == "ppo":
            mask = np.zeros(QUEUE_SLOTS + 1, dtype=bool)
            mask[: len(ordered)] = True
            mask[QUEUE_SLOTS] = True
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)
        idx = int(action)
        if idx >= len(ordered):          # defer/invalid -> severity fallback
            idx = min(range(len(ordered)), key=lambda i: ordered[i]["severity"])
            method = "severity_rule_fallback"
        else:
            method = kind
    else:
        idx = min(range(len(ordered)), key=lambda i: ordered[i]["severity"])
        method = "severity_rule"

    chosen = ordered[idx]
    db.mark_in_treatment(chosen["id"], method)
    return {
        "patient_id": chosen["id"],
        "name": chosen["name"],
        "severity": chosen["severity"],
        "specialist": chosen["specialist"],
        "method": method,
    }


@app.post("/patients/{patient_id}/treated")
def treated(patient_id: int):
    db.mark_treated(patient_id)
    return {"patient_id": patient_id, "status": "treated"}


@app.get("/metrics")
def get_metrics():
    return db.metrics()


@app.get("/symptoms")
def get_symptoms():
    return SYMPTOM_SPECIALISTS


@app.get("/decisions")
def get_decisions(limit: int = 20):
    return {"decisions": db.recent_decisions(limit)}


_DEMO_NAMES = (
    ["Amina", "Kwame", "Grace", "Thabo", "Zainab", "Emeka", "Fatima",
     "Kofi", "Adaeze", "Musa", "Nia", "Sipho", "Layla", "Obi", "Chidera"],
    ["Okafor", "Mensah", "Diallo", "Ndlovu", "Abubakar", "Banda", "Keita",
     "Mwangi", "Chukwu", "Toure", "Nkosi", "Adeyemi"],
)


@app.post("/demo/patients")
def demo_patient():
    """Register one synthetic patient (demo mode / seeding)."""
    from triagerl.patients import generate_patient
    rng = np.random.default_rng()
    p = generate_patient(rng, 0.0)
    data = {
        "name": f"{rng.choice(_DEMO_NAMES[0])} {rng.choice(_DEMO_NAMES[1])}",
        "age": p.age, "gender": p.gender, "symptom": p.symptom,
        "heart_rate": round(p.heart_rate, 1),
        "systolic_bp": round(p.systolic_bp, 1),
        "diastolic_bp": round(p.diastolic_bp, 1),
        "temperature": round(p.temperature, 1),
        "spo2": round(p.spo2, 1),
        "severity": p.severity, "specialist": p.specialist,
    }
    pid = db.insert_patient(data)
    return {"patient_id": pid, "name": data["name"],
            "severity": p.severity, "specialist": p.specialist}


@app.get("/model-info")
def model_info():
    _, kind = _load_model()
    return {"active_model": kind or "severity_rule",
            "label": {"ppo": "MaskablePPO agent", "dqn": "DQN agent"}.get(
                kind, "Severity rule (no model loaded)")}



_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _DIST.exists():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="dashboard")
