"""Synthetic patient generation and rule-based severity assessment.

Patients are generated with clinically plausible vital signs conditioned on an
underlying acuity level, mirroring the dataset variables defined in the
proposal (Section 3.5.3): demographics, clinical indicators, and operational
variables.

Severity levels follow an ESI-like 5-level scale:
    1 = Resuscitation (most critical)
    2 = Emergent
    3 = Urgent
    4 = Less urgent
    5 = Non-urgent
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

# Symptom categories and the specialist each maps to (Specialist
# Recommendation Module, Section 3.4.3).
SYMPTOM_SPECIALISTS = {
    "chest_pain": "Cardiologist",
    "shortness_of_breath": "Pulmonologist",
    "stroke_symptoms": "Neurologist",
    "severe_trauma": "Trauma Surgeon",
    "abdominal_pain": "General Surgeon",
    "fracture": "Orthopedic Specialist",
    "high_fever_child": "Pediatrician",
    "allergic_reaction": "Emergency Physician",
    "headache": "Neurologist",
    "laceration": "Emergency Physician",
    "back_pain": "Orthopedic Specialist",
    "flu_symptoms": "General Practitioner",
}

SYMPTOMS = list(SYMPTOM_SPECIALISTS)

# Probability a given symptom presents at each acuity level (rows sum to 1).
# Index 0 -> severity 1 (critical) ... index 4 -> severity 5 (non-urgent).
SYMPTOM_ACUITY_DIST = {
    "chest_pain":          [0.15, 0.35, 0.30, 0.15, 0.05],
    "shortness_of_breath": [0.15, 0.30, 0.30, 0.20, 0.05],
    "stroke_symptoms":     [0.30, 0.45, 0.20, 0.05, 0.00],
    "severe_trauma":       [0.35, 0.40, 0.20, 0.05, 0.00],
    "abdominal_pain":      [0.03, 0.12, 0.35, 0.35, 0.15],
    "fracture":            [0.02, 0.08, 0.30, 0.40, 0.20],
    "high_fever_child":    [0.05, 0.20, 0.40, 0.25, 0.10],
    "allergic_reaction":   [0.10, 0.25, 0.30, 0.25, 0.10],
    "headache":            [0.02, 0.08, 0.25, 0.40, 0.25],
    "laceration":          [0.01, 0.05, 0.20, 0.40, 0.34],
    "back_pain":           [0.01, 0.04, 0.20, 0.40, 0.35],
    "flu_symptoms":        [0.01, 0.04, 0.15, 0.40, 0.40],
}

# Relative frequency of each presenting symptom in the ED.
SYMPTOM_FREQ = np.array([0.10, 0.09, 0.04, 0.05, 0.14, 0.08,
                         0.07, 0.05, 0.10, 0.10, 0.08, 0.10])
SYMPTOM_FREQ = SYMPTOM_FREQ / SYMPTOM_FREQ.sum()

# Mean treatment duration in minutes by severity (1..5). Critical patients
# occupy resources longer.
TREATMENT_MEAN_MIN = {1: 120, 2: 90, 3: 60, 4: 35, 5: 20}

_id_counter = itertools.count(1)


@dataclass
class Patient:
    """A single ED patient with demographics, vitals, and queue state."""

    patient_id: int
    age: int
    gender: str
    symptom: str
    heart_rate: float
    systolic_bp: float
    diastolic_bp: float
    temperature: float
    spo2: float
    severity: int                 # 1 (critical) .. 5 (non-urgent)
    arrival_time: float           # simulation minutes
    treatment_duration: float     # minutes
    wait_time: float = 0.0        # updated by the environment
    specialist: str = field(default="")

    def __post_init__(self):
        if not self.specialist:
            self.specialist = SYMPTOM_SPECIALISTS.get(
                self.symptom, "Emergency Physician")


def assess_severity(heart_rate: float, systolic_bp: float, spo2: float,
                    temperature: float, age: int) -> int:
    """Rule-based severity assessment from vital signs.

    Implements the Severity Assessment Module (Section 3.4.3): a transparent
    scoring scheme loosely modeled on early-warning scores (NEWS2-style
    thresholds). Returns an ESI-like level 1..5.
    """
    score = 0

    # Heart rate
    if heart_rate <= 40 or heart_rate >= 131:
        score += 3
    elif 111 <= heart_rate <= 130:
        score += 2
    elif 91 <= heart_rate <= 110 or 41 <= heart_rate <= 50:
        score += 1

    # Systolic blood pressure
    if systolic_bp <= 90 or systolic_bp >= 220:
        score += 3
    elif 91 <= systolic_bp <= 100:
        score += 2
    elif 101 <= systolic_bp <= 110:
        score += 1

    # Oxygen saturation
    if spo2 <= 91:
        score += 3
    elif 92 <= spo2 <= 93:
        score += 2
    elif 94 <= spo2 <= 95:
        score += 1

    # Temperature
    if temperature <= 35.0:
        score += 3
    elif temperature >= 39.1:
        score += 2
    elif 38.1 <= temperature <= 39.0 or 35.1 <= temperature <= 36.0:
        score += 1

    # Age adjustment: very young and very old escalate risk
    if age >= 75 or age <= 2:
        score += 1

    if score >= 7:
        return 1
    if score >= 5:
        return 2
    if score >= 3:
        return 3
    if score >= 1:
        return 4
    return 5


def _vitals_for_severity(rng: np.random.Generator, severity: int):
    """Sample plausible vitals given an underlying acuity level."""
    if severity == 1:
        hr = rng.normal(135, 15)
        sbp = rng.normal(85, 12)
        spo2 = rng.normal(87, 3)
        temp = rng.normal(38.5, 1.2)
    elif severity == 2:
        hr = rng.normal(115, 12)
        sbp = rng.normal(100, 12)
        spo2 = rng.normal(92, 2)
        temp = rng.normal(38.2, 1.0)
    elif severity == 3:
        hr = rng.normal(95, 10)
        sbp = rng.normal(115, 12)
        spo2 = rng.normal(95, 1.5)
        temp = rng.normal(37.8, 0.8)
    elif severity == 4:
        hr = rng.normal(85, 8)
        sbp = rng.normal(122, 10)
        spo2 = rng.normal(97, 1.0)
        temp = rng.normal(37.2, 0.5)
    else:
        hr = rng.normal(75, 8)
        sbp = rng.normal(120, 8)
        spo2 = rng.normal(98, 0.8)
        temp = rng.normal(36.8, 0.3)

    hr = float(np.clip(hr, 30, 190))
    sbp = float(np.clip(sbp, 60, 230))
    dbp = float(np.clip(sbp * rng.normal(0.65, 0.05), 40, 130))
    spo2 = float(np.clip(spo2, 70, 100))
    temp = float(np.clip(temp, 34.0, 41.5))
    return hr, sbp, dbp, spo2, temp


def generate_patient(rng: np.random.Generator, arrival_time: float) -> Patient:
    """Generate one synthetic patient arriving at `arrival_time` (minutes)."""
    symptom = SYMPTOMS[rng.choice(len(SYMPTOMS), p=SYMPTOM_FREQ)]

    if symptom == "high_fever_child":
        age = int(rng.integers(1, 13))
    else:
        age = int(np.clip(rng.normal(42, 22), 1, 95))
    gender = "M" if rng.random() < 0.5 else "F"

    # Latent acuity drawn from the symptom's acuity distribution, then vitals
    # sampled consistently with it. Recorded severity comes from the
    # rule-based assessment of those vitals (as the deployed system would).
    latent = int(rng.choice(5, p=SYMPTOM_ACUITY_DIST[symptom])) + 1
    hr, sbp, dbp, spo2, temp = _vitals_for_severity(rng, latent)
    severity = assess_severity(hr, sbp, spo2, temp, age)

    duration = float(np.clip(
        rng.exponential(TREATMENT_MEAN_MIN[severity]),
        10, 6 * TREATMENT_MEAN_MIN[severity]))

    return Patient(
        patient_id=next(_id_counter),
        age=age,
        gender=gender,
        symptom=symptom,
        heart_rate=hr,
        systolic_bp=sbp,
        diastolic_bp=dbp,
        temperature=temp,
        spo2=spo2,
        severity=severity,
        arrival_time=arrival_time,
        treatment_duration=duration,
    )


def arrival_rate(minute_of_day: float, base_rate: float) -> float:
    """Time-varying Poisson arrival rate (patients per minute).

    Models the daily ED demand curve: quiet overnight, morning ramp-up,
    evening peak.
    """
    hour = (minute_of_day / 60.0) % 24
    # Smooth double-peaked profile normalized around 1.0
    profile = (0.55
               + 0.45 * np.exp(-((hour - 11) ** 2) / 18)
               + 0.55 * np.exp(-((hour - 19) ** 2) / 10))
    return base_rate * float(profile)
