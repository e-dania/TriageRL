"""SQLite data layer (Section 3.4.3, Data Layer)."""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(os.environ.get(
    "TRIAGERL_DB",
    Path(__file__).resolve().parent.parent / "triagerl.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    age INTEGER NOT NULL,
    gender TEXT NOT NULL,
    symptom TEXT NOT NULL,
    heart_rate REAL NOT NULL,
    systolic_bp REAL NOT NULL,
    diastolic_bp REAL NOT NULL,
    temperature REAL NOT NULL,
    spo2 REAL NOT NULL,
    severity INTEGER NOT NULL,
    specialist TEXT NOT NULL,
    arrival_ts REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'waiting',   -- waiting | in_treatment | treated
    treated_ts REAL,
    wait_minutes REAL
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    method TEXT NOT NULL,                     -- dqn | severity_rule
    decided_ts REAL NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patients (id)
);
"""


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def insert_patient(data: dict) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO patients
               (name, age, gender, symptom, heart_rate, systolic_bp,
                diastolic_bp, temperature, spo2, severity, specialist,
                arrival_ts)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (data.get("name"), data["age"], data["gender"], data["symptom"],
             data["heart_rate"], data["systolic_bp"], data["diastolic_bp"],
             data["temperature"], data["spo2"], data["severity"],
             data["specialist"], time.time()),
        )
        return cur.lastrowid


def waiting_patients() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM patients WHERE status='waiting' ORDER BY arrival_ts"
        ).fetchall()
    return [dict(r) for r in rows]


def mark_in_treatment(patient_id: int, method: str):
    now = time.time()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT arrival_ts FROM patients WHERE id=?", (patient_id,)
        ).fetchone()
        wait_min = (now - row["arrival_ts"]) / 60.0 if row else None
        conn.execute(
            "UPDATE patients SET status='in_treatment', wait_minutes=? WHERE id=?",
            (wait_min, patient_id),
        )
        conn.execute(
            "INSERT INTO decisions (patient_id, method, decided_ts) VALUES (?,?,?)",
            (patient_id, method, now),
        )


def mark_treated(patient_id: int):
    with get_conn() as conn:
        conn.execute(
            "UPDATE patients SET status='treated', treated_ts=? WHERE id=?",
            (time.time(), patient_id),
        )


def recent_decisions(limit: int = 20) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT d.id, d.patient_id, d.method, d.decided_ts,
                      p.name, p.severity, p.specialist, p.wait_minutes
               FROM decisions d JOIN patients p ON p.id = d.patient_id
               ORDER BY d.decided_ts DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def metrics() -> dict:
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM patients").fetchone()["c"]
        waiting = conn.execute(
            "SELECT COUNT(*) c FROM patients WHERE status='waiting'").fetchone()["c"]
        in_tx = conn.execute(
            "SELECT COUNT(*) c FROM patients WHERE status='in_treatment'").fetchone()["c"]
        treated = conn.execute(
            "SELECT COUNT(*) c FROM patients WHERE status='treated'").fetchone()["c"]
        avg_wait = conn.execute(
            "SELECT AVG(wait_minutes) w FROM patients WHERE wait_minutes IS NOT NULL"
        ).fetchone()["w"]
        crit_wait = conn.execute(
            """SELECT AVG(wait_minutes) w FROM patients
               WHERE wait_minutes IS NOT NULL AND severity <= 2"""
        ).fetchone()["w"]
    return {
        "total_patients": total,
        "waiting": waiting,
        "in_treatment": in_tx,
        "treated": treated,
        "avg_wait_minutes": round(avg_wait, 1) if avg_wait else 0.0,
        "avg_critical_wait_minutes": round(crit_wait, 1) if crit_wait else 0.0,
    }
