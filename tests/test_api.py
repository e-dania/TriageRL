"""Integration + edge-case tests: FastAPI backend.

Testing strategies: end-to-end workflow testing (register -> queue ->
prioritize -> treat -> metrics), input validation / edge cases with
invalid and extreme data values, and error-path testing.
"""

import os
import tempfile

import pytest

# Use an isolated database for the test session
os.environ["TRIAGERL_DB"] = os.path.join(tempfile.gettempdir(),
                                         "triagerl_pytest.db")
if os.path.exists(os.environ["TRIAGERL_DB"]):
    os.remove(os.environ["TRIAGERL_DB"])

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

client = TestClient(app)

VALID_PATIENT = {
    "name": "Test Patient", "age": 45, "gender": "M",
    "symptom": "chest_pain", "heart_rate": 88, "systolic_bp": 125,
    "diastolic_bp": 82, "temperature": 37.1, "spo2": 97,
}


class TestWorkflow:
    """End-to-end happy path."""

    def test_full_patient_lifecycle(self):
        r = client.post("/patients", json=VALID_PATIENT)
        assert r.status_code == 200
        body = r.json()
        assert 1 <= body["severity"] <= 5
        assert body["specialist"] == "Cardiologist"
        pid = body["patient_id"]

        q = client.get("/queue").json()
        assert any(p["id"] == pid for p in q["patients"])

        d = client.post("/prioritize")
        assert d.status_code == 200
        assert d.json()["method"] in (
            "ppo", "dqn", "severity_rule", "severity_rule_fallback")

        t = client.post(f"/patients/{pid}/treated")
        assert t.status_code == 200

        m = client.get("/metrics").json()
        assert m["total_patients"] >= 1

    def test_critical_patient_prioritized_over_mild(self):
        """Different data values: a critical arrival must be chosen before
        an earlier mild arrival."""
        mild = dict(VALID_PATIENT, symptom="laceration", heart_rate=75,
                    systolic_bp=120, spo2=99, temperature=36.8)
        critical = dict(VALID_PATIENT, symptom="stroke_symptoms",
                        heart_rate=138, systolic_bp=84, spo2=87,
                        temperature=39.6, age=82)
        client.post("/patients", json=mild)
        r_crit = client.post("/patients", json=critical)
        assert r_crit.json()["severity"] <= 2

        decision = client.post("/prioritize").json()
        assert decision["severity"] <= 2

    def test_decision_log_records(self):
        d = client.get("/decisions").json()
        assert len(d["decisions"]) >= 1
        assert "method" in d["decisions"][0]


class TestValidationEdgeCases:
    """Invalid and boundary data values must be rejected cleanly (422/400)."""

    @pytest.mark.parametrize("field,value", [
        ("age", -1), ("age", 200),
        ("heart_rate", 5), ("heart_rate", 400),
        ("systolic_bp", 10), ("spo2", 30), ("spo2", 101),
        ("temperature", 20), ("temperature", 50),
        ("gender", "X"),
    ])
    def test_out_of_range_rejected(self, field, value):
        bad = dict(VALID_PATIENT, **{field: value})
        r = client.post("/patients", json=bad)
        assert r.status_code == 422

    def test_unknown_symptom_rejected(self):
        r = client.post("/patients",
                        json=dict(VALID_PATIENT, symptom="not_a_symptom"))
        assert r.status_code == 400

    def test_missing_field_rejected(self):
        incomplete = {k: v for k, v in VALID_PATIENT.items() if k != "age"}
        r = client.post("/patients", json=incomplete)
        assert r.status_code == 422

    def test_boundary_values_accepted(self):
        """Extreme-but-valid vitals must be accepted, not crash."""
        extreme = dict(VALID_PATIENT, age=0, heart_rate=20, systolic_bp=40,
                       diastolic_bp=20, temperature=30.0, spo2=50)
        r = client.post("/patients", json=extreme)
        assert r.status_code == 200
        assert r.json()["severity"] == 1  # maximally deranged vitals

    def test_prioritize_empty_queue_returns_404(self):
        # Drain the queue first
        while True:
            r = client.post("/prioritize")
            if r.status_code == 404:
                break
        assert r.status_code == 404


class TestStaticServing:
    def test_dashboard_served_if_built(self):
        r = client.get("/")
        # 200 with the built SPA, or 404 if frontend/dist absent - never 500
        assert r.status_code in (200, 404)

    def test_model_info(self):
        r = client.get("/model-info")
        assert r.status_code == 200
        assert "active_model" in r.json()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
