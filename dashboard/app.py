"""TriageRL Streamlit dashboard (Section 3.4.3, Web Dashboard).

Run the API first, then:  streamlit run dashboard/app.py
"""

import os

import pandas as pd
import requests
import streamlit as st

API = os.environ.get("TRIAGERL_API", "http://localhost:8000")

st.set_page_config(page_title="TriageRL", page_icon="🏥", layout="wide")
st.title("🏥 TriageRL — Emergency Department Triage")

SEVERITY_LABELS = {1: "1 - Resuscitation", 2: "2 - Emergent", 3: "3 - Urgent",
                   4: "4 - Less urgent", 5: "5 - Non-urgent"}
SEVERITY_COLORS = {1: "🔴", 2: "🟠", 3: "🟡", 4: "🟢", 5: "🔵"}


def api_get(path):
    try:
        r = requests.get(f"{API}{path}", timeout=5)
        return r.json() if r.ok else None
    except requests.ConnectionError:
        return None


def api_post(path, json=None):
    try:
        r = requests.post(f"{API}{path}", json=json, timeout=10)
        return r.json(), r.ok
    except requests.ConnectionError:
        return None, False


symptoms = api_get("/symptoms")
if symptoms is None:
    st.error(f"Cannot reach the TriageRL API at {API}. "
             "Start it with:  uvicorn api.main:app")
    st.stop()

tab_register, tab_queue, tab_metrics = st.tabs(
    ["Register patient", "Queue & prioritization", "Metrics"])

# ---------------------------------------------------------------- register
with tab_register:
    st.subheader("Patient registration")
    with st.form("register"):
        c1, c2, c3 = st.columns(3)
        with c1:
            name = st.text_input("Name (optional)")
            age = st.number_input("Age", 0, 120, 35)
            gender = st.selectbox("Gender", ["M", "F"])
            symptom = st.selectbox("Presenting symptom", list(symptoms))
        with c2:
            heart_rate = st.number_input("Heart rate (bpm)", 20, 250, 80)
            systolic = st.number_input("Systolic BP (mmHg)", 40, 300, 120)
            diastolic = st.number_input("Diastolic BP (mmHg)", 20, 200, 80)
        with c3:
            temperature = st.number_input("Temperature (°C)", 30.0, 45.0, 37.0, 0.1)
            spo2 = st.number_input("SpO₂ (%)", 50, 100, 98)
        if st.form_submit_button("Register", type="primary"):
            body = {"name": name or None, "age": age, "gender": gender,
                    "symptom": symptom, "heart_rate": heart_rate,
                    "systolic_bp": systolic, "diastolic_bp": diastolic,
                    "temperature": temperature, "spo2": spo2}
            resp, ok = api_post("/patients", body)
            if ok:
                sev = resp["severity"]
                st.success(
                    f"Registered patient #{resp['patient_id']} — severity "
                    f"{SEVERITY_COLORS[sev]} {SEVERITY_LABELS[sev]}, "
                    f"recommended specialist: **{resp['specialist']}**")
            else:
                st.error(f"Registration failed: {resp}")

# ------------------------------------------------------------------- queue
with tab_queue:
    st.subheader("Waiting queue")
    left, right = st.columns([3, 1])
    with right:
        if st.button("🚑 Who's next? (prioritize)", type="primary"):
            resp, ok = api_post("/prioritize")
            if ok:
                st.success(
                    f"Treat patient **#{resp['patient_id']}** "
                    f"({resp.get('name') or 'unnamed'}) — severity "
                    f"{resp['severity']}, specialist {resp['specialist']} "
                    f"· decision by `{resp['method']}`")
            else:
                st.warning("No patients waiting.")
        treat_id = st.number_input("Mark treated (patient id)", 0, step=1)
        if st.button("✅ Mark treated") and treat_id:
            api_post(f"/patients/{int(treat_id)}/treated")
            st.rerun()

    queue = api_get("/queue")
    with left:
        if queue and queue["count"]:
            df = pd.DataFrame(queue["patients"])
            df["severity"] = df["severity"].map(
                lambda s: f"{SEVERITY_COLORS[s]} {SEVERITY_LABELS[s]}")
            cols = ["id", "name", "age", "gender", "symptom", "severity",
                    "wait_minutes", "specialist"]
            st.dataframe(df[cols], use_container_width=True, hide_index=True)
        else:
            st.info("Queue is empty.")

# ----------------------------------------------------------------- metrics
with tab_metrics:
    st.subheader("Operational metrics")
    m = api_get("/metrics")
    if m:
        c = st.columns(6)
        c[0].metric("Total patients", m["total_patients"])
        c[1].metric("Waiting", m["waiting"])
        c[2].metric("In treatment", m["in_treatment"])
        c[3].metric("Treated", m["treated"])
        c[4].metric("Avg wait (min)", m["avg_wait_minutes"])
        c[5].metric("Avg critical wait (min)", m["avg_critical_wait_minutes"])
