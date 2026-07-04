import { useCallback, useEffect, useState } from "react";
import { api } from "./api.js";

const SEVERITY = {
  1: { label: "Resuscitation", cls: "sev-1" },
  2: { label: "Emergent", cls: "sev-2" },
  3: { label: "Urgent", cls: "sev-3" },
  4: { label: "Less urgent", cls: "sev-4" },
  5: { label: "Non-urgent", cls: "sev-5" },
};

const METHOD_LABEL = {
  ppo: "PPO agent",
  dqn: "DQN agent",
  severity_rule: "Severity rule",
  severity_rule_fallback: "Rule fallback",
};

const EMPTY_FORM = {
  name: "", age: 35, gender: "M", symptom: "chest_pain",
  heart_rate: 80, systolic_bp: 120, diastolic_bp: 80,
  temperature: 37.0, spo2: 98,
};

const Icon = ({ d, size = 18 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
       stroke="currentColor" strokeWidth="2" strokeLinecap="round"
       strokeLinejoin="round"><path d={d} /></svg>
);
const ICONS = {
  board: "M3 3h7v9H3zM14 3h7v5h-7zM14 12h7v9h-7zM3 16h7v5H3z",
  register: "M12 5v14M5 12h14",
  clock: "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20zM12 6v6l4 2",
  pulse: "M22 12h-4l-3 9L9 3l-3 9H2",
  bed: "M2 4v16M2 8h18a2 2 0 0 1 2 2v10M2 17h20M6 8v9",
  check: "M20 6L9 17l-5-5",
};

export default function App() {
  const [view, setView] = useState("board");
  const [symptoms, setSymptoms] = useState({});
  const [queue, setQueue] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [decisions, setDecisions] = useState([]);
  const [modelInfo, setModelInfo] = useState(null);
  const [toast, setToast] = useState(null);
  const [apiDown, setApiDown] = useState(false);
  const [clock, setClock] = useState(new Date());

  const refresh = useCallback(async () => {
    try {
      const [q, m, d] = await Promise.all([
        api.queue(), api.metrics(), api.decisions(),
      ]);
      setQueue(q.patients);
      setMetrics(m);
      setDecisions(d.decisions);
      setApiDown(false);
    } catch {
      setApiDown(true);
    }
  }, []);

  useEffect(() => {
    api.symptoms().then(setSymptoms).catch(() => setApiDown(true));
    api.modelInfo().then(setModelInfo).catch(() => {});
    refresh();
    const t = setInterval(refresh, 5000);
    const c = setInterval(() => setClock(new Date()), 1000);
    return () => { clearInterval(t); clearInterval(c); };
  }, [refresh]);

  const notify = (msg, kind = "ok") => {
    setToast({ msg, kind });
    setTimeout(() => setToast(null), 6000);
  };

  const criticals = queue.filter((p) => p.severity <= 2).length;

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">T</div>
          <div>
            <div className="brand-name">TriageRL</div>
            <div className="brand-sub">ED Triage Console</div>
          </div>
        </div>
        <nav>
          <button className={view === "board" ? "nav-item active" : "nav-item"}
                  onClick={() => setView("board")}>
            <Icon d={ICONS.board} /> Queue board
          </button>
          <button className={view === "register" ? "nav-item active" : "nav-item"}
                  onClick={() => setView("register")}>
            <Icon d={ICONS.register} /> Register patient
          </button>
        </nav>
        <div className="sidebar-foot">
          <div className={`model-badge ${modelInfo?.active_model || "none"}`}>
            <span className="dot" />
            {modelInfo?.label || "Connecting…"}
          </div>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <h1>{view === "board" ? "Live queue board" : "Patient registration"}</h1>
          <div className="topbar-right">
            {criticals > 0 && view === "board" && (
              <span className="crit-alert">
                {criticals} critical waiting
              </span>
            )}
            <span className="clock">
              {clock.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
            </span>
          </div>
        </header>

        {apiDown && (
          <div className="banner">
            Cannot reach the TriageRL API — start it with <code>uvicorn api.main:app</code>
          </div>
        )}
        {toast && <div className={`toast ${toast.kind}`}>{toast.msg}</div>}

        {view === "board" ? (
          <Board queue={queue} metrics={metrics} decisions={decisions}
                 refresh={refresh} notify={notify} />
        ) : (
          <RegisterForm symptoms={symptoms}
                        onDone={(msg, kind) => { notify(msg, kind); refresh(); }} />
        )}
      </main>
    </div>
  );
}

function Board({ queue, metrics, decisions, refresh, notify }) {
  const [demo, setDemo] = useState(false);

  useEffect(() => {
    if (!demo) return;
    let tick = 0;
    const t = setInterval(async () => {
      tick += 1;
      try {
        await api.demoPatient();
        if (tick % 2 === 0) await api.prioritize().catch(() => {});
        refresh();
      } catch { /* API down; banner already shows */ }
    }, 4000);
    return () => clearInterval(t);
  }, [demo, refresh]);

  const prioritize = async () => {
    try {
      const r = await api.prioritize();
      notify(
        `Next: patient #${r.patient_id}${r.name ? ` (${r.name})` : ""} — ` +
        `severity ${r.severity} (${SEVERITY[r.severity].label}), ` +
        `${r.specialist} · ${METHOD_LABEL[r.method] || r.method}`
      );
      refresh();
    } catch (err) {
      notify(err.message, "err");
    }
  };

  const treat = async (id) => { await api.markTreated(id); refresh(); };

  return (
    <>
      <section className="stats">
        <Stat icon={ICONS.clock} label="Waiting" value={metrics?.waiting ?? "–"} />
        <Stat icon={ICONS.bed} label="In treatment" value={metrics?.in_treatment ?? "–"} />
        <Stat icon={ICONS.check} label="Treated" value={metrics?.treated ?? "–"} />
        <Stat icon={ICONS.clock} label="Avg wait (min)" value={metrics?.avg_wait_minutes ?? "–"} />
        <Stat icon={ICONS.pulse} label="Critical wait (min)"
              value={metrics?.avg_critical_wait_minutes ?? "–"} accent />
      </section>

      <div className="board-grid">
        <section className="card queue-card">
          <div className="card-head">
            <h2>Waiting queue <span className="count">{queue.length}</span></h2>
            <div className="head-actions">
              <button className={demo ? "ghost demo-on" : "ghost"}
                      onClick={() => setDemo(!demo)}>
                {demo ? "\u25a0 Stop demo" : "\u25b6 Demo mode"}
              </button>
              <button className="primary" onClick={prioritize}>
                <Icon d={ICONS.pulse} size={16} /> Who's next?
              </button>
            </div>
          </div>
          {queue.length === 0 ? (
            <p className="empty">Queue is empty — register a patient to begin.</p>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Patient</th><th>Age</th><th>Symptom</th><th>Severity</th>
                  <th>Wait</th><th>Specialist</th><th></th>
                </tr>
              </thead>
              <tbody>
                {queue.map((p) => (
                  <tr key={p.id} className={p.severity <= 2 ? "row-critical" : ""}>
                    <td><strong>#{p.id}</strong> {p.name || ""}</td>
                    <td>{p.age}</td>
                    <td className="cap">{p.symptom.replaceAll("_", " ")}</td>
                    <td>
                      <span className={`pill ${SEVERITY[p.severity].cls}`}>
                        {p.severity} · {SEVERITY[p.severity].label}
                      </span>
                    </td>
                    <td className={p.wait_minutes > 60 ? "wait-long" : ""}>
                      {p.wait_minutes} min
                    </td>
                    <td>{p.specialist}</td>
                    <td>
                      <button className="ghost" onClick={() => treat(p.id)}>
                        Treated
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>

        <section className="card log-card">
          <div className="card-head"><h2>Decision log</h2></div>
          {decisions.length === 0 ? (
            <p className="empty">No prioritization decisions yet.</p>
          ) : (
            <ul className="log">
              {decisions.map((d) => (
                <li key={d.id}>
                  <span className={`pill small ${SEVERITY[d.severity].cls}`}>
                    {d.severity}
                  </span>
                  <div className="log-body">
                    <div>
                      <strong>#{d.patient_id}</strong> {d.name || "unnamed"}
                      <span className="log-spec"> → {d.specialist}</span>
                    </div>
                    <div className="log-meta">
                      {d.wait_minutes != null ? `waited ${d.wait_minutes.toFixed(0)} min · ` : ""}
                      <span className={`method ${d.method}`}>
                        {METHOD_LABEL[d.method] || d.method}
                      </span>
                      {" · "}
                      {new Date(d.decided_ts * 1000).toLocaleTimeString([], {
                        hour: "2-digit", minute: "2-digit" })}
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
    </>
  );
}

function Stat({ icon, label, value, accent }) {
  return (
    <div className={accent ? "stat accent" : "stat"}>
      <div className="stat-icon"><Icon d={icon} /></div>
      <div>
        <div className="stat-value">{value}</div>
        <div className="stat-label">{label}</div>
      </div>
    </div>
  );
}

function RegisterForm({ symptoms, onDone }) {
  const [form, setForm] = useState(EMPTY_FORM);
  const [busy, setBusy] = useState(false);
  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  const submit = async (e) => {
    e.preventDefault();
    setBusy(true);
    try {
      const body = {
        ...form,
        name: form.name || null,
        age: +form.age, heart_rate: +form.heart_rate,
        systolic_bp: +form.systolic_bp, diastolic_bp: +form.diastolic_bp,
        temperature: +form.temperature, spo2: +form.spo2,
      };
      const r = await api.register(body);
      onDone(
        `Registered #${r.patient_id} — assessed severity ${r.severity} ` +
        `(${SEVERITY[r.severity].label}), recommended: ${r.specialist}`
      );
      setForm(EMPTY_FORM);
    } catch (err) {
      onDone(`Registration failed: ${err.message}`, "err");
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="card reg-card" onSubmit={submit}>
      <div className="form-section">
        <h3>Patient details</h3>
        <div className="form-grid">
          <label>Name <span className="opt">(optional)</span>
            <input value={form.name} onChange={(e) => set("name", e.target.value)} />
          </label>
          <label>Age
            <input type="number" min="0" max="120" required value={form.age}
                   onChange={(e) => set("age", e.target.value)} />
          </label>
          <label>Gender
            <select value={form.gender} onChange={(e) => set("gender", e.target.value)}>
              <option value="M">Male</option><option value="F">Female</option>
            </select>
          </label>
          <label>Presenting symptom
            <select value={form.symptom} onChange={(e) => set("symptom", e.target.value)}>
              {Object.keys(symptoms).map((s) => (
                <option key={s} value={s}>{s.replaceAll("_", " ")}</option>
              ))}
            </select>
          </label>
        </div>
      </div>
      <div className="form-section">
        <h3>Vital signs</h3>
        <div className="form-grid">
          <label>Heart rate (bpm)
            <input type="number" min="20" max="250" required value={form.heart_rate}
                   onChange={(e) => set("heart_rate", e.target.value)} />
          </label>
          <label>Systolic BP (mmHg)
            <input type="number" min="40" max="300" required value={form.systolic_bp}
                   onChange={(e) => set("systolic_bp", e.target.value)} />
          </label>
          <label>Diastolic BP (mmHg)
            <input type="number" min="20" max="200" required value={form.diastolic_bp}
                   onChange={(e) => set("diastolic_bp", e.target.value)} />
          </label>
          <label>Temperature (°C)
            <input type="number" step="0.1" min="30" max="45" required
                   value={form.temperature}
                   onChange={(e) => set("temperature", e.target.value)} />
          </label>
          <label>SpO₂ (%)
            <input type="number" min="50" max="100" required value={form.spo2}
                   onChange={(e) => set("spo2", e.target.value)} />
          </label>
        </div>
      </div>
      <button className="primary big" disabled={busy}>
        {busy ? "Registering…" : "Register & assess severity"}
      </button>
    </form>
  );
}
