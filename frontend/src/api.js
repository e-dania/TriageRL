async function request(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const msg = body?.detail || `Request failed (${res.status})`;
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return body;
}

export const api = {
  symptoms: () => request("/symptoms"),
  queue: () => request("/queue"),
  metrics: () => request("/metrics"),
  decisions: () => request("/decisions"),
  modelInfo: () => request("/model-info"),
  register: (patient) =>
    request("/patients", { method: "POST", body: JSON.stringify(patient) }),
  prioritize: () => request("/prioritize", { method: "POST" }),
  demoPatient: () => request("/demo/patients", { method: "POST" }),
  markTreated: (id) => request(`/patients/${id}/treated`, { method: "POST" }),
};
