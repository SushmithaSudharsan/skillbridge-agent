// SkillBridge demo page. Display only: every decision is made by the backend.
// All text is inserted with textContent (never innerHTML), so pasted JD text can't inject markup.
"use strict";

const state = { code: "", samples: null, sessionId: null, view: null };
const $ = (id) => document.getElementById(id);

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function remember(key, value) {
  try { sessionStorage.setItem(key, value); } catch (_) { /* storage blocked: keep it in memory only */ }
}
function recall(key) {
  try { return sessionStorage.getItem(key) || ""; } catch (_) { return ""; }
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", "X-Access-Code": state.code, ...(options.headers || {}) },
  });
  if (!response.ok) {
    let message = `Something went wrong (${response.status}).`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail;
      else if (Array.isArray(body.detail)) message = "Please check: " + body.detail.map((d) => d.loc.slice(-1)[0]).join(", ");
    } catch (_) { /* not JSON */ }
    throw new Error(message);
  }
  return response;
}

function showError(message) {
  const box = $("form-error");
  box.textContent = message;
  box.hidden = !message;
}

// ---------- samples ----------

async function loadSamples() {
  if (!state.code || state.code === state.loadedFor) return;
  try {
    state.samples = await (await api("/api/samples")).json();
  } catch (error) {
    showError(error.message);
    return;
  }
  showError("");
  state.loadedFor = state.code;
  remember("sb-code", state.code);
  const resumes = $("resume-select");
  resumes.replaceChildren(el("option", { value: "", text: "Choose a sample resume…" }),
    ...state.samples.resumes.map((r) => el("option", { value: r.id, text: r.title })));
  const jds = $("jd-select");
  jds.replaceChildren(el("option", { value: "", text: "Use a sample JD…" }),
    ...state.samples.jds.map((j) => el("option", { value: j.id, text: j.title })));
}

function onResumeChange() {
  const sample = state.samples?.resumes.find((r) => r.id === $("resume-select").value);
  $("resume-preview").textContent = sample ? sample.text : "";
  $("resume-preview-wrap").hidden = !sample;
}

function onJdChange() {
  const sample = state.samples?.jds.find((j) => j.id === $("jd-select").value);
  if (!sample) return;
  $("jd-text").value = sample.text;
  $("title").value = sample.title;
  updateCount();
}

function updateCount() {
  $("jd-count").textContent = `${$("jd-text").value.length.toLocaleString()} characters`;
}

// ---------- analyse ----------

const STEPS = ["Reading the job description…", "Reading the resume (masked)…", "Comparing requirement by requirement…",
  "Writing advice and checking every quote…"];

function setLoading(on) {
  $("loading").hidden = !on;
  $("analyze-btn").disabled = on;
  if (!on) return clearInterval(state.ticker);
  let i = 0;
  $("loading-text").textContent = STEPS[0];
  state.ticker = setInterval(() => { i = Math.min(i + 1, STEPS.length - 1); $("loading-text").textContent = STEPS[i]; }, 2500);
}

async function onAnalyze(event) {
  event.preventDefault();
  state.code = $("access-code").value.trim();
  if (!state.samples) await loadSamples();
  const body = {
    sample_id: $("resume-select").value, jd_text: $("jd-text").value.trim(), title: $("title").value.trim(),
    years: Number($("years").value || 0), internship_months: Number($("months").value || 0),
  };
  if (!state.code) return showError("Enter the access code.");
  if (!body.sample_id) return showError("Choose a sample resume.");
  if (body.jd_text.length < 80) return showError("Paste a full job description (at least a few lines).");
  showError("");
  await endSession();
  $("empty-state").hidden = true;
  $("report").hidden = true;
  setLoading(true);
  try {
    const view = await (await api("/api/analyze", { method: "POST", body: JSON.stringify(body) })).json();
    state.sessionId = view.session_id;
    render(view);
  } catch (error) {
    showError(error.message);
    $("empty-state").hidden = false;
  } finally {
    setLoading(false);
  }
}

// ---------- render ----------

const FIT_TONE = { "Strong fit": "good", "Possible fit": "info", "Stretch": "warn", "Not yet": "bad" };
const KIND = { skill: "Skill", task: "Experience", soft: "Way of working" };
const STATUS = { missing: "Not shown", below: "Below", meets: "Meets" };

function render(view) {
  state.view = view;
  $("r-eyebrow").textContent = view.seniority ? `${view.seniority} role` : "Analysis";
  $("r-title").textContent = view.title;
  const fit = view.fit;
  $("fit").className = `fit ${FIT_TONE[fit.verdict] || "info"}`;
  $("fit").replaceChildren(el("h3", { text: fit.verdict }), el("p", { text: fit.advice }),
    el("ul", {}, fit.reasons.map((r) => el("li", { text: r }))));
  const c = view.counts;
  $("tiles").replaceChildren(...[["Requirements", c.requirements], ["Already meet", c.met], ["Gaps", c.gaps],
    ["Must-have gaps", c.must_gaps]].map(([label, n]) => el("div", { class: "tile" }, el("span", { text: label }), el("b", { text: n }))));
  $("summary").textContent = view.summary;

  const BLOCKER = { work_authorisation: "Work authorisation", location: "Location or work arrangement",
    degree: "Degree", experience: "Years of experience" };
  $("blockers").replaceChildren(...(view.blockers && view.blockers.length ? [
    el("h3", { text: "Likely auto-filters: check these first" }),
    el("p", { class: "topics", text: "Employers' systems often filter on these before anyone reads your skills. A hint, not a verdict." }),
    ...view.blockers.map((b) => el("div", { class: "item" },
      el("div", { class: "item-head" }, el("b", { text: BLOCKER[b.kind] }), el("span", { class: `tag ${b.status === "ok" ? "meets" : "missing"}`, text: b.status === "ok" ? "OK" : "Check" })),
      el("p", {}, "The job says: ", el("q", { text: b.jd_quote })), el("p", { text: b.note })))] : []));
  const soft = view.soft || { shown: [], interview: [] };
  const softBlock = (soft.shown.length || soft.interview.length) ? el("div", { class: "item" },
    el("div", { class: "item-head" }, el("b", { text: "Ways of working the job mentions" }), el("span", { class: "tag", text: "Not scored" })),
    soft.shown.length ? el("p", { text: "Your resume already shows: " + soft.shown.map((m) => m.skill).join(", ") + "." }) : null,
    soft.interview.length ? el("p", { text: "Prepare one real example for the interview: " + soft.interview.map((m) => m.skill).join(", ") +
      ". You don't need a soft-skills list on your resume; a bullet that shows what you did is stronger." }) : null) : null;
  const overNote = view.overqualified ? el("div", { class: "item" }, el("p", { text:
    "You have more experience than this role asks for. That isn't a skill gap. Be ready to explain why you want a role at this level." })) : null;
  panel("learn", (view.learn.length ? view.learn.map(requirementItem)
    : [el("p", { class: "empty-tab", text: "Your resume shows every skill and experience this job asks for." })])
    .concat([overNote, softBlock].filter(Boolean)));
  panel("show", view.show_better.length ? view.show_better.map((b) => el("div", { class: "item" },
    el("div", { class: "item-head" }, el("b", { text: b.skill }), el("span", { class: "tag", text: b.why === "context" ? "Present it for this job" : b.why === "implied" ? "Implied by your work" : "Only listed" })),
    el("p", { text: b.text }), b.resume_line ? el("p", { class: "topics" }, "Your line: ", el("q", { text: b.resume_line })) : null))
    : [el("p", { class: "empty-tab", text: "Nothing hidden: what you have is already visible for this job." })]);
  panel("strengths", view.strengths.length ? view.strengths.map(requirementItem)
    : [el("p", { class: "empty-tab", text: "No requirement is clearly shown yet; see Learn and Show better." })]);
  panel("plan", view.learning.length ? view.learning.map((s, i) => el("div", { class: "item" },
    el("div", { class: "item-head" }, el("span", { class: "step-n", text: i + 1 }), el("b", { text: `${s.area} (${s.level})` }),
      el("span", { class: "tag", text: s.reason }), s.suggested ? el("span", { class: "tag ai", text: "AI-suggested" }) : null),
    s.topics.length ? el("p", { class: "topics", text: "Key topics: " + s.topics.join(", ") }) : null))
    : [el("p", { class: "empty-tab", text: "Nothing to learn for this job." })]);
  panel("projects", view.projects.length ? view.projects.map((p) => el("div", { class: "item" },
    el("div", { class: "item-head" }, el("b", { text: p.title }), ...p.skills.map((s) => el("span", { class: "tag", text: s }))),
    el("p", { text: p.build }), el("p", { class: "topics" }, "The job says: ", el("q", { text: p.job_quote }))))
    : [el("p", { class: "empty-tab", text: "No project ideas passed the checks this time." })]);
  renderQuestions(view);

  const usage = view.usage ? ` · ${view.usage.calls} AI calls, about $${view.usage.cost_usd.toFixed(4)}` : "";
  $("footnote").textContent = `${view.removed_claims} AI claim(s) were removed because they couldn't be verified against the text${usage}. ` +
    "Guidance, not a guarantee of employment.";
  $("report").hidden = false;
  selectTab(state.tab || "learn");
}

function requirementItem(m) {
  return el("div", { class: "item" },
    el("div", { class: "item-head" }, el("b", { text: m.skill }), el("span", { class: `tag ${m.status}`, text: STATUS[m.status] }),
      el("span", { class: "tag", text: `${m.current} → ${m.needed}` }), m.kind !== "skill" ? el("span", { class: "tag", text: KIND[m.kind] }) : null,
      m.must_have ? el("span", { class: "tag", text: "Must-have" }) : el("span", { class: "tag", text: "A plus" }),
      m.related ? el("span", { class: "tag", text: `Related: ${m.related.skill}` }) : null),
    el("div", { class: "quotes" },
      el("div", {}, el("span", { text: "The job asks" }), el("q", { text: m.job_quote })),
      el("div", {}, el("span", { text: "Your resume" }), m.resume_quote ? el("q", { text: m.resume_quote }) : el("em", { text: "Not shown" })),
      m.related ? el("div", {}, el("span", { text: `Related (${m.related.level})` }), el("q", { text: m.related.quote }),
        el("em", { text: ` Related is not the same: say how it transfers, and be clear you haven't used ${m.skill} yet.` })) : null));
}

function renderQuestions(view) {
  $("q-count").textContent = view.questions.length || "";
  const changes = view.changes.length ? el("div", { class: "changes" }, el("h3", { text: "What your answers changed" }),
    el("ul", {}, view.changes.map((c) => el("li", {}, el("b", { text: c.skill }), ` ${c.before} → ${c.after}. You said “${c.said}”. ${c.where}`)))) : null;
  const items = view.questions.map((q, i) => el("div", { class: "item question" },
    el("p", { class: "topics", text: q.topic }), el("p", { class: "topics", text: q.issue }), el("p", { class: "q", text: q.question }),
    el("label", { for: `answer-${i}`, text: "Your answer (optional, in your own words)" }),
    (() => { const t = el("textarea", { id: `answer-${i}`, maxlength: "1000" }); t.value = q.answer || ""; return t; })(),
    q.takeaway ? el("p", { class: "note", text: "What this means: " + q.takeaway }) : null,
    q.effects && q.effects.length ? el("p", { class: "note", text: "Changed your plan: " + q.effects.join("; ") }) : null,
    q.rewrite ? el("p", { class: "note rewrite", text: `Suggested line (only your facts): “${q.rewrite}”` }) : null));
  const button = el("button", { class: "primary", type: "button", id: "answers-btn", text: "Update my plan" });
  button.addEventListener("click", onAnswers);
  panel("questions", view.questions.length
    ? [changes, el("p", { class: "muted", text: "Only your answers can add facts. They can raise a level (up to Intermediate) or lower one your resume oversells." }), ...items, button]
    : [el("p", { class: "empty-tab", text: "No questions for this resume." })]);
}

function panel(name, children) {
  document.querySelector(`[data-panel="${name}"]`).replaceChildren(...children.filter(Boolean));
}

function selectTab(name) {
  state.tab = name;
  document.querySelectorAll("[role=tab]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  document.querySelectorAll(".tab-panel").forEach((p) => { p.hidden = p.dataset.panel !== name; });
}

// ---------- answers, report, end ----------

async function onAnswers() {
  const answers = state.view.questions.map((_, i) => $(`answer-${i}`).value.trim());
  const button = $("answers-btn");
  button.disabled = true;
  button.textContent = "Reading your answers…";
  try {
    const view = await (await api(`/api/sessions/${state.sessionId}/answers`, { method: "POST", body: JSON.stringify({ answers }) })).json();
    render(view);
    selectTab("questions");
  } catch (error) {
    alertInline(error.message);
    button.disabled = false;
    button.textContent = "Update my plan";
  }
}

function alertInline(message) {
  const p = el("p", { class: "error", role: "alert", text: message });
  document.querySelector('[data-panel="questions"]').append(p);
}

async function onPdf() {
  try {
    const blob = await (await api(`/api/sessions/${state.sessionId}/report.pdf`)).blob();
    const url = URL.createObjectURL(blob);
    const a = el("a", { href: url, download: "skillbridge-report.pdf" });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) {
    showError(error.message);
  }
}

async function endSession() {
  if (!state.sessionId) return;
  const id = state.sessionId;
  state.sessionId = null;
  try { await fetch(`/api/sessions/${id}/end`, { method: "POST" }); } catch (_) { /* it expires anyway */ }
}

async function onEnd() {
  await endSession();
  $("report").hidden = true;
  $("empty-state").hidden = false;
}

// ---------- wire up ----------

document.addEventListener("DOMContentLoaded", () => {
  state.code = recall("sb-code");
  $("access-code").value = state.code;
  $("resume-preview-wrap").hidden = true;
  $("access-code").addEventListener("change", () => { state.code = $("access-code").value.trim(); loadSamples(); });
  $("resume-select").addEventListener("change", onResumeChange);
  $("jd-select").addEventListener("change", onJdChange);
  $("jd-text").addEventListener("input", updateCount);
  $("analyze-form").addEventListener("submit", onAnalyze);
  $("pdf-btn").addEventListener("click", onPdf);
  $("end-btn").addEventListener("click", onEnd);
  document.querySelectorAll("[role=tab]").forEach((b) => b.addEventListener("click", () => selectTab(b.dataset.tab)));
  window.addEventListener("pagehide", () => {
    if (state.sessionId) navigator.sendBeacon(`/api/sessions/${state.sessionId}/end`);
  });
  loadSamples();
});
