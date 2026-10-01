"""Grounded maintenance-ticket agent.

Pipeline: wav -> anomaly score + evidence (band deviations, condition
indicators, rule-based hypotheses) -> LLM writes the ticket -> grounding check.

The LLM only sees the evidence JSON, never raw audio, and is told to use only
those facts. A post-check rejects any ticket that mentions a frequency that is
not in the evidence and falls back to a deterministic template, so a ticket can
never invent a cause.

Backends (pick with --llm or the LLM_BACKEND env var):
    template   no LLM, deterministic (default; works offline)
    ollama     local model via http://localhost:11434 (e.g. `ollama pull llama3.1`)
    anthropic  Claude API, needs ANTHROPIC_API_KEY

    python -m machine_guardian.ticket_agent --bundle models/fan_section_00.joblib --wav some.wav --llm ollama
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

import joblib
import requests

from .audio_features import load_wav, log_mel
from .explain import fault_hypotheses, signal_stats

PHYSICS_GUARD_Z = 8.0  # envelope-peak prominence (z vs normal) that triggers an alert on its own

SYSTEM = (
    "You are a reliability engineer writing a maintenance ticket for a factory technician. "
    "Use ONLY the facts in the evidence JSON. Do not invent part names, frequencies, numbers or causes. "
    "Hypotheses must be phrased as things to check, not as confirmed diagnoses. "
    "Write 4 short sections in markdown: Summary, Evidence, Likely causes to check, Recommended action. "
    "Under 170 words."
)


def assess(bundle: dict, wav_path: str) -> dict:
    cfg = bundle["cfg"]
    x = load_wav(wav_path, cfg.sr)
    S = log_mel(x, cfg)
    det = bundle["detector"]
    domain = "source" if "target" not in Path(wav_path).name else "target"
    score = float(det.score([S], [domain])[0])
    thr = bundle["threshold"]
    ratio = score / thr if thr > 0 else float("inf")
    if ratio < 1.0:
        status, action = "normal", "No action."
    elif ratio < 1.5:
        status, action = "watch", "Keep monitoring; re-check in the next shift."
    elif ratio < 2.5:
        status, action = "anomalous", "Schedule an inspection within 7 days."
    else:
        status, action = "critical", "Inspect within 48 hours."
    ev = bundle["profile"].evidence(S, signal_stats(x, cfg))
    guard = ev["indicator_z"]["envelope_prominence"] > PHYSICS_GUARD_Z
    raised_by_guard = guard and status in ("normal", "watch")
    if raised_by_guard:
        # hybrid alerting: a strong periodic-impact signature never seen in
        # normal data raises an alert even if the ML score stays low
        status, action = "anomalous", "Schedule an inspection within 7 days (raised by envelope-analysis guard rail)."
    return {
        "machine": f"{bundle['machine']} / {bundle['section']}", "file": Path(wav_path).name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "anomaly_score": round(score, 3), "alert_threshold": round(thr, 3),
        "score_to_threshold": round(ratio, 2), "status": status, "rule_based_action": action,
        "physics_guard_triggered": bool(guard), "raised_by_guard": bool(raised_by_guard),
        **ev, "hypotheses": fault_hypotheses(ev) if status != "normal" else [],
        "_mel": S,
    }


def template_ticket(a: dict) -> str:
    bands = ", ".join(f"~{b['center_hz']} Hz {b['direction']} (z={b['z']})" for b in a["top_bands"][:3])
    hyp = "\n".join(f"- {h}" for h in a["hypotheses"]) or "- None: sound matches normal profile."
    return (
        f"### Maintenance ticket: {a['machine']} ({a['status'].upper()})\n"
        f"**Summary:** Anomaly score {a['anomaly_score']} is {a['score_to_threshold']}x the alert threshold.\n\n"
        f"**Evidence:** largest deviations at {bands}. Shaft speed ~{a['shaft_hz']} Hz; "
        f"high-frequency envelope repeats at {a['envelope_peak_hz']} Hz ({a['envelope_to_shaft_ratio']}x shaft, "
        f"prominence z={a['indicator_z']['envelope_prominence']}). Loudness change {a['loudness_change_db']} dB.\n\n"
        f"**Likely causes to check:**\n{hyp}\n\n"
        f"**Recommended action:** {a['rule_based_action']}\n"
    )


def _ollama(prompt: str) -> str:
    r = requests.post(os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/chat", timeout=120, json={
        "model": os.getenv("OLLAMA_MODEL", "llama3.1"), "stream": False, "options": {"temperature": 0.1},
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]})
    r.raise_for_status()
    return r.json()["message"]["content"]


def _anthropic(prompt: str) -> str:
    r = requests.post("https://api.anthropic.com/v1/messages", timeout=120, headers={
        "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
        "content-type": "application/json"}, json={
        "model": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"), "max_tokens": 600, "temperature": 0.1,
        "system": SYSTEM, "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    return r.json()["content"][0]["text"]


def grounded(text: str, a: dict) -> bool:
    """Every frequency the LLM mentions must be within 15% of an evidence band."""
    allowed = [b["center_hz"] for b in a["top_bands"]] + [a["shaft_hz"], a["envelope_peak_hz"]]
    allowed += [float(n) for h in a["hypotheses"] for n in re.findall(r"(\d+(?:\.\d+)?)\s*Hz", h)]
    for hz in re.findall(r"(\d[\d,]*(?:\.\d+)?)\s*(?:Hz|hz)", text):
        v = float(hz.replace(",", ""))
        if not any(abs(v - b) <= 0.15 * b for b in allowed):
            return False
    return True


def write_ticket(a: dict, backend: str = "template") -> tuple[str, str]:
    """Return (ticket_markdown, backend_actually_used)."""
    if a["status"] == "normal" or backend == "template":
        return template_ticket(a), "template"
    evidence = {k: v for k, v in a.items() if not k.startswith("_")}
    prompt = "Evidence JSON:\n" + json.dumps(evidence, indent=2)
    try:
        text = _ollama(prompt) if backend == "ollama" else _anthropic(prompt)
    except Exception as e:
        return template_ticket(a) + f"\n_(LLM unavailable: {type(e).__name__}; template used)_", "template"
    if not grounded(text, a):
        return template_ticket(a) + "\n_(LLM draft failed the grounding check; template used)_", "template"
    return text, backend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--llm", default=os.getenv("LLM_BACKEND", "template"), choices=["template", "ollama", "anthropic"])
    args = ap.parse_args()
    a = assess(joblib.load(args.bundle), args.wav)
    text, used = write_ticket(a, args.llm)
    print(text)
    print(f"\n[backend: {used}]")


if __name__ == "__main__":
    main()
