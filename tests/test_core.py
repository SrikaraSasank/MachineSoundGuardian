"""Unit tests for the parts that must never silently break.

    python -m pytest tests -q        (or: python tests/test_core.py)
Needs the smoke-test models (run scripts/smoke_test.py once first).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from machine_guardian import ticket_agent as ta  # noqa: E402
from machine_guardian.metrics import pauc  # noqa: E402


def _assessment():
    wav = sorted((ROOT / "data/synthetic/fan/test").glob("section_00_source_test_anomaly_*.wav"))[0]
    return ta.assess(joblib.load(ROOT / "models/fan_section_00.joblib"), str(wav))


def test_grounding_rejects_invented_frequency(monkeypatch=None):
    a = _assessment()
    fake = "Summary: bearing fault. Evidence: strong peak at 7300 Hz."  # 7300 Hz is not in the evidence
    ta._anthropic = lambda prompt: fake
    text, used = ta.write_ticket(a, "anthropic")
    assert used == "template" and "grounding check" in text


def test_grounding_accepts_supported_frequency():
    a = _assessment()
    ok = f"Summary: impacts at {a['envelope_peak_hz']} Hz, shaft {a['shaft_hz']} Hz. Check the bearing."
    ta._anthropic = lambda prompt: ok
    text, used = ta.write_ticket(a, "anthropic")
    assert used == "anthropic" and text == ok


def test_llm_failure_falls_back():
    a = _assessment()

    def boom(prompt):
        raise ConnectionError("offline")
    ta._ollama = boom
    text, used = ta.write_ticket(a, "ollama")
    assert used == "template" and "LLM unavailable" in text


def test_pauc_perfect_and_random():
    y = np.r_[np.zeros(500), np.ones(500)]
    assert abs(pauc(y, y) - 1.0) < 1e-9
    s = np.random.default_rng(0).random(1000)
    assert 0.4 < pauc(y, s) < 0.6  # DCASE / McClish-standardised pAUC: random scorer ~= 0.5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
