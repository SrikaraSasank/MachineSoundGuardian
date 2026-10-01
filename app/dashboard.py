"""Streamlit dashboard.

    streamlit run app/dashboard.py
"""
import glob
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from machine_guardian.ticket_agent import assess, write_ticket  # noqa: E402

st.set_page_config(page_title="Machine Sound Guardian", layout="wide")
st.title("Machine Sound Guardian")
st.caption("Acoustic anomaly detection, envelope-analysis diagnostics and grounded maintenance tickets")

tab_live, tab_trend, tab_line = st.tabs(["Inspect a clip", "Fleet monitor", "Production line (Bosch)"])

bundles = sorted(glob.glob(str(ROOT / "models" / "*.joblib")))
COLORS = {"normal": "green", "watch": "orange", "anomalous": "red", "critical": "red"}

# ------------------------------------------------------------------ single clip
with tab_live:
    if not bundles:
        st.warning("No trained models yet. Run `python scripts/smoke_test.py` or `evaluate_audio` first.")
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            bpath = st.selectbox("Machine model", bundles, format_func=lambda p: Path(p).stem)
            bundle = joblib.load(bpath)
            up = st.file_uploader("Upload a .wav", type=["wav"])
            examples = sorted(glob.glob(str(ROOT / "data" / "**" / bundle["machine"] / "test" /
                                            f"{bundle['section']}_*.wav"), recursive=True))
            pick = st.selectbox("...or pick a test clip", examples, format_func=lambda p: Path(p).name) if examples else None
            backend = st.radio("Ticket writer", ["template", "ollama", "anthropic"], horizontal=True)
        wav = None
        if up is not None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(up.read())
            wav = tmp.name
        elif pick:
            wav = pick
        if wav:
            a = assess(bundle, wav)
            with c1:
                st.audio(wav)
                st.metric("Anomaly score / threshold", f"{a['score_to_threshold']}x")
                st.markdown(f"Status: :{COLORS[a['status']]}[**{a['status'].upper()}**]")
                if a["raised_by_guard"]:
                    st.info("Raised by the envelope-analysis guard rail (the ML score alone stayed below the alert level).")
                elif a["physics_guard_triggered"]:
                    st.caption("Envelope analysis independently confirms a periodic fault signature.")
            with c2:
                fig, ax = plt.subplots(figsize=(9, 3.2))
                ax.imshow(a["_mel"], origin="lower", aspect="auto", cmap="magma")
                ax.set_xlabel("frame")
                ax.set_ylabel("mel band")
                ax.set_title("Log-mel spectrogram")
                st.pyplot(fig)
                ticket, used = write_ticket(a, backend)
                st.markdown(ticket)
                st.caption(f"ticket written by: {used}")
                with st.expander("Evidence passed to the ticket writer"):
                    st.json({k: v for k, v in a.items() if not k.startswith("_")})

# ------------------------------------------------------------------ fleet trend
with tab_trend:
    files = sorted(glob.glob(str(ROOT / "results" / "scores_*.csv")))
    if not files:
        st.info("Run evaluate_audio to populate scores.")
    else:
        f = st.selectbox("Score file", files, format_func=lambda p: Path(p).stem)
        df = pd.read_csv(f)
        sec = st.selectbox("Section", sorted(df.section.unique()))
        d = df[df.section == sec].sample(frac=1, random_state=0).reset_index(drop=True)
        # replay test clips as a stream: normals first, then a degradation period
        d = pd.concat([d[d.label == 0], d[d.label == 1]]).reset_index(drop=True)
        d["rolling"] = d.score.rolling(10, min_periods=1).median()
        st.line_chart(d[["score", "rolling"]])
        st.caption("Clips replayed in time order: healthy period, then a fault develops. "
                   "Rolling median smooths single noisy clips before alerting.")
        m = pd.read_csv(ROOT / "results" / "audio_metrics.csv")
        st.dataframe(m.pivot_table(index=["machine", "section"], columns="detector", values="auc_source").round(3))

# ------------------------------------------------------------------ Bosch line
with tab_line:
    sj = ROOT / "results" / "bosch_summary.json"
    if not sj.exists():
        st.info("Run `python -m machine_guardian.tabular_bosch` first.")
    else:
        s = json.loads(sj.read_text())
        st.caption(s["source"])
        k = st.columns(4)
        k[0].metric("Parts", f"{s['n_parts']:,}")
        k[1].metric("Failure rate", f"{s['failure_rate']:.2%}")
        k[2].metric("MCC", s["mcc"])
        k[3].metric("PR-AUC", s["pr_auc"], f"random = {s['pr_auc_random']}")
        png = ROOT / "results" / "bosch_station_importance.png"
        if png.exists():
            st.image(str(png))
        st.dataframe(pd.DataFrame(s["top_drivers"]))
