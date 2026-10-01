"""Turn your real results into resume bullets.

    python -m machine_guardian.resume

Reads results/*.json and fills the numbers in. It refuses to fill numbers that
come from the synthetic smoke-test data, so nothing fake ends up on your CV.
"""
from __future__ import annotations

import json
from pathlib import Path

R = Path("results")


def _load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(v):
    return f"{100 * v:.1f}%"


def main():
    audio, bosch, edge = _load("audio_summary.json"), _load("bosch_summary.json"), _load("edge_benchmark.json")
    bullets, todo = [], []

    if audio and "synthetic" not in audio["meta"]["dataset"].lower():
        det = audio["detectors"]
        best = max((k for k in det if k != "knn_naive"), key=lambda k: det[k]["official_score"])
        b = det[best]
        names = {"idcnn": "self-supervised machine-ID CNN", "ae": "dense autoencoder", "knn": "domain-normalised kNN",
                 "pca": "PCA reconstruction", "gmm": "Gaussian mixture"}
        line = (f"Built an unsupervised acoustic anomaly detector for industrial machines "
                f"({', '.join(audio['meta']['machines'])}; {audio['meta']['n_clips']:,} clips, "
                f"{audio['meta']['n_sections']} machine sections) trained only on normal sounds; the "
                f"{names.get(best, best)} reached {pct(b['auc_source'])} AUC")
        if b.get("auc_target") is not None:
            line += f" (source) / {pct(b['auc_target'])} (shifted target domain)"
        line += f" and {pct(b['pauc'])} pAUC on the DCASE benchmark."
        bullets.append(line)
        if "knn" in det and "knn_naive" in det and det["knn"]["auc_target"] and det["knn_naive"]["auc_target"]:
            gain = det["knn"]["auc_target"] - det["knn_naive"]["auc_target"]
            bullets.append(
                f"Tackled domain shift (new speeds, noise, operating conditions) with per-domain normalisation "
                f"using only a few target-domain normal clips, changing target-domain AUC by "
                f"{100 * gain:+.1f} points vs. a pooled baseline.")
    else:
        todo.append("audio: run evaluate_audio on real DCASE/MIMII data")

    if bosch and "synthetic" not in bosch["source"].lower():
        top = [d["driver"] for d in bosch["top_drivers"] if not d["driver"].startswith("[")][:2]
        bullets.append(
            f"Predicted part failures on Bosch's production-line dataset ({bosch['n_parts']:,} parts, "
            f"{bosch['n_features']:,} features, {pct(bosch['failure_rate'])} failure rate) with "
            f"{bosch['model'].split(' ')[0]} and process-flow features: MCC {bosch['mcc']:.3f}, "
            f"PR-AUC {bosch['pr_auc']:.3f} ({bosch['pr_auc'] / bosch['pr_auc_random']:.0f}x random); "
            f"SHAP attribution traced failures to stations {' and '.join(top)}.")
    else:
        todo.append("bosch: run tabular_bosch on the Kaggle data")

    if edge:
        bullets.append(
            f"Exported the CNN to INT8 ONNX ({edge['model_mb']} MB) for edge inference: "
            f"{edge['latency_ms_median']} ms per {edge['clip_seconds']:.0f}-second clip on {edge['device']} "
            f"(real-time factor {edge['real_time_factor']}).")
    else:
        todo.append("edge: run edge_export export + bench")

    bullets.append(
        "Added envelope-analysis diagnostics (shaft speed vs. impact frequency) that label likely causes "
        "(bearing wear, imbalance, rubbing), and an LLM agent that writes maintenance tickets from that "
        "evidence only, with a grounding check that rejects any claim not supported by the measurements.")

    print("Machine Sound Guardian: Acoustic Predictive Maintenance with LLM Diagnostics")
    print("Python, PyTorch, scikit-learn, LightGBM, SHAP, ONNX Runtime, Streamlit, LLM APIs\n")
    for b in bullets:
        print(f"- {b}")
    if todo:
        print("\nStill missing real results for:\n  " + "\n  ".join(todo))


if __name__ == "__main__":
    main()
