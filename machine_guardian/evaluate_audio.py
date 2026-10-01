"""Train + evaluate anomaly detectors per machine type and section.

Example:
    python -m machine_guardian.evaluate_audio --data data/dcase2023/dev --detectors knn,knn_naive,pca,gmm,ae,idcnn

Writes:
    results/audio_metrics.csv   one row per machine/section/detector
    results/audio_summary.json  averaged metrics per detector (feeds resume.py)
    models/<machine>.joblib     a deployable bundle for the dashboard / agent
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .audio_features import FeatureConfig
from .data import scan
from .detectors import DETECTORS, DomainKNNDetector, extract
from .explain import NormalProfile
from .metrics import auc, official_score, pauc, threshold_from_normals

SKLEARN_PER_SECTION = {"pca", "gmm", "knn", "knn_naive"}


def make_detector(name, cfg, epochs):
    if name == "knn_naive":
        d = DomainKNNDetector(domain_aware=False)
        d.name = "knn_naive"
        return d
    if name == "knn":
        return DomainKNNDetector()
    if name in DETECTORS:
        return DETECTORS[name](cfg)
    from .torch_models import TORCH_DETECTORS  # imported lazily: torch is optional
    kw = {"epochs": epochs} if epochs else {}
    return TORCH_DETECTORS[name](cfg, **kw)


def _fit_score(det, tr_m, tr_d, tr_s, te_m, te_d, te_s, tr_a=None):
    if det.name in ("idcnn",):
        # auxiliary labels = section x domain x operating attributes (if present)
        det.fit(tr_m, tr_a, tr_s)
        return det.score(te_m, te_d, te_s), det.score(tr_m, tr_d, tr_s)
    det.fit(tr_m, tr_d)
    s_tr = getattr(det, "train_scores_", None)
    if s_tr is None or not np.isfinite(s_tr).all():
        s_tr = det.score(tr_m, tr_d)
    return det.score(te_m, te_d), s_tr


def section_metrics(y, s, dom):
    """DCASE 2023 convention: AUC per domain = that domain's normals vs all
    anomalies; pAUC over everything."""
    y, s, dom = np.asarray(y), np.asarray(s), np.asarray(dom)
    out = {"pauc": pauc(y, s)}
    for d in ("source", "target"):
        m = (dom == d) | (y == 1)
        has_normals = ((dom == d) & (y == 0)).any()
        out[f"auc_{d}"] = auc(y[m], s[m]) if has_normals and len(set(y[m])) == 2 else None
    return out


def run(args):
    cfg = FeatureConfig()
    df = scan(args.data, args.layout)
    if args.machines:
        df = df[df.machine.isin(args.machines.split(","))]
    df = df[(df.split == "train") | df.label.notna()]  # need labels to evaluate
    names = args.detectors.split(",")
    rows, out_dir, model_dir = [], Path(args.out), Path(args.models)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    timing = {}

    for machine, mdf in df.groupby("machine"):
        print(f"\n=== {machine}: {len(mdf)} clips ===")
        mels, stats = extract(mdf.path, cfg, cache_dir=args.cache)
        mdf = mdf.reset_index(drop=True)
        mdf["mel"], mdf["stats"] = mels, stats
        tr_all, te_all = mdf[mdf.split == "train"], mdf[mdf.split == "test"]

        for name in names:
            t0 = time.time()
            groups = tr_all.groupby("section") if name in SKLEARN_PER_SECTION else [("__all__", tr_all)]
            scores = pd.Series(index=te_all.index, dtype=float)
            for sec, tr in groups:
                te = te_all if sec == "__all__" else te_all[te_all.section == sec]
                det = make_detector(name, cfg, args.epochs)
                tr_a = [f"{d}|{a}" for d, a in zip(tr.domain, tr.get("attr", [""] * len(tr)))]
                s_te, s_tr = _fit_score(det, list(tr.mel), list(tr.domain), list(tr.section),
                                        list(te.mel), list(te.domain), list(te.section), tr_a)
                scores.loc[te.index] = s_te
                if name == args.deploy and sec != "__all__":
                    profile = NormalProfile().fit(list(tr.mel), list(tr.stats), cfg)
                    joblib.dump({"detector": det, "profile": profile, "cfg": cfg,
                                 "threshold": threshold_from_normals(s_tr), "machine": machine,
                                 "section": sec, "detector_name": name},
                                model_dir / f"{machine}_{sec}.joblib")
            timing[name] = timing.get(name, 0) + time.time() - t0
            for sec, te in te_all.groupby("section"):
                m = section_metrics(te.label.astype(int), scores.loc[te.index], te.domain)
                rows.append({"machine": machine, "section": sec, "detector": name, **m})
                fmt = lambda v: "  -  " if v is None else f"{v:.3f}"
                print(f"  {name:10s} {sec:11s} AUC-src {fmt(m['auc_source'])}  "
                      f"AUC-tgt {fmt(m['auc_target'])}  pAUC {fmt(m['pauc'])}")
            te_scores = te_all[["path", "machine", "section", "domain", "label"]].assign(score=scores, detector=name)
            te_scores.to_csv(out_dir / f"scores_{machine}_{name}.csv", index=False)

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "audio_metrics.csv", index=False)
    summary = {}
    for name, g in res.groupby("detector"):
        src, tgt = g.auc_source.dropna(), g.auc_target.dropna()
        summary[name] = {
            "auc_source": round(float(src.mean()), 4) if len(src) else None,
            "auc_target": round(float(tgt.mean()), 4) if len(tgt) else None,
            "pauc": round(float(g.pauc.mean()), 4),
            "official_score": round(official_score(src, tgt, g.pauc), 4),
            "train_eval_seconds": round(timing[name], 1),
        }
    meta = {"dataset": str(args.data), "machines": sorted(res.machine.unique().tolist()),
            "n_sections": int(res[["machine", "section"]].drop_duplicates().shape[0]),
            "n_clips": int(len(df))}
    (out_dir / "audio_summary.json").write_text(json.dumps({"meta": meta, "detectors": summary}, indent=2))
    print("\nSummary:\n" + json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root")
    ap.add_argument("--layout", default="auto", choices=["auto", "dcase", "mimii"])
    ap.add_argument("--machines", default="", help="comma-separated subset, e.g. fan,pump")
    ap.add_argument("--detectors", default="knn,knn_naive,pca,gmm")
    ap.add_argument("--deploy", default="knn", help="detector to save for the dashboard")
    ap.add_argument("--epochs", type=int, default=0, help="override epochs for torch models")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", default="models")
    ap.add_argument("--cache", default="cache")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
