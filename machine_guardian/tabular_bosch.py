"""Bosch Production Line Performance: predict which parts fail QC, and find
which stations drive failures.

Data (Kaggle competition "bosch-production-line-performance"):
    train_numeric.csv  ~1.18M parts x 968 anonymised measurements + Response
    train_date.csv     timestamps for each measurement (same column naming)
Columns are named L{line}_S{station}_F{feature} / _D{feature}.

Example:
    python -m machine_guardian.tabular_bosch --data data/bosch --nrows 300000
    python -m machine_guardian.tabular_bosch --synthetic          # smoke test

Writes results/bosch_summary.json, results/bosch_station_importance.csv,
results/bosch_station_importance.png
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    import lightgbm as lgb
except ImportError:  # sklearn fallback keeps the pipeline runnable anywhere
    lgb = None

STATION = re.compile(r"^(L\d+_S\d+)_")


# --------------------------------------------------------------------------- loading
def load_bosch(data_dir: str, nrows: int | None = None, chunksize: int = 100_000):
    """Load numeric data as float32, plus per-part start/end time from the date file.

    train_date.csv is ~2.9 GB, so it is streamed in chunks and reduced to
    (Id, t_start, t_end) on the fly; only those three columns are kept.
    """
    d = Path(data_dir)
    head = pd.read_csv(d / "train_numeric.csv", nrows=1)
    dtypes = {c: np.float32 for c in head.columns if c not in ("Id", "Response")}
    num = pd.read_csv(d / "train_numeric.csv", nrows=nrows, dtype=dtypes)
    date_path = d / "train_date.csv"
    if not date_path.exists():
        return num, None
    dh = pd.read_csv(date_path, nrows=1).columns
    parts = []
    for ch in pd.read_csv(date_path, nrows=nrows, chunksize=chunksize,
                          dtype={c: np.float32 for c in dh if c != "Id"}):
        t = ch.drop(columns="Id")
        parts.append(pd.DataFrame({"Id": ch.Id.values, "t_start": t.min(axis=1).values, "t_end": t.max(axis=1).values}))
    return num, pd.concat(parts, ignore_index=True)


def make_synthetic(n: int = 40_000, seed: int = 0):
    """Bosch-shaped toy data: sparse station measurements, ~0.6% failures that
    are driven by two stations (L1_S24, L3_S32) and long cycle times."""
    rng = np.random.default_rng(seed)
    stations = [f"L{l}_S{s}" for l, rng_s in [(0, range(0, 12)), (1, range(24, 26)),
                                              (2, range(26, 29)), (3, range(29, 52))] for s in rng_s]
    cols, data = [], {}
    route = rng.random((n, len(stations))) < 0.45  # which stations each part visits
    for j, st in enumerate(stations):
        for f in range(4):
            name = f"{st}_F{j * 4 + f}"
            v = rng.normal(0, 0.1, n).astype(np.float32)
            v[~route[:, j]] = np.nan
            data[name] = v
            cols.append(name)
    num = pd.DataFrame(data)
    num.insert(0, "Id", np.arange(1, n + 1) * 2)
    start = np.sort(rng.uniform(0, 1700, n)).astype(np.float32)
    dur = rng.gamma(2.0, 1.5, n).astype(np.float32)
    risk = (
        -6.2
        + 18 * np.nan_to_num(num["L3_S32_F" + str(stations.index("L3_S32") * 4)].values)
        + 14 * np.abs(np.nan_to_num(num["L1_S24_F" + str(stations.index("L1_S24") * 4 + 1)].values))
        + 0.25 * dur
    )
    num["Response"] = (rng.random(n) < 1 / (1 + np.exp(-risk))).astype(int)
    date = pd.DataFrame({"Id": num.Id, "t_start": start, "t_end": start + dur})
    return num, date


# --------------------------------------------------------------------------- features
def engineer(num: pd.DataFrame, date: pd.DataFrame | None) -> pd.DataFrame:
    X = num.drop(columns=["Response"], errors="ignore").copy()
    meas = [c for c in X.columns if c != "Id"]
    X["n_measured"] = X[meas].notna().sum(axis=1).astype(np.float32)
    # process path: which stations a part went through (hashed route signature)
    stations = sorted({STATION.match(c).group(1) for c in meas if STATION.match(c)})
    visited = np.stack([X[[c for c in meas if c.startswith(s + "_")]].notna().any(axis=1).values
                        for s in stations], axis=1)
    X["n_stations"] = visited.sum(1)
    route = pd.Series([r.tobytes() for r in np.packbits(visited, axis=1)], index=X.index)
    X["route_frequency"] = route.map(route.value_counts()).astype(np.float32)  # rare routes = rework/unusual flow
    if date is not None:
        t = date.set_index("Id").reindex(X.Id)
        X["t_start"] = t.t_start.values
        X["t_end"] = t.t_end.values
        X["cycle_time"] = X.t_end - X.t_start
        # neighbours in production order: parts made back-to-back share conditions
        order = X.sort_values(["t_start", "Id"]).index
        ids = X.loc[order, "Id"].values
        X.loc[order, "id_gap_prev"] = np.r_[np.nan, np.diff(ids)]
        X.loc[order, "id_gap_next"] = np.r_[np.diff(ids), np.nan]
        X["parts_same_start"] = X.groupby("t_start")["Id"].transform("count")
    return X.drop(columns=["Id"])


# --------------------------------------------------------------------------- modelling
def best_mcc_threshold(y, p):
    grid = np.quantile(p, np.linspace(0.90, 0.9995, 300))
    scores = [matthews_corrcoef(y, p >= t) for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def _model(pos_weight, seed):
    if lgb is not None:
        return lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
            scale_pos_weight=pos_weight, reg_lambda=1.0, random_state=seed, verbose=-1)
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=63,
                                          class_weight={0: 1, 1: pos_weight}, random_state=seed)


def contributions(model, X: pd.DataFrame, y=None) -> pd.Series:
    """Mean |SHAP| per feature. LightGBM computes exact TreeSHAP natively
    (pred_contrib), so no extra dependency is needed."""
    if lgb is not None:
        sv = model.booster_.predict(X, pred_contrib=True)
        return pd.Series(np.abs(sv[:, :-1]).mean(0), index=X.columns)
    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, X, y, scoring="average_precision", n_repeats=3, random_state=0, n_jobs=-1)
    return pd.Series(np.clip(r.importances_mean, 0, None), index=X.columns)


def station_of(feature: str) -> str:
    m = STATION.match(feature)
    return m.group(1) if m else f"[engineered] {feature}"


def run(args):
    t0 = time.time()
    if args.synthetic:
        num, date = make_synthetic()
        source = "synthetic (smoke test)"
    else:
        num, date = load_bosch(args.data, args.nrows)
        source = f"Kaggle Bosch Production Line Performance ({args.data})"
    y = num["Response"].values.astype(int)
    X = engineer(num, date)
    print(f"Data: {X.shape[0]:,} parts, {X.shape[1]:,} features, failure rate {y.mean():.3%}")

    pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1)) ** 0.5  # sqrt: calmer than full ratio
    oof = np.zeros(len(y))
    skf = StratifiedKFold(args.folds, shuffle=True, random_state=0)
    models = []
    for k, (tr, va) in enumerate(skf.split(X, y)):
        m = _model(pos_weight, k).fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        models.append((m, va))
        print(f"  fold {k}: AUC {roc_auc_score(y[va], oof[va]):.4f}")

    thr, mcc = best_mcc_threshold(y, oof)
    flagged = oof >= thr
    summary = {
        "source": source, "model": "LightGBM" if lgb else "HistGradientBoosting (LightGBM not installed)",
        "n_parts": int(len(y)), "n_features": int(X.shape[1]), "failure_rate": round(float(y.mean()), 5),
        "cv_folds": args.folds, "auc": round(roc_auc_score(y, oof), 4),
        "pr_auc": round(average_precision_score(y, oof), 4), "pr_auc_random": round(float(y.mean()), 4),
        "mcc": round(mcc, 4), "threshold": thr,
        "recall_at_threshold": round(float(flagged[y == 1].mean()), 4),
        "precision_at_threshold": round(float(y[flagged].mean()) if flagged.any() else 0.0, 4),
        "pct_parts_flagged": round(float(flagged.mean()), 5),
    }

    # explanations from the last fold's model on its validation rows
    m, va = models[-1]
    Xs = X.iloc[va]
    if len(Xs) > args.shap_rows:
        Xs = Xs.sample(args.shap_rows, random_state=0)
    imp = contributions(m, Xs, y[Xs.index] if lgb is None else None)
    st = imp.groupby(imp.index.map(station_of)).sum().sort_values(ascending=False)
    st = (st / st.sum()).rename("share_of_attribution")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    st.to_csv(out / "bosch_station_importance.csv")
    summary["top_drivers"] = [{"driver": k, "share": round(float(v), 4)} for k, v in st.head(8).items()]
    summary["runtime_seconds"] = round(time.time() - t0, 1)
    (out / "bosch_summary.json").write_text(json.dumps(summary, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top = st.head(12)[::-1]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.barh(top.index, top.values, color="#2b6cb0")
        ax.set_xlabel("Share of mean |SHAP| attribution")
        ax.set_title("What drives part failures (by station)")
        fig.tight_layout()
        fig.savefig(out / "bosch_station_importance.png", dpi=150)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bosch")
    ap.add_argument("--nrows", type=int, default=None, help="limit rows (e.g. 300000) for faster runs")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--shap-rows", type=int, default=20000)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="results")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
