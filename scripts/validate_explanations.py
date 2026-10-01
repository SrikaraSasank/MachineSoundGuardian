"""Check that the rule-based fault hypotheses match the injected faults
on the synthetic set (sanity check for the explanation layer).

    python scripts/validate_explanations.py
"""
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from machine_guardian.ticket_agent import assess  # noqa: E402


def guess(a):
    h = " ".join(a["hypotheses"]).lower()
    if a["status"] in ("normal", "watch"):
        return "none"
    for key, lab in (("bearing", "bearing"), ("imbalance", "imbalance"), ("tone", "whine")):
        if key in h:
            return lab
    return "unclear"


def main(root="data/synthetic", models="models"):
    gt = pd.read_csv(f"{root}/fault_ground_truth.csv")
    rows = []
    for bpath in sorted(glob.glob(f"{models}/*.joblib")):
        b = joblib.load(bpath)
        files = glob.glob(f"{root}/{b['machine']}/test/{b['section']}_*_test_*.wav")
        for f in files:
            a = assess(b, f)
            name = Path(f).name
            fault = gt[(gt.file == name) & (gt.machine == b["machine"])].fault.iloc[0]
            rows.append({"fault": fault, "guess": guess(a), "guard": a["physics_guard_triggered"],
                         "ml_flag": a["score_to_threshold"] >= 1.5,
                         **{f"z_{k}": v for k, v in a["indicator_z"].items()}})
    df = pd.DataFrame(rows)
    print(pd.crosstab(df.fault, df.guess))
    print("\nMedian indicator z by true fault:\n",
          df.groupby("fault")[["z_kurtosis_hp", "z_envelope_prominence"]].median())
    acc = (df.fault == df.guess).mean()
    det = df[df.fault != "none"]
    print(f"\nAgreement with injected fault (all clips): {acc:.1%}")
    print(f"Correct cause among anomalies that were flagged: "
          f"{(det[det.guess != 'none'].fault == det[det.guess != 'none'].guess).mean():.1%}")
    anom, norm = df.fault != "none", df.fault == "none"
    hybrid = df.ml_flag | df.guard
    print(f"Detection recall  ML only: {df.ml_flag[anom].mean():.1%}   ML + physics guard: {hybrid[anom].mean():.1%}")
    print(f"False-alarm rate  ML only: {df.ml_flag[norm].mean():.1%}   ML + physics guard: {hybrid[norm].mean():.1%}")
    return df


if __name__ == "__main__":
    main()
