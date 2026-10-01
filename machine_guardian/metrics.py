"""Evaluation metrics following the DCASE Task 2 protocol."""
from __future__ import annotations

import numpy as np
from scipy.stats import hmean
from sklearn.metrics import roc_auc_score, roc_curve


def auc(y, s) -> float:
    return float(roc_auc_score(y, s))


def pauc(y, s, max_fpr: float = 0.1) -> float:
    """Partial AUC over FPR in [0, max_fpr], normalised by max_fpr.

    DCASE uses the *uncorrected* pAUC (sklearn's `max_fpr` applies the McClish
    correction, which gives different numbers), so we integrate the ROC directly.
    """
    fpr, tpr, _ = roc_curve(y, s)
    stop = np.searchsorted(fpr, max_fpr, side="right")
    x = np.concatenate([fpr[:stop], [max_fpr]])
    y_ = np.concatenate([tpr[:stop], [np.interp(max_fpr, fpr, tpr)]])
    return float(np.trapezoid(y_, x) / max_fpr)


def official_score(aucs_source, aucs_target, paucs) -> float:
    """DCASE 2022+ score: harmonic mean of source AUCs, target AUCs and pAUCs."""
    vals = [v for v in list(aucs_source) + list(aucs_target) + list(paucs) if v is not None]
    return float(hmean(np.clip(vals, 1e-6, None)))


def threshold_from_normals(train_scores, quantile: float = 0.95) -> float:
    """Decision threshold = quantile of scores on normal training clips.

    Uses only normal data, so it is available in a real factory where faults
    are rare and unlabelled.
    """
    return float(np.quantile(train_scores, quantile))
