"""Evaluation metrics following the DCASE Task 2 protocol."""
from __future__ import annotations

import numpy as np
from scipy.stats import hmean
from sklearn.metrics import roc_auc_score


def auc(y, s) -> float:
    return float(roc_auc_score(y, s))


def pauc(y, s, max_fpr: float = 0.1) -> float:
    """Partial AUC over FPR in [0, max_fpr], exactly as the official DCASE
    Task 2 evaluator computes it: scikit-learn's `roc_auc_score(max_fpr=...)`,
    which applies the McClish standardisation (random scorer = 0.5,
    perfect = 1.0). This makes our numbers directly comparable to the
    published baseline tables.
    """
    return float(roc_auc_score(y, s, max_fpr=max_fpr))


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
