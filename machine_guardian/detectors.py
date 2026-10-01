"""Unsupervised anomaly detectors that need only scikit-learn.

All detectors are trained on NORMAL clips only and output a score where higher
means more anomalous.

  pca  - frame-level reconstruction error (linear analogue of the DCASE AE baseline)
  gmm  - frame-level negative log-likelihood under a Gaussian mixture
  knn  - clip-embedding nearest-neighbour distance with per-domain
         normalisation (our domain-shift fix; uses the few target-domain
         normal clips DCASE 2022+ provides)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .audio_features import FeatureConfig, clip_embedding, load_wav, log_mel, stack_frames
from .explain import signal_stats


# --------------------------------------------------------------------------- features
def _extract_one(path: str, cfg: FeatureConfig):
    x = load_wav(path, cfg.sr)
    return log_mel(x, cfg), signal_stats(x, cfg)


def extract(paths, cfg: FeatureConfig = FeatureConfig(), cache_dir: str | None = "cache", n_jobs: int = -1):
    """Log-mels + condition indicators for many files, cached to disk as .npz."""
    paths = list(paths)
    cache = None
    if cache_dir:
        key = hashlib.md5(("|".join(paths) + repr(cfg)).encode()).hexdigest()[:16]
        cache = Path(cache_dir) / f"feat_{key}.npz"
        if cache.exists():
            d = np.load(cache, allow_pickle=True)
            return list(d["mels"]), list(d["stats"])
    out = Parallel(n_jobs=n_jobs)(delayed(_extract_one)(p, cfg) for p in paths)
    mels, stats = [o[0] for o in out], [o[1] for o in out]
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        arr = np.empty(len(mels), dtype=object)
        arr[:] = mels
        np.savez(cache, mels=arr, stats=np.array(stats, dtype=object))
    return mels, stats


def _frames(mels, frames, max_rows=200_000, seed=0):
    X = np.concatenate([stack_frames(S, frames) for S in mels])
    if len(X) > max_rows:
        X = X[np.random.default_rng(seed).choice(len(X), max_rows, replace=False)]
    return X


# --------------------------------------------------------------------------- detectors
class PCADetector:
    name = "pca"

    def __init__(self, cfg=FeatureConfig(), var=0.9):
        self.cfg, self.var = cfg, var

    def fit(self, mels, domains=None):
        X = _frames(mels, self.cfg.frames)
        self.scaler = StandardScaler().fit(X)
        self.pca = PCA(n_components=self.var, svd_solver="full", random_state=0).fit(self.scaler.transform(X))
        return self

    def score(self, mels, domains=None):
        out = []
        for S in mels:
            Z = self.scaler.transform(stack_frames(S, self.cfg.frames))
            R = self.pca.inverse_transform(self.pca.transform(Z))
            out.append(float(((Z - R) ** 2).mean()))
        return np.array(out)


class GMMDetector:
    name = "gmm"

    def __init__(self, cfg=FeatureConfig(), n_components=8, dims=32):
        self.cfg, self.k, self.dims = cfg, n_components, dims

    def fit(self, mels, domains=None):
        X = _frames(mels, self.cfg.frames, max_rows=60_000)
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        self.pca = PCA(n_components=min(self.dims, Z.shape[1]), random_state=0).fit(Z)
        self.gmm = GaussianMixture(self.k, covariance_type="diag", random_state=0, reg_covar=1e-3)
        self.gmm.fit(self.pca.transform(Z))
        return self

    def score(self, mels, domains=None):
        return np.array([
            -float(self.gmm.score(self.pca.transform(self.scaler.transform(stack_frames(S, self.cfg.frames)))))
            for S in mels
        ])


class DomainKNNDetector:
    """kNN on clip embeddings with per-domain standardisation.

    Source and target normals are standardised separately and each domain's
    distances are divided by that domain's typical normal-to-normal distance.
    A test clip's score is the smallest normalised distance across domains,
    so a clip only needs to look normal for *one* operating condition.
    Pass `domain_aware=False` to get the naive baseline for comparison.
    """
    name = "knn"

    def __init__(self, k=1, domain_aware=True):
        self.k, self.domain_aware = k, domain_aware

    def fit(self, mels, domains=None):
        E = np.stack([clip_embedding(S) for S in mels])
        domains = np.array(domains if (domains is not None and self.domain_aware) else ["all"] * len(E))
        self.models = {}
        loo = np.full(len(E), np.inf)
        for d in np.unique(domains):
            Ed = E[domains == d]
            sc = StandardScaler().fit(Ed)
            Zd = sc.transform(Ed)
            k = min(self.k + 1, len(Zd))
            nn = NearestNeighbors(n_neighbors=k).fit(Zd)
            dist, _ = nn.kneighbors(Zd)  # first neighbour is the point itself
            ref = float(np.median(dist[:, 1:].mean(1))) if k > 1 else 1.0
            ref = max(ref, 1e-6)
            if k > 1:
                loo[domains == d] = dist[:, 1:].mean(1) / ref
            self.models[d] = (sc, NearestNeighbors(n_neighbors=min(self.k, len(Zd))).fit(Zd), ref)
        # leave-one-out scores of the training normals, used to set the alert threshold
        self.train_scores_ = loo
        return self

    def score(self, mels, domains=None):
        E = np.stack([clip_embedding(S) for S in mels])
        per_domain = []
        for sc, nn, ref in self.models.values():
            dist, _ = nn.kneighbors(sc.transform(E))
            per_domain.append(dist.mean(1) / ref)
        return np.min(np.stack(per_domain), axis=0)


DETECTORS = {"pca": PCADetector, "gmm": GMMDetector, "knn": DomainKNNDetector}
