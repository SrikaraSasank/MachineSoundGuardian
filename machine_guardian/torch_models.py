"""Deep detectors (PyTorch). Train on your GPU: these are the headline models.

  ae    - dense autoencoder on 5-frame log-mel context (the official DCASE
          baseline architecture), score = reconstruction error
  idcnn - self-supervised auxiliary-classification CNN: learns to tell the
          machine sections / operating domains apart using normal data only.
          Anomalous sounds break those learned cues, so the score is the
          negative log-probability of the clip's true section. This family of
          methods is what most top DCASE systems build on.

Both expose the same fit(mels, domains, sections) / score(...) interface as
the scikit-learn detectors.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .audio_features import FeatureConfig, stack_frames

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_T = 64  # frames per CNN patch (~2 s at hop 512 / 16 kHz)


# --------------------------------------------------------------------------- autoencoder
class DenseAE(nn.Module):
    def __init__(self, d_in: int, hidden: int = 128, bottleneck: int = 8):
        super().__init__()

        def block(i, o):
            return [nn.Linear(i, o), nn.BatchNorm1d(o), nn.ReLU()]

        self.net = nn.Sequential(
            *block(d_in, hidden), *block(hidden, hidden), *block(hidden, hidden), *block(hidden, hidden),
            *block(hidden, bottleneck),
            *block(bottleneck, hidden), *block(hidden, hidden), *block(hidden, hidden), *block(hidden, hidden),
            nn.Linear(hidden, d_in),
        )

    def forward(self, x):
        return self.net(x)


class AEDetector:
    name = "ae"

    def __init__(self, cfg=FeatureConfig(), epochs=100, batch=512, lr=1e-3):
        self.cfg, self.epochs, self.batch, self.lr = cfg, epochs, batch, lr

    def fit(self, mels, domains=None, sections=None):
        X = np.concatenate([stack_frames(S, self.cfg.frames) for S in mels])
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        X = torch.from_numpy((X - self.mu) / self.sd).float()
        self.model = DenseAE(X.shape[1]).to(DEVICE)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        dl = DataLoader(TensorDataset(X), batch_size=self.batch, shuffle=True, drop_last=True)
        self.model.train()
        for ep in range(self.epochs):
            tot = 0.0
            for (xb,) in dl:
                xb = xb.to(DEVICE)
                loss = F.mse_loss(self.model(xb), xb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(xb)
            if ep % 10 == 0 or ep == self.epochs - 1:
                print(f"  [ae] epoch {ep:3d}  loss {tot / len(X):.4f}")
        return self

    @torch.no_grad()
    def score(self, mels, domains=None, sections=None):
        self.model.eval()
        out = []
        for S in mels:
            x = torch.from_numpy((stack_frames(S, self.cfg.frames) - self.mu) / self.sd).float().to(DEVICE)
            out.append(float(F.mse_loss(self.model(x), x).item()))
        return np.array(out)


# --------------------------------------------------------------------------- ID classifier
class SectionCNN(nn.Module):
    """Small CNN over (1, n_mels, PATCH_T) log-mel patches. ~0.4 M parameters,
    small enough for real-time CPU / Raspberry Pi inference after INT8."""

    def __init__(self, n_classes: int, n_mels: int = 128, emb: int = 128):
        super().__init__()

        def conv(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.MaxPool2d(2))

        self.features = nn.Sequential(conv(1, 16), conv(16, 32), conv(32, 64), conv(64, 128))
        self.emb = nn.Linear(128, emb)
        self.head = nn.Linear(emb, n_classes)

    def forward(self, x):  # x: (B, 1, n_mels, T)
        h = self.features(x).mean(dim=(2, 3))
        return self.head(F.relu(self.emb(h)))


def _patches(S: np.ndarray, hop: int = PATCH_T // 2) -> np.ndarray:
    T = S.shape[1]
    if T < PATCH_T:
        S = np.pad(S, ((0, 0), (0, PATCH_T - T)), mode="edge")
        T = PATCH_T
    starts = range(0, T - PATCH_T + 1, hop)
    return np.stack([S[:, s:s + PATCH_T] for s in starts])


class IDCNNDetector:
    name = "idcnn"

    def __init__(self, cfg=FeatureConfig(), epochs=40, batch=64, lr=1e-3, mixup=0.2):
        self.cfg, self.epochs, self.batch, self.lr, self.mixup = cfg, epochs, batch, lr, mixup

    def _label(self, section, domain):
        return f"{section}|{domain}"

    def fit(self, mels, domains, sections):
        labels = [self._label(s, d) for s, d in zip(sections, domains)]
        self.classes = sorted(set(labels))
        if len(self.classes) < 2:
            raise ValueError("idcnn needs >= 2 sections/domains; use ae/knn for single-ID data")
        cls_idx = {c: i for i, c in enumerate(self.classes)}
        P, Y = [], []
        for S, lab in zip(mels, labels):
            p = _patches(S)
            P.append(p)
            Y += [cls_idx[lab]] * len(p)
        P = np.concatenate(P)
        self.mu, self.sd = float(P.mean()), float(P.std() + 1e-6)
        X = torch.from_numpy((P - self.mu) / self.sd).float().unsqueeze(1)
        Y = torch.tensor(Y)
        # class-balanced weights: target domains have only a few clips
        counts = torch.bincount(Y, minlength=len(self.classes)).float()
        weight = (counts.sum() / (counts + 1)).to(DEVICE)
        weight = weight / weight.mean()
        self.model = SectionCNN(len(self.classes), self.cfg.n_mels).to(DEVICE)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        dl = DataLoader(TensorDataset(X, Y), batch_size=self.batch, shuffle=True, drop_last=True)
        for ep in range(self.epochs):
            self.model.train()
            tot = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                lam = np.random.beta(self.mixup, self.mixup) if self.mixup else 1.0
                perm = torch.randperm(len(xb), device=DEVICE)
                logits = self.model(lam * xb + (1 - lam) * xb[perm])
                loss = lam * F.cross_entropy(logits, yb, weight=weight) + \
                    (1 - lam) * F.cross_entropy(logits, yb[perm], weight=weight)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(xb)
            sched.step()
            if ep % 5 == 0 or ep == self.epochs - 1:
                print(f"  [idcnn] epoch {ep:3d}  loss {tot / len(X):.4f}")
        self.cls_idx = cls_idx
        return self

    @torch.no_grad()
    def score(self, mels, domains, sections):
        self.model.eval()
        out = []
        for S, s, d in zip(mels, sections, domains):
            x = torch.from_numpy((_patches(S) - self.mu) / self.sd).float().unsqueeze(1).to(DEVICE)
            logp = F.log_softmax(self.model(x), dim=1)
            # a clip is normal if it looks like its own section in *either* domain
            idx = [self.cls_idx[c] for c in self.cls_idx if c.split("|")[0] == s]
            out.append(float(-logp[:, idx].logsumexp(1).mean().item()))
        return np.array(out)

    def export_inputs(self):
        """Example input for ONNX export (one patch)."""
        return torch.zeros(1, 1, self.cfg.n_mels, PATCH_T, device=DEVICE)


TORCH_DETECTORS = {"ae": AEDetector, "idcnn": IDCNNDetector}
