"""Dataset discovery for DCASE 2020-2024 Task 2, raw MIMII and the synthetic set.

Every file becomes a `Clip` record so the rest of the code never cares about
the original folder layout.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Clip:
    path: str
    machine: str        # fan, pump, valve, slider, ToyCar, ...
    section: str        # machine ID (2020) or section (2021+)
    split: str          # train | test
    label: int | None   # 0 normal, 1 anomaly, None unknown (eval sets)
    domain: str         # source | target
    attr: str = ""      # operating attributes from the file name (DCASE 2022+), e.g. "vel_8_noise_1"


_DCASE20 = re.compile(r"(normal|anomaly)_id_(\d+)_\d+\.wav$")
_DCASE21 = re.compile(r"section_(\d+)_(source|target)_(train|test)_(?:(normal|anomaly)_)?\d+_?(.*)\.wav$")


def _parse_dcase(path: Path, machine: str, split: str) -> Clip | None:
    name = path.name
    m = _DCASE20.search(name)
    if m:
        return Clip(str(path), machine, f"id_{m.group(2)}", split,
                    int(m.group(1) == "anomaly"), "source")
    m = _DCASE21.search(name)
    if m:
        lab = m.group(4)
        return Clip(str(path), machine, f"section_{m.group(1)}", split,
                    None if lab is None else int(lab == "anomaly"), m.group(2), m.group(5) or "")
    return None


def scan_dcase(root: str | Path) -> pd.DataFrame:
    """root/<machine>/{train,test}/*.wav (DCASE dev/additional/eval layouts)."""
    rows = []
    for split_dir in Path(root).glob("*/*"):
        if split_dir.name not in ("train", "test") or not split_dir.is_dir():
            continue
        machine = split_dir.parent.name
        for wav in sorted(split_dir.glob("*.wav")):
            c = _parse_dcase(wav, machine, split_dir.name)
            if c:
                rows.append(c.__dict__)
    return pd.DataFrame(rows)


def scan_mimii(root: str | Path, test_normals: int = 300, seed: int = 0) -> pd.DataFrame:
    """Raw MIMII: root/<machine>/id_XX/{normal,abnormal}/*.wav.

    MIMII has no official split, so we follow the DCASE protocol: per machine ID,
    hold out `test_normals` normal clips (or as many as there are anomalies, if
    fewer) plus all abnormal clips for testing; the remaining normals train.
    """
    rng = random.Random(seed)
    rows = []
    for id_dir in sorted(Path(root).glob("*/id_*")):
        machine = id_dir.parent.name
        normals = sorted((id_dir / "normal").glob("*.wav"))
        abnormals = sorted((id_dir / "abnormal").glob("*.wav"))
        rng.shuffle(normals)
        k = min(test_normals, len(abnormals)) or test_normals
        for i, p in enumerate(normals):
            rows.append(Clip(str(p), machine, id_dir.name, "test" if i < k else "train", 0, "source").__dict__)
        for p in abnormals:
            rows.append(Clip(str(p), machine, id_dir.name, "test", 1, "source").__dict__)
    return pd.DataFrame(rows)


def scan(root: str | Path, layout: str = "auto") -> pd.DataFrame:
    root = Path(root)
    if layout == "auto":
        layout = "mimii" if any(root.glob("*/id_*/normal")) else "dcase"
    df = scan_dcase(root) if layout == "dcase" else scan_mimii(root)
    if df.empty:
        raise FileNotFoundError(f"No clips found under {root} (layout={layout}).")
    return df
