"""Edge deployment: export the CNN to ONNX, quantise to INT8 and benchmark.

    # 1) train + export (needs torch; run on your laptop GPU)
    python -m machine_guardian.edge_export export --data data/dcase2023/dev --machine fan --epochs 40
    # 2) benchmark (needs only onnxruntime + numpy; run this ON the Raspberry Pi)
    python -m machine_guardian.edge_export bench --model models/fan_idcnn.int8.onnx

The benchmark times the full path a device would run: wav samples -> log-mel
(numpy) -> ONNX model, and reports per-clip latency and real-time factor.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from .audio_features import FeatureConfig, log_mel

PATCH_T = 64


def export(args):
    import torch

    from .data import scan
    from .detectors import extract
    from .torch_models import IDCNNDetector

    cfg = FeatureConfig()
    df = scan(args.data)
    df = df[(df.machine == args.machine) & (df.split == "train")]
    mels, _ = extract(df.path, cfg, cache_dir=args.cache)
    aux = [f"{d}|{a}" for d, a in zip(df.domain, df.attr)]  # section x domain x attribute labels
    det = IDCNNDetector(cfg, epochs=args.epochs).fit(mels, aux, list(df.section))

    out = Path(args.models)
    out.mkdir(parents=True, exist_ok=True)
    fp32 = out / f"{args.machine}_idcnn.onnx"
    det.model.eval()
    kw = dict(input_names=["logmel"], output_names=["logits"],
              dynamic_axes={"logmel": {0: "batch"}}, opset_version=17)
    try:  # newer PyTorch defaults to the dynamo exporter; the classic one handles this CNN fine
        torch.onnx.export(det.model, det.export_inputs(), str(fp32), dynamo=False, **kw)
    except TypeError:  # older PyTorch without the `dynamo` argument
        torch.onnx.export(det.model, det.export_inputs(), str(fp32), **kw)
    int8 = out / f"{args.machine}_idcnn.int8.onnx"
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    except Exception as e:
        print(f"INT8 quantisation failed ({e}); benchmark the FP32 model instead.")
    meta = {"classes": det.classes, "mu": det.mu, "sd": det.sd, "patch_t": PATCH_T}
    (out / f"{args.machine}_idcnn.meta.json").write_text(json.dumps(meta))
    sizes = {p.name: round(p.stat().st_size / 1e6, 2) for p in (fp32, int8) if p.exists()}
    print("Model sizes (MB):", sizes)


def bench(args):
    import onnxruntime as ort

    cfg = FeatureConfig()
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(args.model, so, providers=["CPUExecutionProvider"])
    meta_path = Path(args.model.replace(".int8.onnx", ".onnx")).with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"mu": 0.0, "sd": 1.0}
    clip = (np.random.default_rng(0).standard_normal(int(cfg.sr * args.seconds)) * 0.1).astype(np.float32)

    def run_once():
        S = log_mel(clip, cfg)
        T = S.shape[1]
        P = np.stack([S[:, s:s + PATCH_T] for s in range(0, T - PATCH_T + 1, PATCH_T // 2)])
        x = ((P - meta["mu"]) / meta["sd"])[:, None].astype(np.float32)
        return sess.run(None, {sess.get_inputs()[0].name: x})

    for _ in range(5):
        run_once()
    times = []
    for _ in range(args.repeats):
        t = time.perf_counter()
        run_once()
        times.append((time.perf_counter() - t) * 1000)
    res = {
        "model": os.path.basename(args.model), "model_mb": round(os.path.getsize(args.model) / 1e6, 2),
        "device": f"{platform.machine()} / {platform.processor() or platform.platform()}",
        "threads": args.threads, "clip_seconds": args.seconds,
        "latency_ms_median": round(float(np.median(times)), 1),
        "latency_ms_p95": round(float(np.percentile(times, 95)), 1),
        "real_time_factor": round(float(np.median(times)) / 1000 / args.seconds, 4),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "edge_benchmark.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--data", required=True)
    e.add_argument("--machine", required=True)
    e.add_argument("--epochs", type=int, default=40)
    e.add_argument("--models", default="models")
    e.add_argument("--cache", default="cache")
    b = sub.add_parser("bench")
    b.add_argument("--model", required=True)
    b.add_argument("--seconds", type=float, default=10.0)
    b.add_argument("--repeats", type=int, default=50)
    b.add_argument("--threads", type=int, default=4)
    b.add_argument("--out", default="results")
    args = ap.parse_args()
    export(args) if args.cmd == "export" else bench(args)


if __name__ == "__main__":
    main()
