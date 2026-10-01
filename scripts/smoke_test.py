"""End-to-end smoke test on synthetic data (about 2-4 minutes on a laptop CPU).

    python scripts/smoke_test.py

Checks every stage runs. Numbers from this run are NOT real results.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sh(*args):
    print("\n$", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


has_torch = importlib.util.find_spec("torch") is not None
has_onnx = importlib.util.find_spec("onnx") is not None

sh("-m", "machine_guardian.synth", "--out", "data/synthetic", "--sections", "2")
dets = "knn,knn_naive,pca,gmm" + (",ae,idcnn" if has_torch else "")
sh("-m", "machine_guardian.evaluate_audio", "--data", "data/synthetic", "--detectors", dets,
   "--epochs", "3", "--out", "results", "--models", "models")
sh("scripts/validate_explanations.py")
sh("-m", "machine_guardian.tabular_bosch", "--synthetic", "--folds", "3", "--shap-rows", "3000", "--out", "results")
wav = sorted((ROOT / "data/synthetic/fan/test").glob("section_00_source_test_anomaly_*.wav"))[0]
sh("-m", "machine_guardian.ticket_agent", "--bundle", "models/fan_section_00.joblib", "--wav", str(wav))
if has_torch and has_onnx:
    sh("-m", "machine_guardian.edge_export", "export", "--data", "data/synthetic", "--machine", "fan", "--epochs", "2")
    sh("-m", "machine_guardian.edge_export", "bench", "--model", "models/fan_idcnn.int8.onnx",
       "--repeats", "10", "--out", "results")
else:
    print("\n(torch/onnx not installed: skipped deep models and edge export)")
print("\nSMOKE TEST PASSED")
