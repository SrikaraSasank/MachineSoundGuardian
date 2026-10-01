# Machine Sound Guardian

**Acoustic predictive maintenance with physics-based diagnostics and a grounded LLM ticket agent.**

Listen to industrial machines (fans, pumps, valves, slide rails), flag developing faults using only
recordings of *healthy* operation, explain which component is the likely cause, and write a maintenance
ticket a technician can act on. A second module predicts part failures on a real Bosch production line
and traces them back to the stations responsible.

```
 wav ──► log-mel (numpy) ──► anomaly detector ──► score vs. threshold ─┐
   │                        kNN / PCA / GMM / AE / ID-CNN              │
   └──► envelope analysis ──► shaft speed, impact frequency ──► physics guard rail
                                     │                                  │
                                     ▼                                  ▼
                           evidence JSON  ──► LLM (Ollama / Claude) ──► grounding check ──► ticket
                                                    └──── fails? ──► deterministic template

 Bosch line data ──► process-flow features ──► LightGBM (5-fold CV) ──► MCC / PR-AUC
                                                     └──► TreeSHAP ──► failure drivers by station
```

## What is in here

| Module | What it does |
|---|---|
| `audio_features.py` | Log-mel spectrograms in pure numpy/scipy, so the same code runs on a laptop and a Raspberry Pi |
| `detectors.py` | Unsupervised detectors trained on normal sound only: PCA reconstruction, GMM, and a **domain-normalised kNN** that adapts to new operating conditions from a handful of clips |
| `torch_models.py` | GPU models: the DCASE dense autoencoder baseline, and a **self-supervised ID-classifier CNN** (learns to recognise machine section, domain and operating attributes; anomalies break those cues) with mixup and class balancing |
| `evaluate_audio.py` | DCASE protocol: per-section AUC (source / target domain), pAUC (FPR ≤ 0.1, uncorrected), official harmonic-mean score |
| `explain.py` | **Envelope analysis** from vibration engineering: shaft speed vs. the repetition frequency of high-frequency impacts separates imbalance (1×), bearing defects (non-integer multiple), and blade-pass / gear-mesh (integer multiple) |
| `ticket_agent.py` | Hybrid alerting (ML score + physics guard rail), LLM ticket writer (Ollama locally or Claude API), **grounding check** that rejects any frequency the LLM mentions that isn't in the evidence |
| `tabular_bosch.py` | Bosch Production Line Performance (1.18 M parts, 968 features, ~0.58 % failures): process-route, cycle-time and production-order features, LightGBM, MCC-optimal threshold, native TreeSHAP aggregated per station |
| `edge_export.py` | PyTorch → ONNX → INT8; benchmark the full wav → prediction path on any CPU |
| `app/dashboard.py` | Streamlit: inspect a clip, replay a fleet over time, view the Bosch line analysis |
| `resume.py` | Prints resume bullets using **your** real results (refuses synthetic numbers) |

## Quick start (Windows, ~5 minutes)

```powershell
python -m venv .venv; .venv\Scripts\activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA build for an RTX GPU
python scripts\smoke_test.py          # whole pipeline on synthetic data
python -m pytest tests -q
streamlit run app\dashboard.py
```

The smoke test generates synthetic machine sounds with injected bearing, imbalance and whine faults.
It proves the code runs; **its numbers are not results**.

## Real data

1. **DCASE Task 2** (machine-condition monitoring). Download the *development* dataset for the 2023 task
   (7 machine types, source and target domains) from the DCASE challenge site (dcase.community → Challenge 2023 →
   Task 2; files are on Zenodo). The 2020 dev set and raw MIMII also work.
   Unzip so the layout is `data\dcase\<machine>\train\*.wav` and `data\dcase\<machine>\test\*.wav`
   (for MIMII: `data\mimii\<machine>\id_00\normal\*.wav` and pass `--layout mimii`).
2. **Bosch**: `pip install kaggle`, accept the competition rules on Kaggle, then
   `kaggle competitions download -c bosch-production-line-performance -p data\bosch` and unzip
   `train_numeric.csv` and `train_date.csv`.
3. Run everything: `.\run_all.ps1` (or the individual commands inside it).

Approximate cost on a 32 GB laptop with an RTX 4060: audio sklearn models minutes; AE + ID-CNN roughly
20–60 min per machine type; Bosch full data 30–60 min (use `--nrows 500000` for a quick pass).

## Results

Fill these from `results\audio_summary.json`, `results\bosch_summary.json`, `results\edge_benchmark.json`,
or just run `python -m machine_guardian.resume`.

| Detector | AUC source | AUC target | pAUC | Official score |
|---|---|---|---|---|
| kNN (pooled baseline) | | | | |
| kNN (domain-normalised) | | | | |
| PCA / GMM | | | | |
| Dense AE (DCASE baseline arch.) | | | | |
| ID-classifier CNN | | | | |

| Bosch | Value |
|---|---|
| MCC (5-fold OOF) | |
| PR-AUC (random = failure rate) | |
| Top failure stations (SHAP) | |

| Edge | Value |
|---|---|
| INT8 model size | |
| Latency per 10 s clip | |

## What was verified, and how

* Full pipeline (synthetic audio → detectors → explanations → tickets; synthetic Bosch-format CSVs →
  loader → LightGBM/fallback → SHAP) runs end to end; unit tests cover the grounding check, LLM failure
  fallback and the pAUC metric.
* On synthetic data with known planted causes: station attribution recovered both planted failure
  stations; envelope analysis named the correct fault type for ~98 % of flagged clips; the physics
  guard rail raised recall from 82.5 % to 98.8 % with no extra false alarms. These checks validate the
  logic, not real-world accuracy: real faults are messier, so report real-data numbers only.
* The PyTorch models and ONNX export are written against the standard APIs but must be run on your
  machine (GPU) to produce results.

## Interview talking points

* **Why unsupervised?** Faults are rare and varied; you can record months of normal operation but never
  every failure. Training on normal data only matches how a plant actually works.
* **Domain shift** is the deployment problem: a new motor speed or a noisier hall looks "anomalous" to a
  naive model. Per-domain normalisation with ~10 target clips is cheap and needs no labels.
* **pAUC** matters more than AUC in a factory: it measures detection while false alarms stay below 10 %,
  and false alarms are what make technicians ignore a system.
* **MCC** instead of accuracy for Bosch: with 0.58 % failures, predicting "pass" for everything is 99.4 %
  accurate and useless.
* **Grounding**: the LLM never sees audio, only measured evidence, and its output is checked against it.
  If the check fails the system falls back to a template, so a ticket can never invent a cause.
* **Hybrid ML + physics**: the learned score catches anything unusual; envelope analysis adds a cause
  and a guard rail for strong, well-understood fault signatures.

## Limitations

* Rule thresholds in `explain.py` were set on synthetic signals; tune them on labelled field data before
  trusting cause labels.
* One microphone channel is used; MIMII's 8 channels could support beamforming / localisation.
* The Bosch features are anonymised, so station attribution says *where* to look, not *what* to fix.
