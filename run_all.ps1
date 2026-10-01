# Full pipeline on real data (Windows PowerShell). Run from the project folder.
# Prerequisite folders:
#   data\dcase\<machine>\{train,test}\*.wav      (DCASE 2023 Task 2 dev set, or 2020)
#   data\bosch\train_numeric.csv, train_date.csv (Kaggle Bosch Production Line Performance)
$ErrorActionPreference = "Stop"

Write-Host "1/4 Audio anomaly detection (sklearn + GPU deep models)"
python -m machine_guardian.evaluate_audio --data data\dcase --detectors knn,knn_naive,pca,gmm,ae,idcnn

Write-Host "2/4 Bosch failure prediction (use --nrows 500000 if RAM is tight)"
python -m machine_guardian.tabular_bosch --data data\bosch

Write-Host "3/4 Edge export (pick one machine)"
python -m machine_guardian.edge_export export --data data\dcase --machine fan --epochs 40
python -m machine_guardian.edge_export bench --model models\fan_idcnn.int8.onnx

Write-Host "4/4 Resume bullets from your real numbers"
python -m machine_guardian.resume
