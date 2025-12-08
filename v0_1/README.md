## Quickstart

```bash
# 1) clone repo
git clone https://github.com/parkjunho12/edge-snn-robot-template.git
cd edge-snn-robot-template

# 2) Python env
python -m venv .venv && source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# 3) Download datasets
# ex) s1.mat is training dataset
python scripts/download_data.py --mat s1.mat

# s2, s3.mat can download too
python scripts/download_data.py --mat s2.mat
python scripts/download_data.py --mat s3.mat

# 4) Download artifacts
python scripts/download_data.py --artifacts

# 5) Train mat file (Optional)
# DATA_PATH = {mat file for train} ex) "./src/data/s1.mat"
#   encoding_type = "{encoding_type}" ex) "latency", "rate", "delta"
#   num_steps = 20
#
# output: output/(emg_scaler.pkl, label_encoder.pkl, spiking_tcn_{encoding_type}_best.pth, tcn_{encoding_type}_best.pth, snn_{encoding_type}_best.pth, hybrid_{encoding_type}_best.pth, emg_meta.json)
python -m  v0_1.scripts.train_ninapro_offline

# 6) Test single window(mat file) [Optional]
python -m v0_1.scripts.test_emg_model \
  --artifact-dir ./output/{encoding_type} \
  --model-prefix spiking_tcn \
  --sample-from-mat ./src/data/s1.mat \
  --sample-index 100

# 7) Compare models [Optional]
python -m v0_1.scripts.compare_models \
  --artifact-dir ./output/{encoding_type} \
  --model-prefixes snn,tcn,hybrid,spiking_tcn \
  --mat-path ./src/data/s1.mat \
  --max-samples 200 \
  --shuffle

# 8) Export onnx and compare with pytorch
python -m v0_1.scripts.export_onnx_emg_models \
  --artifact-dir ./output/{encoding_type} \
  --model-prefixes snn,tcn,hybrid,spiking_tcn --sample-from-mat ./src/data/s1.mat \
  --sample-index 124

# 9) Validate onnx (Optional)
python -m v0_1.scripts.onnx_model_eval \       
  --artifact-dir ./output/{encoding_type} \  
  --model-prefix spiking_tcn \                                                     
  --mat-path ./src/data/s1.mat \
  --val-ratio 0.2 \
  --batch-size 64 \
  --max-samples 5000
```

### Compare model accuracy

```bash
Summary Accuracy - rate encoding (s1.mat)
    - snn              (SNNClassifier) : 20/200 = 0.1000
    - tcn              (TCNClassifier) : 193/200 = 0.9650
    - hybrid           (HybridTCNSNN) : 193/200 = 0.9650
    - spiking_tcn      (SpikingTCN  ) : 151/200 = 0.7550

Summary Accuracy - rate encoding (s2.mat)
    - snn              (SNNClassifier) : 29/200 = 0.1450
    - tcn              (TCNClassifier) : 78/200 = 0.3900
    - hybrid           (HybridTCNSNN) : 68/200 = 0.3400
    - spiking_tcn      (SpikingTCN  ) : 82/200 = 0.4100

Summary Accuracy - rate encoding (s3.mat)
    - snn              (SNNClassifier) : 22/200 = 0.1100
    - tcn              (TCNClassifier) : 106/200 = 0.5300
    - hybrid           (HybridTCNSNN) : 102/200 = 0.5100
    - spiking_tcn      (SpikingTCN  ) : 111/200 = 0.5550

=== Comparison complete ===

Summary Accuracy - delta encoding (s1.mat)
    - snn              (SNNClassifier) : 86/200 = 0.4300
    - tcn              (TCNClassifier) : 194/200 = 0.9700
    - hybrid           (HybridTCNSNN) : 187/200 = 0.9350
    - spiking_tcn      (SpikingTCN  ) : 31/200 = 0.1550

Summary Accuracy - delta encoding (s2.mat)
    - snn              (SNNClassifier) : 57/200 = 0.2850
    - tcn              (TCNClassifier) : 68/200 = 0.3400
    - hybrid           (HybridTCNSNN) : 60/200 = 0.3000
    - spiking_tcn      (SpikingTCN  ) : 36/200 = 0.1800

Summary Accuracy - delta encoding (s3.mat)
    - snn              (SNNClassifier) : 71/200 = 0.3550
    - tcn              (TCNClassifier) : 101/200 = 0.5050
    - hybrid           (HybridTCNSNN) : 103/200 = 0.5150
    - spiking_tcn      (SpikingTCN  ) : 38/200 = 0.1900

=== Comparison complete ===

Summary Accuracy - latency encoding (s1.mat)
    - snn              (SNNClassifier) : 137/200 = 0.6850
    - tcn              (TCNClassifier) : 194/200 = 0.9700
    - hybrid           (HybridTCNSNN) : 194/200 = 0.9700

Summary Accuracy - latency encoding (s2.mat)
    - snn              (SNNClassifier) : 52/200 = 0.2600
    - tcn              (TCNClassifier) : 79/200 = 0.3950
    - hybrid           (HybridTCNSNN) : 67/200 = 0.3350

=== Comparison complete ===

```



### Subject-wise Generalization (2 kHz NinaPro sEMG, TCN model)

The TCN baseline was trained **only on subject S1** (2 kHz NinaPro sEMG, gestures {1, 3, 4, 6, 9, 10, 11}).  
We then exported the model to ONNX and evaluated it on different subjects using the ONNX runtime.

**Evaluation accuracy (ONNX model):**

| Train subject | Eval subject | Accuracy | Notes                    |
|---------------|-------------|----------|--------------------------|
| S1            | S1          | **95.47%** | Intra-subject (personalized) |
| S1            | S2          | 41.14%   | Cross-subject            |
| S1            | S3          | 53.20%   | Cross-subject            |

Per-class recall (S1 → S1):

- class_idx 0 (raw_label = 1): 97.87%  
- class_idx 1 (raw_label = 3): 95.42%  
- class_idx 2 (raw_label = 4): 92.24%  
- class_idx 3 (raw_label = 6): 94.20%  
- class_idx 4 (raw_label = 9): 95.73%  
- class_idx 5 (raw_label = 10): 94.94%  
- class_idx 6 (raw_label = 11): 97.85%

Per-class recall (S1 → S2):

- class_idx 0 (raw_label = 1): 49.58%  
- class_idx 1 (raw_label = 3): 51.50%  
- class_idx 2 (raw_label = 4): 39.92%  
- class_idx 3 (raw_label = 6): 11.85%  
- class_idx 4 (raw_label = 9): 22.73%  
- class_idx 5 (raw_label = 10): 70.78%  
- class_idx 6 (raw_label = 11): 38.20%

Per-class recall (S1 → S3):

- class_idx 0 (raw_label = 1): 60.52%  
- class_idx 1 (raw_label = 3): 49.37%  
- class_idx 2 (raw_label = 4): 32.78%  
- class_idx 3 (raw_label = 6): 40.44%  
- class_idx 4 (raw_label = 9): 51.44%  
- class_idx 5 (raw_label = 10): 71.97%  
- class_idx 6 (raw_label = 11): 66.52%

**Interpretation**

- The model achieves **high intra-subject performance** on S1 (≈95% accuracy).
- When evaluated on unseen subjects (S2, S3) **without any re-training or calibration**, the accuracy drops to ~41–53%, which is expected due to strong inter-subject variability in sEMG.
- At this stage (v0.2), the TCN can be viewed as a **subject-specific controller**.  
  Future work includes:
  - multi-subject training (S1+S2+S3+…),
  - subject-wise normalization,
  - and quick per-subject fine-tuning for calibration.

### Subject-wise Generalization (2 kHz NinaPro sEMG, Hybrid model)

The Hybrid baseline was trained **only on subject S1** (2 kHz NinaPro sEMG.

#### RATE Encoding
**Evaluation accuracy (ONNX model - rate encoding):**

| Train subject | Eval subject | Accuracy | Notes                    |
|---------------|-------------|----------|--------------------------|
| S1            | S1          | **95.54%** | Intra-subject (personalized) |
| S1            | S2          | 34.66%  | Cross-subject            |
| S1            | S3          | 50.03%  | Cross-subject            |

Per-class recall (S1 → S1):
  - class_idx 0 (raw_label=1): recall = 96.60%
  - class_idx 1 (raw_label=3): recall = 94.17%
  - class_idx 2 (raw_label=4): recall = 96.12%
  - class_idx 3 (raw_label=6): recall = 95.54%
  - class_idx 4 (raw_label=9): recall = 97.01%
  - class_idx 5 (raw_label=10): recall = 94.94%
  - class_idx 6 (raw_label=11): recall = 94.42%

Per-class recall (S1 → S2):
  - class_idx 0 (raw_label=1): recall = 39.83%
  - class_idx 1 (raw_label=3): recall = 31.76%
  - class_idx 2 (raw_label=4): recall = 49.58%
  - class_idx 3 (raw_label=6): recall = 9.95%
  - class_idx 4 (raw_label=9): recall = 26.03%
  - class_idx 5 (raw_label=10): recall = 71.19%
  - class_idx 6 (raw_label=11): recall = 10.30%

Per-class recall (S1 → S3):
  - class_idx 0 (raw_label=1): recall = 48.93%
  - class_idx 1 (raw_label=3): recall = 56.54%
  - class_idx 2 (raw_label=4): recall = 41.08%
  - class_idx 3 (raw_label=6): recall = 28.89%
  - class_idx 4 (raw_label=9): recall = 53.09%
  - class_idx 5 (raw_label=10): recall = 72.38%
  - class_idx 6 (raw_label=11): recall = 47.96%

#### DELTA Encoding
**Evaluation accuracy (ONNX model - delta encoding):**

| Train subject | Eval subject | Accuracy | Notes                    |
|---------------|-------------|----------|--------------------------|
| S1            | S1          | **94.31%** | Intra-subject (personalized) |
| S1            | S2          | 32.33%  | Cross-subject            |
| S1            | S3          | 48.69%  | Cross-subject            |

Per-class recall (S1 → S1):
  - class_idx 0 (raw_label=1): recall = 96.60%
  - class_idx 1 (raw_label=3): recall = 96.67%
  - class_idx 2 (raw_label=4): recall = 96.98%
  - class_idx 3 (raw_label=6): recall = 91.96%
  - class_idx 4 (raw_label=9): recall = 86.75%
  - class_idx 5 (raw_label=10): recall = 92.83%
  - class_idx 6 (raw_label=11): recall = 98.28%

Per-class recall (S1 → S2):

  - class_idx 0 (raw_label=1): recall = 31.78%
  - class_idx 1 (raw_label=3): recall = 48.93%
  - class_idx 2 (raw_label=4): recall = 54.20%
  - class_idx 3 (raw_label=6): recall = 4.74%
  - class_idx 4 (raw_label=9): recall = 2.48%
  - class_idx 5 (raw_label=10): recall = 65.43%
  - class_idx 6 (raw_label=11): recall = 15.45%

Per-class recall (S1 → S3):

  - class_idx 0 (raw_label=1): recall = 54.94%
  - class_idx 1 (raw_label=3): recall = 62.03%
  - class_idx 2 (raw_label=4): recall = 48.55%
  - class_idx 3 (raw_label=6): recall = 24.44%
  - class_idx 4 (raw_label=9): recall = 23.46%
  - class_idx 5 (raw_label=10): recall = 70.29%
  - class_idx 6 (raw_label=11): recall = 57.01%

#### LATENCY Encoding
**Evaluation accuracy (ONNX model - latency encoding):**

| Train subject | Eval subject | Accuracy | Notes                    |
|---------------|-------------|----------|--------------------------|
| S1            | S1          | **94.19%** | Intra-subject (personalized) |
| S1            | S2          | 35.39%  | Cross-subject            |
| S1            | S3          | 51.56%  | Cross-subject            |

Per-class recall (S1 → S1):
  - class_idx 0 (raw_label=1): recall = 97.02%
  - class_idx 1 (raw_label=3): recall = 94.17%
  - class_idx 2 (raw_label=4): recall = 94.40%
  - class_idx 3 (raw_label=6): recall = 95.98%
  - class_idx 4 (raw_label=9): recall = 97.01%
  - class_idx 5 (raw_label=10): recall = 85.23%
  - class_idx 6 (raw_label=11): recall = 95.71%

Per-class recall (S1 → S2):

  - class_idx 0 (raw_label=1): recall = 47.03%
  - class_idx 1 (raw_label=3): recall = 30.47%
  - class_idx 2 (raw_label=4): recall = 47.06%
  - class_idx 3 (raw_label=6): recall = 8.06%
  - class_idx 4 (raw_label=9): recall = 26.03%
  - class_idx 5 (raw_label=10): recall = 76.95%
  - class_idx 6 (raw_label=11): recall = 7.73%

Per-class recall (S1 → S3):

  - class_idx 0 (raw_label=1): recall = 50.64%
  - class_idx 1 (raw_label=3): recall = 45.57%
  - class_idx 2 (raw_label=4): recall = 36.93%
  - class_idx 3 (raw_label=6): recall = 44.00%
  - class_idx 4 (raw_label=9): recall = 65.84%
  - class_idx 5 (raw_label=10): recall = 70.29%
  - class_idx 6 (raw_label=11): recall = 46.61%

### Subject-wise Generalization (2 kHz NinaPro sEMG, SpikingTCN model)

The SpikingTCN baseline was trained **only on subject S1** (2 kHz NinaPro sEMG.

#### RATE Encoding
**Evaluation accuracy (ONNX model - rate encoding):**

| Train subject | Eval subject | Accuracy | Notes                    |
|---------------|-------------|----------|--------------------------|
| S1            | S1          | **67.95%** | Intra-subject (personalized) |
| S1            | S2          | 44.07%  | Cross-subject            |
| S1            | S3          | 55.28%  | Cross-subject            |

Per-class recall (S1 → S1):
  - class_idx 0 (raw_label=1): recall = 84.26%
  - class_idx 1 (raw_label=3): recall = 60.42%
  - class_idx 2 (raw_label=4): recall = 62.93%
  - class_idx 3 (raw_label=6): recall = 57.14%
  - class_idx 4 (raw_label=9): recall = 68.80%
  - class_idx 5 (raw_label=10): recall = 63.71%
  - class_idx 6 (raw_label=11): recall = 78.11%

Per-class recall (S1 → S2):

  - class_idx 0 (raw_label=1): recall = 54.24%
  - class_idx 1 (raw_label=3): recall = 47.21%
  - class_idx 2 (raw_label=4): recall = 76.05%
  - class_idx 3 (raw_label=6): recall = 10.43%
  - class_idx 4 (raw_label=9): recall = 22.73%
  - class_idx 5 (raw_label=10): recall = 53.91%
  - class_idx 6 (raw_label=11): recall = 40.34%

Per-class recall (S1 → S3):

  - class_idx 0 (raw_label=1): recall = 64.81%
  - class_idx 1 (raw_label=3): recall = 43.04%
  - class_idx 2 (raw_label=4): recall = 48.13%
  - class_idx 3 (raw_label=6): recall = 46.22%
  - class_idx 4 (raw_label=9): recall = 59.26%
  - class_idx 5 (raw_label=10): recall = 60.25%
  - class_idx 6 (raw_label=11): recall = 65.61%