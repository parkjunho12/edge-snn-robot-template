#!/usr/bin/env python
"""
onnx_model_eval.py

ONNX EMG 모델 평가 스크립트.

- NinaPro DB6 .mat 파일에서 EMG/레이블 로드
- 윈도우 생성 (preprocess_data_for_networks와 동일한 방식)
- 스케일링 + ONNX 추론
- accuracy / confusion matrix / per-class recall 출력

사용 예:

    python -m scripts.onnx_model_eval \
        --artifact-dir ./output/rate \
        --model-prefix spiking_tcn \
        --mat-path ./src/data/s1.mat \
        --val-ratio 0.2 \
        --max-samples 5000

"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
from sklearn.metrics import accuracy_score, confusion_matrix

from src.emg_io.data_src.ninapro import (
    load_ninapro_data,
    preprocess_data_for_networks,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ONNX EMG model on NinaPro DB6."
    )

    parser.add_argument(
        "--artifact-dir",
        type=str,
        required=True,
        help="Directory containing ONNX + meta + scaler + label_encoder artifacts.",
    )
    parser.add_argument(
        "--model-prefix",
        type=str,
        default="spiking_tcn",
        help="Model prefix used for artifacts (e.g., snn, tcn, hybrid, spiking_tcn).",
    )
    parser.add_argument(
        "--mat-path",
        type=str,
        required=True,
        help="Path to NinaPro .mat file (e.g., ./src/data/s1.mat).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Ratio of windows used for validation (0~1). Remainder is treated as 'train' and 무시.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="ONNX inference batch size.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional max number of validation windows to evaluate (for quick test).",
    )

    return parser.parse_args()


def load_artifacts(artifact_dir: Path, prefix: str):
    """meta.json + scaler + label_encoder + onnx 세트 로드"""

    # 1) meta
    meta_path = artifact_dir / f"{prefix}_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta JSON not found: {meta_path}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    encoding_type = meta.get("encoding_type", "rate")

    # 2) scaler
    scaler_path = artifact_dir / f"emg_scaler_{encoding_type}.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler pickle not found: {scaler_path}")
    scaler = joblib.load(scaler_path)

    # 3) label encoder
    label_encoder_path = artifact_dir / f"label_encoder_{encoding_type}.pkl"
    if not label_encoder_path.exists():
        raise FileNotFoundError(f"LabelEncoder pickle not found: {label_encoder_path}")
    label_encoder = joblib.load(label_encoder_path)

    # 4) ONNX 모델
    onnx_path = artifact_dir / f"{prefix}_inference.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    return meta, scaler, label_encoder, onnx_path


def scale_windows(X: np.ndarray, scaler, meta: dict) -> np.ndarray:
    """
    윈도우 묶음 [N, T, C]에 대해 scaler를 적용.
    - 학습 시와 동일하게: 각 타임스텝을 feature 벡터로 보고 StandardScaler 적용.
    """
    N, T, C = X.shape
    X_flat = X.reshape(-1, C)  # [N*T, C]
    X_scaled_flat = scaler.transform(X_flat)
    X_scaled = X_scaled_flat.reshape(N, T, C)
    return X_scaled.astype(np.float32)


def build_onnx_session(onnx_path: Path) -> ort.InferenceSession:
    """ONNX Runtime 세션 생성"""
    sess_options = ort.SessionOptions()
    sess = ort.InferenceSession(
        onnx_path.as_posix(),
        sess_options,
        providers=["CPUExecutionProvider"],
    )
    return sess


def run_onnx_inference(sess: ort.InferenceSession, X: np.ndarray, batch_size: int = 64):
    """
    ONNX 모델로 배치 추론 수행.
    X: [N, T, C]
    return: logits [N, num_classes]
    """
    all_logits = []

    N = X.shape[0]
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = X[start:end]  # [B, T, C]

        ort_inputs = {"emg": batch}
        ort_outputs = sess.run(None, ort_inputs)
        logits = ort_outputs[0]  # [B, num_classes]
        all_logits.append(logits)

    return np.concatenate(all_logits, axis=0)


def main():
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    mat_path = Path(args.mat_path)
    prefix = args.model_prefix

    print("\n[1] Loading artifacts...")
    meta, scaler, label_encoder, onnx_path = load_artifacts(artifact_dir, prefix)

    window_size = int(meta["window_size"])
    overlap = int(meta.get("overlap", window_size // 2))
    num_channels = int(meta["num_channels"])
    num_classes = int(meta["num_classes"])
    encoding_type = meta.get("encoding_type", "rate")

    print(f"    - model_prefix : {prefix}")
    print(f"    - model_name   : {meta.get('model_name')}")
    print(f"    - encoding     : {encoding_type}")
    print(f"    - window_size  : {window_size}")
    print(f"    - overlap      : {overlap}")
    print(f"    - num_channels : {num_channels}")
    print(f"    - num_classes  : {num_classes}")
    print(f"    - ONNX path    : {onnx_path}")

    print("\n[2] Loading NinaPro data...")
    if not mat_path.exists():
        raise FileNotFoundError(f".mat file not found: {mat_path}")

    emg_raw, labels_raw = load_ninapro_data(mat_path.as_posix())
    print(f"    - raw EMG shape   : {emg_raw.shape}  # [N_samples, C]")
    print(f"    - raw labels shape: {labels_raw.shape}")
    print(f"    - unique labels   : {np.unique(labels_raw)}")

    # 윈도우 기반 시퀀스 생성 (훈련과 동일한 함수 사용)
    print("\n[3] Building windows (preprocess_data_for_networks)...")
    X_win, y_win_raw = preprocess_data_for_networks(
        emg_raw,
        labels_raw,
        window_size=window_size,
        overlap=overlap,
    )
    print(f"    - windows shape   : {X_win.shape}   # [N_win, T, C]")
    print(f"    - window labels   : {y_win_raw.shape}")

    # 채널 수 체크
    if X_win.shape[-1] != num_channels:
        raise ValueError(
            f"Channel mismatch: X_win.shape[-1]={X_win.shape[-1]}, meta.num_channels={num_channels}"
        )

    # 라벨 인덱스 (label_encoder 기준)
    y_win_idx = label_encoder.transform(y_win_raw.astype(int))
    print(f"    - label_encoder classes_: {label_encoder.classes_}")
    print(f"    - y_win_idx range       : [{y_win_idx.min()}, {y_win_idx.max()}]")

    # Train/Val split (simple shuffle)
    print("\n[4] Splitting train/val...")
    N = X_win.shape[0]
    indices = np.arange(N)
    rng = np.random.default_rng(seed=42)
    rng.shuffle(indices)

    val_size = int(N * args.val_ratio)
    val_indices = indices[:val_size]
    # train_indices = indices[val_size:]  # 지금은 안 씀

    X_val = X_win[val_indices]
    y_val_idx = y_win_idx[val_indices]

    if args.max_samples is not None:
        X_val = X_val[: args.max_samples]
        y_val_idx = y_val_idx[: args.max_samples]

    print(f"    - total windows: {N}")
    print(f"    - val windows  : {X_val.shape[0]}")

    # 스케일링
    print("\n[5] Scaling validation windows...")
    X_val_scaled = scale_windows(X_val, scaler, meta)  # [N_val, T, C]

    # ONNX 세션
    print("\n[6] Building ONNX session...")
    sess = build_onnx_session(onnx_path)

    # 추론
    print("\n[7] Running ONNX inference on validation set...")
    logits_val = run_onnx_inference(sess, X_val_scaled, batch_size=args.batch_size)
    preds_val_idx = np.argmax(logits_val, axis=1)

    # 메트릭
    print("\n[8] Computing metrics...")
    acc = accuracy_score(y_val_idx, preds_val_idx)
    cm = confusion_matrix(y_val_idx, preds_val_idx, labels=np.arange(num_classes))

    # 클래스별 리콜
    per_class_recall = []
    for k in range(num_classes):
        true_pos = cm[k, k]
        total_true = cm[k, :].sum()
        recall_k = true_pos / total_true if total_true > 0 else 0.0
        per_class_recall.append(recall_k)

    print("\n================ Evaluation Result (ONNX model) ================")
    print(f"Validation accuracy: {acc * 100:.2f}%")
    print("\nConfusion Matrix (rows = true, cols = pred, index = class idx):")
    print(cm)

    print("\nPer-class recall:")
    for idx, r in enumerate(per_class_recall):
        raw_label = label_encoder.classes_[idx]
        print(f"  - class_idx {idx} (raw_label={raw_label}): recall = {r * 100:.2f}%")

    print(
        "\nDone. (This is ONNX-only evaluation; PyTorch와 수치 다르게 나와도, "
        "여기 accuracy가 괜찮으면 'ONNX 모델 자체'는 믿고 써도 됨.)"
    )


if __name__ == "__main__":
    main()
