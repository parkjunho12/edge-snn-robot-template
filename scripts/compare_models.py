# scripts/compare_models.py

import argparse
from pathlib import Path

import numpy as np

from src.infer_server.emg_artifacts import (
    load_emg_model,
    preprocess_emg_window,
    infer_single_window,
)
from src.emg_io.data_src.ninapro import (
    load_ninapro_data,
    preprocess_data_for_networks,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare multiple EMG models (SNN/TCN/Hybrid/SpikingTCN) on same windows"
    )

    parser.add_argument(
        "--artifact-dir",
        type=str,
        required=True,
        help="Directory with exported artifacts (e.g., ./output/rate)",
    )
    parser.add_argument(
        "--model-prefixes",
        type=str,
        default="snn_rate,tcn,hybrid_rate,spiking_tcn_rate",
        help="Comma-separated model prefixes (e.g., snn_rate,tcn,hybrid_rate,spiking_tcn_rate)",
    )
    parser.add_argument(
        "--mat-path",
        type=str,
        required=True,
        help="Path to Ninapro .mat file (e.g., ./src/data/s2.mat)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of windows to evaluate (0 = use all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on (cpu or cuda)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="If set, randomly shuffle windows before picking max-samples",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    device = args.device
    prefixes = [p.strip() for p in args.model_prefixes.split(",") if p.strip()]

    print("\n[1] Loading models & artifacts...")
    models = {}
    metas = {}
    # 같은 encoding_type 기반 scaler/label_encoder를 썼다고 가정하지만,
    # 각 prefix별로 따로 로드해도 동일 파일을 읽게 됨
    scalers = {}
    label_encoders = {}

    for prefix in prefixes:
        print(f"    - Loading [{prefix}] ...")
        model, scaler, label_encoder, meta = load_emg_model(
            artifact_dir, prefix=prefix, device=device
        )
        models[prefix] = model
        metas[prefix] = meta
        scalers[prefix] = scaler
        label_encoders[prefix] = label_encoder

        print(f"      * model_name  : {meta.get('model_name')}")
        print(f"      * encoding    : {meta.get('encoding_type')}")
        print(f"      * window_size : {meta.get('window_size')}")
        print(f"      * num_channels: {meta.get('num_channels')}")
        print(f"      * num_classes : {meta.get('num_classes')}")

    # 윈도우/채널 수는 모든 모델에서 같다고 가정하고 첫 모델 기준으로 사용
    first_meta = next(iter(metas.values()))
    window_size = int(first_meta["window_size"])
    num_channels = int(first_meta["num_channels"])

    print(f"\n[2] Loading Ninapro .mat: {args.mat_path}")
    emg_raw, labels_raw = load_ninapro_data(args.mat_path)
    print(f"    - raw EMG shape   : {emg_raw.shape}")
    print(f"    - raw labels shape: {labels_raw.shape}")

    print("\n[3] Windowing with preprocess_data_for_networks (same as training)...")
    X_win, y_win = preprocess_data_for_networks(
        emg_raw,
        labels_raw,
        window_size=window_size,
        overlap=100,  # 학습 때와 동일하게
    )
    num_windows = len(X_win)
    print(f"    - Generated windows: {X_win.shape}  # [N, T, C]")

    if num_windows == 0:
        raise RuntimeError("No windows generated. Check preprocessing settings.")

    indices = np.arange(num_windows)
    if args.shuffle:
        np.random.shuffle(indices)

    max_samples = args.max_samples if args.max_samples > 0 else num_windows
    max_samples = min(max_samples, num_windows)
    indices = indices[:max_samples]

    print(f"\n[4] Evaluating {len(prefixes)} models on {max_samples} windows...")

    # 모델별 accuracy 카운터
    correct_counts = {prefix: 0 for prefix in prefixes}
    total = max_samples

    for k, idx in enumerate(indices):
        emg_window = X_win[idx]  # [T, C]
        true_label_raw = int(y_win[idx])

        # label_encoder는 prefix마다 동일한 매핑이어야 하지만,
        # 안전하게 첫 모델의 encoder를 기준으로 사용
        first_prefix = prefixes[0]
        first_le = label_encoders[first_prefix]
        true_idx = int(first_le.transform([true_label_raw])[0])

        print(f"\n[Sample {k+1}/{total}] window_idx={idx}")
        print(f"    - GT raw label : {true_label_raw}")
        print(f"    - GT class idx : {true_idx}")

        # 전처리는 scaler/메타도 모델마다 동일해야 하지만,
        # 역시 첫 모델 기준으로 사용 (모두 같은 scaler/메타를 썼다는 가정)
        scaler = scalers[first_prefix]
        meta = metas[first_prefix]
        emg_tensor = preprocess_emg_window(emg_window, scaler, meta)

        # 각 모델 추론
        for prefix in prefixes:
            model = models[prefix]
            le = label_encoders[prefix]

            pred_idx, pred_label, conf, _ = infer_single_window(
                model, emg_tensor, le, device=device
            )
            is_correct = pred_idx == true_idx
            if is_correct:
                correct_counts[prefix] += 1

            print(
                f"    [{prefix}] pred_idx={pred_idx:2d}, "
                f"label={pred_label}, conf={conf:.4f}, correct={is_correct}"
            )

    # 요약 accuracy
    print("\n[5] Summary Accuracy")
    for prefix in prefixes:
        acc = correct_counts[prefix] / total
        model_name = metas[prefix].get("model_name")
        print(
            f"    - {prefix:16s} ({model_name:12s}) : "
            f"{correct_counts[prefix]}/{total} = {acc:.4f}"
        )

    print("\n=== Comparison complete ===")


if __name__ == "__main__":
    main()
