# scripts/test_emg_model.py

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
    parser = argparse.ArgumentParser(description="Test EMG models (SNN/TCN/Hybrid/SpikingTCN)")

    parser.add_argument(
        "--artifact-dir",
        type=str,
        required=True,
        help="Directory with exported artifacts (e.g., ./output/rate)",
    )
    parser.add_argument(
        "--model-prefix",
        type=str,
        required=True,
        help="Model prefix (e.g., snn_rate, tcn, hybrid_rate, spiking_tcn_rate)",
    )

    # 입력 소스
    parser.add_argument(
        "--sample-from-mat",
        type=str,
        default=None,
        help="Path to Ninapro .mat file (use training-like windowing)",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Window index when using --sample-from-mat",
    )
    parser.add_argument(
        "--emg-npy",
        type=str,
        default=None,
        help="Path to .npy EMG window (shape: [T, C])",
    )
    parser.add_argument(
        "--use-dummy",
        action="store_true",
        help="Use dummy random EMG window if no input specified",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run on (cpu or cuda)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    device = args.device

    # 1) 모델 + 스케일러 + 라벨인코더 + meta 로드
    print("\n[1] Loading model & artifacts...")
    model, scaler, label_encoder, meta = load_emg_model(
        artifact_dir, prefix=args.model_prefix, device=device
    )

    print(f"    - model_name  : {meta.get('model_name')}")
    print(f"    - encoding    : {meta.get('encoding_type')}")
    print(f"    - window_size : {meta.get('window_size')}")
    print(f"    - num_channels: {meta.get('num_channels')}")
    print(f"    - num_classes : {meta.get('num_classes')}")

    window_size = int(meta["window_size"])
    num_channels = int(meta["num_channels"])

    # 2) EMG 윈도우 준비
    emg_window = None
    true_label_raw = None

    # 2-1) mat에서 뽑기 (training과 동일한 preprocess_data_for_networks 사용)
    if args.sample_from_mat is not None:
        print(f"\n[2] Loading sample from mat: {args.sample_from_mat}")
        emg_raw, labels_raw = load_ninapro_data(args.sample_from_mat)
        print(f"    - raw EMG shape   : {emg_raw.shape}")
        print(f"    - raw labels shape: {labels_raw.shape}")

        X_win, y_win = preprocess_data_for_networks(
            emg_raw,
            labels_raw,
            window_size=window_size,
            overlap=100,
        )
        print(f"    - Generated windows: {X_win.shape}")

        if args.sample_index < 0 or args.sample_index >= len(X_win):
            raise IndexError(
                f"sample_index {args.sample_index} out of range (0 ~ {len(X_win)-1})"
            )

        emg_window = X_win[args.sample_index]
        true_label_raw = int(y_win[args.sample_index])

        print(f"    - Selected window[{args.sample_index}], shape: {emg_window.shape}")
        print(f"    - Raw GT label: {true_label_raw}")

    # 2-2) npy에서 로드
    elif args.emg_npy is not None:
        print(f"\n[2] Loading EMG window from npy: {args.emg_npy}")
        emg_window = np.load(args.emg_npy)
        print(f"    - EMG window shape: {emg_window.shape}")

    # 2-3) dummy
    elif args.use_dummy:
        print("\n[2] Using dummy EMG window")
        emg_window = np.random.randn(window_size, num_channels).astype(np.float32)
        print(f"    - Dummy EMG shape: {emg_window.shape}")

    else:
        raise ValueError(
            "No EMG source specified. Use one of: "
            "--sample-from-mat, --emg-npy, or --use-dummy"
        )

    # 3) 전처리 → 텐서
    print("\n[3] Preprocessing EMG window (scaling + tensor conversion)...")
    emg_tensor = preprocess_emg_window(emg_window, scaler, meta)
    print(f"    - Tensor shape: {tuple(emg_tensor.shape)}  # [1, T, C]")

    # 4) 추론
    print("\n[4] Running inference...")
    pred_idx, pred_label, conf, probs = infer_single_window(
        model, emg_tensor, label_encoder, device=device
    )

    print("\n[5] Result")
    print("    - Predicted class index :", pred_idx)
    print("    - Predicted label       :", pred_label)
    print(f"    - Confidence (softmax)  : {conf:.4f}")
    print("    - Probabilities         :", probs)

    # 5-1) GT 비교 (mat 모드인 경우에만)
    if true_label_raw is not None:
        true_idx = int(label_encoder.transform([true_label_raw])[0])
        is_correct = (pred_idx == true_idx)

        print("\n[6] Ground Truth Check")
        print("    - Ground truth raw label   :", true_label_raw)
        print("    - Ground truth class index :", true_idx)
        print("    - Prediction matches GT?   :", is_correct)

    print("\n=== Test complete ===")


if __name__ == "__main__":
    main()
