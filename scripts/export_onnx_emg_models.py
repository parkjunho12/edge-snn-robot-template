# scripts/export_onnx_emg_models.py

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

from src.infer_server.emg_artifacts import (
    load_emg_model,
    preprocess_emg_window,
)
from src.emg_io.data_src.ninapro import (
    load_ninapro_data,
    preprocess_data_for_networks,
)

from src.emg_io.spiking_tcn_onnx_wrapper import (
    SpikingTCNONNXWrapper,
    export_spiking_tcn_full,
    verify_spiking_tcn_export,
)


class ExportWrapper(nn.Module):
    """
    ONNX export 시, model(x) → logits 만 내보내도록 래핑.
    SNN/Hybrid/SpikingTCN처럼 (logits, spikes)를 반환하는 경우도 지원.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        try:
            logits = self.model(x, return_spikes=False)
        except ValueError:
            logits = self.model(x)
        except TypeError:
            logits = self.model(x)
        return logits


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export multiple EMG models (SNN/TCN/Hybrid/SpikingTCN) to ONNX"
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
        default="snn,tcn,hybrid,spiking_tcn",
        help="Comma-separated model prefixes to export "
        "(e.g., snn_rate,tcn,hybrid_rate,spiking_tcn_rate)",
    )
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
        help="Device to use for export (cpu or cuda)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )

    return parser.parse_args()


def compare_outputs(pth_model, onnx_path, test_input, device="cpu"):
    """PyTorch 모델과 ONNX 모델의 출력 비교"""

    # 1. PyTorch 출력
    pth_model.eval()
    with torch.no_grad():
        if isinstance(test_input, np.ndarray):
            test_tensor = torch.from_numpy(test_input).float().to(device)
        else:
            test_tensor = test_input.to(device)

        try:
            pth_output = pth_model(test_tensor, return_spikes=False)
        except (ValueError, TypeError):
            pth_output = pth_model(test_tensor)

    # 2. ONNX 출력
    sess_options = ort.SessionOptions()
    ort_session = ort.InferenceSession(
        onnx_path.as_posix(),
        sess_options,
        providers=["CPUExecutionProvider"],
    )
    onnx_input = test_tensor.cpu().numpy()
    onnx_output = ort_session.run(None, {"emg": onnx_input})[0]

    # 3. 비교
    pth_np = pth_output.cpu().numpy()

    print("=" * 60)
    print("PyTorch vs ONNX Comparison")
    print("=" * 60)
    print(f"PyTorch output shape: {pth_np.shape}")
    print(f"ONNX output shape:    {onnx_output.shape}")
    print(f"\nPyTorch output:\n{pth_np}")
    print(f"\nONNX output:\n{onnx_output}")

    # 차이 계산
    abs_diff = np.abs(pth_np - onnx_output)
    rel_diff = abs_diff / (np.abs(pth_np) + 1e-7)

    print(f"\n--- Difference Statistics ---")
    print(f"Max absolute diff: {np.max(abs_diff):.6f}")
    print(f"Mean absolute diff: {np.mean(abs_diff):.6f}")
    print(f"Max relative diff: {np.max(rel_diff):.6f}")
    print(f"Mean relative diff: {np.mean(rel_diff):.6f}")

    # Prediction 비교
    pth_pred = np.argmax(pth_np, axis=1)
    onnx_pred = np.argmax(onnx_output, axis=1)

    print(f"\n--- Predictions ---")
    print(f"PyTorch prediction: {pth_pred}")
    print(f"ONNX prediction:    {onnx_pred}")
    print(f"Match: {np.array_equal(pth_pred, onnx_pred)}")

    return {
        "pth_output": pth_np,
        "onnx_output": onnx_output,
        "max_abs_diff": np.max(abs_diff),
        "mean_abs_diff": np.mean(abs_diff),
        "predictions_match": np.array_equal(pth_pred, onnx_pred),
    }


def main():
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    device = args.device
    prefixes = [p.strip() for p in args.model_prefixes.split(",") if p.strip()]

    print("\n[1] Exporting models to ONNX...")
    for prefix in prefixes:
        print(f"\n=== [{prefix}] ===")

        # 1) 모델 + scaler + meta 로드
        model, scaler, label_encoder, meta = load_emg_model(
            artifact_dir, prefix=prefix, device=device
        )

        window_size = int(meta["window_size"])
        num_channels = int(meta["num_channels"])
        num_classes = int(meta["num_classes"])
        print(f"    - model_name  : {meta.get('model_name')}")
        print(f"    - encoding    : {meta.get('encoding_type')}")
        print(f"    - window_size : {window_size}")
        print(f"    - num_channels: {num_channels}")
        print(f"    - num_classes : {num_classes}")

        # 2) dummy window + 전처리
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

            print(
                f"    - Selected window[{args.sample_index}], shape: {emg_window.shape}"
            )
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

        emg_tensor = preprocess_emg_window(emg_window, scaler, meta).to(device)
        print(f"    - Dummy tensor shape: {tuple(emg_tensor.shape)}  # [1, T, C]")

        # 3) ONNX export
        model.eval()

        wrapper = ExportWrapper(model).to(device)
        wrapper.eval()
        onnx_path = artifact_dir / f"{prefix}_inference.onnx"
        print(f"    - Exporting ONNX to: {onnx_path}")

        torch.onnx.export(
            wrapper,
            emg_tensor,
            onnx_path.as_posix(),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["emg"],
            output_names=["logits"],
            dynamic_shapes=(
                {  # 첫 번째 인자 x에 대한 동적 차원
                    0: torch.export.Dim("batch"),
                    2: torch.export.Dim("time"),
                },
            ),
        )
        print("    ✅ ONNX export done")
        print(f"    - Verifying ONNX export...")
        compare_outputs(wrapper, onnx_path, emg_tensor, device)

    print("\n=== All ONNX exports complete ===")


if __name__ == "__main__":
    main()
