# scripts/compare_onnx_vs_pt.py

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

from src.infer_server.emg_artifacts import (
    load_emg_model,
    preprocess_emg_window,
)


class ExportWrapper(nn.Module):
    """
    PyTorch 쪽도 ONNX와 동일한 인터페이스(logits만 반환)로 맞추기 위한 래퍼.
    - forward(x, return_spikes=False) -> (logits, spikes)
    - forward(x, return_spikes=False) -> logits
    - forward(x) -> logits
    세 경우 모두 지원.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        # case 1: forward(x, return_spikes=False) -> (logits, spikes)
        try:
            logits, _ = self.model(x, return_spikes=False)
        # case 2: forward(x, return_spikes=False) -> logits  (언팩 실패)
        except (ValueError, TypeError):
            logits = self.model(x)
        return logits


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare PyTorch (ExportWrapper) vs ONNX outputs over random EMG windows"
    )

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
        help="Model prefix (e.g., spiking_tcn_rate, tcn, snn_rate, hybrid_rate)",
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default=None,
        help="Path to ONNX file (default: <artifact-dir>/<prefix>_inference.onnx)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=5,
        help="Number of random inputs to test",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Allowed max absolute difference threshold",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for PyTorch model (cpu or cuda)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-trial statistics",
    )

    return parser.parse_args()


def run_pt_onnx_once(
    pt_model: nn.Module,
    ort_session: ort.InferenceSession,
    emg_tensor: torch.Tensor,
    device: str = "cpu",
):
    """
    단일 입력(emg_tensor)에 대해:
      - PyTorch(ExportWrapper 기준) 출력
      - ONNX 출력
    을 비교하고, 통계값 리턴.
    """
    pt_model.eval()
    emg_tensor = emg_tensor.to(device)

    with torch.no_grad():
        pt_logits = pt_model(
            emg_tensor
        )  # ExportWrapper 기준이라 return_spikes 인자 불필요

    pt_out = pt_logits.cpu().numpy()

    ort_inputs = {"emg": emg_tensor.cpu().numpy()}
    onnx_out = ort_session.run(None, ort_inputs)[0]

    abs_diff = np.abs(pt_out - onnx_out)
    rel_diff = abs_diff / (np.abs(pt_out) + 1e-7)

    return {
        "pt_out": pt_out,
        "onnx_out": onnx_out,
        "max_abs": float(abs_diff.max()),
        "mean_abs": float(abs_diff.mean()),
        "max_rel": float(rel_diff.max()),
        "mean_rel": float(rel_diff.mean()),
        "pt_pred": np.argmax(pt_out, axis=1),
        "onnx_pred": np.argmax(onnx_out, axis=1),
    }


def main():
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    device = args.device
    prefix = args.model_prefix

    # --------------------------------------------------------
    # [1] PyTorch 모델 + scaler + meta 로드
    # --------------------------------------------------------
    print("\n[1] Loading PyTorch model & artifacts...")
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

    # ExportWrapper로 래핑 (export_onnx_emg_models.py와 동일한 기준)
    wrapped_model = ExportWrapper(model).to(device)
    wrapped_model.eval()

    # --------------------------------------------------------
    # [2] ONNX 세션 로드
    # --------------------------------------------------------
    onnx_path = (
        Path(args.onnx_path)
        if args.onnx_path is not None
        else artifact_dir / f"{prefix}_inference.onnx"
    )

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    print(f"\n[2] Loading ONNX model from: {onnx_path}")
    sess_options = ort.SessionOptions()
    # 비교 조건 통일을 위해 CPUExecutionProvider 고정
    sess = ort.InferenceSession(
        onnx_path.as_posix(),
        sess_options,
        providers=["CPUExecutionProvider"],
    )

    # --------------------------------------------------------
    # [3] 랜덤 입력으로 PyTorch vs ONNX 비교
    # --------------------------------------------------------
    print("\n[3] Running comparisons over random EMG windows...")
    global_max_diff = 0.0
    global_max_rel = 0.0
    mismatch_count = 0

    for t in range(args.num_trials):
        # 3-1) dummy EMG window 생성
        dummy_window = np.random.randn(window_size, num_channels).astype(np.float32)

        # 3-2) scaler + meta 기반 전처리 → [1, T, C] 텐서
        emg_tensor = preprocess_emg_window(dummy_window, scaler, meta)
        emg_tensor = emg_tensor.to(device)

        # 3-3) PyTorch vs ONNX 비교
        stats = run_pt_onnx_once(wrapped_model, sess, emg_tensor, device=device)
        max_diff = stats["max_abs"]
        max_rel = stats["max_rel"]
        global_max_diff = max(global_max_diff, max_diff)
        global_max_rel = max(global_max_rel, max_rel)

        preds_match = np.array_equal(stats["pt_pred"], stats["onnx_pred"])
        if not preds_match:
            mismatch_count += 1

        if args.verbose:
            print(f"\n  [Trial {t+1}/{args.num_trials}]")
            print(f"    - max |pt - onnx|  = {max_diff:.6e}")
            print(f"    - mean |pt - onnx| = {stats['mean_abs']:.6e}")
            print(f"    - max rel diff     = {max_rel:.6e}")
            print(f"    - mean rel diff    = {stats['mean_rel']:.6e}")
            print(f"    - pt_pred          = {stats['pt_pred']}")
            print(f"    - onnx_pred        = {stats['onnx_pred']}")
            print(f"    - preds match?     = {preds_match}")

    # --------------------------------------------------------
    # [4] 요약 + 종료 코드
    # --------------------------------------------------------
    print("\n[4] Summary")
    print(f"  - Global max |pt - onnx| = {global_max_diff:.6e}")
    print(f"  - Global max rel diff    = {global_max_rel:.6e}")
    print(f"  - Threshold (atol)       = {args.atol:.6e}")
    print(f"  - Num trials             = {args.num_trials}")
    print(f"  - Prediction mismatches  = {mismatch_count}")

    if global_max_diff <= args.atol:
        print("✅ OK: PyTorch(ExportWrapper) and ONNX match within tolerance.")
        sys.exit(0)
    else:
        print("❌ MISMATCH: max difference exceeds tolerance!")
        sys.exit(1)


if __name__ == "__main__":
    main()
