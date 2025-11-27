# scripts/test_onnx_v0_2s.py
import argparse
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

from src.models.spiking_tcn import SpikingTCNClassifier
from src.settings import Settings


def load_torch_model(checkpoint_path: Path, device: str = "cpu"):
    settings = Settings()
    model = SpikingTCNClassifier(
        emg_channels=settings.emg_ch,
        window_size=settings.emg_win,
        num_classes=settings.num_classes,
    )
    state = torch.load(checkpoint_path, map_location=device)
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_onnx_session(onnx_path: Path):
    # CPU 기반 세션 옵션
    sess_options = ort.SessionOptions()
    sess = ort.InferenceSession(onnx_path.as_posix(), sess_options, providers=["CPUExecutionProvider"])
    return sess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/model_v0.2s.pth",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default="artifacts/model_v0.2s.onnx",
    )
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    device = "cpu"
    settings = Settings()
    emg_win = settings.emg_win
    emg_ch = settings.emg_ch

    model = load_torch_model(Path(args.checkpoint), device=device)
    sess = load_onnx_session(Path(args.onnx))

    max_diff_global = 0.0

    for t in range(args.num_trials):
        # 동일한 입력으로 테스트
        x = torch.randn(1, emg_win, emg_ch, device=device, dtype=torch.float32)
        with torch.no_grad():
            pt_out = model(x)  # [1, num_classes] 가정
        pt_out_np = pt_out.cpu().numpy()

        # ONNX 입력은 numpy여야 함
        ort_inputs = {"emg": x.cpu().numpy()}
        ort_outs = sess.run(None, ort_inputs)
        onnx_out_np = ort_outs[0]

        diff = np.abs(pt_out_np - onnx_out_np)
        max_diff = diff.max()
        max_diff_global = max(max_diff_global, max_diff)

        print(f"[Trial {t}] max |pt - onnx| = {max_diff:.6e}")

    print("==== Summary ====")
    print(f"Global max |pt - onnx| = {max_diff_global:.6e}")
    if max_diff_global <= args.atol:
        print(f"✅ OK: max diff ≤ {args.atol}")
    else:
        print(f"❌ Too large diff (> {args.atol})")


if __name__ == "__main__":
    main()
