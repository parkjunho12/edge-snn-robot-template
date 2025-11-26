# scripts/export_onnx_v0_2s.py
import argparse
from pathlib import Path

import torch
import numpy as np

from src.models.spiking_tcn import SpikingTCN  # 네가 실제로 쓰는 클래스 이름으로 수정
from src.settings import Settings  # pydantic Settings 가정

SPIKING_TCN_CHANNELS = [64, 128, 256]
SPIKING_TCN_KERNEL_SIZE = 3
SPIKING_TCN_DROPOUT = 0.2
SPIKING_TCN_BETA = 0.94
SPIKING_TCN_V_TH = 1.0


def load_model(checkpoint_path: Path, device: str = "cpu") -> torch.nn.Module:
    """v0.2s PyTorch 모델 로드"""
    settings = Settings()  # 환경변수 / .env 에서 읽음

    model = SpikingTCNClassifier(
        emg_channels=settings.emg_ch,
        window_size=settings.emg_win,
        num_classes=settings.num_classes,  # Settings에 있다고 가정
    )
    state = torch.load(checkpoint_path, map_location=device)
    # state dict 안에 'model' 키가 있을 수도 있고, 없을 수도 있으니 둘 다 처리
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="output/model_v0.2s.pth",
        help="Trained PyTorch checkpoint path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/model_v0.2s.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    onnx_path = Path(args.output)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cpu"  # ONNX export는 CPU 기준으로 하는 게 속편함
    model = load_model(ckpt_path, device=device)

    # Settings에서 window/ch 가져오기
    settings = Settings()
    emg_win = settings.emg_win
    emg_ch = settings.emg_ch

    # [batch, time, channel] 형태 더미 입력 (float32)
    dummy_input = torch.randn(1, emg_win, emg_ch, device=device, dtype=torch.float32)

    # 필요시: (B, C, T)로 받는 모델이면 permute 해줘야 함
    # dummy_input = dummy_input.permute(0, 2, 1).contiguous()

    print(f"Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path.as_posix(),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["emg"],
        output_names=["logits"],
        dynamic_axes={
            "emg": {0: "batch_size", 1: "time_steps"},
            "logits": {0: "batch_size"},
        },
    )
    print("✅ ONNX export done:", onnx_path)


if __name__ == "__main__":
    main()
