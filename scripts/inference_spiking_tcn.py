"""
inference_spiking_tcn.py

SpikingTCN 배포용 추론 예제 스크립트

전제:
  ./output/<encoding_type>/ 안에 다음 4개 아티팩트가 존재한다고 가정
    - spiking_tcn_<encoding_type>_best.pth
    - emg_scaler.pkl
    - label_encoder.pkl
    - emg_meta.json

사용 예:
  1) 더미 EMG 윈도우로 테스트
      python inference_spiking_tcn.py --artifact-dir ./output/latency

  2) 직접 수집한 EMG 윈도우 (numpy .npy 파일)
      # np.save("emg_window.npy", emg_window)  # shape: [window_size, num_channels]
      python inference_spiking_tcn.py \
        --artifact-dir ./output/latency \
        --emg-npy ./emg_window.npy
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

from src.models.spiking_tcn import SpikingTCN

from src.emg_io.data_src.ninapro import (
    load_ninapro_data,
    preprocess_data_for_networks,
)


# 🔧 SpikingTCN 하이퍼파라미터 (훈련 때와 동일해야 함!)
#   → 필요하면 emg_meta.json에 같이 저장하도록 나중에 확장 가능
SPIKING_TCN_CHANNELS = [64, 128, 256]
SPIKING_TCN_KERNEL_SIZE = 3
SPIKING_TCN_DROPOUT = 0.2
SPIKING_TCN_BETA = 0.94
SPIKING_TCN_V_TH = 1.0


def load_artifacts(artifact_dir: Path):
    """배포용 아티팩트 4개 로드"""
    artifact_dir = Path(artifact_dir)

    # 1) 메타 정보
    meta_path = artifact_dir / "emg_meta.json"
    with meta_path.open("r", encoding="utf-8") as f:
        emg_meta = json.load(f)

    # 2) 스케일러
    scaler_path = artifact_dir / "emg_scaler.pkl"
    scaler = joblib.load(scaler_path)

    # 3) LabelEncoder
    label_encoder_path = artifact_dir / "label_encoder.pkl"
    label_encoder = joblib.load(label_encoder_path)

    # 4) 모델 가중치 (.pth)
    #    파일명: spiking_tcn_<encoding_type>_best.pth
    encoding_type = emg_meta.get("encoding_type", "latency")
    model_path = artifact_dir / f"spiking_tcn_{encoding_type}_best.pth"
    state_dict = torch.load(model_path, map_location="cpu")

    return emg_meta, scaler, label_encoder, state_dict


def build_model_from_meta(emg_meta, state_dict):
    """emg_meta 정보를 이용해 SpikingTCN 모델 구조 생성 + weight 로드"""

    num_inputs = int(emg_meta["num_channels"])
    num_classes = int(emg_meta["num_classes"])
    num_steps = int(emg_meta["num_steps"])

    model = SpikingTCN(
        num_inputs=num_inputs,
        num_channels=SPIKING_TCN_CHANNELS,
        num_classes=num_classes,
        kernel_size=SPIKING_TCN_KERNEL_SIZE,
        dropout=SPIKING_TCN_DROPOUT,
        timesteps=num_steps,
        beta=SPIKING_TCN_BETA,
        v_th=SPIKING_TCN_V_TH,
    )

    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_emg_window(emg_window: np.ndarray, scaler, emg_meta):
    """
    EMG 윈도우 하나를 받아서:
      1) shape 체크
      2) z-score 정규화
      3) PyTorch tensor [1, T, C] 로 변환
    """
    window_size = int(emg_meta["window_size"])
    num_channels = int(emg_meta["num_channels"])

    if emg_window.shape != (window_size, num_channels):
        raise ValueError(
            f"EMG window shape mismatch: expected ({window_size}, {num_channels}), "
            f"but got {emg_window.shape}"
        )

    # [T, C] → [T*C, C] 꼴은 이미 [T, C]라서 그대로 flatten 후 scaler 적용
    emg_flat = emg_window.reshape(-1, num_channels)
    emg_scaled_flat = scaler.transform(emg_flat)
    emg_scaled = emg_scaled_flat.reshape(1, window_size, num_channels)  # [1, T, C]

    emg_tensor = torch.from_numpy(emg_scaled.astype(np.float32))
    return emg_tensor


def run_inference(model, emg_tensor: torch.Tensor, label_encoder):
    """
    단일 EMG 윈도우에 대해 추론 수행:
      - logits → softmax → 최고 확률 클래스
      - 클래스 인덱스 및 라벨 문자열 반환
    """
    with torch.no_grad():
        logits = model(emg_tensor)  # shape: [1, num_classes]
        probs = torch.softmax(logits, dim=-1)
        conf, pred_idx = torch.max(probs, dim=-1)

    pred_idx = int(pred_idx.item())
    conf = float(conf.item())

    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    return pred_idx, pred_label, conf, probs.numpy().squeeze()


def parse_args():
    parser = argparse.ArgumentParser(description="SpikingTCN EMG Inference Script")

    parser.add_argument(
        "--artifact-dir",
        type=str,
        required=True,
        help="Directory containing .pth, scaler, label_encoder, emg_meta.json",
    )
    parser.add_argument(
        "--emg-npy",
        type=str,
        default=None,
        help="Optional path to .npy file with EMG window (shape: [window_size, num_channels])",
    )
    parser.add_argument(
        "--use-dummy",
        action="store_true",
        help="Use dummy random EMG window instead of loading from file",
    )
    parser.add_argument(
    "--sample-from-mat",
    type=str,
    default=None,
    help="Load EMG window from a .mat file (e.g., ./src/data/s2.mat)",
    )

    return parser.parse_args()

class ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, _ = self.model(x, return_spikes=True)
        return logits


def main():
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    sample_index = 1500
    emg_meta, scaler, label_encoder, state_dict = load_artifacts(artifact_dir)
    print(f"    - encoding_type: {emg_meta.get('encoding_type')}")
    print(f"    - window_size: {emg_meta.get('window_size')}")
    print(f"    - num_channels: {emg_meta.get('num_channels')}")
    print(f"    - num_classes: {emg_meta.get('num_classes')}")

    print("\n[2] Building SpikingTCN model...")
    model = build_model_from_meta(emg_meta, state_dict)
    print("    - Model ready (eval mode).")
    
    window_size = int(emg_meta["window_size"])
    num_channels = int(emg_meta["num_channels"])

    # 3-1) MAT 파일에서 window 추출
    if args.sample_from_mat is not None:
        print(f"\n[3] Loading EMG sample window from mat: {args.sample_from_mat}")

        # Raw load
        emg_raw, labels_raw = load_ninapro_data(args.sample_from_mat)
        print(f"    - raw EMG shape: {emg_raw.shape}")
        print(f"    - raw labels shape: {labels_raw.shape}")

        # Windowing (training과 동일)
        X_win, y_win = preprocess_data_for_networks(
            emg_raw,
            labels_raw,
            window_size=window_size,
            overlap=100,
        )
        print(f"    - Generated windows: {X_win.shape}")

        # 첫 번째 윈도우 사용 (or index 변경)
        emg_window = X_win[sample_index]
        true_label_raw = int(y_win[sample_index])
        print(f"    - Selected window[0], shape: {emg_window.shape}")
        

    # 3-2) NPY 로드
    elif args.emg_npy is not None:
        print(f"\n[3] Loading EMG window from file: {args.emg_npy}")
        emg_window = np.load(args.emg_npy)
        print(f"    - Loaded EMG shape: {emg_window.shape}")

    # 3-3) Dummy window
    else:
        print("\n[3] No EMG window provided.")
        print("    Using dummy EMG window for testing.")
        emg_window = np.random.randn(window_size, num_channels).astype(np.float32)
        print(f"    - Dummy EMG shape: {emg_window.shape}")

    # -------------------------------------------------
    # 전처리 + 추론
    # -------------------------------------------------
    print("\n[4] Preprocessing EMG window (scaling + tensor conversion)...")
    emg_tensor = preprocess_emg_window(emg_window, scaler, emg_meta)
    print(f"    - Tensor shape: {tuple(emg_tensor.shape)}  # [1, T, C]")

    print("\n[5] Running inference...")
    pred_idx, pred_label, conf, probs = run_inference(model, emg_tensor, label_encoder)

    print("\n[6] Result")
    print("    - Predicted class index :", pred_idx)
    print("    - Predicted label       :", pred_label)
    print(f"    - Confidence (softmax)  : {conf:.4f}")
    print("    - Probabilities         :", probs)
    

    # ✅ sample-from-mat 모드일 때만 GT 비교
    if true_label_raw is not None:
        # label_encoder는 s1 기준으로 fit 되어있지만,
        # s2도 같은 stimulus ID 스페이스(0~17)이므로 같은 매핑 사용 가능.
        true_idx = int(label_encoder.transform([true_label_raw])[0])
        is_correct = (pred_idx == true_idx)

        print("\n[7] Ground Truth Check")
        print("    - Ground truth class index    :", true_idx)
        print("    - Ground truth raw label      :", true_label_raw)
        print("    - Prediction matches GT?      :", is_correct)

    print("\n=== Inference complete ===")
    onnx_path = artifact_dir / "spiking_tcn_inference.onnx"
    
    print(f"Exporting to ONNX: {onnx_path}")
    model_export = ExportWrapper(model)
    torch.onnx.export(
        model_export,
        emg_tensor,
        onnx_path.as_posix(),
        export_params=True,
        opset_version=17,
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
