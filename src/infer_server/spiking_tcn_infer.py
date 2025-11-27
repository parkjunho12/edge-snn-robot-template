# src/infer_server/spiking_tcn_infer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import json
import joblib
import numpy as np
import torch
import torch.nn as nn

from src.models.spiking_tcn import SpikingTCN
from src.emg_io.data_src.ninapro import (
    load_ninapro_data,
    preprocess_data_for_networks,
)


# ------------------------------------------------------------
# SpikingTCN 하이퍼파라미터 (훈련 때와 동일해야 함)
# ------------------------------------------------------------
@dataclass
class SpikingTCNHyperParams:
    channels: Tuple[int, ...] = (64, 128, 256)
    kernel_size: int = 3
    dropout: float = 0.2
    beta: float = 0.94
    v_th: float = 1.0


# ============================================================
# 1. Artifact & Model 로딩
# ============================================================
def load_artifacts(artifact_dir: Path):
    """
    artifact_dir 안에서 다음 네 가지를 로드:
      - emg_meta.json
      - emg_scaler.pkl
      - label_encoder.pkl
      - spiking_tcn_<encoding_type>_best.pth
    """
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

    # 4) 모델 가중치
    encoding_type = emg_meta.get("encoding_type", "latency")
    model_path = artifact_dir / f"spiking_tcn_{encoding_type}_best.pth"
    state_dict = torch.load(model_path, map_location="cpu")

    return emg_meta, scaler, label_encoder, state_dict


def build_model_from_meta(
    emg_meta: dict,
    state_dict: dict,
    hparams: Optional[SpikingTCNHyperParams] = None,
    device: str = "cpu",
) -> nn.Module:
    """
    emg_meta + 하이퍼파라미터로 SpikingTCN 인스턴스 생성 후,
    state_dict 로드하고 eval 모드로 반환.
    """
    if hparams is None:
        hparams = SpikingTCNHyperParams()

    num_inputs = int(emg_meta["num_channels"])
    num_classes = int(emg_meta["num_classes"])
    num_steps = int(emg_meta["num_steps"])

    model = SpikingTCN(
        num_inputs=num_inputs,
        num_channels=list(hparams.channels),
        num_classes=num_classes,
        kernel_size=hparams.kernel_size,
        dropout=hparams.dropout,
        timesteps=num_steps,
        beta=hparams.beta,
        v_th=hparams.v_th,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# ============================================================
# 2. EMG 윈도우 준비 (from .mat / .npy / dummy)
# ============================================================
def load_window_from_mat(
    mat_path: Path,
    window_size: int,
    num_channels: int,
    sample_index: int = 0,
    overlap: int = 100,
):
    """
    Ninapro .mat 파일에서 EMG/label을 읽고
    -> training과 동일한 preprocess_data_for_networks()로 windowing 후
    -> sample_index 번째 윈도우와 해당 라벨을 반환
    """
    emg_raw, labels_raw = load_ninapro_data(mat_path)

    X_win, y_win = preprocess_data_for_networks(
        emg_raw,
        labels_raw,
        window_size=window_size,
        overlap=overlap,
    )

    if sample_index < 0 or sample_index >= len(X_win):
        raise IndexError(
            f"sample_index {sample_index} out of range (0 ~ {len(X_win)-1})"
        )

    emg_window = X_win[sample_index]
    true_label_raw = int(y_win[sample_index])

    if emg_window.shape != (window_size, num_channels):
        raise ValueError(
            f"Window shape mismatch: expected ({window_size}, {num_channels}), "
            f"got {emg_window.shape}"
        )

    return emg_window, true_label_raw


def load_window_from_npy(npy_path: Path):
    """
    np.save() 해 둔 EMG 윈도우 (.npy) 로딩
    - shape: [T, C] 이어야 함
    """
    emg_window = np.load(npy_path)
    return emg_window


def make_dummy_window(window_size: int, num_channels: int):
    """
    랜덤 dummy EMG window 생성 (네트워크/아티팩트 sanity check 용)
    """
    return np.random.randn(window_size, num_channels).astype(np.float32)


# ============================================================
# 3. EMG 윈도우 전처리 (scaler 적용 + tensor 변환)
# ============================================================
def preprocess_emg_window(
    emg_window: np.ndarray,
    scaler,
    emg_meta: dict,
) -> torch.Tensor:
    """
    EMG 윈도우 하나를 받아서:
      1) shape 체크
      2) z-score 정규화 (훈련 때 사용한 scaler)
      3) PyTorch tensor [1, T, C] 반환
    """
    window_size = int(emg_meta["window_size"])
    num_channels = int(emg_meta["num_channels"])

    if emg_window.shape != (window_size, num_channels):
        raise ValueError(
            f"EMG window shape mismatch: "
            f"expected ({window_size}, {num_channels}), but got {emg_window.shape}"
        )

    emg_flat = emg_window.reshape(-1, num_channels)  # [T, C]
    emg_scaled_flat = scaler.transform(emg_flat)
    emg_scaled = emg_scaled_flat.reshape(1, window_size, num_channels)  # [1, T, C]

    emg_tensor = torch.from_numpy(emg_scaled.astype(np.float32))
    return emg_tensor


# ============================================================
# 4. 추론
# ============================================================
def run_inference(
    model: nn.Module,
    emg_tensor: torch.Tensor,
    label_encoder,
    device: str = "cpu",
):
    """
    단일 EMG 윈도우에 대해 추론:
      - logits → softmax → predicted index/label/conf/probs
    SpikingTCN이 (logits, spikes) or logits 만 반환하는 경우 모두 처리.
    """
    emg_tensor = emg_tensor.to(device)

    with torch.no_grad():
        try:
            logits, _ = model(emg_tensor, return_spikes=False)
        except TypeError:
            logits = model(emg_tensor)

        probs = torch.softmax(logits, dim=-1)
        conf, pred_idx = torch.max(probs, dim=-1)

    pred_idx = int(pred_idx.item())
    conf = float(conf.item())
    probs_np = probs.cpu().numpy().squeeze()

    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    return pred_idx, pred_label, conf, probs_np


# ============================================================
# 5. ONNX Export용 Wrapper
# ============================================================
class ExportWrapper(nn.Module):
    """
    ONNX export 시, model(x) → logits 만 내보내도록 래핑.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        try:
            logits, _ = self.model(x, return_spikes=False)
        except TypeError:
            logits = self.model(x)
        return logits
