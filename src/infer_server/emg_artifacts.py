# src/infer_server/emg_artifacts.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import json
import joblib
import numpy as np
import torch
import torch.nn as nn

# 👉 실제 네 프로젝트에 있는 클래스 이름/경로로 맞춰줘
from src.models.snn_core import SNNClassifier
from src.models.tcn import TCNClassifier
from src.models.hybrid_tcnsnn import HybridTCNSNN
from src.models.spiking_tcn import SpikingTCN


# -------------------------------------------------------------------
# 0. 하이퍼파라미터 (훈련 때와 반드시 동일해야 함)
#    필요하면 meta.json에 같이 저장해서 여기서 안 하드코딩하게 바꿔도 됨
# -------------------------------------------------------------------
@dataclass
class SNNParams:
    hidden_dim: int = 128
    num_layers: int = 2
    beta: float = 0.9


@dataclass
class TCNParams:
    channels: Tuple[int, ...] = (64, 128, 256)
    kernel_size: int = 3
    dropout: float = 0.2


@dataclass
class HybridParams:
    tcn_channels: Tuple[int, ...] = (64, 128, 256)
    kernel_size: int = 3
    dropout: float = 0.2
    beta: float = 0.94
    v_th: float = 1.0
    timesteps: int = 20


@dataclass
class SpikingTCNParams:
    channels: Tuple[int, ...] = (64, 128, 256)
    kernel_size: int = 3
    dropout: float = 0.2
    beta: float = 0.94
    v_th: float = 1.0


# -------------------------------------------------------------------
# 1. 모델 빌더 registry
# -------------------------------------------------------------------
def _build_snn(meta: dict, params: SNNParams, device: str) -> nn.Module:
    num_channels = int(meta["num_channels"])
    num_classes = int(meta["num_classes"])
    num_steps = int(meta.get("num_steps", 20))

    model = SNNClassifier(
        num_inputs=num_channels,
        num_classes=num_classes,
        hidden_dim=params.hidden_dim,
        num_layers=params.num_layers,
        timesteps=num_steps,
        beta=params.beta,
    )
    return model.to(device)


def _build_tcn(meta: dict, params: TCNParams, device: str) -> nn.Module:
    num_channels = int(meta["num_channels"])
    num_classes = int(meta["num_classes"])

    model = TCNClassifier(
        num_inputs=num_channels,
        num_channels=list(params.channels),
        num_classes=num_classes,
        kernel_size=params.kernel_size,
        dropout=params.dropout,
    )
    return model.to(device)


def _build_hybrid(meta: dict, params: HybridParams, device: str) -> nn.Module:
    num_channels = int(meta["num_channels"])
    num_classes = int(meta["num_classes"])
    num_steps = int(meta.get("num_steps", params.timesteps))

    model = HybridTCNSNN(
        num_inputs=num_channels,
        num_channels=list(params.tcn_channels),
        num_classes=num_classes,
        kernel_size=params.kernel_size,
        dropout=params.dropout,
        timesteps=num_steps,
        beta=params.beta,
        v_th=params.v_th,
    )
    return model.to(device)


def _build_spiking_tcn(meta: dict, params: SpikingTCNParams, device: str) -> nn.Module:
    num_channels = int(meta["num_channels"])
    num_classes = int(meta["num_classes"])
    num_steps = int(meta.get("num_steps", 20))

    model = SpikingTCN(
        num_inputs=num_channels,
        num_channels=list(params.channels),
        num_classes=num_classes,
        kernel_size=params.kernel_size,
        dropout=params.dropout,
        timesteps=num_steps,
        beta=params.beta,
        v_th=params.v_th,
    )
    return model.to(device)


# model_name (meta["model_name"]) → (빌더 함수, default_params)
MODEL_BUILDERS = {
    "SNNClassifier": ( _build_snn,        SNNParams() ),
    "TCNClassifier": ( _build_tcn,        TCNParams() ),
    "HybridTCNSNN":  ( _build_hybrid,     HybridParams() ),
    "SpikingTCN":    ( _build_spiking_tcn,SpikingTCNParams() ),
}


# -------------------------------------------------------------------
# 2. 아티팩트 로더
# -------------------------------------------------------------------
def load_emg_model(
    artifact_dir: Path | str,
    prefix: str,
    device: str = "cpu",
):
    """
    prefix 기준으로:
      - <prefix>_meta.json
      - <prefix>_best.pth
      - emg_scaler_{encoding_type}.pkl
      - label_encoder_{encoding_type}.pkl
    전부 로드하고, PyTorch 모델 + 전처리 객체까지 리턴.

    예:
      model, scaler, le, meta = load_emg_model("./output/rate", prefix="spiking_tcn_rate")
    """
    artifact_dir = Path(artifact_dir)

    # 1) meta.json
    meta_path = artifact_dir / f"{prefix}_meta.json"
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    model_name = meta["model_name"]
    encoding_type = meta.get("encoding_type", "rate")

    # 2) scaler, label encoder
    scaler_path = artifact_dir / f"emg_scaler_{encoding_type}.pkl"
    label_encoder_path = artifact_dir / f"label_encoder_{encoding_type}.pkl"

    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)

    # 3) state_dict (.pth)
    weight_path = artifact_dir / f"{prefix}_best.pth"
    state_dict = torch.load(weight_path, map_location=device)

    # 4) 모델 생성
    if model_name not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model_name [{model_name}] in meta; "
            f"expected one of {list(MODEL_BUILDERS.keys())}"
        )

    build_fn, default_params = MODEL_BUILDERS[model_name]
    model = build_fn(meta, default_params, device=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, scaler, label_encoder, meta


# -------------------------------------------------------------------
# 3. EMG 윈도우 전처리 + 추론
# -------------------------------------------------------------------
def preprocess_emg_window(emg_window: np.ndarray, scaler, meta: dict) -> torch.Tensor:
    """
    EMG 윈도우 하나를:
      - shape 체크
      - scaler (StandardScaler) 적용
      - [1, T, C] PyTorch tensor 로 반환
    """
    window_size = int(meta["window_size"])
    num_channels = int(meta["num_channels"])

    if emg_window.shape != (window_size, num_channels):
        raise ValueError(
            f"EMG window shape mismatch: "
            f"expected ({window_size}, {num_channels}), got {emg_window.shape}"
        )

    flat = emg_window.reshape(-1, num_channels)  # [T, C]
    flat_scaled = scaler.transform(flat)
    scaled = flat_scaled.reshape(1, window_size, num_channels)

    return torch.from_numpy(scaled.astype(np.float32))


def infer_single_window(
    model: nn.Module,
    emg_tensor: torch.Tensor,
    label_encoder,
    device: str = "cpu",
):
    """
    단일 윈도우에 대해:
      - model forward (logits or (logits, spikes) 모두 지원)
      - softmax → predicted idx, label, confidence, full probs
    """
    emg_tensor = emg_tensor.to(device)

    with torch.no_grad():
        try:
            logits = model(emg_tensor, return_spikes=False)
        except TypeError:
            logits = model(emg_tensor)

        probs = torch.softmax(logits, dim=-1)
        conf, pred_idx = torch.max(probs, dim=-1)

    pred_idx = int(pred_idx.item())
    conf = float(conf.item())
    probs_np = probs.cpu().numpy().squeeze()

    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    
    
    return pred_idx, pred_label, conf, probs_np
