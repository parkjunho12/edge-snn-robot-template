# src/config.py
from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings
from src.emg_io.emg_stream import EMGMode
from src.emg_io.data_src.ninapro import NinaProConfig


class Settings(BaseSettings):
    emg_mode: EMGMode = EMGMode.NINAPRO
    emg_port: str | None = None
    emg_win: int = 200
    emg_ch: int = 16
    emg_fs: int = 2000

    ninapro_path: Path | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def build_ninapro_cfg(settings: Settings) -> NinaProConfig:
    cfg = NinaProConfig(
        path=Path("../data/s1.mat"),
        fs=2000,
        win=200,
        hop=200,
        ch=12,
        key="emg",
        label_key="labels",
        windowed=True,   # ★ 그대로 load
    )
    if settings.ninapro_path is None:
        raise RuntimeError("NinaPro path not configured")
    return cfg
