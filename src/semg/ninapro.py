from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat   # ★ 중요


@dataclass
class NinaProConfig:
    path: Path
    fs: int            # sampling rate (Hz), e.g. 2000
    win: int           # window size (samples)
    hop: int           # hop size (samples)
    ch: int | None = None      # optional: auto-detect if None
    key: str = "emg"           # .mat 내부에서 EMG 데이터가 들어있는 key
    label_key: Optional[str] = None   # g. "restimulus" / "labels"
    windowed: bool = False     # 이미 [N, win, ch] 형태인지?


@dataclass
class NinaProWindows:
    samples: NDArray[np.float32]      # [N, win, ch]
    labels: Optional[NDArray[np.int64]] = None


def load_ninapro_mat(cfg: NinaProConfig) -> NinaProWindows:
    """
    Loads NinaPro .mat and returns windowed EMG.
    Supports both:
        - raw shape [time, ch]
        - windowed shape [N, win, ch]
    """
    data = loadmat(cfg.path)

    if cfg.key not in data:
        raise KeyError(f"Key '{cfg.key}' not found in .mat file")

    emg = np.asarray(data[cfg.key]).astype(np.float32)  # maybe [time, ch] or [N, win, ch]
    print(f"[NinaPro] Loaded key '{cfg.key}', shape = {emg.shape}")

    # 이미 윈도우된 데이터라면 그대로
    if cfg.windowed:
        assert emg.ndim == 3, f"Expected [N, win, ch], got {emg.shape}"
        if cfg.ch is not None:
            assert emg.shape[2] == cfg.ch
        return NinaProWindows(samples=emg)

    # Raw → windowing
    assert emg.ndim == 2, f"Expected raw EMG [time, ch], got {emg.shape}"
    time, ch = emg.shape
    if cfg.ch is not None:
        assert ch == cfg.ch

    windows = []
    labels = []

    # if label key exists → align labels to windows
    raw_labels = None
    if cfg.label_key and cfg.label_key in data:
        raw_labels = np.asarray(data[cfg.label_key]).reshape(-1)

    for start in range(0, time - cfg.win + 1, cfg.hop):
        end = start + cfg.win
        w = emg[start:end]      # [win, ch]
        windows.append(w)

        if raw_labels is not None:
            # Majority vote or last
            lbl = int(raw_labels[start:end].mean().round())
            labels.append(lbl)

    windows = np.stack(windows)      # [N, win, ch]
    labels = np.array(labels) if labels else None

    return NinaProWindows(samples=windows, labels=labels)
