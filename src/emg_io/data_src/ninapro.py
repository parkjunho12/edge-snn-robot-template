from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat  # ★ 중요

from scipy.signal import butter, filtfilt


@dataclass
class NinaProConfig:
    path: Path
    fs: int  # sampling rate (Hz), e.g. 2000
    win: int  # window size (samples)
    hop: int  # hop size (samples)
    ch: int | None = None  # optional: auto-detect if None
    key: str = "emg"  # .mat 내부에서 EMG 데이터가 들어있는 key
    label_key: Optional[str] = None  # g. "restimulus" / "labels"
    windowed: bool = False  # 이미 [N, win, ch] 형태인지?


@dataclass
class NinaProWindows:
    samples: NDArray[np.float32]  # [N, win, ch]
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

    emg = np.asarray(data[cfg.key]).astype(
        np.float32
    )  # maybe [time, ch] or [N, win, ch]
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
        w = emg[start:end]  # [win, ch]
        windows.append(w)

        if raw_labels is not None:
            # Majority vote or last
            lbl = int(raw_labels[start:end].mean().round())
            labels.append(lbl)

    windows = np.stack(windows)  # [N, win, ch]
    labels = np.array(labels) if labels else None

    return NinaProWindows(samples=windows, labels=labels)


def load_ninapro_data(file_path):
    """NinaPro 데이터를 로드하고 전처리하는 함수"""
    try:
        # .mat 파일 로드
        data = loadmat(file_path)

        print("Available keys in the data:", list(data.keys()))

        # EMG 데이터 추출
        if "emg" in data:
            emg_data = data["emg"]
        elif "data" in data:
            emg_data = data["data"]
        else:
            data_keys = [k for k in data.keys() if not k.startswith("__")]
            emg_data = data[data_keys[0]]

        # 라벨 데이터 추출
        if "stimulus" in data:
            labels = data["stimulus"].flatten()
        elif "restimulus" in data:
            labels = data["restimulus"].flatten()
        elif "glove" in data:
            labels = data["glove"]
            if labels.ndim > 1:
                labels = labels[:, 0]
        else:
            label_keys = [
                k
                for k in data.keys()
                if "stimulus" in k.lower() or "label" in k.lower()
            ]
            if label_keys:
                labels = data[label_keys[0]].flatten()
            else:
                data_keys = [k for k in data.keys() if not k.startswith("__")]
                labels = (
                    data[data_keys[1]].flatten()
                    if len(data_keys) > 1
                    else np.zeros(emg_data.shape[0])
                )

        print(f"EMG data shape: {emg_data.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Unique labels: {np.unique(labels)}")

        return emg_data, labels

    except Exception as e:
        print(f"Error loading data: {e}")
        print("Generating sample data for demonstration...")

        # 샘플 데이터 생성
        n_samples = 10000
        n_channels = 12
        emg_data = np.random.randn(n_samples, n_channels) * 0.1

        # EMG 신호처럼 보이도록 시간적 상관관계 추가
        for i in range(n_channels):
            emg_data[:, i] += np.sin(np.linspace(0, 100 * np.pi, n_samples) + i) * 0.05
            emg_data[:, i] += (
                np.convolve(np.random.randn(n_samples), np.ones(5) / 5, mode="same")
                * 0.02
            )

        labels = np.random.randint(0, 7, n_samples)  # 0-6 클래스
        return emg_data, labels


def bandpass_filter(signal, lowcut=20, highcut=450, fs=2000, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal, axis=0)


def preprocess_data_for_networks(emg_data, labels, window_size=200, overlap=100):
    """네트워크를 위한 EMG 데이터 전처리"""
    # 레이블이 0인 rest 구간 제거 (선택사항)
    non_zero_mask = labels != 0
    emg_data = emg_data[non_zero_mask]
    labels = labels[non_zero_mask]

    # 윈도우 기반 시퀀스 생성
    windowed_sequences = []
    windowed_labels = []

    step_size = window_size - overlap
    results = {}
    for i in range(0, len(emg_data) - window_size + 1, step_size):
        window = emg_data[i:i + window_size]
        window_label = labels[i:i + window_size]
        for j in range(0, len(window_label)):
            if results.get(f"{window_label[j]}") is not None:
                results[f"{window_label[j]}"] = results[f"{window_label[j]}"] + 1
            else:
                results[f"{window_label[j]}"] = 1

        # 윈도우 내에서 가장 빈번한 라벨 사용
        unique_labels, counts = np.unique(window_label, return_counts=True)
        dominant_label = unique_labels[np.argmax(counts)]
        windowed_sequences.append(window)
        windowed_labels.append(dominant_label)
    print(results)

    return np.array(windowed_sequences), np.array(windowed_labels)
