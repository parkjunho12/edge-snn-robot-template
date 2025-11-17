# src/pipes/sync_mux.py
from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

Stamp = float
Frame = np.ndarray
Emg = np.ndarray
StampedFrame = Tuple[Stamp, Frame]
StampedEmg = Tuple[Stamp, Emg]


class SyncMux:
    def __init__(self, tol_ms: float = 20, maxlen: int = 120):
        self.frames: Deque[StampedFrame] = deque(maxlen=maxlen)
        self.emgs: Deque[StampedEmg] = deque(maxlen=maxlen)
        self.tol: float = tol_ms / 1000.0

    def push_frame(self, ts: float, frame: np.ndarray) -> None:
        self.frames.append((ts, frame))

    def push_emg(self, ts: float, emg: np.ndarray) -> None:
        self.emgs.append((ts, emg))

    def nearest(self) -> Optional[Tuple[Stamp, Frame, Emg]]:
        # 최근 데이터의 ts로 상대 스트림에서 최근접 매칭
        if not self.frames or not self.emgs:
            return None
        tsf, f = self.frames[-1]
        tse, e = self.emgs[-1]
        # 더 최신 쪽 기준으로 매칭
        target_ts = max(tsf, tse)
        cand_f = min(self.frames, key=lambda x: abs(x[0] - target_ts))
        cand_e = min(self.emgs, key=lambda x: abs(x[0] - target_ts))
        if abs(cand_f[0] - cand_e[0]) <= self.tol:
            return (target_ts, cand_f[1], cand_e[1])
        return None
