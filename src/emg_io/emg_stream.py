# src/io/emg_stream.py
from __future__ import annotations

import time
from typing import AsyncIterator, List, NamedTuple

import numpy as np
import serial
import serial.tools.list_ports as lp


class EMG(NamedTuple):
    ts: float
    samples: np.ndarray


def find_port(vendor_hint: str | None = None) -> str:
    for p in lp.comports():
        if (not vendor_hint) or vendor_hint in (p.manufacturer or ""):
            return p.device
    raise RuntimeError("EMG device not found")


class EMGStream:
    def __init__(self, port: str | None = None, baud: int = 115200, win: int = 200, ch: int = 8):
        self.port, self.baud, self.win, self.ch = port, baud, win, ch

    async def stream(self) -> AsyncIterator[EMG]:
        ser = serial.Serial(self.port or find_port(), self.baud, timeout=0.001)
        buf: List[np.ndarray] = []
        while True:
            _ = ser.read(ser.in_waiting or 1)
            # TODO: 패킷 파서/CRC
            # buf에 샘플 누적 후 윈도우 당 EMG 반환
            if len(buf) >= self.win:
                yield EMG(ts=time.time(), samples=np.array(buf[-self.win :]))
