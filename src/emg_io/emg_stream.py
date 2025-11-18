# src/io/emg_stream.py
from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator, NamedTuple, Protocol, Optional

import numpy as np
from numpy.typing import NDArray
import serial
import serial.tools.list_ports as lp

from src.data.ninapro import NinaProConfig, NinaProWindows, load_ninapro_mat


class EMG(NamedTuple):
    ts: float
    samples: NDArray[np.float32]


def find_port(vendor_hint: str | None = None) -> str:
    for p in lp.comports():
        if (not vendor_hint) or vendor_hint in (p.manufacturer or ""):
            return p.device
    raise RuntimeError("EMG device not found")


class EMGStream(Protocol):
    """Common async EMG stream interface."""


    async def stream(self) -> AsyncIterator[EMG]: ...


class NinaProEMGStream(EMGStream):
    
    
    def __init__(self, cfg: NinaProConfig, realtime: bool = True):
        self.cfg = cfg
        self.realtime = realtime
        self._windows: Optional[NinaProWindows] = None


    def _lazy_load(self) -> NinaProWindows:
        if self._windows is None:
            self._windows = load_ninapro_mat(self.cfg)  # ★ 여기만 변경
        return self._windows


    async def stream(self):
        win_data = self._lazy_load()
        emg = win_data.samples
        N, win, ch = emg.shape

        dt = self.cfg.hop / self.cfg.fs
        t0 = time.time()

        for i in range(N):
            ts = t0 + i * dt
            yield EMG(ts=ts, samples=emg[i])
            if self.realtime:
                await asyncio.sleep(dt)


class RealtimeEMGStream:
    """Read EMG from serial device in (soft) realtime."""


    def __init__(
        self,
        port: str | None = None,
        baud: int = 115200,
        win: int = 200,
        ch: int = 8,
        fs: int = 2000,
    ) -> None:
        self.port = port
        self.baud = baud
        self.win = win
        self.ch = ch
        self.fs = fs


    def _bytes_to_samples(self, buf: bytearray) -> NDArray[np.float32]:
        bytes_per_sample = 2
        needed = self.win * self.ch * bytes_per_sample
        if len(buf) < needed:
            raise ValueError("buffer too small")
        window = buf[-needed:]
        arr = np.frombuffer(window, dtype="<i2")  # int16
        return arr.astype(np.float32).reshape(self.win, self.ch)


    async def stream(self) -> AsyncIterator[EMG]:
        port = self.port or find_port()
        ser = serial.Serial(port, self.baud, timeout=0.001)
        buf = bytearray()
        dt = self.win / self.fs
        try:
            while True:
                chunk = await asyncio.to_thread(ser.read, ser.in_waiting or 1)
                if chunk:
                    buf.extend(chunk)
                bytes_per_sample = 2
                needed = self.win * self.ch * bytes_per_sample
                if len(buf) >= needed:
                    samples = self._bytes_to_samples(buf)
                    yield EMG(ts=time.time(), samples=samples)
                    await asyncio.sleep(dt)
        finally:
            ser.close()


class DummyEMGStream:
    """Synthetic EMG generator for tests and dev."""


    def __init__(
        self,
        win: int = 200,
        ch: int = 8,
        fs: int = 2000,
        realtime: bool = False,
    ) -> None:
        self.win = win
        self.ch = ch
        self.fs = fs
        self.realtime = realtime


    async def stream(self) -> AsyncIterator[EMG]:
        dt = self.win / self.fs
        t = time.time()
        while True:
            samples = np.random.randn(self.win, self.ch)
            .astype(np.float32)
            yield EMG(ts=t, samples=samples)
            t += dt
            if self.realtime:
                await asyncio.sleep(dt)


from enum import Enum
from pathlib import Path


class EMGMode(str, Enum):
    DUMMY = "dummy"
    REALTIME = "realtime"
    NINAPRO = "ninapro"


def get_emg_stream(mode: EMGMode, **kwargs) -> EMGStream:
    if mode == EMGMode.DUMMY:
        return DummyEMGStream(**kwargs)
    if mode == EMGMode.REALTIME:
        return RealtimeEMGStream(**kwargs)
    if mode == EMGMode.NINAPRO:
        ninapro_cfg = kwargs["ninapro_cfg"]
        return NinaProEMGStream(ninapro_cfg, 
                                realtime=kwargs.get("realtime", True))
    raise ValueError(f"Unknown EMG mode: {mode}")
