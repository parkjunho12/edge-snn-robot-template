# tests/test_smoke.py
from __future__ import annotations

import numpy as np

from src.emg_io.emg_stream import EMGStream
from src.emg_io.vision_camera import VisionCamera
from src.pipes.sync_mux import SyncMux


def test_imports() -> None:
    # Classes should be importable
    assert VisionCamera is not None
    assert EMGStream is not None
    assert SyncMux is not None


def test_sync_mux_nearest() -> None:
    # Create multiplexer with 50 ms tolerance
    mux = SyncMux(tol_ms=50.0)

    # Push synthetic data — timestamps differ by 30 ms
    mux.push_frame(1.00, np.zeros((2, 2), dtype=np.uint8))
    mux.push_emg(1.03, np.zeros((8,), dtype=np.float32))

    fused = mux.nearest()
    assert fused is not None
    ts, frame, emg = fused
    assert isinstance(ts, float)
    assert isinstance(frame, np.ndarray)
    assert isinstance(emg, np.ndarray)
