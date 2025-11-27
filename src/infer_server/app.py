import time

import psutil
import torch
from fastapi import FastAPI
from src.config import Settings, build_ninapro_cfg
from src.emg_io.emg_stream import EMGMode, get_emg_stream
from pydantic import BaseModel

from src.models.hybrid_tcnsnn import HybridTCNSNN

settings = Settings()
app = FastAPI(title="Edge SNN Robot Dashboard")
model = HybridTCNSNN(input_size=16, num_classes=7)

if settings.emg_mode == EMGMode.NINAPRO:
    ninapro_cfg = build_ninapro_cfg(settings)
    emg_stream = get_emg_stream(EMGMode.NINAPRO, ninapro_cfg=ninapro_cfg)
elif settings.emg_mode == EMGMode.REALTIME:
    emg_stream = get_emg_stream(
        EMGMode.REALTIME,
        port=settings.emg_port,
        win=settings.emg_win,
        ch=settings.emg_ch,
        fs=settings.emg_fs,
    )
else:
    emg_stream = get_emg_stream(
        EMGMode.DUMMY, win=settings.emg_win, ch=settings.emg_ch, fs=settings.emg_fs
    )


class InferenceInput(BaseModel):
    batch: int = 1
    channels: int = 8
    length: int = 64
    steps: int = 1


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/infer")
def infer(inp: InferenceInput) -> dict[str, str]:
    x = torch.rand(inp.batch, inp.channels, inp.length)
    t0 = time.perf_counter()
    with torch.inference_mode():
        z, s = model(x, num_steps=inp.steps)
    dt = (time.perf_counter() - t0) * 1000.0
    spikes = float((s > 0).sum().item())
    return {
        "latency_ms": str(dt),
        "spikes": str(spikes),
        "cpu_percent": str(psutil.cpu_percent(interval=None)),
        "shape": str(list(z.shape)),
    }


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "edge-snn-robot-template infer server"}
