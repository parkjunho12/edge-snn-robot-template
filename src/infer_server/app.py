import time

import psutil
import torch
from fastapi import FastAPI
from pydantic import BaseModel

from src.models.hybrid_tcnsnn import HybridTCNSNN

app = FastAPI(title="Edge SNN Robot Dashboard")
model = HybridTCNSNN()


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
        "latency_ms": dt,
        "spikes": spikes,
        "cpu_percent": psutil.cpu_percent(interval=None),
        "shape": list(z.shape),
    }


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "edge-snn-robot-template infer server"}
