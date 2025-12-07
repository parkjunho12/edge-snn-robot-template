import time
import asyncio
from typing import Optional, AsyncGenerator
from pathlib import Path

import json
import psutil
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from src.config import Settings, build_ninapro_cfg
from src.emg_io.emg_stream import EMGMode, get_emg_stream, EMGStream
from src.models.hybrid_tcnsnn import HybridTCNSNN

from src.infer_server.emg_artifacts import (
    load_emg_model,
    preprocess_emg_window,
    infer_single_window,
)

settings = Settings()
settings.ninapro_path = "../data/s1.mat"
app = FastAPI(title="Edge SNN Robot Dashboard")

origins = ["*"]  # 또는 ["http://localhost:5500", "http://127.0.0.1:5500"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],      # POST, GET, OPTIONS 등 모두 허용
    allow_headers=["*"],      # Content-Type 같은 헤더 허용
)

# PyTorch model (original)

artifact_dir = Path("./output/rate")
model, scaler, label_encoder, meta = load_emg_model(
        artifact_dir, prefix="tcn", device="cpu"
)
snn_model, snn_scaler, snn_label_encoder, snn_meta = load_emg_model(
        artifact_dir, prefix="snn", device="cpu"
)
hybrid_model, hybrid_scaler, hybrid_label_encoder, hybrid_meta = load_emg_model(
        artifact_dir, prefix="hybrid", device="cpu"
)
spiking_tcn_model, spiking_tcn_scaler, spiking_tcn_label_encoder, spiking_tcn_meta = load_emg_model(
        artifact_dir, prefix="spiking_tcn", device="cpu"
)
window_size = int(meta["window_size"])
num_channels = int(meta["num_channels"])
batch_size = 1
# TensorRT runtime (for optimized inference)
current_emg_mode: EMGMode = settings.emg_mode

ninapro_cfg = build_ninapro_cfg(settings)
ninapro_emg_stream = get_emg_stream(EMGMode.NINAPRO, ninapro_cfg=ninapro_cfg)

def build_emg_stream(mode: EMGMode, settings: Settings) -> EMGStream:
    if settings.emg_mode == EMGMode.NINAPRO:
        print("Using NINAPRO EMG stream")
        emg_stream = ninapro_emg_stream
    elif settings.emg_mode == EMGMode.REALTIME:
        emg_stream = get_emg_stream(
            EMGMode.REALTIME,
            port=settings.emg_port,
            win=settings.emg_win,
            ch=settings.emg_ch,
            fs=settings.emg_fs,
        )
    else:
        print("Using DUMMY EMG stream")
        emg_stream = get_emg_stream(
            EMGMode.DUMMY, win=settings.emg_win, ch=settings.emg_ch, fs=settings.emg_fs
        )
    return emg_stream

emg_stream: EMGStream = build_emg_stream(current_emg_mode, settings)


class InferenceInput(BaseModel):
    batch: int = 1
    channels: int = 16
    length: int = 200

class InferInput(BaseModel):
    encoding_type: str = "rate"
    model_prefix: str = "tcn"
    device: str = "cpu"


class StreamConfig(BaseModel):
    duration_seconds: Optional[float] = 10  # None = infinite stream
    use_tensorrt: bool = False
    fps: int = 30  # Frames per second
    preprocess: bool = True  # Apply normalization
    emg_mode: EMGMode | None = EMGMode.DUMMY  # Optional EMG mode override
    model_type: str = "TCN"  # "TCN", "SNN", "Hybrid", "SpikingTCN"

class StreamResponse(BaseModel):
    timestamp: float
    latency_ms: float
    prediction: int
    confidence: float
    shape: list
    backend: str  # "tensorrt" or "pytorch"


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint"""
    return {
        "status": "ok",
        "tensorrt_available": trt_runtime is not None,
        "emg_mode": settings.emg_mode.value,
    }


@app.post("/infer")
def infer(inp: InferInput) -> dict[str, str]:
    """Single inference with PyTorch model"""

    emg_window = None
    emg_window = np.random.randn(window_size, num_channels).astype(np.float32)
    print(f"    - Dummy EMG shape: {emg_window.shape}")
    emg_tensor = preprocess_emg_window(emg_window, scaler, meta)
    t0 = time.perf_counter()
    pred_idx, pred_label, conf, probs = infer_single_window(
        model, emg_tensor, label_encoder, device=inp.device
    )
    dt = (time.perf_counter() - t0) * 1000.0
    return {
        "latency_ms": str(dt),
        "cpu_percent": str(psutil.cpu_percent(interval=None)),
        "pred_idx": str(pred_idx),
        "pred_label": str(pred_label),
        "conf": str(conf),
        "probs": str(probs),
    }


async def emg_stream_generator(
    config: StreamConfig
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields EMG inference results as Server-Sent Events (SSE)
    """
    start_time = time.time()
    frame_count = 0
    frame_interval = 1.0 / config.fps

    # Select backend
    trt_runtime = None
    use_trt = config.use_tensorrt and trt_runtime is not None
    backend = "tensorrt" if use_trt else "pytorch"
    
    if config.emg_mode != settings.emg_mode.value:
        settings.emg_mode = EMGMode(config.emg_mode)
    
    if config.model_type == "TCN":
        cur_model = model
        cur_scaler = scaler
        cur_meta = meta
    elif config.model_type == "SNN":
        cur_model = snn_model
        cur_scaler = snn_scaler
        cur_meta = snn_meta
    elif config.model_type == "Hybrid":
        cur_model = hybrid_model
        cur_scaler = hybrid_scaler
        cur_meta = hybrid_meta
    elif config.model_type == "SpikingTCN":
        cur_model = spiking_tcn_model
        cur_scaler = spiking_tcn_scaler
        cur_meta = spiking_tcn_meta

    try:
        # 🔹 EMG 스트림을 비동기 반복
        async for emg in emg_stream.stream():
            # duration 제한 체크
            if config.duration_seconds is not None:
                elapsed = time.time() - start_time
                
                if elapsed >= config.duration_seconds:
                    # 마지막 종료 메시지
                    completion_response = {
                        "status": "completed",
                        "total_frames": frame_count,
                        "duration_seconds": elapsed,
                        "avg_fps": frame_count / elapsed if frame_count > 0 and elapsed > 0 else 0,
                    }
                    yield f"data: {json.dumps(completion_response)}\n\n"
                    return  # 🔥 여기서 스트림 종료

            frame_start = time.time()

            # EMG data: [win, ch] = [200, 16]
            emg_data = emg.samples

            # Preprocess
            if config.preprocess:
                emg_tensor = preprocess_emg_window(emg_data, cur_scaler, cur_meta)
            else:
                emg_tensor = emg_data

            # Run inference
            inference_start = time.perf_counter()
            
            with torch.inference_mode():
                z = cur_model(emg_tensor)  # num_steps 생략/내부에서 처리한다고 가정

            # Prediction
            prediction = int(torch.argmax(z, dim=-1).item())
            confidence = float(torch.max(torch.softmax(z, dim=-1)).item())
            output_shape = list(z.shape)

            inference_time = (time.perf_counter() - inference_start) * 1000.0

            # SSE payload
            response = {
                "status": "running",
                "timestamp": time.time(),
                "frame": frame_count,
                "latency_ms": round(inference_time, 3),
                "prediction": prediction,
                "confidence": round(confidence, 4),
                "shape": output_shape,
                "backend": backend,
                "cpu_percent": psutil.cpu_percent(interval=None),
                # 디버그용 tensor shape도 여기에 포함
                "emg_tensor_shape": list(emg_tensor.shape),
            }

            # SSE 포맷으로 전송
            yield f"data: {json.dumps(response)}\n\n"

            frame_count += 1

            # FPS 유지
            frame_elapsed = time.time() - frame_start
            if frame_elapsed < frame_interval:
                await asyncio.sleep(frame_interval - frame_elapsed)

    except Exception as e:
        error_response = {
            "error": str(e),
            "timestamp": time.time(),
            "frame": frame_count,
        }
        yield f"data: {json.dumps(error_response)}\n\n"
    finally:
        # Send completion message
        completion_response = {
            "status": "completed",
            "total_frames": frame_count,
            "duration_seconds": time.time() - start_time,
            "avg_fps": frame_count / (time.time() - start_time) if frame_count > 0 else 0
        }
        yield f"data: {json.dumps(completion_response)}\n\n"



@app.post("/infer/stream")
async def infer_stream(config: StreamConfig = StreamConfig()):
    """
    Streaming inference endpoint using Server-Sent Events (SSE)
    
    Returns real-time EMG inference results as they are processed.
    
    Example usage:
    ```python
    import requests
    
    response = requests.post(
        "http://localhost:8000/infer/stream",
        json={"duration_seconds": 10, "use_tensorrt": True, "fps": 30},
        stream=True
    )
    
    for line in response.iter_lines():
        if line:
            print(line.decode())
    ```
    
    Or with curl:
    ```bash
    curl -X POST "http://localhost:8000/infer/stream" \
         -H "Content-Type: application/json" \
         -d '{"duration_seconds": 10, "use_tensorrt": true, "fps": 30}' \
         -N
    ```
    """
    global emg_stream, current_emg_mode
    if config.emg_mode is not None and config.emg_mode != current_emg_mode:
        # 모드 변경
        current_emg_mode = config.emg_mode
        # 새로운 EMG 스트림 재생성
        emg_stream = build_emg_stream(current_emg_mode, settings)
    return StreamingResponse(
        emg_stream_generator(config),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@app.get("/infer/stream/stats")
def stream_stats() -> dict:
    """Get statistics about the streaming capabilities"""
    return {
        "tensorrt_available": trt_runtime is not None,
        "emg_mode": settings.emg_mode.value,
        "emg_window_size": settings.emg_win,
        "emg_channels": settings.emg_ch,
        "emg_sampling_rate": settings.emg_fs,
        "max_throughput_qps": 7032.89 if trt_runtime else "unknown",
        "expected_latency_ms": 0.13 if trt_runtime else "unknown",
    }


@app.post("/benchmark")
def benchmark(num_iterations: int = 1000) -> dict:
    """Benchmark inference performance"""
    results = {"pytorch": None, "tensorrt": None}
    
    # PyTorch benchmark
    x = torch.rand(1, 16, 200)
    latencies_pytorch = []
    
    for _ in range(num_iterations):
        t0 = time.perf_counter()
        with torch.inference_mode():
            z, s = model(x)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies_pytorch.append(dt)
    
    results["pytorch"] = {
        "mean_latency_ms": float(np.mean(latencies_pytorch)),
        "median_latency_ms": float(np.median(latencies_pytorch)),
        "p99_latency_ms": float(np.percentile(latencies_pytorch, 99)),
        "throughput_qps": 1000.0 / np.mean(latencies_pytorch),
    }
    
    # TensorRT benchmark
    if trt_runtime:
        trt_results = trt_runtime.benchmark(
            input_shape=(1, 200, 16),
            num_iterations=num_iterations,
            warmup_iterations=50
        )
        results["tensorrt"] = trt_results
    
    return results


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "message": "edge-snn-robot-template infer server",
        "version": "2.0",
        "features": [
            "PyTorch inference",
            "TensorRT acceleration",
            "Real-time EMG streaming",
            "Server-Sent Events (SSE)",
        ]
    }