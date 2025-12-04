import time
import asyncio
from typing import Optional, AsyncGenerator
from pathlib import Path

import psutil
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.config import Settings, build_ninapro_cfg
from src.emg_io.emg_stream import EMGMode, get_emg_stream
from src.models.hybrid_tcnsnn import HybridTCNSNN
from src.infer_server.runtime_trt import TRTRuntime

from src.infer_server.emg_artifacts import (
    load_emg_model,
    preprocess_emg_window,
    infer_single_window,
)

settings = Settings()
settings.ninapro_path = "../data/s1.mat"
app = FastAPI(title="Edge SNN Robot Dashboard")

# PyTorch model (original)

artifact_dir = Path("./output/rate")
model, scaler, label_encoder, meta = load_emg_model(
        artifact_dir, prefix="tcn", device="cpu"
)
window_size = int(meta["window_size"])
num_channels = int(meta["num_channels"])
batch_size = 1
# TensorRT runtime (for optimized inference)
trt_runtime: Optional[TRTRuntime] = None
try:
    trt_runtime = TRTRuntime("output/rate/model_tcn_fp16.plan")
    trt_runtime.warmup(input_shape=(batch_size, window_size, num_channels), num_iterations=10)
    print("✓ TensorRT runtime loaded successfully")
except Exception as e:
    print(f"⚠ TensorRT runtime not available: {e}")
    print("  Falling back to PyTorch inference")

# EMG stream setup
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
    channels: int = 16
    length: int = 200

class InferInput(BaseModel):
    encoding_type: str = "rate"
    model_prefix: str = "tcn"
    device: str = "cpu"


class StreamConfig(BaseModel):
    duration_seconds: Optional[float] = None  # None = infinite stream
    use_tensorrt: bool = True
    fps: int = 30  # Frames per second
    preprocess: bool = True  # Apply normalization


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


@app.post("/infer/tensorrt")
def infer_tensorrt(inp: InferenceInput) -> dict[str, str]:
    """Single inference with TensorRT runtime"""
    if trt_runtime is None:
        raise HTTPException(status_code=503, detail="TensorRT runtime not available")
    
    # Create dummy input matching TensorRT expected shape [batch, sequence, channels]
   
    input_shape = (inp.batch, inp.length, inp.channels)
    x = np.random.randn(*input_shape).astype(np.float32)
    
    t0 = time.perf_counter()
    output = trt_runtime.infer(x)
    dt = (time.perf_counter() - t0) * 1000.0
    
    # Get prediction
    prediction = int(np.argmax(output))
    confidence = float(np.max(output))
    
    return {
        "latency_ms": str(dt),
        "prediction": str(prediction),
        "confidence": str(confidence),
        "cpu_percent": str(psutil.cpu_percent(interval=None)),
        "shape": str(list(output.shape)),
        "backend": "tensorrt"
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
    use_trt = config.use_tensorrt and trt_runtime is not None
    backend = "tensorrt" if use_trt else "pytorch"
    
    try:
        while True:
            # Check duration limit
            if config.duration_seconds is not None:
                elapsed = time.time() - start_time
                if elapsed >= config.duration_seconds:
                    break
            
            frame_start = time.time()
            
            # Get EMG data from stream
            try:
                emg_data = next(emg_stream)  # Shape: [sequence_length, channels]
            except StopIteration:
                break
            
            # Preprocess
            if config.preprocess:
                # Z-score normalization
                emg_normalized = (emg_data - emg_data.mean(axis=0)) / (emg_data.std(axis=0) + 1e-8)
            else:
                emg_normalized = emg_data
            
            # Run inference
            inference_start = time.perf_counter()
            
            if use_trt:
                # TensorRT inference
                # Add batch dimension: [1, sequence_length, channels]
                emg_batch = np.expand_dims(emg_normalized, axis=0).astype(np.float32)
                output = trt_runtime.infer(emg_batch)
                
                # Get prediction
                prediction = int(np.argmax(output))
                confidence = float(np.max(output))
                output_shape = list(output.shape)
                
            else:
                # PyTorch inference
                # Convert to torch tensor: [batch, channels, sequence_length]
                emg_tensor = torch.from_numpy(emg_normalized).float()
                emg_tensor = emg_tensor.transpose(0, 1).unsqueeze(0)  # [1, channels, sequence_length]
                
                with torch.inference_mode():
                    z, s = model(emg_tensor, num_steps=1)
                
                # Get prediction
                prediction = int(torch.argmax(z, dim=-1).item())
                confidence = float(torch.max(torch.softmax(z, dim=-1)).item())
                output_shape = list(z.shape)
            
            inference_time = (time.perf_counter() - inference_start) * 1000.0
            
            # Prepare response
            response = {
                "timestamp": time.time(),
                "frame": frame_count,
                "latency_ms": round(inference_time, 3),
                "prediction": prediction,
                "confidence": round(confidence, 4),
                "shape": output_shape,
                "backend": backend,
                "cpu_percent": psutil.cpu_percent(interval=None),
            }
            
            # Yield as SSE format
            import json
            yield f"data: {json.dumps(response)}\n\n"
            
            frame_count += 1
            
            # Maintain target FPS
            frame_elapsed = time.time() - frame_start
            if frame_elapsed < frame_interval:
                await asyncio.sleep(frame_interval - frame_elapsed)
    
    except Exception as e:
        import json
        error_response = {
            "error": str(e),
            "timestamp": time.time(),
            "frame": frame_count
        }
        yield f"data: {json.dumps(error_response)}\n\n"
    
    finally:
        # Send completion message
        import json
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