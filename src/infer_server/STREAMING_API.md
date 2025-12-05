# Streaming Inference API Documentation

Real-time EMG inference server with TensorRT acceleration and Server-Sent Events (SSE) streaming.

## 🚀 Quick Start

### 1. Start the Server

```bash
# Install dependencies
pip install fastapi uvicorn tensorrt pycuda numpy torch psutil

# Run the server
uvicorn src.infer_server.app:app --reload --host 0.0.0.0 --port 8000
```

### 2. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Single inference (PyTorch)
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"batch": 1, "channels": 16, "length": 200, "steps": 1}'

# Single inference (TensorRT)
curl -X POST http://localhost:8000/infer/tensorrt \
  -H "Content-Type: application/json" \
  -d '{"batch": 1, "channels": 16, "length": 200}'

# Streaming inference
curl -X POST http://localhost:8000/infer/stream \
  -H "Content-Type: application/json" \
  -d '{"duration_seconds": 10, "use_tensorrt": true, "fps": 30}' \
  -N
```

### 3. Use the Dashboard

Open `examples/dashboard.html` in your browser to visualize real-time inference results.

## 📡 API Endpoints

### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "tensorrt_available": true,
  "emg_mode": "dummy"
}
```

### POST `/infer`

Single inference with PyTorch model.

**Request:**
```json
{
  "batch": 1,
  "channels": 16,
  "length": 200,
  "steps": 1
}
```

**Response:**
```json
{
  "latency_ms": "2.456",
  "spikes": "128.0",
  "cpu_percent": "15.2",
  "shape": "[1, 7]"
}
```

### POST `/infer/tensorrt`

Single inference with TensorRT runtime.

**Request:**
```json
{
  "batch": 1,
  "channels": 16,
  "length": 200
}
```

**Response:**
```json
{
  "latency_ms": "0.132",
  "prediction": "3",
  "confidence": "0.8542",
  "cpu_percent": "8.4",
  "shape": "[1, 7]",
  "backend": "tensorrt"
}
```

### POST `/infer/stream`

Streaming inference with Server-Sent Events (SSE).

**Request:**
```json
{
  "duration_seconds": 30.0,
  "use_tensorrt": true,
  "fps": 30,
  "preprocess": true
}
```

**Parameters:**
- `duration_seconds` (float, optional): Duration to stream. `null` = infinite stream
- `use_tensorrt` (bool): Use TensorRT backend if available
- `fps` (int): Target frames per second
- `preprocess` (bool): Apply z-score normalization

**Response Stream:**

Each event has the format:
```
data: {"timestamp": 1234567890.123, "frame": 42, "latency_ms": 0.132, ...}
```

**Event Data Structure:**
```json
{
  "timestamp": 1234567890.123,
  "frame": 42,
  "latency_ms": 0.132,
  "prediction": 3,
  "confidence": 0.8542,
  "shape": [1, 7],
  "backend": "tensorrt",
  "cpu_percent": 8.4
}
```

**Completion Event:**
```json
{
  "status": "completed",
  "total_frames": 900,
  "duration_seconds": 30.0,
  "avg_fps": 30.0
}
```

**Error Event:**
```json
{
  "error": "Error message",
  "timestamp": 1234567890.123,
  "frame": 42
}
```

### GET `/infer/stream/stats`

Get streaming statistics and capabilities.

**Response:**
```json
{
  "tensorrt_available": true,
  "emg_mode": "dummy",
  "emg_window_size": 200,
  "emg_channels": 16,
  "emg_sampling_rate": 2000,
  "max_throughput_qps": 7032.89,
  "expected_latency_ms": 0.13
}
```

### POST `/benchmark`

Benchmark inference performance.

**Parameters:**
- `num_iterations` (int): Number of iterations (default: 1000)

**Response:**
```json
{
  "pytorch": {
    "mean_latency_ms": 2.456,
    "median_latency_ms": 2.401,
    "p99_latency_ms": 3.142,
    "throughput_qps": 407.16
  },
  "tensorrt": {
    "mean_latency_ms": 0.132,
    "median_latency_ms": 0.128,
    "p99_latency_ms": 0.163,
    "throughput_qps": 7575.76
  }
}
```

## 🐍 Python Client

### Basic Usage

```python
from examples.streaming_client import StreamingInferenceClient

# Create client
client = StreamingInferenceClient("http://localhost:8000")

# Health check
health = client.health_check()
print(health)

# Single inference
result = client.single_inference_tensorrt()
print(f"Prediction: {result['prediction']}, Latency: {result['latency_ms']}ms")

# Stream inference
client.stream_inference(
    duration_seconds=10.0,
    use_tensorrt=True,
    fps=30
)
```

### Custom Callback

```python
def my_callback(data):
    if 'prediction' in data:
        print(f"Frame {data['frame']}: Prediction={data['prediction']}, "
              f"Confidence={data['confidence']:.2%}")

client.stream_inference(
    duration_seconds=5.0,
    use_tensorrt=True,
    fps=30,
    callback=my_callback
)
```

### Command Line Interface

```bash
# Run all examples
python examples/streaming_client.py

# Run specific example
python examples/streaming_client.py --example 4

# Quick streaming
python examples/streaming_client.py --stream --duration 30 --fps 30

# Use PyTorch backend
python examples/streaming_client.py --stream --duration 10 --no-tensorrt

# Custom server URL
python examples/streaming_client.py --url http://192.168.1.100:8000 --stream
```

## 🌐 JavaScript/Browser Client

### Using Fetch API

```javascript
const config = {
    duration_seconds: 10.0,
    use_tensorrt: true,
    fps: 30,
    preprocess: true
};

fetch('http://localhost:8000/infer/stream', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify(config)
})
.then(response => {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    function read() {
        reader.read().then(({done, value}) => {
            if (done) return;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');

            lines.forEach(line => {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.substring(6));
                    console.log('Prediction:', data.prediction);
                    console.log('Confidence:', data.confidence);
                }
            });

            read();
        });
    }

    read();
});
```

### HTML Dashboard

Open `examples/dashboard.html` in a web browser for a full-featured visualization dashboard with:
- Real-time prediction display
- Latency monitoring
- Prediction distribution charts
- Event logging
- Performance metrics

## 🔧 Configuration

### Environment Variables

Set these in your `.env` file or environment:

```bash
# EMG Stream Configuration
EMG_MODE=dummy  # Options: dummy, realtime, ninapro
EMG_PORT=/dev/ttyUSB0  # For realtime mode
EMG_WIN=200  # Window size
EMG_CH=16  # Number of channels
EMG_FS=2000  # Sampling frequency

# TensorRT Engine Path
TENSORRT_ENGINE=output/rate/model_fp16.plan
```

### Code Configuration

In `src/infer_server/app.py`:

```python
from src.config import Settings

settings = Settings()
# Modify settings as needed
```

## 📊 Performance

### Benchmarks (NVIDIA GPU)

| Backend  | Mean Latency | P99 Latency | Throughput |
|----------|-------------|-------------|------------|
| PyTorch  | ~2.5 ms     | ~3.1 ms     | ~400 QPS   |
| TensorRT | ~0.13 ms    | ~0.16 ms    | ~7,000 QPS |

**Speedup: ~19x with TensorRT**

### Streaming Performance

- Maximum FPS: 60+ (limited by EMG sampling rate)
- Typical latency: 0.13-0.16 ms per frame (TensorRT)
- CPU usage: <10% with TensorRT
- Memory footprint: <500 MB

## 🛠️ Troubleshooting

### TensorRT Not Available

**Symptom:** Server logs show "TensorRT runtime not available"

**Solution:**
```bash
# Build TensorRT engine
trtexec --onnx=output/rate/tcn_inference.onnx \
        --saveEngine=output/rate/model_fp16.plan \
        --fp16 \
        --minShapes=emg:1x200x16 \
        --optShapes=emg:1x200x16 \
        --maxShapes=emg:1x200x16
```

### Connection Refused

**Symptom:** Client cannot connect to server

**Solution:**
```bash
# Check if server is running
curl http://localhost:8000/health

# Start server if needed
uvicorn src.infer_server.app:app --host 0.0.0.0 --port 8000
```

### Slow Streaming

**Symptom:** FPS lower than expected

**Solutions:**
1. Use TensorRT backend: `"use_tensorrt": true`
2. Reduce FPS: `"fps": 10`
3. Warm up the engine first:
   ```python
   trt_runtime.warmup(input_shape=(1, 200, 16), num_iterations=10)
   ```

### CORS Issues (Browser)

**Symptom:** Browser blocks requests from different origin

**Solution:** Add CORS middleware in `app.py`:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📝 Examples

All examples are in the `examples/` directory:

1. **streaming_client.py** - Python client with 6 complete examples
2. **dashboard.html** - Web-based visualization dashboard
3. **example_trt_inference.py** - TensorRT runtime examples

Run examples:
```bash
# Python client examples
python examples/streaming_client.py

# Open dashboard in browser
open examples/dashboard.html  # macOS
xdg-open examples/dashboard.html  # Linux
start examples/dashboard.html  # Windows
```

## 🔐 Security Considerations

For production deployment:

1. **Add authentication**: Use API keys or OAuth2
2. **Rate limiting**: Prevent abuse with rate limits
3. **Input validation**: Validate all input parameters
4. **HTTPS**: Use TLS for encrypted communication
5. **Network isolation**: Run on private network or VPN

Example with API key:
```python
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

API_KEY = "your-secret-key"
api_key_header = APIKeyHeader(name="X-API-Key")

@app.post("/infer/stream")
async def infer_stream(
    config: StreamConfig,
    api_key: str = Security(api_key_header)
):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    # ... rest of the code
```

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Server-Sent Events Specification](https://html.spec.whatwg.org/multipage/server-sent-events.html)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Chart.js Documentation](https://www.chartjs.org/)

## 📄 License

MIT License - See LICENSE file for details