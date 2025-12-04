# edge-snn-robot-template

A production-ready template for **SNN/TCN-based edge robotics** projects (ROS2 + snnTorch + FastAPI + Docker).

Focus areas:

- **Latency-aware control** (ms-level end-to-end, **target p95 < 30 ms**)
- **Spike / energy metrics** (spike counts, firing rate, synaptic events)
- **Edge deployment** (Jetson / x86 + Docker) _(Raspberry Pi optional)_
- **Signal-driven control:** sEMG as primary input\*\* (IMU optional, vision removed in v0.2s)

## ✨ What this repo gives you

- A **Python “core loop”**: sensor stream → encoder → SNN/TCN → control command
- A minimal **FastAPI inference server** (batch/stream) with hooks for dashboards
- Hooks for **ROS2 nodes** (mobile base or robot arm)
- Tooling for **latency and spike-based energy evaluation**
- A **deployable Docker image** for edge devices

You can treat this as a starting point for:

- EMG-driven **robot arm / manipulator control**
- EMG/IMU-driven mobile robots (TurtleBot, diff-drive) _(optional IMU)_
- Simulation-only pipelines (Gazebo, fake sensors) for algorithm work

## 🧩 Folder layout

- `src/`
  - `models/` – TCN / SNN / Hybrid TCN–SNN models (PyTorch + snnTorch)
  - `control/` – low-latency control loop (policy → command)
  - `infer_server/` – FastAPI app exposing `/infer` and `/health`
  - `io/` – encoders/decoders (e.g. EMG/IMU → spikes, sliding windows)
  - `metrics/` – latency & spike/energy counters, tegrastats parser
- `ros2_ws/`
  - ROS2 nodes to bridge topics ↔ inference server
  - Messages/services for commands & sensor streams
- `eval/`
  - Latency benchmark scripts
  - Spike/energy metric tooling
- `deploy/`
  - Dockerfile + `docker-compose.yml` for edge devices (Pi / Jetson / x86)
  - Entrypoint script + env template
- `firmware/`
  - ESP32 / OpenMV example stubs for low-level I/O
- `docs/`
  - MkDocs skeleton (architecture notes, BOM, wiring examples)
- `tests/`
  - `pytest` smoke tests (imports + simple forward pass)
- `notebooks/`
  - Prototyping & analysis notebooks (optional)

CI:

- Lint (ruff)
- Type-check (mypy)
- Unit tests (pytest)
- Docker build

## Quickstart

```bash
# 1) clone repo
git clone https://github.com/parkjunho12/edge-snn-robot-template.git
cd edge-snn-robot-template

# 2) Python env
python -m venv .venv && source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# 3) Download datasets
# ex) s1.mat is training dataset
python scripts/download_data.py --mat s1.mat

# s2, s3.mat can download too
python scripts/download_data.py --mat s2.mat
python scripts/download_data.py --mat s3.mat

# 4) Download artifacts
python scripts/download_data.py --artifacts

cd v0_1
# v0_1 README.md read

cd ..

# 5) Test single window(mat file) [Optional]
# 6) Compare models [Optional]
# 7) Export onnx and compare with pytorch
# 8) Validate onnx (Optional)

cd v0_2
# v0_2 README.md read

cd ..

# 9) 

# 9) Run inference server (FastAPI)
uvicorn src.infer_server.app:app --reload --host 0.0.0.0 --port 8000
# Endpoints:
#   GET  /health
#   POST /infer/run         # batch
#   WS   /infer/stream      # sEMG windows → predictions
#   WS   /emg/stream        # raw/processed sEMG (optional)

# 10) ROS2 (optional)
source /opt/ros/humble/setup.bash
cd ros2_ws
colcon build --symlink-install


# 11) Docker build (edge)
docker build -t edge-snn-robot:dev .
docker compose -f deploy/docker-compose.yml up
```

## Architecture

```mermaid
flowchart LR
  SENS[sEMG] --> PREPROC[notch/z-score]
  PREPROC[notch/z-score] --> ENC[Spike / Window Encoder]
  ENC --> HYB[Hybrid TCN-SNN Inference]
  HYB --> CTRL[Controller : PID / Policy]
  CTRL --> ACT[Robot Actuators]


  HYB --> MET[Metrics: latency, spikes, energy]
  MET --> API_NODE[FastAPI Dashboard]
```

## Safety & Calibration

- Dead-man: when RMS < rest_threshold → hold/stop
- Velocity & acceleration clamps per joint
- 5-min user calibration: rest → MVC (3×3s) → quick gestures → verify (>90% acc)

## Roadmap (6 months)

- ### ([v0.1](https://github.com/parkjunho12/edge-snn-robot-template/tree/main/v0_1)): minimal SNN control loop + metrics ([tag](https://github.com/parkjunho12/edge-snn-robot-template/releases/tag/v0.1.16))


- ### ([v0.2](https://github.com/parkjunho12/edge-snn-robot-template/tree/main/v0_2)): **sEMG-only input + INT8/TensorRT**   (working)

✓ Requirement: p95 < 30 ms → PASSED  
✗ Hybrid/SNN models: ONNX/TensorRT export pending, Spikegen encoding problem

- ### v0.3: edge container + dashboard (soon)
- ### v0.4: Hybrid TCN–SNN ablation/results (soon)
- ### v0.5: robustness + fail-safe (soon)
- ### v1.0: report + kit release (soon)


## License

MIT (see `LICENSE`).
