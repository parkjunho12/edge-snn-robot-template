# edge-snn-robot-template — v0.2 (TCN + TensorRT INT8)

> **v0.2 goal:**  
> Real-time sEMG-only gesture classification using a TCN model, exported to ONNX and deployed with TensorRT (INT8),  
> with a working latency target **p95 < 30 ms**.

This version focuses on a **TCN baseline** trained on a single subject (S1) from the 2 kHz NinaPro sEMG dataset,  
exported to ONNX, and compiled to a TensorRT INT8 engine for real-time inference.


## TensorRT INT8 Deployment

The ONNX model (tcn_inference.onnx) is compiled to a TensorRT INT8 engine using trtexec.

###  Build command

```bash

# 1) TensorRT Download - Compatible version for your computer, (% Need CUDA %)
wget "https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.9.0/tars/TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-12.8.tar.gz"   -O TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-12.8.tar.gz 

tar -xzvf TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-12.8.tar.gz

VOLUME="/{your_root_folder}"

TRT_DIR=$(ls -d TensorRT-*/ | head -n 1)

echo "export LD_LIBRARY_PATH=$VOLUME/tensorrt/$TRT_DIR/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc

echo "export PATH=$VOLUME/tensorrt/$TRT_DIR/bin:\$PATH" >> ~/.bashrc

source ~/.bashrc


# 2) TensorRT INT8 Deployment

trtexec \
  --onnx=output/rate/tcn_inference.onnx \
  --saveEngine=output/rate/model_tcn_int8.plan \
  --memPoolSize=workspace:4096 \
  --int8 \
  --minShapes=emg:1x200x16 \
  --optShapes=emg:1x200x16 \
  --maxShapes=emg:1x200x16 \
  --useCudaGraph

```

## Latency results (INT8, batch=1)

### trtexec performance summary

- Throughput: 8222.71 qps

- Latency:
  - min        = 0.1266 ms
  - max        = 1.8063 ms
  - mean       = 0.1319 ms
  - median     = 0.1315 ms
  - p90        = 0.1342 ms
  - p95        = 0.1345 ms
  - p99        = 0.1384 ms

## v0.2 latency goal

#### Target: p95 < 30 ms (working real-time inference)

#### Result (INT8): p95 ≈ 0.135 ms
- ✅ v0.2 latency requirement is exceeded by a large margin
(≈ 220× faster than the 30 ms target).

- This means that on the test hardware:

    - Most inferences complete in ≈0.13 ms, 
    - Even worst-case (max) is ≈1.8 ms,
    - Making the TCN INT8 engine effectively negligible in a typical 10–20 ms control loop budget.

## 4. Version Status (v0.2)

| Component            | Status                                                                                          |
|----------------------|--------------------------------------------------------------------------------------------------|
| TCN (sEMG-only)      | ✅ Trained on S1, ONNX export, TensorRT INT8, p95 < 30 ms                                       |
| Spiking TCN          | ⚠️ PyTorch OK, **TensorRT not supported (spikegen Bernoulli sampling breaks ONNX/TensorRT)**    |
| Hybrid TCN–SNN       | ⚠️ PyTorch OK, **TensorRT not supported (spikegen Bernoulli sampling breaks ONNX/TensorRT)**    |
| SNN-only classifier  | ⚠️ PyTorch OK, **TensorRT not supported (spikegen Bernoulli sampling breaks ONNX/TensorRT)**    |


### 4.1 Limitations: spikegen-based SNN / Spiking TCN / Hybrid

The SNN, Spiking TCN, and Hybrid TCN–SNN models in this project use
`snntorch.spikegen` with **Bernoulli sampling** to generate stochastic spikes
from continuous-valued sEMG features.

This has two important consequences for deployment:

1. **Data-dependent randomness and control flow**

   - Bernoulli spike generation introduces **data-dependent random sampling**
     (e.g., sampling spikes from `p ~ Bernoulli(rate)` inside the forward pass).
   - The new `torch.export`-based ONNX/TensorRT path in PyTorch/TensorRT
     cannot reliably trace or compile these stochastic, data-dependent branches.
   - As a result, `torch.onnx.export` / `torch.export` either fails or produces
     graphs that TensorRT cannot build into an engine.

2. **TensorRT support is currently unavailable**

   - Because of the `spikegen` Bernoulli code, the SNN / Spiking TCN / Hybrid
     models **cannot be exported to a clean, static ONNX graph** that TensorRT
     accepts.
   - For v0.2, **only the deterministic TCN baseline** is fully supported in the
     PyTorch → ONNX → TensorRT (INT8) pipeline.


> TL;DR:  
> At v0.2, **SNN / Spiking TCN / Hybrid models run only in PyTorch**.  
> TensorRT deployment is available **only for the TCN baseline** due to
> limitations in exporting Bernoulli-based spike generation to ONNX/TensorRT.

