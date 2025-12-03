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