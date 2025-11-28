# scripts/build_trt_engine_v0_2s.py

import argparse
import subprocess
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Build TensorRT engine (v0.2s) from ONNX"
    )
    p.add_argument(
        "--onnx-path",
        type=str,
        required=True,
        help="Path to ONNX model (e.g., ./output/rate/spiking_tcn_rate_inference.onnx)",
    )
    p.add_argument(
        "--engine-path",
        type=str,
        default="model_v0.2s_int8.plan",
        help="Output TensorRT engine path (.plan)",
    )
    p.add_argument(
        "--int8",
        action="store_true",
        help="Enable INT8 mode (requires calibration or Q/DQ ONNX)",
    )
    p.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 mode",
    )
    p.add_argument(
        "--workspace",
        type=int,
        default=4096,
        help="Max workspace size in MB (for TensorRT builder)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    onnx_path = Path(args.onnx_path)
    engine_path = Path(args.engine_path)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    cmd = [
        "trtexec",
        f"--onnx={onnx_path.as_posix()}",
        f"--saveEngine={engine_path.as_posix()}",
        "--explicitBatch",
        f"--workspace={args.workspace}",
    ]

    # precision options
    if args.fp16:
        cmd.append("--fp16")
    if args.int8:
        cmd.append("--int8")
        # ⚠️ 여기서 실제 INT8을 쓰려면:
        # 1) ONNX에 Q/DQ 노드가 이미 박혀있거나
        # 2) --calib 옵션 + 캘리브레이션용 캐시/데이터가 있어야 함
        # 지금은 구조만 잡아두고, 나중에 calibration 추가하는 편이 편함.

    print("\n[build_trt_engine_v0_2s] Running command:")
    print(" ", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("\n❌ trtexec failed!")
        print(result.stdout)
        print(result.stderr)
        raise SystemExit(result.returncode)

    print("\n✅ TensorRT engine build complete!")
    print(f"   → {engine_path}")


if __name__ == "__main__":
    main()
