"""
Spiking TCN ONNX Export Wrapper
상태를 명시적으로 입력/출력으로 관리하여 ONNX 호환성 확보
"""

import torch
import torch.nn as nn


class SpikingTCNONNXWrapper(nn.Module):
    """
    Spiking TCN을 ONNX로 변환하기 위한 Wrapper
    - 모든 시간 스텝을 하나의 forward pass에 unroll
    - Membrane state를 명시적으로 초기화
    """

    def __init__(self, spiking_tcn_model):
        super().__init__()
        self.model = spiking_tcn_model
        self.timesteps = spiking_tcn_model.timesteps

        # eval 모드 고정
        self.model.eval()

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, features) - 원본 EMG 입력

        Returns:
            logits: (batch_size, num_classes) - 분류 출력
        """
        batch_size = x.shape[0]
        device = x.device

        # 1. Spike Encoding
        spikes = self.model.encoder(x)  # (T, B, seq_len, C_in)

        # 2. Membrane states 명시적 초기화
        mem_states = []
        for blk in self.model.blocks:
            # init_leaky() 대신 명시적으로 zeros 생성
            # LIF membrane shape: (B, C, seq_len)
            mem1 = torch.zeros(
                batch_size,
                blk.conv1.out_channels,
                x.shape[1],
                device=device,
                dtype=x.dtype,
            )
            mem2 = torch.zeros(
                batch_size,
                blk.conv2.out_channels,
                x.shape[1],
                device=device,
                dtype=x.dtype,
            )
            mem_states.append((mem1, mem2))

        # 3. Time loop unrolling (ONNX는 이를 graph로 변환)
        logits_sum = torch.zeros(
            batch_size,
            self.model.classifier[-1].out_features,
            device=device,
            dtype=x.dtype,
        )

        for t in range(self.timesteps):
            cur = spikes[t]  # (B, seq_len, C_in)
            cur = cur.transpose(1, 2).contiguous()  # (B, C_in, seq_len)

            # Blocks forward
            for i, blk in enumerate(self.model.blocks):
                cur, mem1, mem2 = blk(cur, *mem_states[i])
                mem_states[i] = (mem1, mem2)

            # Classification
            pooled = cur.mean(dim=2)  # (B, C_last)
            logits_sum = logits_sum + self.model.classifier(pooled)

        # 4. Average over timesteps
        logits = logits_sum / self.timesteps

        return logits


class SpikingTCNStatefulONNX(nn.Module):
    """
    대안: Single timestep 추론용 (상태를 입력으로 받음)
    실시간 시스템에서 유용 - 각 timestep마다 호출
    """

    def __init__(self, spiking_tcn_model):
        super().__init__()
        self.model = spiking_tcn_model
        self.num_blocks = len(self.model.blocks)

    def get_initial_states(self, batch_size, seq_len, device="cpu"):
        """초기 membrane states 생성"""
        states = []
        for blk in self.model.blocks:
            mem1 = torch.zeros(
                batch_size, blk.conv1.out_channels, seq_len, device=device
            )
            mem2 = torch.zeros(
                batch_size, blk.conv2.out_channels, seq_len, device=device
            )
            states.extend([mem1, mem2])
        return states

    def forward(self, x, t, *mem_states):
        """
        Single timestep forward

        Args:
            x: (B, seq_len, C_in) - 원본 EMG 입력
            t: int - 현재 timestep index (0 ~ timesteps-1)
            *mem_states: 각 block의 (mem1, mem2) flatten된 list

        Returns:
            logits: (B, num_classes)
            *new_mem_states: 업데이트된 membrane states
        """
        # Spike encoding for single timestep
        spikes = self.model.encoder(x)  # (T, B, seq_len, C_in)
        cur = spikes[t]
        cur = cur.transpose(1, 2).contiguous()

        # Reconstruct mem_states
        mem_list = []
        for i in range(0, len(mem_states), 2):
            mem_list.append((mem_states[i], mem_states[i + 1]))

        # Forward through blocks
        new_mem_states = []
        for i, blk in enumerate(self.model.blocks):
            cur, mem1, mem2 = blk(cur, *mem_list[i])
            new_mem_states.extend([mem1, mem2])

        # Classification
        pooled = cur.mean(dim=2)
        logits = self.model.classifier(pooled)

        return logits, *new_mem_states


# =====================================
# Export 함수들
# =====================================


def export_spiking_tcn_full(model, save_path, example_input, opset_version=17):
    """
    전체 timesteps를 unroll한 ONNX 모델 export

    Args:
        model: SpikingTCN 모델
        save_path: 저장 경로
        example_input: (batch_size, seq_len, features) 예제 입력
        opset_version: ONNX opset version
    """
    wrapper = SpikingTCNONNXWrapper(model)
    wrapper.eval()

    print(f"Exporting full unrolled Spiking TCN to {save_path}")
    print(f"  - Timesteps: {wrapper.timesteps}")
    print(f"  - Input shape: {example_input.shape}")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example_input,
            save_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["emg"],
            output_names=["logits"],
            dynamic_axes={
                "emg": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            training=torch.onnx.TrainingMode.EVAL,
            verbose=False,
        )

    print("✅ Export complete")


def export_spiking_tcn_stateful(model, save_path, example_input, opset_version=17):
    """
    Single timestep 추론용 stateful ONNX 모델 export (고급)

    주의: 이 방식은 외부에서 state 관리가 필요함
    """
    wrapper = SpikingTCNStatefulONNX(model)
    wrapper.eval()

    batch_size, seq_len, _ = example_input.shape
    device = example_input.device

    # Example inputs
    t = torch.tensor([0], dtype=torch.long)
    mem_states = wrapper.get_initial_states(batch_size, seq_len, device)
    example_inputs = (example_input, t, *mem_states)

    # Dynamic axes for states
    dynamic_axes = {
        "emg": {0: "batch_size"},
        "t": {},  # scalar
        "logits": {0: "batch_size"},
    }

    input_names = ["emg", "t"]
    output_names = ["logits"]

    for i in range(len(mem_states)):
        input_names.append(f"mem_state_{i}")
        output_names.append(f"new_mem_state_{i}")
        dynamic_axes[f"mem_state_{i}"] = {0: "batch_size"}
        dynamic_axes[f"new_mem_state_{i}"] = {0: "batch_size"}

    print(f"Exporting stateful Spiking TCN to {save_path}")
    print(f"  - Input shape: {example_input.shape}")
    print(f"  - Num membrane states: {len(mem_states)}")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example_inputs,
            save_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=False,  # State는 constant folding 안 함
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            training=torch.onnx.TrainingMode.EVAL,
            verbose=False,
        )

    print("✅ Export complete")
    print("\n⚠️  이 모델은 외부에서 timestep loop와 state 관리가 필요합니다")


def verify_spiking_tcn_export(model, onnx_path, example_input, tolerance=1e-2):
    """
    ONNX export 검증

    Args:
        model: 원본 SpikingTCN
        onnx_path: ONNX 파일 경로
        example_input: 테스트 입력
        tolerance: 허용 오차
    """
    import onnxruntime as ort
    import numpy as np

    # PyTorch 출력
    model.eval()
    with torch.no_grad():
        pth_output = model(example_input, return_spikes=False)

    # ONNX 출력
    ort_session = ort.InferenceSession(onnx_path)
    onnx_input = example_input.cpu().numpy()
    onnx_output = ort_session.run(None, {"emg": onnx_input})[0]

    pth_np = pth_output.cpu().numpy()

    # 비교
    print("\n" + "=" * 60)
    print("Spiking TCN ONNX Verification")
    print("=" * 60)

    abs_diff = np.abs(pth_np - onnx_output)
    rel_diff = abs_diff / (np.abs(pth_np) + 1e-7)

    print(f"PyTorch output:\n{pth_np}")
    print(f"\nONNX output:\n{onnx_output}")
    print(f"\nMax absolute diff: {np.max(abs_diff):.6f}")
    print(f"Mean absolute diff: {np.mean(abs_diff):.6f}")
    print(f"Max relative diff: {np.max(rel_diff):.6f}")

    # Prediction 비교
    pth_pred = np.argmax(pth_np, axis=1)
    onnx_pred = np.argmax(onnx_output, axis=1)

    print(f"\nPyTorch prediction: {pth_pred}")
    print(f"ONNX prediction:    {onnx_pred}")
    print(f"Predictions match:  {np.array_equal(pth_pred, onnx_pred)}")

    # Pass/Fail
    if np.max(abs_diff) < tolerance and np.array_equal(pth_pred, onnx_pred):
        print(f"\n✅ PASS (tolerance: {tolerance})")
        return True
    else:
        print(f"\n❌ FAIL (tolerance: {tolerance})")
        return False


# =====================================
# 사용 예시
# =====================================

if __name__ == "__main__":
    # 예제 모델 생성
    from spiking_tcn import SpikingTCN

    model = SpikingTCN(
        num_inputs=12,
        num_channels=[64, 128, 256],
        num_classes=7,
        kernel_size=3,
        dropout=0.2,
        timesteps=10,
    )
    model.eval()

    # 예제 입력
    example_input = torch.randn(1, 200, 12)  # (batch, seq_len, channels)

    # Export
    export_spiking_tcn_full(model, "spiking_tcn.onnx", example_input, opset_version=17)

    # 검증
    verify_spiking_tcn_export(model, "spiking_tcn.onnx", example_input, tolerance=1e-2)
