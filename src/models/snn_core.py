from __future__ import annotations

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import snntorch.spikegen as spikegen


class SpikeEncoder(nn.Module):
    """연속 신호를 스파이크로 변환하는 인코더"""

    def __init__(
        self,
        encoding_type="latency",
        num_steps=10,
        latency_linear=False,
        latency_threshold=0.75,
        per_channel_norm=True,
    ):
        super(SpikeEncoder, self).__init__()
        self.encoding_type = encoding_type
        self.num_steps = num_steps
        self.latency_linear = latency_linear
        self.latency_threshold = latency_threshold
        self.per_channel_norm = per_channel_norm

    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        batch_size, seq_len, features = x.shape
        if self.encoding_type == "delta":
            # 1) (B,T,C) 그대로 delta에 투입
            #    snntorch.spikegen.delta는 (B,T,C) -> (B,T,C) 형태 반환(버전에 따라 다를 수 있으니 아래 체크 권장)
            spikes = spikegen.delta(x, threshold=0.1)  # (B,T,C)

            # 2) 시간 차원 첫 번째로: (T,B,C)
            spikes = spikes.transpose(0, 1).contiguous()

            return spikes  # (T, B, C)
        if self.encoding_type == "rate":
            # Rate encoding: 입력 크기에 비례하는 스파이크 확률
            # 입력을 0-1 범위로 정규화
            p = torch.sigmoid(x)  # (B,T,C)

            spikes = spikegen.rate(p, num_steps=self.num_steps)  # (num_steps, B, T, C)
            return spikes

        elif self.encoding_type == "latency":
            # Latency encoding
            if self.latency_threshold > 0:
                x = torch.where(x < self.latency_threshold, torch.zeros_like(x), x)

            # --- Normalization ---
            if self.per_channel_norm:
                # 채널별 min-max 정규화
                x_min = x.min(dim=1, keepdim=True)[0]  # (B,1,C)
                x_max = x.max(dim=1, keepdim=True)[0]  # (B,1,C)
                batch_norm = (x - x_min) / (x_max - x_min + 1e-8)
            else:
                # 배치 전체 min-max
                batch_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)

            # --- Latency Encoding ---
            spikes = spikegen.latency(
                batch_norm,
                num_steps=self.num_steps,
                normalize=True,
                linear=self.latency_linear,
            )
            return spikes  # (num_steps, B, T, C)

        else:  # 'delta'
            # Delta modulation
            spikes = spikegen.delta(x, threshold=0.1)  # (B,T,C)

            # 2) 시간 차원 첫 번째로: (T,B,C)
            spikes = spikes.transpose(0, 1).contiguous()

            return spikes  # (T, B, C)


class SNNBlock(nn.Module):
    """SNN 블록 (LIF 뉴런 사용)"""

    def __init__(self, input_size, hidden_size, num_steps=10, beta=0.9, threshold=1.0):
        super(SNNBlock, self).__init__()

        self.num_steps = num_steps
        self.hidden_size = hidden_size

        # Linear layer
        self.fc = nn.Linear(input_size, hidden_size)

        # LIF neuron
        self.lif = snn.Leaky(
            beta=beta, threshold=threshold, spike_grad=surrogate.fast_sigmoid()
        )

    def forward(self, x):  # x: (T,B,input_size)
        T, B, _ = x.shape  # ❗ self.num_steps 대신 T 사용
        mem = self.lif.init_leaky()
        spk_rec, mem_rec = [], []
        for t in range(T):
            cur = self.fc(x[t])  # (B,H)
            spk, mem = self.lif(cur, mem)
            spk_rec.append(spk)
            mem_rec.append(mem)
        return torch.stack(spk_rec, 0), torch.stack(mem_rec, 0)


class SpikingNeuralNetwork(nn.Module):
    """Multi-layer SNN"""

    def __init__(
        self,
        input_size,
        hidden_sizes,
        num_steps=10,
        beta=0.9,
        threshold=1.0,
        encoding_type="latency",
    ):
        super(SpikingNeuralNetwork, self).__init__()

        self.num_steps = num_steps
        self.layers = nn.ModuleList()

        # Input encoding
        self.encoder = SpikeEncoder(
            encoding_type=encoding_type, num_steps=num_steps, per_channel_norm=True
        )

        # SNN layers
        layer_sizes = [input_size] + hidden_sizes
        for i in range(len(hidden_sizes)):
            self.layers.append(
                SNNBlock(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                    num_steps=num_steps,
                    beta=beta,
                    threshold=threshold,
                )
            )

    def forward(self, x):
        B, T, F = x.shape

        spikes = self.encoder(x)  # (num_steps, B, T_seq, C)  # latency 기준
        spikes = spikes.mean(dim=2).contiguous()  # (num_steps, B, C)

        for layer in self.layers:
            spikes, _ = layer(spikes)  # (num_steps, B, H)
        return spikes


class SNNClassifier(nn.Module):
    """SNN 기반 EMG 분류기"""

    def __init__(
        self,
        input_size,
        num_classes,
        hidden_sizes=[64, 128, 256],
        num_steps=10,
        beta=0.9,
        threshold=0.9,
        encoding_type="latency",
    ):
        super(SNNClassifier, self).__init__()

        self.num_steps = num_steps
        self.snn = SpikingNeuralNetwork(
            input_size, hidden_sizes, num_steps, beta, threshold, encoding_type
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], num_classes)

    def forward(self, x):
        batch_size, seq_len, features = x.shape

        # Get spike outputs from SNN
        spikes = self.snn(x)  # (num_steps, batch_size, hidden_size)

        spike_rates = spikes.mean(dim=0)  # (B, H)  <-- 시간축 평균

        output = self.output_layer(spike_rates)

        return output
