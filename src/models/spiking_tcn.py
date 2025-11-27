from __future__ import annotations

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torch.nn.utils.parametrizations import weight_norm

from .snn_core import SpikeEncoder

spike_grad = surrogate.fast_sigmoid(slope=25)


class Chomp1d(nn.Module):
    """
    Remove extra padding from the right side
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous() if self.chomp_size > 0 else x


class SpikingTemporalBlock(nn.Module):
    """Spiking 버전 Temporal Block (막전위 유지)"""

    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0.2,
        beta=0.9,
        v_th=1.0,
    ):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.lif1 = snn.Leaky(
            beta=beta, threshold=v_th, spike_grad=spike_grad, init_hidden=False
        )
        self.do1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.lif2 = snn.Leaky(
            beta=beta, threshold=v_th, spike_grad=spike_grad, init_hidden=False
        )
        self.do2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )

    def forward(self, x, mem1, mem2, return_spk=False):
        x1 = self.chomp1(self.conv1(x))
        spk1, mem1 = self.lif1(x1, mem1)
        x1 = self.do1(spk1)

        x2 = self.chomp2(self.conv2(x1))
        spk2, mem2 = self.lif2(x2, mem2)
        out = self.do2(spk2)

        res = x if self.downsample is None else self.downsample(x)
        y = out + res

        if return_spk:
            # spk2: (B, C, T_seq) — 마지막 LIF의 바이너리 스파이크
            return y, mem1, mem2, spk2
        else:
            return y, mem1, mem2


# ======================
# Spiking Temporal ConvNet
# ======================


class SpikingTCN(nn.Module):
    """에너지 효율적인 Spiking-TCN (막전위 유지)"""

    def __init__(
        self,
        num_inputs,
        num_channels,
        num_classes,
        kernel_size=2,
        dropout=0.2,
        timesteps=10,
        beta=0.9,
        v_th=1.0,
    ):
        super().__init__()
        self.timesteps = timesteps
        self.encoder = SpikeEncoder(encoding_type="rate", num_steps=timesteps)

        self.blocks = nn.ModuleList()
        for i in range(len(num_channels)):
            dilation = 2**i
            in_c = num_inputs if i == 0 else num_channels[i - 1]
            out_c = num_channels[i]
            pad = (kernel_size - 1) * dilation
            self.blocks.append(
                SpikingTemporalBlock(
                    in_c,
                    out_c,
                    kernel_size,
                    1,
                    dilation,
                    pad,
                    dropout=dropout,
                    beta=beta,
                    v_th=v_th,
                )
            )

        self.classifier = nn.Sequential(
            nn.Linear(num_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, return_spikes=False):
        spikes = self.encoder(x)  # (T, B, F_in)
        mem_states = [
            (blk.lif1.init_leaky(), blk.lif2.init_leaky()) for blk in self.blocks
        ]

        logits_sum = 0.0
        spk_tbC_list = []  # 여기 모아 T×B×C_last 로 반환

        for t in range(self.timesteps):
            cur = spikes[t]  # (B, T_seq, C_in)
            cur = cur.transpose(1, 2).contiguous()  # (B, C_in, T_seq)

            for i, blk in enumerate(self.blocks):
                last_block = (i == len(self.blocks) - 1) and return_spikes
                if last_block:
                    cur, m1, m2, spk2 = blk(
                        cur, *mem_states[i], return_spk=True
                    )  # spk2: (B,C_last,T_seq)
                else:
                    cur, m1, m2 = blk(cur, *mem_states[i])
                mem_states[i] = (m1, m2)

            # 분류용 풀링
            pooled = cur.mean(dim=2)
            logits_sum += self.classifier(pooled)

            # 비교용 스파이크 시퀀스: (T,B,C_last)
            if return_spikes:
                # 래스터/히스토그램 비교를 위해 conv 시간축을 이진 합성:
                # conv 시간축 중 "한 번이라도 쏘면 1" 로 축약  => (B, C_last)
                spk_frame = (spk2 > 0).float().mean(dim=2)
                spk_tbC_list.append(spk_frame)

        logits = logits_sum / self.timesteps

        if return_spikes:
            spk_tbC = torch.stack(spk_tbC_list, dim=0)  # (T, B, C_last)
            # 원래 통계들에 같이 넣어서 보낼 수도 있고,
            # 최소한 spk_tbC만 넘겨도 시각화는 동일 루틴 재사용 가능
            stats = {"spk_tbC": spk_tbC}
            return logits, stats

        return logits
