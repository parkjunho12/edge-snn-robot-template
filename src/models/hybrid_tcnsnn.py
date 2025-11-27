from __future__ import annotations

import torch
import torch.nn as nn

from .tcn import TemporalConvNet
from .snn_core import SpikingNeuralNetwork


class HybridTCNSNN(nn.Module):
    """TCN과 SNN을 결합한 하이브리드 모델"""

    def __init__(
        self,
        input_size,
        num_classes,
        tcn_channels=[64, 128, 256],
        snn_hidden_sizes=[64, 128, 256],
        num_steps=10,
        kernel_size=3,
        dropout=0.2,
        encoding_type="rate",
        beta=0.9,
        threshold=1.0,
    ):
        super(HybridTCNSNN, self).__init__()

        self.num_steps = num_steps

        # TCN branch
        self.tcn = TemporalConvNet(input_size, tcn_channels, kernel_size, dropout)

        # SNN branch
        self.snn = SpikingNeuralNetwork(
            input_size,
            snn_hidden_sizes,
            num_steps=num_steps,
            beta=beta,
            encoding_type=encoding_type,
            threshold=threshold,
        )

        # Feature fusion
        combined_size = tcn_channels[-1] + snn_hidden_sizes[-1]
        self.fusion = nn.Sequential(
            nn.Linear(combined_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # TCN branch
        tcn_out = self.tcn(x)
        # print(tcn_out.shape)
        tcn_pooled = tcn_out.mean(dim=1)  # Global average pooling

        # SNN branch
        snn_spikes = self.snn(x)
        # print(snn_spikes.shape)
        snn_rates = snn_spikes.mean(dim=0)

        # Combine features
        combined = torch.cat([tcn_pooled, snn_rates], dim=1)

        # Final classification
        output = self.fusion(combined)
        return output
