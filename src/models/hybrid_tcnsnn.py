import snntorch as snn
import torch
import torch.nn as nn
from snntorch import surrogate
from torch import Tensor


class TCNBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, d: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, padding=d * (k - 1) // 2, dilation=d),
            nn.ReLU(),
            nn.Conv1d(c_out, c_out, k, padding=d * (k - 1) // 2, dilation=d),
        )
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        y = self.net(x) + self.res(x)
        return self.act(y)


class HybridTCNSNN(nn.Module):
    def __init__(self, c_in: int = 8, hidden: int = 32, classes: int = 3, seq: int = 64) -> None:
        super().__init__()
        self.tcn1 = TCNBlock(c_in, hidden, k=3, d=1)
        self.tsn = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())
        self.readout = nn.Linear(hidden * seq, classes)
        self.seq = seq

    def forward(self, x: Tensor, num_steps: int = 1) -> tuple[Tensor, Tensor]:
        # x: [B, C, T]
        h = self.tcn1(x)
        spk_seq = []
        mem = torch.zeros_like(h)
        for _ in range(num_steps):
            spk, mem = self.tsn(h, mem)
            spk_seq.append(spk)
        s = torch.stack(spk_seq).mean(0)  # [B, hidden, T]
        z = self.readout(s.flatten(1))
        return z, s
