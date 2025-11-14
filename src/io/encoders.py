from typing import Tuple
import torch


def rate_encode(x: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
    """Simple rate encoding: [B,C] -> [T,B,C] Bernoulli spikes."""
    x = x.clamp(0, 1)
    T = num_steps
    probs = x.unsqueeze(0).repeat(T, 1, 1)
    return torch.bernoulli(probs)


def window_signal(
    x: torch.Tensor, win: int = 32, hop: int = 16
) -> Tuple[torch.Tensor, int]:
    """Naive windowing for 1D signals. Returns [N, win], count."""
    xs = []
    for i in range(0, x.shape[-1] - win + 1, hop):
        xs.append(x[..., i: i + win])
    if not xs:
        return x[..., :win].unsqueeze(0), 1
    return torch.stack(xs), len(xs)
