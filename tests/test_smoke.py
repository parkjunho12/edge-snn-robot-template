from src.models.hybrid_tcnsnn import HybridTCNSNN
import torch


def test_forward():
    m = HybridTCNSNN()
    x = torch.rand(2, 8, 64)
    z, s = m(x, num_steps=2)
    assert z.shape[0] == 2
    assert s.shape[0] == 2
