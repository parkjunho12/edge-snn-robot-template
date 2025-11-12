import torch

def spike_count(spike_tensor: torch.Tensor) -> int:
    return int((spike_tensor>0).sum().item())

def synaptic_events(spike_tensor: torch.Tensor, fanout:int=16) -> int:
    return spike_count(spike_tensor) * fanout

def energy_score(spikes:int, syn:int, macs:int, alpha=1.0, beta=0.1, gamma=0.01):
    return alpha*spikes + beta*syn + gamma*macs
