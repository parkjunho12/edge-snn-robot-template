import torch
from src.models.hybrid_tcnsnn import HybridTCNSNN

if __name__ == "__main__":
    model = HybridTCNSNN()
    x = torch.randn(1, 8, 64)
    torch.onnx.export(
        model,
        (x, 1),
        "deploy/model.onnx",
        input_names=["x", "steps"],
        output_names=["logits", "spike_map"],
        opset_version=17,
        dynamic_axes={"x": {0: "B"}},
    )
    print("Saved to deploy/model.onnx")
