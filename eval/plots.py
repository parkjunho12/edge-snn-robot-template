from __future__ import annotations

import os
from typing import Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.models.spiking_tcn import SpikingTCN
from src.models.snn_core import SNNClassifier
from src.models.hybrid_tcnsnn import HybridTCNSNN


def plot_training_history(histories, model_names):
    """훈련 과정 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    colors = ["blue", "red", "green", "orange", "purple"]

    for i, (history, name) in enumerate(zip(histories, model_names)):
        color = colors[i % len(colors)]

        # Training accuracy
        axes[0, 0].plot(history["train_acc"], label=f"{name}", color=color, linewidth=2)

        # Validation accuracy
        axes[0, 1].plot(history["val_acc"], label=f"{name}", color=color, linewidth=2)

        # Training loss
        axes[1, 0].plot(
            history["train_loss"], label=f"{name}", color=color, linewidth=2
        )

        # Validation loss
        axes[1, 1].plot(history["val_loss"], label=f"{name}", color=color, linewidth=2)

    axes[0, 0].set_title("Training Accuracy", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title("Validation Accuracy", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title("Training Loss", fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title("Validation Loss", fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Model Training Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    count_path = os.path.join("./output/latency", "model_trainint_latency4.png")
    plt.savefig(count_path)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    """혼동 행렬 시각화"""
    cm = confusion_matrix(y_true, y_pred)

    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"{model_name} - Confusion Matrix", fontsize=16, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    count_path = os.path.join("./output/latency", "rasterplot_latency4.png")
    plt.savefig(count_path)
    plt.close(fig)


def visualize_snn_spikes(model, x, title_prefix="Model"):
    model.eval()
    with torch.no_grad():
        # 공통: (T, B, N) 형태의 spike 시퀀스를 확보
        if isinstance(model, SpikingTCN):
            logits, stats = model(x, return_spikes=True)
            spk = stats["spk_tbC"]  # (T, B, C_last)
        elif isinstance(model, SNNClassifier):
            spk = model.snn(x)  # (T, B, H)
        elif isinstance(model, HybridTCNSNN):
            spk = model.snn(x)  # (T, B, H)
        else:
            raise TypeError("SpikingTCN 또는 SNNClassifier만 지원")

        T, B, N = spk.shape
        # 발화율/카운트
        counts_per_neuron = spk.sum(dim=(0, 1))  # (N,)
        rate_per_neuron = counts_per_neuron / T  # 배치 평균된 타임스텝당 발화율
        total_spikes = counts_per_neuron.sum().item()

        print(f"=== {title_prefix} Spike Stats ===")
        print(f"T={T}, B={B}, N={N}")
        print(f"Total spikes: {total_spikes:.0f}")
        print(f"Mean firing rate per neuron: {rate_per_neuron.mean():.4f}")
        print(f"Max firing rate per neuron:  {rate_per_neuron.max():.4f}")

        os.makedirs("./output/latency", exist_ok=True)

        # 히스토그램 (두 모델 동일)
        fig = plt.figure()
        plt.hist(rate_per_neuron.cpu().numpy(), bins=40)
        plt.title(f"{title_prefix} Neuron Firing Rate Distribution")
        plt.xlabel("Firing rate (spikes / timestep)")
        plt.ylabel("Neuron count")
        plt.savefig(
            os.path.join("./output/latency", f"{title_prefix}_latency_hist.png")
        )
        plt.close(fig)

        # 래스터 (두 모델 동일)
        fig = plt.figure(figsize=(10, 5))
        spk_cpu = spk.cpu()
        max_neurons = min(N, 100)
        for h in range(max_neurons):
            # (T, B)에서 nonzero: [t, b]
            nz = (spk_cpu[:, :, h] > 0.5).nonzero(as_tuple=False)
            if nz.numel() == 0:
                continue
            # 배치 인덱스를 살짝 오프셋으로 그려 겹침 방지
            plt.scatter(
                nz[:, 0],
                h + nz[:, 1] * 0.1,
                marker="|",
                s=250,
                alpha=0.9,
                linewidths=2,
                color="black",
            )
        plt.title(f"{title_prefix} Raster (first {max_neurons} neurons)")
        plt.xlabel("Time step")
        plt.ylabel("Neuron index")
        plt.savefig(os.path.join("./output/latency", f"{title_prefix}_raster.png"))
        plt.close(fig)


def plot_model_comparison_results(results):
    """모델 성능 비교 결과 시각화"""
    model_names = list(results.keys())
    accuracies = [results[name]["test_acc"] for name in model_names]
    f1_macros = [results[name]["f1_macro"] for name in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy")
    bars2 = ax.bar(x + width / 2, f1_macros, width, label="F1-macro")

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.5,
                f"{height:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    plt.ylabel("Score (%)")
    plt.title("Model Performance Comparison", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    plt.ylim(0, 100)
    ax.legend(loc="upper right")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    count_path = os.path.join("./output/latency", "model_performance_com_latency4.png")
    plt.savefig(count_path)
    plt.close(fig)
