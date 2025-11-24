import warnings

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import torch
from torch.utils.data import DataLoader

from src.emg_io.data_src.emg_dataset import EMGDataset
from src.emg_io.data_src.ninapro import (
    load_ninapro_data,
    preprocess_data_for_networks,
)  # bandpass_filter 필요시 여기서 import
from src.models.snn_core import SNNClassifier
from src.models.tcn import TCNClassifier
from src.models.hybrid_tcnsnn import HybridTCNSNN
from src.models.spiking_tcn import SpikingTCN
from eval.train_loop import train_model, evaluate_model
from eval.plots import (
    plot_training_history,
    plot_model_comparison_results,
    plot_confusion_matrix,
    visualize_snn_spikes,
)

warnings.filterwarnings("ignore")


def main():
    # Check if CUDA is available

    # 데이터 경로 설정
    DATA_PATH = "./src/data/s1.mat"
    encoding_type = "latency"
    num_steps = 20
    print("=== EMG Classification with TCN and SNN using PyTorch ===")

    # 1. 데이터 로드
    print("\n1. Loading data...")
    emg_data, labels = load_ninapro_data(DATA_PATH)

    # emg_data = bandpass_filter(emg_data, lowcut=20, highcut=450, fs=2000)

    # 2. 데이터 전처리
    print("\n2. Preprocessing data...")
    X, y = preprocess_data_for_networks(emg_data, labels, window_size=200, overlap=100)

    print(f"Sequence data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of classes: {len(np.unique(y))}")

    # 3. 라벨 인코딩
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(np.unique(y_encoded))
    class_names = [f"Gesture {i}" for i in range(num_classes)]

    # 4. 데이터 분할
    print("\n3. Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # 5. 데이터 정규화
    print("\n4. Normalizing data...")
    scaler = StandardScaler()

    # 2D로 변형 후 스케일링, 다시 3D로 복원
    X_train_res = X_train.reshape(-1, X_train.shape[-1])
    X_val_res = X_val.reshape(-1, X_val.shape[-1])
    X_test_res = X_test.reshape(-1, X_test.shape[-1])

    X_train_scaled = scaler.fit_transform(X_train_res).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val_res).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test_res).reshape(X_test.shape)

    # 6. DataLoader 생성
    print("\n5. Creating DataLoaders...")
    train_dataset = EMGDataset(X_train_scaled, y_train)
    val_dataset = EMGDataset(X_val_scaled, y_val)
    test_dataset = EMGDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 7. 모델 학습 및 평가
    print("\n6. Training Models...")
    results = {}
    histories = []

    # SNN
    print("\n--- Training SNN Model ---")
    snn_model = SNNClassifier(
        input_size=X_train.shape[-1],
        num_classes=num_classes,
        encoding_type=encoding_type,
        num_steps=num_steps,
        beta=0.95,
        threshold=0.48,
    )
    snn_model, snn_history = train_model(
        snn_model, train_loader, val_loader, title_prefix="SNN"
    )
    snn_test_acc, snn_f1_macro, snn_f1_weighted, snn_preds, snn_targets = (
        evaluate_model(snn_model, test_loader)
    )
    results["SNN(latency)"] = {
        "test_acc": snn_test_acc,
        "f1_macro": snn_f1_macro,
        "f1_weighted": snn_f1_weighted,
        "preds": snn_preds,
        "targets": snn_targets,
    }
    histories.append(snn_history)

    # TCN
    print("\n--- Training TCN Model ---")
    tcn_model = TCNClassifier(input_size=X_train.shape[-1], num_classes=num_classes)
    tcn_model, tcn_history = train_model(
        tcn_model, train_loader, val_loader, title_prefix="TCN"
    )
    tcn_test_acc, tcn_f1_macro, tcn_f1_weighted, tcn_preds, tcn_targets = (
        evaluate_model(tcn_model, test_loader)
    )
    results["TCN"] = {
        "test_acc": tcn_test_acc,
        "f1_macro": tcn_f1_macro,
        "f1_weighted": tcn_f1_weighted,
        "preds": tcn_preds,
        "targets": tcn_targets,
    }
    histories.append(tcn_history)

    print("\n--- Training Hybrid TCN-SNN Model ---")
    hybrid_fusion_model = HybridTCNSNN(
        input_size=X_train.shape[-1],
        num_classes=num_classes,
        encoding_type=encoding_type,
        num_steps=num_steps,
        beta=0.95,
        threshold=0.48,
    )
    hybrid_fusion_model, hybrid_fusion_history = train_model(
        hybrid_fusion_model, train_loader, val_loader, title_prefix="Hybrid"
    )
    (
        hybrid_fusion_test_acc,
        hybrid_fusion_f1_macro,
        hybrid_fusion_f1_weighted,
        hybrid_fusion_preds,
        hybrid_fusion_targets,
    ) = evaluate_model(hybrid_fusion_model, test_loader)
    results["Hybrid(latency)"] = {
        "test_acc": hybrid_fusion_test_acc,
        "f1_macro": hybrid_fusion_f1_macro,
        "f1_weighted": hybrid_fusion_f1_weighted,
        "preds": hybrid_fusion_preds,
        "targets": hybrid_fusion_targets,
    }
    histories.append(hybrid_fusion_history)

    # Hybrid
    print("\n--- Training SpikingTCN Model ---")
    hybrid_model = SpikingTCN(
        num_inputs=X_train.shape[-1],  # 채널 수(=특징 수)
        num_channels=[64, 128, 256],  # 기존 TCN 채널 구성 그대로
        num_classes=num_classes,
        kernel_size=3,
        dropout=0.2,
        timesteps=num_steps,  # 6~16 권장 (낮출수록 지연/연산↓)
        beta=0.94,  # EMG는 0.9~0.99가 전이 유지에 유리
        v_th=1.0,
    )
    hybrid_model, hybrid_history = train_model(
        hybrid_model, train_loader, val_loader, title_prefix="SpikingTCN"
    )
    (
        hybrid_test_acc,
        hybrid_f1_macro,
        hybrid_f1_weighted,
        hybrid_preds,
        hybrid_targets,
    ) = evaluate_model(hybrid_model, test_loader)
    results["SpikingTCN(latency)"] = {
        "test_acc": hybrid_test_acc,
        "f1_macro": hybrid_f1_macro,
        "f1_weighted": hybrid_f1_weighted,
        "preds": hybrid_preds,
        "targets": hybrid_targets,
    }
    histories.append(hybrid_history)

    # 8. 학습 결과 시각화
    print("\n7. Plotting Training History...")
    plot_training_history(
        histories, ["SNN(latency)", "TCN", "Hybrid(latency)", "SpikingTCN(latency)"]
    )

    print("\n8. Plotting Test Accuracy Comparison...")
    plot_model_comparison_results(results)

    print("\n9. Confusion Matrices")
    for model_name, result in results.items():
        plot_confusion_matrix(
            result["targets"], result["preds"], class_names, model_name
        )

    # 10. SNN Spike 시각화
    print("\n10. Visualizing Spikes (Hybrid Model)...")
    sample_input = torch.FloatTensor(X_test_scaled[:1])

    visualize_snn_spikes(snn_model, sample_input, title_prefix="SNNClassifier(latency)")

    visualize_snn_spikes(
        hybrid_fusion_model, sample_input, title_prefix="Hybrid(latency)"
    )

    visualize_snn_spikes(hybrid_model, sample_input, title_prefix="SpikingTCN(latency)")

    print("\n=== All Done ===")
    return hybrid_model, hybrid_history, scaler, label_encoder


if __name__ == "__main__":
    model, history, scaler, label_encoder = main()
