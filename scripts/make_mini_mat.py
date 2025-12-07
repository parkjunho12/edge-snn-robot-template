#!/usr/bin/env python3
from pathlib import Path
import scipy.io as sio

SRC = Path("src/data/s1.mat")
DST = Path("src/data/s1_mini_10s.mat")

print(f"[LOAD] Loading mat: {SRC}")
mat = sio.loadmat(SRC, squeeze_me=True, struct_as_record=False)

print(f"[INFO] Keys in mat: {list(mat.keys())}")

emg = mat["emg"]          # [T, C]
labels = mat["stimulus"]  # [T]

print(f"[INFO] emg shape    : {emg.shape}")
print(f"[INFO] stimulus shape: {labels.shape}")

# ------- 10초 만큼만 자르기 (fs=2000 가정) -------
fs = 2000
seconds = 10
N = min(fs * seconds, emg.shape[0])

emg_small = emg[:N, :]
labels_small = labels[:N]

print(f"[CUT] mini emg shape    : {emg_small.shape}")
print(f"[CUT] mini labels shape : {labels_small.shape}")

# ------- 새로운 mat으로 저장 -------
print(f"[SAVE] {DST}")
sio.savemat(
    DST,
    {
        "emg": emg_small,
        "stimulus": labels_small,
    },
    do_compression=True,
)
print("[DONE]")
