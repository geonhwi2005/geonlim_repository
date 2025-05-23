# dataset_loader.py
import os
import cv2
import numpy as np
import pandas as pd

# 문자 집합 및 파라미터
alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
max_str_len = 24                 # 최대 레이블 길이
num_of_characters = len(alphabets) + 1  # +1: CTC blank
num_of_timestamps = 64           # downsample=32 적용 시 최대 타임스텝
TARGET_H = 256
MAX_W = 64

def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.dropna(subset=['IDENTITY'], inplace=True)
    df = df[df['IDENTITY'] != 'UNREADABLE']
    df['IDENTITY'] = df['IDENTITY'].str.upper()
    df.reset_index(drop=True, inplace=True)
    return df

def preprocess(img, target_h=TARGET_H, max_w=MAX_W):
    h, w = img.shape
    new_w = int(w * (target_h / h))
    if new_w > max_w:
        new_w = max_w
    img = cv2.resize(img, (new_w, target_h))
    canvas = np.ones((target_h, max_w), dtype=np.uint8) * 255
    canvas[:, :new_w] = img
    return canvas  # 회전 제거, (H, W)

def label_to_num(label):
    return np.array([alphabets.find(ch) for ch in label], dtype=np.int32)

def load_data(df, img_dir, size, downsample=32):
    X, Y = [], []
    input_len, label_len = [], []

    for i in range(min(size, len(df))):
        img = cv2.imread(os.path.join(img_dir, df.loc[i, 'FILENAME']), cv2.IMREAD_GRAYSCALE)
        img = preprocess(img) / 255.0                     # (256, 64)
        X.append(img[..., None])                          # (256, 64, 1)

        text = df.loc[i, 'IDENTITY']
        label_idx = label_to_num(text)
        if len(label_idx) < max_str_len:
            pad = np.full((max_str_len - len(label_idx),), len(alphabets), dtype=np.int32)
            label_idx = np.concatenate([label_idx, pad])
        else:
            label_idx = label_idx[:max_str_len]
        Y.append(label_idx)

        T = img.shape[1] // downsample
        input_len.append(T)
        label_len.append(min(len(text), max_str_len))

    X = np.stack(X, axis=0)
    Y = np.stack(Y, axis=0)
    return (
        X,
        Y,
        np.array(input_len, dtype=np.int32)[:, None],
        np.array(label_len, dtype=np.int32)[:, None]
    )
