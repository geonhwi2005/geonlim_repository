# dataset/dataset_loader.py

import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence

# --- 문자 집합 및 파라미터 ---
alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
max_str_len = 24
num_of_characters = len(alphabets)+1

# --- CSV 로드 및 전처리 함수 ---
def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.dropna(subset=['IDENTITY'], inplace=True)
    df = df[df['IDENTITY'] != 'UNREADABLE']
    df['IDENTITY'] = df['IDENTITY'].str.upper()
    df.reset_index(drop=True, inplace=True)
    return df

def preprocess(img, target_h=64, max_w=256):
    h, w = img.shape
    new_w = int(w * (target_h / h))
    if new_w > max_w:
        new_w = max_w
    img = cv2.resize(img, (new_w, target_h))
    canvas = np.ones((target_h, max_w), dtype=np.uint8) * 255
    canvas[:, :new_w] = img
    return canvas

def label_to_num(label):
    return np.array([alphabets.find(ch) for ch in label], dtype=np.int32)

def num_to_label(nums):
    s = ""
    for idx in nums:
        if idx < 0 or idx >= len(alphabets):
            break
        s += alphabets[idx]
    return s

# --- CPU 최적화 데이터 제너레이터 ---
class DataGenerator(Sequence):
    def __init__(self, df, img_dir, batch_size, target_h=64, max_w=256):
        self.df = df
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.target_h = target_h
        self.max_w = max_w
        self.indices = np.arange(len(df))

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_idx]

        X, Y, input_len, label_len = [], [], [], []

        for _, row in batch_df.iterrows():
            # 이미지 로드 + 전처리
            img_path = os.path.join(self.img_dir, row['FILENAME'])
            if not os.path.exists(img_path):
                continue
                
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            img = preprocess(img, self.target_h, self.max_w).astype(np.float32)
            img /= np.float32(255.0)
            X.append(img[..., np.newaxis])

            # 레이블 인코딩
            nums = label_to_num(row['IDENTITY'])

            # MobileNetV3Small의 다운샘플링 비율 (일반적으로 16배)
            T = self.max_w // 16  # 256 // 16 = 16

            # 레이블이 너무 길면 잘라내기
            if len(nums) >= T:
                nums = nums[:T-1]

            Y.append(nums)
            input_len.append(T)
            label_len.append(len(nums))

        # 배치가 비어있으면 더미 데이터 반환
        if not X:
            X = [np.zeros((self.target_h, self.max_w, 1), dtype=np.float32)]
            Y = [np.array([0], dtype=np.int32)]
            input_len = [16]
            label_len = [1]

        X = np.array(X, dtype=np.float32)

        # Y 패딩
        Y_padded = np.full((len(Y), max_str_len), fill_value=0, dtype=np.int32)
        for i, seq in enumerate(Y):
            Y_padded[i, :len(seq)] = seq

        inputs = [
            X,
            Y_padded,
            np.array(input_len, dtype=np.int32)[:, None],
            np.array(label_len, dtype=np.int32)[:, None]
        ]

        outputs = np.zeros((len(X),), dtype=np.float32)

        return inputs, outputs
