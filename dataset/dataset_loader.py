# dataset/dataset_loader.py

import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence

# --- 문자 집합 및 파라미터 ---
alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
max_str_len = 24 # 최대 시퀀스(레이블) 길이
num_of_characters = len(alphabets)+1 # +1 for CTC blank

# --- CSV 로드 및 전처리 함수 ---
def load_csv(csv_path):
    """
    CSV 파일 로드 → NaN/UNREADABLE 제거 → 대문자 변환 → 인덱스 리셋
    """
    df = pd.read_csv(csv_path)
    df.dropna(subset=['IDENTITY'], inplace=True)
    df = df[df['IDENTITY'] != 'UNREADABLE']
    df['IDENTITY'] = df['IDENTITY'].str.upper()
    df.reset_index(drop=True, inplace=True)
    return df

def preprocess(img, target_h=64, max_w=256):  # 기본값 수정
    """
    그레이스케일 이미지 리사이즈 → 패딩 캔버스 삽입 (회전 제거)
    Returns: np.ndarray shape=(target_h, max_w), dtype=uint8
    """
    h, w = img.shape
    new_w = int(w * (target_h / h))
    if new_w > max_w:
        new_w = max_w
    
    img = cv2.resize(img, (new_w, target_h))
    canvas = np.ones((target_h, max_w), dtype=np.uint8) * 255
    canvas[:, :new_w] = img
    
    # 90도 회전 제거 - 원본 shape 유지 (64, 256)
    return canvas

def label_to_num(label):
    """
    문자열 레이블 → 숫자 인덱스 배열
    """
    return np.array([alphabets.find(ch) for ch in label], dtype=np.int32)

def num_to_label(nums):
    """
    숫자 인덱스 배열 → 문자열 레이블
    (CTC 디코딩 후 사용하는 유틸)
    """
    s = ""
    for idx in nums:
        if idx < 0 or idx >= len(alphabets):
            break
        s += alphabets[idx]
    return s

# --- 배치 단위 데이터 제너레이터 ---
class DataGenerator(Sequence):
    def __init__(self, df, img_dir, batch_size, target_h=64, max_w=256):  # 기본값 수정
        self.df = df
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.target_h = target_h
        self.max_w = max_w
        self.indices = np.arange(len(df))

    def __len__(self):
        # 한 에폭당 배치 개수
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        # 배치에 해당하는 행 인덱스, DataFrame 슬라이스
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_idx]

        X, Y, input_len, label_len = [], [], [], []

        for _, row in batch_df.iterrows():
            # 1) 이미지 로드 + 전처리
            img = cv2.imread(
                os.path.join(self.img_dir, row['FILENAME']),
                cv2.IMREAD_GRAYSCALE)
            img = preprocess(img, self.target_h, self.max_w).astype(np.float32)
            img /= np.float32(255.0)
            
            # 채널 차원 추가(H, W) → (H, W, 1)
            X.append(img[..., np.newaxis])

            # 2) 레이블 숫자 인코딩
            nums = label_to_num(row['IDENTITY'])

            # 3) CTC용 길이 계산 - DenseNet121 실제 출력 길이
            # DenseNet121: 5번 다운샘플링 (2^5 = 32)
            # 입력 (64, 256) → DenseNet → (2, 8)
            T = self.max_w // 32  # 256 // 32 = 8 (2가 아님!)

            # 레이블이 너무 길면 잘라내기
            if len(nums) >= T:
                nums = nums[:T-1] # CTC blank를 위한 여유 공간

            Y.append(nums)
            input_len.append(T)
            label_len.append(len(nums))

        # NumPy array 변환 및 패딩
        X = np.array(X, dtype=np.float32) # (batch, H, W, 1)
        
        # Y를 (batch, max_str_len)로 패딩, 빈 공간은 -1
        Y_padded = np.full((len(Y), max_str_len), fill_value=0, dtype=np.int32)
        for i, seq in enumerate(Y):
            Y_padded[i, :len(seq)] = seq

        inputs = [
            X,
            Y_padded,
            np.array(input_len, dtype=np.int32)[:, None],
            np.array(label_len, dtype=np.int32)[:, None]
        ]

        # CTC loss용 dummy output
        outputs = np.zeros((len(X),), dtype=np.float32)
        return inputs, outputs
