import os
import cv2
import numpy as np
import pandas as pd

# 문자 집합 및 파라미터
alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
max_str_len = 24                     # 최대 레이블 길이
num_of_characters = len(alphabets) + 1  # +1: CTC blank
num_of_timestamps = 64               # RNN 타임스텝 수

def load_csv(csv_path):
    """CSV 파일 로드 → NaN, 'UNREADABLE' 레코드 제거 → 대문자 변환"""
    df = pd.read_csv(csv_path)
    df.dropna(subset=['IDENTITY'], inplace=True)
    df = df[df['IDENTITY'] != 'UNREADABLE']
    df['IDENTITY'] = df['IDENTITY'].str.upper()
    df.reset_index(drop=True, inplace=True)
    return df

def preprocess(img, target_h=256, max_w=1024):
    h, w = img.shape
    new_w = int(w * (target_h / h))
    if new_w > max_w:
        new_w = max_w
    img = cv2.resize(img, (new_w, target_h))
    canvas = np.ones((target_h, max_w), dtype=np.uint8) * 255
    canvas[:, :new_w] = img
    return cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)

def label_to_num(label):
    """문자열 레이블 → 숫자 인덱스 배열"""
    return np.array([alphabets.find(ch) for ch in label], dtype=np.int32)

def num_to_label(nums):
    """숫자 인덱스 배열 → 문자열 레이블"""
    s = ""
    for idx in nums:
        if idx == -1:
            break
        s += alphabets[idx]
    return s

def load_data(df, img_dir, size, downsample=32):
    """
    이미지 전처리 → 모델 입력(X),
    레이블 숫자화(Y), 입력 길이(input_len), 레이블 길이(label_len) 반환
    """
    X, Y = [], []
    input_len, label_len = [], []

    # 기본 크기 파라미터
    target_h, max_w = 256, 1024

    for i in range(size):
        img = cv2.imread(
            os.path.join(img_dir, df.loc[i, 'FILENAME']),
            cv2.IMREAD_GRAYSCALE
        )
        img = preprocess(img, target_h, max_w) / 255.0
        X.append(img)

        text = df.loc[i, 'IDENTITY']
        num_label = label_to_num(text)
        Y.append(num_label)

        label_len.append(len(num_label))
        T = img.shape[1] // downsample
        input_len.append(T)

    # 배열 변환 및 레이블 패딩
    X = np.array(X).reshape(-1, target_h, max_w, 1)
    Y_padded = np.full((len(Y), max_str_len), fill_value=-1, dtype=np.int32)
    for idx, seq in enumerate(Y):
        length = len(seq)
        Y_padded[idx, :length] = seq

    return X, Y_padded, np.array(input_len)[:, None], np.array(label_len)[:, None]
