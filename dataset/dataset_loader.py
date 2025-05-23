import os
import cv2
import numpy as np
import pandas as pd

# 문자 집합 및 파라미터
alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
max_str_len = 24            # 최대 레이블 길이
num_of_characters = len(alphabets) + 1  # +1: CTC blank
num_of_timestamps = 64      # RNN 타임스텝 수

def load_csv(csv_path):
    """CSV 파일 로드 → NaN, 'UNREADABLE' 레코드 제거 → 대문자 변환"""
    df = pd.read_csv(csv_path)
    df.dropna(subset=['IDENTITY'], inplace=True)
    df = df[df['IDENTITY'] != 'UNREADABLE']
    df['IDENTITY'] = df['IDENTITY'].str.upper()
    df.reset_index(drop=True, inplace=True)
    return df

def preprocess(img):
    """이미지 크롭/패딩 → (64,256) → 90도 회전"""
    h, w = img.shape
    canvas = np.ones((64, 256), dtype=np.uint8) * 255
    if w > 256:
        img = img[:, :256]
    if h > 64:
        img = img[:64, :]
    canvas[:img.shape[0], :img.shape[1]] = img
    return cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)

def label_to_num(label):
    """문자열 레이블 → 숫자 인덱스 배열"""
    return np.array([alphabets.find(ch) for ch in label], dtype=np.int32)

def num_to_label(nums):
    """숫자 인덱스 배열 → 문자열 레이블"""
    s = ""
    for idx in nums:
        if idx == -1:  # CTC blank
            break
        s += alphabets[idx]
    return s

def load_data(df, img_dir, size):
    """
    DataFrame, 이미지 디렉토리, 샘플 수 → 
    (X, Y, input_len, label_len) 반환
    """
    X, Y = [], []
    for i in range(size):
        fname = df.loc[i, 'FILENAME']
        img = cv2.imread(os.path.join(img_dir, fname), cv2.IMREAD_GRAYSCALE)
        img = preprocess(img) / 255.0
        X.append(img)
    X = np.array(X).reshape(-1, 256, 64, 1)

    Y = np.ones((size, max_str_len), dtype=np.int32) * -1
    input_len = np.ones((size, 1), dtype=np.int32) * (num_of_timestamps - 2)
    label_len = np.zeros((size, 1), dtype=np.int32)
    for i in range(size):
        text = df.loc[i, 'IDENTITY']
        label_len[i] = len(text)
        Y[i, :len(text)] = label_to_num(text)

    return X, Y, input_len, label_len

