import os
import numpy as np
from keras import backend as K
from keras.models import load_model
from dataset.dataset_loader import (
    load_csv, load_data,
    num_of_characters, max_str_len, num_to_label, alphabets
)
from models.CRNN_Model import build_crnn

def levenshtein(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,
                           dp[i][j-1] + 1,
                           dp[i-1][j-1] + cost)
    return dp[m][n]

def decode_predictions(model, X):
    preds = model.predict(X)  # shape: (batch, timesteps, num_chars)
    input_len = np.ones(preds.shape[0]) * preds.shape[1]
    # greedy CTC 디코딩
    decoded, _ = K.ctc_decode(preds, input_length=input_len, greedy=True)
    decoded = K.get_value(decoded[0])
    texts = []
    for seq in decoded:
        # -1 인덱스(CTC blank) 및 반복 제거
        chars = [alphabets[idx] for idx in seq if idx >= 0]
        texts.append(''.join(chars))
    return texts

def evaluate(model, df, img_dir, size):
    # 데이터 로드
    X, Y, _, _ = load_data(df, img_dir, size)
    # 실제 문자열 목록
    gt_texts = [num_to_label(seq) for seq in Y]
    # 예측
    pred_texts = decode_predictions(model, X)
    # 지표 계산
    seq_acc = np.mean([g == p for g, p in zip(gt_texts, pred_texts)])
    cer_scores = [levenshtein(g, p)/max(len(g), 1)
                  for g, p in zip(gt_texts, pred_texts)]
    print(f"Sequence Accuracy: {seq_acc:.4f}")
    print(f"Mean CER: {np.mean(cer_scores):.4f}")
    # 샘플 출력
    for i in range(min(10, size)):
        print(f"GT: {gt_texts[i]}  |  Pred: {pred_texts[i]}")

if __name__ == "__main__":
    # 설정
    ROOT = os.path.dirname(__file__)
    # 검증 CSV 및 디렉터리 경로
    valid_csv = os.path.join(
        ROOT, "dataset/handwriting-recognition/written_name_validation_v2.csv"
    )
    valid_dir = os.path.join(
        ROOT, "dataset/handwriting-recognition/validation_v2/validation"
    )
    # 모델 빌드 및 가중치 로드
    base_model = build_crnn(input_shape=(256, 64, 1))
    base_model.load_weights("best_resnet_crnn.h5")
    # 평가 실행
    df_valid = load_csv(valid_csv)
    evaluate(base_model, df_valid, valid_dir, size=len(df_valid))
