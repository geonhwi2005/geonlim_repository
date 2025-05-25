import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from dataset.dataset_loader import (
    load_csv, preprocess, label_to_num, num_to_label, 
    alphabets, max_str_len, DataGenerator
)
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time

# CPU 최적화 설정
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

def decode_batch_predictions(y_pred, input_length):
    """
    CTC 디코딩을 통해 예측 결과를 문자열로 변환
    """
    decoded_texts = []
    
    for i in range(y_pred.shape[0]):
        # CTC 그리디 디코딩
        input_len = input_length[i][0]
        prediction = y_pred[i][:input_len]
        
        # 가장 높은 확률의 인덱스 선택
        decoded_indices = np.argmax(prediction, axis=1)
        
        # CTC blank 제거 및 연속된 같은 문자 제거
        decoded_text = ""
        prev_idx = -1
        
        for idx in decoded_indices:
            if idx != prev_idx and idx < len(alphabets):  # blank(마지막 인덱스) 제외
                decoded_text += alphabets[idx]
            prev_idx = idx
            
        decoded_texts.append(decoded_text)
    
    return decoded_texts

def calculate_edit_distance(str1, str2):
    """
    두 문자열 간의 편집 거리(Levenshtein distance) 계산
    """
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]

def evaluate_model(model_path, test_csv, test_dir, batch_size=4):
    """
    CPU 환경에 최적화된 모델 평가 함수
    """
    print(f"Loading model from {model_path}...")
    
    # 모델 로드 (CPU 환경)
    try:
        model = load_model(model_path, compile=False)
        print("Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # 테스트 데이터 로드
    print(f"Loading test data from {test_csv}...")
    test_df = load_csv(test_csv)
    
    # CPU 환경용 소규모 테스트셋
    if len(test_df) > 500:
        test_df = test_df.sample(n=500, random_state=42)
        print(f"Using subset of {len(test_df)} samples for CPU evaluation")
    
    # 테스트 데이터 제너레이터
    test_gen = DataGenerator(
        test_df,
        test_dir,
        batch_size=batch_size,
        target_h=64,
        max_w=256
    )
    
    print(f"Starting evaluation on {len(test_df)} samples...")
    
    # 평가 메트릭 초기화
    total_samples = 0
    exact_matches = 0
    total_edit_distance = 0
    character_accuracy = 0
    total_characters = 0
    
    predictions = []
    ground_truths = []
    inference_times = []
    
    # 배치별 평가
    for batch_idx in range(len(test_gen)):
        if batch_idx % 10 == 0:
            print(f"Processing batch {batch_idx+1}/{len(test_gen)}")
        
        # 배치 데이터 로드
        batch_data, _ = test_gen[batch_idx]
        images, labels, input_lengths, label_lengths = batch_data
        
        # CPU 추론 시간 측정
        start_time = time.time()
        with tf.device('/CPU:0'):
            y_pred = model.predict(images, verbose=0)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        # 예측 결과 디코딩
        decoded_preds = decode_batch_predictions(y_pred, input_lengths)
        
        # 실제 레이블 변환
        batch_ground_truths = []
        for i in range(len(labels)):
            label_len = label_lengths[i][0]
            true_indices = labels[i][:label_len]
            true_text = num_to_label(true_indices)
            batch_ground_truths.append(true_text)
        
        # 메트릭 계산
        for pred, truth in zip(decoded_preds, batch_ground_truths):
            predictions.append(pred)
            ground_truths.append(truth)
            total_samples += 1
            
            # 정확한 매치 확인
            if pred == truth:
                exact_matches += 1
            
            # 편집 거리 계산
            edit_dist = calculate_edit_distance(pred, truth)
            total_edit_distance += edit_dist
            
            # 문자 단위 정확도
            max_len = max(len(pred), len(truth))
            if max_len > 0:
                char_acc = 1 - (edit_dist / max_len)
                character_accuracy += char_acc
                total_characters += max_len
    
    # 최종 메트릭 계산
    exact_accuracy = exact_matches / total_samples
    avg_edit_distance = total_edit_distance / total_samples
    avg_character_accuracy = character_accuracy / total_samples
    avg_inference_time = np.mean(inference_times)
    
    # 결과 출력
    print("\n" + "="*50)
    print("EVALUATION RESULTS (CPU Environment)")
    print("="*50)
    print(f"Total samples evaluated: {total_samples}")
    print(f"Exact match accuracy: {exact_accuracy:.4f} ({exact_matches}/{total_samples})")
    print(f"Average edit distance: {avg_edit_distance:.4f}")
    print(f"Average character accuracy: {avg_character_accuracy:.4f}")
    print(f"Average inference time per batch: {avg_inference_time:.4f}s")
    print(f"Inference speed: {total_samples/sum(inference_times):.2f} samples/sec")
    
    # 예측 샘플 출력
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)
    for i in range(min(10, len(predictions))):
        print(f"Ground Truth: '{ground_truths[i]}'")
        print(f"Prediction:   '{predictions[i]}'")
        print(f"Match: {'✓' if predictions[i] == ground_truths[i] else '✗'}")
        print("-" * 30)
    
    return {
        'exact_accuracy': exact_accuracy,
        'avg_edit_distance': avg_edit_distance,
        'avg_character_accuracy': avg_character_accuracy,
        'avg_inference_time': avg_inference_time,
        'predictions': predictions,
        'ground_truths': ground_truths
    }

def evaluate_single_image(model_path, image_path):
    """
    단일 이미지에 대한 예측 함수
    """
    # 모델 로드
    model = load_model(model_path, compile=False)
    
    # 이미지 로드 및 전처리
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Cannot load image: {image_path}")
        return None
    
    # 전처리
    processed_img = preprocess(img, target_h=64, max_w=256)
    processed_img = processed_img.astype(np.float32) / 255.0
    processed_img = processed_img[..., np.newaxis]  # 채널 차원 추가
    processed_img = np.expand_dims(processed_img, axis=0)  # 배치 차원 추가
    
    # 예측
    with tf.device('/CPU:0'):
        y_pred = model.predict(processed_img, verbose=0)
    
    # 디코딩
    input_length = np.array([[processed_img.shape[2] // 16]])  # MobileNetV3 다운샘플링
    decoded_text = decode_batch_predictions(y_pred, input_length)[0]
    
    # 결과 시각화
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(processed_img[0, :, :, 0], cmap='gray')
    plt.title(f'Processed Image\nPrediction: "{decoded_text}"')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return decoded_text

if __name__ == '__main__':
    # CPU 환경용 평가 실행
    model_path = 'best_mobilenetv3_small_crnn_cpu.h5'  # 훈련된 모델 경로
    test_csv = 'dataset/handwriting-recognition/written_name_validation_v2.csv'
    test_dir = 'dataset/handwriting-recognition/validation_v2/validation'
    
    # 모델 평가
    if os.path.exists(model_path):
        results = evaluate_model(
            model_path=model_path,
            test_csv=test_csv,
            test_dir=test_dir,
            batch_size=4  # CPU 환경용 작은 배치
        )
        
        if results:
            print(f"\nFinal Results Summary:")
            print(f"Exact Accuracy: {results['exact_accuracy']:.4f}")
            print(f"Character Accuracy: {results['avg_character_accuracy']:.4f}")
    else:
        print(f"Model file not found: {model_path}")
        print("Please train the model first using CRNN_Model.py")
    
    # 단일 이미지 테스트 예제
    # single_image_path = 'test_image.png'
    # if os.path.exists(single_image_path):
    #     prediction = evaluate_single_image(model_path, single_image_path)
    #     print(f"Single image prediction: '{prediction}'")
