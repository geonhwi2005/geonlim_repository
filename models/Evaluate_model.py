# Evaluation.py
import numpy as np
import cv2
import editdistance
from tensorflow.keras.models import load_model
from dataset.dataset_loader import load_csv, DataGenerator, num_to_label
import tensorflow.keras.backend as K

def ctc_decode_predictions(y_pred, input_length):
    """CTC 디코딩을 통한 예측 결과 변환"""
    # Greedy 디코딩
    decoded = K.ctc_decode(y_pred, input_length, greedy=True)[0][0]
    return K.eval(decoded)

def calculate_accuracy(y_true_labels, y_pred_labels):
    """문자열 정확도 계산"""
    correct = 0
    total = len(y_true_labels)
    
    for true_label, pred_label in zip(y_true_labels, y_pred_labels):
        if true_label == pred_label:
            correct += 1
    
    return correct / total

def calculate_edit_distance(y_true_labels, y_pred_labels):
    """편집 거리 기반 유사도 계산"""
    total_distance = 0
    total_length = 0
    
    for true_label, pred_label in zip(y_true_labels, y_pred_labels):
        distance = editdistance.eval(true_label, pred_label)
        total_distance += distance
        total_length += max(len(true_label), len(pred_label))
    
    # 정규화된 편집 거리 (0에 가까울수록 좋음)
    normalized_distance = total_distance / total_length if total_length > 0 else 0
    similarity = 1 - normalized_distance
    
    return similarity

def evaluate_model(model_path, test_csv, test_dir):
    """모델 평가 메인 함수"""
    
    # 1. 모델 로드
    print("모델 로딩 중...")
    model = load_model(model_path)
    
    # 2. 테스트 데이터 로드
    print("테스트 데이터 로딩 중...")
    test_df = load_csv(test_csv)
    test_gen = DataGenerator(
        test_df,
        test_dir,
        batch_size=32,
        target_h=64,
        max_w=256
    )
    
    # 3. 예측 수행
    print("예측 수행 중...")
    y_true_labels = []
    y_pred_labels = []
    
    for i in range(len(test_gen)):
        batch_data, _ = test_gen[i]
        batch_images = batch_data[0]
        batch_true_labels = batch_data[1]
        batch_input_lengths = batch_data[2]
        
        # 모델 예측
        predictions = model.predict(batch_images)
        
        # CTC 디코딩
        decoded_preds = ctc_decode_predictions(predictions, batch_input_lengths)
        
        # 결과 변환
        for j, (true_label, pred_indices) in enumerate(zip(batch_true_labels, decoded_preds)):
            # True label 변환
            true_text = num_to_label(true_label[true_label >= 0])
            
            # Predicted label 변환
            pred_text = num_to_label(pred_indices[pred_indices >= 0])
            
            y_true_labels.append(true_text)
            y_pred_labels.append(pred_text)
            
            # 진행 상황 출력 (처음 10개만)
            if len(y_true_labels) <= 10:
                print(f"Sample {len(y_true_labels)}: True='{true_text}', Pred='{pred_text}'")
    
    # 4. 평가 지표 계산
    print("\n=== 평가 결과 ===")
    
    # 정확도
    accuracy = calculate_accuracy(y_true_labels, y_pred_labels)
    print(f"정확도 (Exact Match): {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 편집 거리 기반 유사도
    similarity = calculate_edit_distance(y_true_labels, y_pred_labels)
    print(f"편집 거리 유사도: {similarity:.4f} ({similarity*100:.2f}%)")
    
    # 5. 상세 분석
    print(f"\n총 테스트 샘플 수: {len(y_true_labels)}")
    
    # 길이별 정확도
    length_accuracies = {}
    for true_label, pred_label in zip(y_true_labels, y_pred_labels):
        length = len(true_label)
        if length not in length_accuracies:
            length_accuracies[length] = {'correct': 0, 'total': 0}
        
        length_accuracies[length]['total'] += 1
        if true_label == pred_label:
            length_accuracies[length]['correct'] += 1
    
    print("\n길이별 정확도:")
    for length in sorted(length_accuracies.keys()):
        stats = length_accuracies[length]
        acc = stats['correct'] / stats['total']
        print(f"  길이 {length}: {acc:.4f} ({stats['correct']}/{stats['total']})")

if __name__ == '__main__':
    # 평가 실행
    model_path = 'best_densenet_crnn.h5'  # 학습된 모델 경로
    test_csv = 'dataset/handwriting-recognition/written_name_test_v2.csv'
    test_dir = 'dataset/handwriting-recognition/test_v2/test'
    
    evaluate_model(model_path, test_csv, test_dir)
