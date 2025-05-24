# Evaluation.py (수정 버전)
import numpy as np
import cv2
import editdistance
import tensorflow as tf
from tensorflow.keras.models import load_model
from dataset.dataset_loader import load_csv, DataGenerator, num_to_label
from models.CRNN_Model import ctc_lambda_func  # Custom loss import

def ctc_decode_predictions(y_pred, input_length):
    """TensorFlow 2.x 호환 CTC 디코딩"""
    # Tensor를 numpy로 변환
    if isinstance(input_length, tf.Tensor):
        input_length = input_length.numpy()
    
    # CTC 디코딩 (TensorFlow 2.x 방식)
    decoded_dense, _ = tf.nn.ctc_greedy_decoder(
        inputs=tf.transpose(y_pred, [1, 0, 2]),  # (time, batch, classes)
        sequence_length=input_length.flatten()
    )
    
    # SparseTensor를 dense로 변환
    decoded_dense = tf.sparse.to_dense(decoded_dense[0], default_value=-1)
    return decoded_dense.numpy()

def evaluate_model(model_path, test_csv, test_dir):
    """모델 평가 메인 함수 (수정 버전)"""
    
    # 1. 모델 로드 (Custom objects 포함)
    print("모델 로딩 중...")
    try:
        # Base model 로드 시도
        model = load_model(model_path)
    except:
        # Custom CTC loss가 포함된 경우
        model = load_model(model_path, custom_objects={'ctc_lambda_func': ctc_lambda_func})
    
    print(f"모델 입력 shape: {model.input_shape}")
    print(f"모델 출력 shape: {model.output_shape}")
    
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
        
        # 모델 예측 (base model만 사용)
        predictions = model.predict(batch_images, verbose=0)
        
        # CTC 디코딩 (수정된 방식)
        decoded_preds = ctc_decode_predictions(predictions, batch_input_lengths)
        
        # 결과 변환
        for j in range(len(batch_true_labels)):
            # True label 변환
            true_label = batch_true_labels[j]
            true_text = num_to_label(true_label[true_label >= 0])
            
            # Predicted label 변환
            if j < len(decoded_preds):
                pred_indices = decoded_preds[j]
                pred_text = num_to_label(pred_indices[pred_indices >= 0])
            else:
                pred_text = ""
            
            y_true_labels.append(true_text)
            y_pred_labels.append(pred_text)
            
            # 진행 상황 출력 (처음 10개만)
            if len(y_true_labels) <= 10:
                print(f"Sample {len(y_true_labels)}: True='{true_text}', Pred='{pred_text}'")
    
    # 4. 평가 지표 계산
    print("\n=== 평가 결과 ===")
    
    # 정확도
    correct = sum(1 for true, pred in zip(y_true_labels, y_pred_labels) if true == pred)
    accuracy = correct / len(y_true_labels)
    print(f"정확도 (Exact Match): {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 편집 거리 기반 유사도
    total_distance = 0
    total_length = 0
    for true_label, pred_label in zip(y_true_labels, y_pred_labels):
        distance = editdistance.eval(true_label, pred_label)
        total_distance += distance
        total_length += max(len(true_label), len(pred_label))
    
    similarity = 1 - (total_distance / total_length) if total_length > 0 else 0
    print(f"편집 거리 유사도: {similarity:.4f} ({similarity*100:.2f}%)")
    
    print(f"\n총 테스트 샘플 수: {len(y_true_labels)}")
    return accuracy, similarity

if __name__ == '__main__':
    # 평가 실행 - base model 경로로 수정
    model_path = 'final_densenet_crnn_rtx3070.h5'  # base model 사용
    test_csv = 'dataset/handwriting-recognition/written_name_test_v2.csv'
    test_dir = 'dataset/handwriting-recognition/test_v2/test'
    
    evaluate_model(model_path, test_csv, test_dir)
