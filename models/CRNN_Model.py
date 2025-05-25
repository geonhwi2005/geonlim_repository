import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Lambda, Dense, Bidirectional, LSTM, Activation,
    Concatenate, Permute, Reshape
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV3Small  # Large -> Small로 변경
from dataset.dataset_loader import load_csv, num_of_characters, max_str_len
from dataset.dataset_loader import DataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import gc


# CPU 최적화 설정
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    
    # TensorFlow 2.x 호환 CTC loss - dense padded labels 사용
    return tf.reduce_mean(
        tf.nn.ctc_loss(
            labels=labels,              # dense tensor 사용
            logits=y_pred,             # 로짓
            label_length=tf.cast(tf.squeeze(label_length), tf.int32),   # 레이블 길이
            logit_length=tf.cast(tf.squeeze(input_length), tf.int32),   # 입력 길이
            logits_time_major=False,    # 배치가 첫 번째 차원
            blank_index=-1              # 마지막 인덱스가 blank
        )
    )

def build_crnn(input_shape):
    inp = Input(shape=input_shape, name='input')  # (64, 256, 1)
    x = Concatenate(axis=-1)([inp, inp, inp])  # (64, 256, 3)
    
    # CPU 환경용 MobileNetV3Small 사용
    mobilenet = MobileNetV3Small(
        include_top=False,
        weights='imagenet',
        input_tensor=x,
        input_shape=(64, 256, 3),
        alpha=0.75,  # 모델 크기 더 축소 (1.0 -> 0.75)
        minimalistic=False,  # 단순화된 블록 사용
        include_preprocessing=False
    )
    
    x = mobilenet.output  # (batch, H', W', C)
    print(f"MobileNetV3Small output shape: {x.shape}")
    
    x = Permute((2, 1, 3))(x)  # (batch, W', H', C)
    static = x.shape
    x = Reshape((static[1], static[2] * static[3]))(x)  # (batch, W', H'*C)
    
    # CPU 환경용 축소된 네트워크
    x = Dense(32, activation='gelu', kernel_initializer='he_normal')(x)  # 64 -> 32
    x = Bidirectional(LSTM(128, return_sequences=True))(x)  # 256 -> 128
    x = Bidirectional(LSTM(128, return_sequences=True))(x)  # 256 -> 128
    x = Dense(num_of_characters, kernel_initializer='he_normal')(x)
    # y_pred = Activation('softmax', name='softmax')(x)
    
    return Model(inputs=inp, outputs= x, name='MobileNetV3Small_CRNN_CPU')

def build_train_model(base_model):
    labels = Input(name='labels', shape=[max_str_len], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')
    
    ctc_loss = Lambda(
        ctc_lambda_func,
        output_shape=(1,),
        name='ctc'
    )([base_model.output, labels, input_length, label_length])
    
    return Model(
        inputs=[base_model.input, labels, input_length, label_length],
        outputs=ctc_loss
    )

def train():
    # CPU 환경용 경로 설정 - Kaggle 데이터셋만 사용
    kaggle_train_csv = 'dataset/handwriting-recognition/written_name_train_v2.csv'
    kaggle_valid_csv = 'dataset/handwriting-recognition/written_name_validation_v2.csv'
    kaggle_train_dir = 'dataset/handwriting-recognition/train_v2/train'
    kaggle_valid_dir = 'dataset/handwriting-recognition/validation_v2/validation'
    
    # CPU 환경용 소규모 데이터셋
    print("Loading Kaggle dataset (CPU optimized)...")
    kaggle_train_df = load_csv(kaggle_train_csv).sample(n=5000)   
    kaggle_valid_df = load_csv(kaggle_valid_csv).sample(n=1000)   
    
    train_df = kaggle_train_df
    valid_df = kaggle_valid_df
    print(f"CPU optimized dataset - Train: {len(train_df)}, Valid: {len(valid_df)}")
    
    # CPU 환경용 소규모 배치
    train_gen = DataGenerator(
        train_df,
        kaggle_train_dir,
        batch_size=8,      
        target_h=64,
        max_w=256
    )
    
    valid_gen = DataGenerator(
        valid_df,
        kaggle_valid_dir,
        batch_size=16,     
        target_h=64,
        max_w=256
    )
    
    # 모델 준비
    print("Building MobileNetV3Small-CRNN model for CPU...")
    base_model = build_crnn(input_shape=(64, 256, 1))
    train_model = build_train_model(base_model)
    
    # CPU 환경용 컴파일
    train_model.compile(
        optimizer=Adam(learning_rate=5e-4),  # 1e-3 -> 5e-4 (더 보수적)
        loss={'ctc': lambda y_true, y_pred: y_pred},
        jit_compile=False
    )
    
    # CPU 환경용 콜백
    callbacks = [
        ModelCheckpoint('best_mobilenetv3_small_crnn_cpu.h5', save_best_only=True, verbose=1),
        ReduceLROnPlateau(
            monitor='val_loss',
            patience=3,      
            factor=0.7,      
            min_lr=1e-7,
            verbose=1
        ),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    ]
    
    # CPU 환경용 학습 설정
    print("Starting CPU training...")
    print(f"Total training steps: {len(train_gen)} per epoch")
    print(f"Total validation steps: {len(valid_gen)} per epoch")
    print("Warning: CPU training will be significantly slower!")
    
    train_model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=30,
        steps_per_epoch=len(train_gen),
        validation_steps=len(valid_gen),
        callbacks=callbacks,
        verbose=1,
        workers=1,
        use_multiprocessing=False,
        max_queue_size=2  # 메모리 사용량 제한
    )
    del train_model
    gc.collect()
    
    # 모델 저장
    print("Saving final CPU model...")
    # base_model.save('final_mobilenetv3_small_crnn_cpu.h5')
    print("CPU training completed!")

if __name__ == '__main__':
    train()
