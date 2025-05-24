import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Lambda, Dense, Bidirectional, LSTM, Activation,
    Concatenate, Permute, Reshape
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet121
from dataset.dataset_loader import load_csv, num_of_characters, max_str_len
from dataset.dataset_loader import DataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def build_crnn(input_shape):
    inp = Input(shape=input_shape, name='input')                 # (128, 512, 1)
    x = Concatenate(axis=-1)([inp, inp, inp])                    # (128, 512, 3)
    densenet = DenseNet121(
        include_top=False,
        weights='imagenet',
        input_tensor=x
    )
    x = densenet.output   # (batch, H', W', C)
    print(f"DenseNet output shape: {x.shape}")
     
    x = Permute((2, 1, 3))(x)                                   # (batch, W', H', C)
    static = x.shape
    x = Reshape((static[1], static[2] * static[3]))(x)          # (batch, W', H'*C)
    
    x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Dense(num_of_characters, kernel_initializer='he_normal')(x)
    y_pred = Activation('softmax', name='softmax')(x)
    return Model(inputs=inp, outputs=y_pred, name='DenseNet_CRNN')

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
    
    # 경로 설정
    train_csv = 'dataset/handwriting-recognition/written_name_train_v2.csv'
    valid_csv = 'dataset/handwriting-recognition/written_name_validation_v2.csv'
    train_dir = 'dataset/handwriting-recognition/train_v2/train'
    valid_dir = 'dataset/handwriting-recognition/validation_v2/validation'

    # 데이터 로드
    train_df = load_csv(train_csv).sample(n = 8000)
    valid_df = load_csv(valid_csv).sample(n = 1000)

    train_gen = DataGenerator(
        train_df,
        train_dir,
        batch_size=32,  # 메모리 절약을 위해 줄임
        target_h=64,
        max_w=256
    )

    valid_gen = DataGenerator(
        valid_df,
        valid_dir,
        batch_size=64,
        target_h=64,
        max_w=256
    )

    # 모델 준비 - 올바른 input_shape
    base_model = build_crnn(input_shape=(64, 256, 1))  # (128 ,512 , 1)
    train_model = build_train_model(base_model)
    
    train_model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss={'ctc': lambda y_true, y_pred: y_pred}
    )

    # 콜백 설정
    callbacks = [
        ModelCheckpoint('best_model.h5', save_best_only=True, verbose=1),
        ReduceLROnPlateau(patience=3, verbose=1)
    ]

    # 학습 - 한 번만 호출
    train_model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=20,
        steps_per_epoch=len(train_gen),
        validation_steps=len(valid_gen),
        callbacks=callbacks,
        verbose=1    )
    # 저장
    base_model.save('best_densenet_crnn.h5')
    
if __name__ == '__main__':
    train()
