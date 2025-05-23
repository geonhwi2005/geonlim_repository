import sys
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import (Input, Lambda, Dense,
                          Bidirectional, LSTM, Activation)
from keras.optimizers import Adam
from keras.applications import ResNet50
from dataset.dataset_loader import load_csv, load_data, num_of_characters, max_str_len
from keras.layers import Concatenate
import tensorflow.keras.backend as K
from keras.layers import Lambda
from keras.layers import Permute, Reshape
import tensorflow as tf

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def build_crnn(input_shape):
    # 1. Input 및 채널 복제 (1→3)
    inp = Input(shape=input_shape, name='input')
    x = Concatenate(axis=-1)([inp, inp, inp])

    # 2. ResNet50 백본 (include_top=False)
    resnet = ResNet50(include_top=False,
                      weights='imagenet',
                      input_tensor=x)
    x = resnet.output  # (batch, H', W', 2048)

    # 3. (batch,H',W',C) → (batch, W', H'*C)로 permute & reshape
    # permute
    x = Permute((2, 1, 3))(x)  
    # reshape: static shape 인수가 필요하다면 int_shape 사용
    static = K.int_shape(x)  # (None, W′, H′, C)
    x = Reshape((static[1], static[2] * static[3]))(x)
    # 4. 시퀀스 처리를 위한 FC + BiLSTM
    x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Bidirectional(LSTM(256, return_sequences=True))(x)

    # 5. 문자 클래스 예측
    x = Dense(num_of_characters, kernel_initializer='he_normal')(x)
    y_pred = Activation('softmax')(x)

    return Model(inputs=inp, outputs=y_pred)

def build_train_model(base_model):
    labels = Input(name='labels', shape=[max_str_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [base_model.output, labels, input_length, label_length])
    return Model(
        inputs=[base_model.input, labels, input_length, label_length],
        outputs=ctc_loss)

def train():
    # 경로 설정
    train_csv = 'dataset/handwriting-recognition/written_name_train_v2.csv'
    valid_csv = 'dataset/handwriting-recognition/written_name_validation_v2.csv'
    train_dir = 'dataset/handwriting-recognition/train_v2/train'
    valid_dir = 'dataset/handwriting-recognition/validation_v2/validation'

    # 데이터 로드
    train_df = load_csv(train_csv)
    valid_df = load_csv(valid_csv)
    train_x, train_y, train_input_len, train_label_len = load_data(train_df, train_dir, 30000)
    valid_x, valid_y, valid_input_len, valid_label_len = load_data(valid_df, valid_dir, 3000)

    # 모델 빌드 및 컴파일
    base_model = build_crnn(input_shape=(256, 64, 1))
    train_model = build_train_model(base_model)
    train_model.compile(
        optimizer=Adam(lr=1e-4),
        loss={'ctc': lambda y_true, y_pred: y_pred})

    # 학습
    train_model.fit(
        x=[train_x, train_y, train_input_len, train_label_len],
        y=np.zeros(30000),
        validation_data=(
            [valid_x, valid_y, valid_input_len, valid_label_len],
            np.zeros(3000)
        ),
        epochs=60, batch_size=128)

    # 베이스 모델 저장
    base_model.save('best_resnet_crnn.h5')

if __name__ == '__main__':
    train()
