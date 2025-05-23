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

from dataset.dataset_loader import load_csv, load_data, num_of_characters, max_str_len

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def build_crnn(input_shape):
    inp = Input(shape=input_shape, name='input')                 # (H, W, 1)
    x = Concatenate(axis=-1)([inp, inp, inp])                    # (H, W, 3)
    densenet = DenseNet121(
        include_top=False,
        weights='imagenet',
        input_tensor=x
    )
    x = densenet.output                                         # (batch, H', W', C)
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
    train_df = load_csv(train_csv)
    valid_df = load_csv(valid_csv)
    train_x, train_y, train_input_len, train_label_len = load_data(
        train_df, train_dir, size=30000
    )
    valid_x, valid_y, valid_input_len, valid_label_len = load_data(
        valid_df, valid_dir, size=3000
    )

    # 모델 준비
    base_model = build_crnn(input_shape=(256, 64, 1))
    train_model = build_train_model(base_model)
    train_model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss={'ctc': lambda y_true, y_pred: y_pred}
    )

    # 학습
    train_model.fit(
        x=[train_x, train_y, train_input_len, train_label_len],
        y=np.zeros(len(train_x)),
        validation_data=(
            [valid_x, valid_y, valid_input_len, valid_label_len],
            np.zeros(len(valid_x))
        ),
        epochs=2,
        batch_size=128
    )

    # 저장
    base_model.save('best_densenet_crnn.h5')

if __name__ == '__main__':
    train()
