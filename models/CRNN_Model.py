<<<<<<< HEAD
# models/CRNN_Model.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
from tensorflow import keras
import pandas as pd

from dataset.dataset_loader import DataLoader, academic_data_loader
from models.CRNN_Model import build_academic_model

def build_academic_model(image_width, image_height, num_chars, max_label_len):
    """
    Builds a CRNN model based on MobileNetV3 and Bi-LSTM for handwriting recognition.
    """
    # Input layers
    input_img = layers.Input(shape=(image_height, image_width, 1), name="image", dtype="float32")
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # MobileNetV3 backbone
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(image_height, image_width, 1),
=======
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
from tensorflow.keras.applications import DenseNet121
from dataset.dataset_loader import load_csv, num_of_characters, max_str_len
from dataset.dataset_loader import DataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # TensorFlow 2.x 호환성 개선
    return tf.reduce_mean(
        tf.compat.v1.nn.ctc_loss(
            labels=labels,
            inputs=y_pred,
            sequence_length=tf.cast(tf.squeeze(input_length), tf.int32)
        )
    )

def build_crnn(input_shape):
    inp = Input(shape=input_shape, name='input') # (64, 256, 1)
    x = Concatenate(axis=-1)([inp, inp, inp]) # (64, 256, 3)
    
    densenet = DenseNet121(
>>>>>>> 3dc3c155bd117d90c00fde6e54a01068df22ea2a
        include_top=False,
        weights=None, # Training from scratch
    )
    
<<<<<<< HEAD
    # Use an intermediate output of the model as CNN features.
    # The 'expanded_conv_10/expand' layer is a good source of rich feature maps.
    cnn_features = base_model.get_layer('expanded_conv_10/expand').output
    
    # Reshape the CNN output to fit the RNN input (batch, time_steps, features)
    shape = tf.shape(cnn_features)
    x = layers.Reshape(target_shape=(shape[1] * shape[2], shape[3]))(cnn_features)
    
    # RNN layers (Bi-LSTM)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    
    # Output layer for CTC Loss
    output = layers.Dense(num_chars + 1, activation="softmax", name="dense2")(x)

    # Define the Keras model
    model = keras.Model(
        inputs=input_img,
        outputs=output,
        name="Academic_MobileNetV3_BiLSTM_Model"
=======
    x = densenet.output # (batch, H', W', C)
    print(f"DenseNet output shape: {x.shape}")
    
    x = Permute((2, 1, 3))(x) # (batch, W', H', C)
    static = x.shape
    x = Reshape((static[1], static[2] * static[3]))(x) # (batch, W', H'*C)
    
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
>>>>>>> 3dc3c155bd117d90c00fde6e54a01068df22ea2a
    )
    
    return model

# --- Configuration ---
DATA_DIR = '/content/drive/MyDrive/iam_data'
CSV_PATH = '/content/drive/MyDrive/iam_dataset_IMPROVED.csv'
MODEL_SAVE_PATH = '/content/drive/MyDrive/academic_complete_model_best_small.keras'

EPOCHS = 80
BATCH_SIZE = 32
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 128
LEARNING_RATE = 2e-3

# --- CTC Loss Function ---
def CTCLoss(y_true, y_pred):
    """Custom CTC Loss function for Keras."""
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

# --- Main Training Function ---
def train():
<<<<<<< HEAD
    # 1. Prepare the data
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        df = academic_data_loader(DATA_DIR, CSV_PATH)

    # Create a DataLoader instance and split the data
    data_loader = DataLoader(df, BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT)
    train_ds, val_ds, test_ds = data_loader.split_data()
    
    # 2. Build the model
    num_chars = len(data_loader.char_to_num.get_vocabulary())
    model = build_academic_model(IMAGE_WIDTH, IMAGE_HEIGHT, num_chars, data_loader.MAX_LABEL_LEN)
    
    # 3. Compile the model
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss=CTCLoss)
    
    print("--- Starting Training ---")
    model.summary()

    # 4. Set up callbacks
    # Save the model with the lowest validation loss
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH, monitor='val_loss', save_best_only=True, verbose=1
    )
    # Dynamically adjust the learning rate
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
    )
    # Stop training early if no improvement
    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
    )
    
    # 5. Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, reduce_lr_cb, early_stopping_cb]
    )
    
    print("--- Training Finished ---")
    
    # 6. Final evaluation on the test set
    print("--- Final Model Evaluation (Test Set) ---")
    results = model.evaluate(test_ds)
    print(f"Test Loss: {results}")

if __name__ == "__main__":
=======
    # 경로 설정 - Kaggle과 IAM 데이터셋 모두 사용
    # Kaggle 경로
    kaggle_train_csv = 'dataset/handwriting-recognition/written_name_train_v2.csv'
    kaggle_valid_csv = 'dataset/handwriting-recognition/written_name_validation_v2.csv'
    kaggle_train_dir = 'dataset/handwriting-recognition/train_v2/train'
    kaggle_valid_dir = 'dataset/handwriting-recognition/validation_v2/validation'
    
    # IAM 경로 (실제 경로에 맞게 수정 필요)
    iam_train_csv = 'dataset/iam-handwriting/train.csv'
    iam_valid_csv = 'dataset/iam-handwriting/validation.csv'
    iam_train_dir = 'dataset/iam-handwriting/train'
    iam_valid_dir = 'dataset/iam-handwriting/validation'
    
    # 데이터 로드 및 합치기
    print("Loading Kaggle dataset...")
    kaggle_train_df = load_csv(kaggle_train_csv).sample(n=35000)  # Kaggle에서 3.5만개
    kaggle_valid_df = load_csv(kaggle_valid_csv).sample(n=7000)   # Kaggle에서 7천개
    
    print("Loading IAM dataset...")
    try:
        iam_train_df = load_csv(iam_train_csv).sample(n=40000)    # IAM에서 4만개
        iam_valid_df = load_csv(iam_valid_csv).sample(n=8000)     # IAM에서 8천개
        
        # 두 데이터셋 합치기
        train_df = pd.concat([kaggle_train_df, iam_train_df], ignore_index=True)
        valid_df = pd.concat([kaggle_valid_df, iam_valid_df], ignore_index=True)
        
        print(f"Combined dataset - Train: {len(train_df)}, Valid: {len(valid_df)}")
        
    except FileNotFoundError:
        print("IAM dataset not found. Using only Kaggle dataset...")
        train_df = kaggle_train_df
        valid_df = kaggle_valid_df
        print(f"Kaggle only - Train: {len(train_df)}, Valid: {len(valid_df)}")
    
    # 데이터 제너레이터 생성 - RTX 3070 최적화
    train_gen = DataGenerator(
        train_df,
        kaggle_train_dir,  # 기본 디렉토리 (IAM 사용시 수정 필요)
        batch_size=64,     # RTX 3070 8GB에 최적화
        target_h=64,
        max_w=256
    )
    
    valid_gen = DataGenerator(
        valid_df,
        kaggle_valid_dir,  # 기본 디렉토리 (IAM 사용시 수정 필요)
        batch_size=128,    # validation은 더 큰 배치 가능
        target_h=64,
        max_w=256
    )
    
    # 모델 준비
    print("Building CRNN model...")
    base_model = build_crnn(input_shape=(64, 256, 1))
    train_model = build_train_model(base_model)
    
    # 컴파일 - RTX 3070에 최적화된 학습률
    train_model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss={'ctc': lambda y_true, y_pred: y_pred},
        jit_compile=False  # XLA 관련 에러 방지
    )

    # 콜백 설정
    callbacks = [
        ModelCheckpoint('best_densenet_crnn_rtx3070.h5', save_best_only=True, verbose=1),
        ReduceLROnPlateau(
            monitor='val_loss',
            patience=5,        # 더 긴 patience
            factor=0.5,        # 학습률 절반으로 감소
            min_lr=1e-6,       # 최소 학습률
            verbose=1
        )
    ]
    
    # 학습 - 3시간 기준 최적화
    print("Starting training...")
    print(f"Total training steps: {len(train_gen)} per epoch")
    print(f"Total validation steps: {len(valid_gen)} per epoch")
    
    train_model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=35,         # RTX 3070으로 더 많은 에포크 가능
        steps_per_epoch=len(train_gen),
        validation_steps=len(valid_gen),
        callbacks=callbacks,
        verbose=1
    )
    
    # 모델 저장
    print("Saving final model...")
    base_model.save('final_densenet_crnn_rtx3070.h5')
    print("Training completed!")

if __name__ == '__main__':
>>>>>>> 3dc3c155bd117d90c00fde6e54a01068df22ea2a
    train()
