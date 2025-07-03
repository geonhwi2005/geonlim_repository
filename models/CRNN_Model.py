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
        include_top=False,
        weights=None, # Training from scratch
    )
    
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
    train()
