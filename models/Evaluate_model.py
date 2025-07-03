# Evaluate_model.py

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataset.dataset_loader import DataLoader
from train import CTCLoss # Import the custom loss function for loading

# --- Configuration ---
MODEL_PATH = 'your model path'
CSV_PATH = 'your csv path'
BATCH_SIZE = 32
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 128

def decode_batch_predictions(pred, num_to_char):
    """Decodes raw model predictions into human-readable text."""
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use CTC decoder
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :num_to_char.vocabulary_size()
    ]
    # Convert numerical indices back to characters
    output_text = []
    for res in results:
        res_text = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res_text)
    return output_text

def evaluate():
    # 1. Prepare the data
    df = pd.read_csv(CSV_PATH)
    data_loader = DataLoader(df, BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT)
    _, _, test_ds = data_loader.split_data() # Use only the test dataset

    # 2. Load the trained model
    # Load with custom objects dictionary
    custom_objects = {"CTCLoss": CTCLoss}
    model = keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    print("✅ Model loaded successfully!")

    # 3. Evaluate the model on the test set
    print("📊 Evaluating on the test dataset...")
    results = model.evaluate(test_ds)
    print(f"Test Loss: {results}")

    # 4. Visualize predictions
    for batch in test_ds.take(1):
        batch_images = batch["image"]
        batch_labels = batch["label"]

        preds = model.predict(batch_images)
        pred_texts = decode_batch_predictions(preds, data_loader.num_to_char)

        orig_texts = []
        for label in batch_labels:
            label_text = tf.strings.reduce_join(data_loader.num_to_char(label)).numpy().decode("utf-8")
            orig_texts.append(label_text.replace('[UNK]', '').strip())

        _, axes = plt.subplots(4, 4, figsize=(15, 8))

        for i in range(min(16, len(pred_texts))):
            img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
            img = img.T
            
            title = f"Pred: {pred_texts[i]}\nTrue: {orig_texts[i]}"
            
            ax = axes[i // 4, i % 4]
            ax.imshow(img, cmap="gray")
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()
