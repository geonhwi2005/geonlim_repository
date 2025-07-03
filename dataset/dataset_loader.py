# dataset/dataset_loader.py

import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import StringLookup

# --- Configuration ---
IAM_DATA_PATH = 'your data location'
CSV_PATH = 'your csv location'
BATCH_SIZE = 32
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 128
MAX_LABEL_LEN = 0 # This will be updated after loading the data

def get_image_paths_from_df(df):
    """Scans for actual image file paths and maps them to the DataFrame."""
    base_path = os.path.join(IAM_DATA_PATH, 'data')
    all_image_paths = {os.path.basename(p): p for p in glob.glob(os.path.join(base_path, "*", "*.png"))}

    def map_path(img_id):
        img_name = f"{img_id}.png"
        return all_image_paths.get(img_name)

    df['image_path'] = df['image_id'].apply(map_path)
    return df.dropna(subset=['image_path'])

def academic_data_loader(data_dir, output_csv_path):
    """
    Parses the words.txt from the IAM dataset, preprocesses it, and saves it
    as a CSV file containing 'image_id' and 'text'.
    """
    words_path = os.path.join(data_dir, 'words.txt')

    lines = []
    with open(words_path, 'r') as f:
        for line in f:
            if not line.startswith("#"):
                lines.append(line.strip().split())

    # Create and filter the DataFrame
    df = pd.DataFrame(lines, columns=['id', 'status', 'graylevel', 'x', 'y', 'w', 'h', 'tag', 'text'])
    df = df[df['status'] == 'ok']

    # Extract image ID and text
    df['image_id'] = df['id'].apply(lambda x: '-'.join(x.split('-')[:2]))

    # Group text by line
    line_texts = df.groupby('image_id')['text'].apply(lambda x: ' '.join(x)).reset_index()
    line_texts.rename(columns={'text': 'text_line'}, inplace=True)
    
    # Map to actual file paths and filter invalid data
    line_texts = get_image_paths_from_df(line_texts)

    line_texts.to_csv(output_csv_path, index=False)
    print(f"🎉 Final IAM CSV created! Saved to: {output_csv_path}")
    return line_texts

class DataLoader:
    """Handles data loading, preprocessing, and splitting."""
    def __init__(self, df, batch_size, image_width, image_height):
        self.df = df
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height

        # Set up the character set and StringLookup layers
        self.characters = sorted(list(set(char for text in self.df['text'] for char in text)))
        global MAX_LABEL_LEN
        MAX_LABEL_LEN = max(map(len, self.df['text']))

        self.char_to_num = StringLookup(vocabulary=list(self.characters), mask_token=None)
        self.num_to_char = StringLookup(vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True)

    def preprocess_image(self, image_path):
        """Reads, resizes, and normalizes an image."""
        image = tf.io.read_file(image_path)
        image = tf.io.decode_png(image, 1)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [self.image_height, self.image_width])
        return image

    def vectorize_label(self, label):
        """Converts a text label into a numerical vector and applies padding."""
        label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        length = tf.shape(label)[0]
        pad_amount = MAX_LABEL_LEN - length
        label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=0)
        return label

    def process_data(self, image_path, label):
        """Preprocesses an image-label pair for model input."""
        image = self.preprocess_image(image_path)
        label = self.vectorize_label(label)
        return {"image": image, "label": label}

    def get_dataset(self, dataframe):
        """Creates a tf.data.Dataset object from a DataFrame."""
        dataset = tf.data.Dataset.from_tensor_slices(
            (dataframe["image_path"].values, dataframe["text"].values)
        )
        dataset = (
            dataset.map(self.process_data, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        return dataset

    def split_data(self, test_size=0.1, val_size=0.1):
        """Splits the data into training, validation, and test sets."""
        test_samples = self.df.sample(frac=test_size, random_state=42)
        train_val_samples = self.df.drop(test_samples.index)

        val_samples = train_val_samples.sample(frac=val_size / (1 - test_size), random_state=42)
        train_samples = train_val_samples.drop(val_samples.index)

        return self.get_dataset(train_samples), self.get_dataset(val_samples), self.get_dataset(test_samples)
