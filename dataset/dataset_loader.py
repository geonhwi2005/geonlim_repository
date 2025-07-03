# dataset/dataset_loader.py

import os
import glob
import pandas as pd
<<<<<<< HEAD
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
=======
from tensorflow.keras.utils import Sequence

# --- 문자 집합 및 파라미터 ---
alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
max_str_len = 24 # 최대 시퀀스(레이블) 길이
num_of_characters = len(alphabets)+1 # +1 for CTC blank

# --- CSV 로드 및 전처리 함수 ---
def load_csv(csv_path):
    """
    CSV 파일 로드 → NaN/UNREADABLE 제거 → 대문자 변환 → 인덱스 리셋
    """
    df = pd.read_csv(csv_path)
    df.dropna(subset=['IDENTITY'], inplace=True)
    df = df[df['IDENTITY'] != 'UNREADABLE']
    df['IDENTITY'] = df['IDENTITY'].str.upper()
    df.reset_index(drop=True, inplace=True)
    return df

def preprocess(img, target_h=64, max_w=256):  # 기본값 수정
    """
    그레이스케일 이미지 리사이즈 → 패딩 캔버스 삽입 (회전 제거)
    Returns: np.ndarray shape=(target_h, max_w), dtype=uint8
    """
    h, w = img.shape
    new_w = int(w * (target_h / h))
    if new_w > max_w:
        new_w = max_w
    
    img = cv2.resize(img, (new_w, target_h))
    canvas = np.ones((target_h, max_w), dtype=np.uint8) * 255
    canvas[:, :new_w] = img
    
    # 90도 회전 제거 - 원본 shape 유지 (64, 256)
    return canvas

def label_to_num(label):
    """
    문자열 레이블 → 숫자 인덱스 배열
    """
    return np.array([alphabets.find(ch) for ch in label], dtype=np.int32)

def num_to_label(nums):
    """
    숫자 인덱스 배열 → 문자열 레이블
    (CTC 디코딩 후 사용하는 유틸)
    """
    s = ""
    for idx in nums:
        if idx < 0 or idx >= len(alphabets):
            break
        s += alphabets[idx]
    return s

# --- 배치 단위 데이터 제너레이터 ---
class DataGenerator(Sequence):
    def __init__(self, df, img_dir, batch_size, target_h=64, max_w=256):  # 기본값 수정
        self.df = df
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.target_h = target_h
        self.max_w = max_w
        self.indices = np.arange(len(df))

    def __len__(self):
        # 한 에폭당 배치 개수
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        # 배치에 해당하는 행 인덱스, DataFrame 슬라이스
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_idx]

        X, Y, input_len, label_len = [], [], [], []

        for _, row in batch_df.iterrows():
            # 1) 이미지 로드 + 전처리
            img = cv2.imread(
                os.path.join(self.img_dir, row['FILENAME']),
                cv2.IMREAD_GRAYSCALE)
            img = preprocess(img, self.target_h, self.max_w).astype(np.float32)
            img /= np.float32(255.0)
            
            # 채널 차원 추가(H, W) → (H, W, 1)
            X.append(img[..., np.newaxis])

            # 2) 레이블 숫자 인코딩
            nums = label_to_num(row['IDENTITY'])

            # 3) CTC용 길이 계산 - DenseNet121 실제 출력 길이
            # DenseNet121: 5번 다운샘플링 (2^5 = 32)
            # 입력 (64, 256) → DenseNet → (2, 8)
            T = self.max_w // 32  # 256 // 32 = 8 (2가 아님!)

            # 레이블이 너무 길면 잘라내기
            if len(nums) >= T:
                nums = nums[:T-1] # CTC blank를 위한 여유 공간

            Y.append(nums)
            input_len.append(T)
            label_len.append(len(nums))

        # NumPy array 변환 및 패딩
        X = np.array(X, dtype=np.float32) # (batch, H, W, 1)
        
        # Y를 (batch, max_str_len)로 패딩, 빈 공간은 -1
        Y_padded = np.full((len(Y), max_str_len), fill_value=0, dtype=np.int32)
        for i, seq in enumerate(Y):
            Y_padded[i, :len(seq)] = seq

        inputs = [
            X,
            Y_padded,
            np.array(input_len, dtype=np.int32)[:, None],
            np.array(label_len, dtype=np.int32)[:, None]
        ]

        # CTC loss용 dummy output
        outputs = np.zeros((len(X),), dtype=np.float32)
        return inputs, outputs
>>>>>>> 3dc3c155bd117d90c00fde6e54a01068df22ea2a
