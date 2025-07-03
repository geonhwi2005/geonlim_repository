CRNN handwriting recognition model
TensorFlow Model Quantization for Academic Research
📋 Project Overview
This project provides a practical implementation of 8-bit quantization-aware training on a Keras model using the TensorFlow Model Optimization Toolkit (tfmot). The primary goal is to optimize a pre-trained academic model (academic_complete_model_best_small.keras) for deployment on resource-constrained environments by reducing its size and accelerating inference, while aiming to maintain high accuracy.

This implementation is designed to be run in a Google Colab environment.

✨ Core Features
Model Optimization: Implements 8-bit quantization to reduce model size by up to 4x.

TensorFlow Integration: Utilizes the official tensorflow-model-optimization library for robust and standardized quantization.

Environment: Pre-configured for seamless execution on Google Colab, including Google Drive integration for model loading.

Focus Model: Specifically tailored for optimizing the Academic_MobileNetV3_BiLSTM_Attention model architecture.

📚 Tech Stack
Language: Python 3

Frameworks & Libraries:

TensorFlow (2.18.0)

TensorFlow Model Optimization (0.8.0)

NumPy, Pandas, OpenCV, Matplotlib

🚀 Getting Started
Prerequisites
A Google account to use Google Colab.

The pre-trained model file (academic_complete_model_best_small.keras) uploaded to your Google Drive.

Installation & Setup
Clone the Repository

bash
git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY_NAME>.git
cd <YOUR_REPOSITORY_NAME>
Open in Google Colab
Upload the coejeoghwa.ipynb notebook to your Google Colab environment.

Run Setup Cells
Execute the initial cells in the notebook to mount your Google Drive and install the required dependencies.

python
# Mount Google Drive to access the model file
from google.colab import drive
drive.mount('/content/drive')

# Install necessary libraries
!pip install tensorflow-model-optimization
Note: The notebook includes a command to restart the Colab runtime after installation, which is crucial for the new libraries to be recognized.

🔧 Usage
The core logic for model quantization is located in the coejeoghwa.ipynb notebook.

Load the Pre-trained Model
Update the path to your Keras model file stored in Google Drive.

python
import tensorflow as tf

# Ensure the path is correct
model_path = '/content/drive/MyDrive/academic_complete_model_best_small.keras'
model = tf.keras.models.load_model(model_path)
Apply Quantization
Use tfmot to apply quantization-aware wrappers to the model layers and re-compile it.

python
import tensorflow_model_optimization as tfmot

quantize_model = tfmot.quantization.keras.quantize_model
q_aware_model = quantize_model(model)

# Re-compile the model with the quantization layers
q_aware_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

q_aware_model.summary()
⚠️ Troubleshooting & Known Issues
Model Deserialization Error
When loading the Keras model, you might encounter a TypeError: Could not deserialize class 'Functional'. This typically occurs due to one of the following reasons:

Version Mismatch: The TensorFlow or Keras version used to save the model is different from the version in the current environment.

Custom Objects: If the model contains custom layers, functions, or objects, they must be registered with Keras during the loading process.

Complex Architecture: The model architecture, Academic_MobileNetV3_BiLSTM_Attention, is a complex functional model which can be sensitive to environment changes.

Solution:
Ensure your Colab environment uses a TensorFlow version compatible with the one used to create the .keras file. If custom objects are present, use the custom_objects argument in tf.keras.models.load_model().

📈 Future Work
Full Training Pipeline: Implement the full training loop for the quantization-aware model to fine-tune it and recover any accuracy lost during the initial conversion.

Performance Benchmarking: Add a comprehensive evaluation step to compare the original model and the quantized model on:

Model size (in MB)

Inference latency (in ms)

Accuracy and other relevant metrics

TFLite Conversion: Extend the project to convert the final quantized model to the TensorFlow Lite (.tflite) format for on-device deployment.

👤 Author
임건휘 (Geon-hwi Im)

Student, Department of Data Science, Hanyang University

GitHub: [geonhwi2005]

Email: [rjsgnl9316@gmail.com]
