# Import Data Science Libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import itertools
import random

# Import visualization libraries
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

# Tensorflow Libraries
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Model

# System libraries
from pathlib import Path
import os.path

# Metrics
from sklearn.metrics import classification_report, confusion_matrix

def create_model(input_shape=(224, 224, 3), num_classes=2, learning_rate=5e-5):
    """
    CNN + MobileNetV2: Fine-Tuned for smoking detection.
    Updated with stronger augmentation to handle night/bright scenes.
    """
    
    # === TĂNG CƯỜNG DATA AUGMENTATION (Đã cập nhật theo yêu cầu) ===
    data_augmentation = tf.keras.Sequential([
        layers.Resizing(input_shape[0], input_shape[1]),
        # layers.Rescaling(1./255), # Loại bỏ vì train.py đã dùng preprocess_input
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1), 
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")
    
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    
    # 1. Augmentation (Auto-disabled during inference)
    x = data_augmentation(inputs)
    
    # 2. Rescaling chuyển về [-1, 1] — (Đã tích hợp trong generator ở train.py)
    # x = Rescaling(scale=1./127.5, offset=-1, name="rescaling")(x)
    
    # 3. Base model MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=input_shape,
        pooling='max' # Sử dụng Max Pooling như đoạn code
    )
    base_model._name = "mobilenetv2_base"
    
    base_model.trainable = False # Đóng băng hoàn toàn theo yêu cầu
        
    x = base_model(x, training=False) # Keep BatchNorm in inference mode
    
    # 5. Top Layers theo cấu trúc bạn yêu cầu
    x = Dense(256, activation='relu', name="dense_1")(x)
    x = Dropout(0.45, name="dropout_1")(x)
    x = Dense(256, activation='relu', name="dense_2")(x)
    x = Dropout(0.45, name="dropout_2")(x)
    
    # 6. Predictions (Softmax)
    outputs = Dense(num_classes, activation='softmax', name="predictions")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Smoker_Improved")
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    model = create_model()
    model.summary()
