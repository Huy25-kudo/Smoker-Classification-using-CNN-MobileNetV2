import os
import sys

# Khắc phục lỗi ModuleNotFoundError: No module named 'src' khi chạy 'python src/train.py'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from src.model import create_model
from sklearn.utils import class_weight
import numpy as np
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# --- CONFIGURATION ---
DATA_DIR = r"D:\Dev\XLA\smoking_detection_ai\data\processed_dataset"  # Thẳng vào folder chứa smoking/not_smoking
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 100 
LEARNING_RATE = 1e-5 # Cực thấp cho fine-tuning để tránh làm hỏng weights imagenet 
MODEL_SAVE_PATH = os.path.join("logs", "mobilenetv2_smoker.keras")

def train_model():
    # 1. Loading data & Preparing DataFrame
    print(f"[INFO] Dang quet du lieu tu {DATA_DIR}...")
    
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Khong tim thay folder du lieu tai: {DATA_DIR}")
    
    # Tao image_df tu thu muc
    filepaths = []
    labels = []
    class_names = ['not_smoking', 'smoking']
    
    for cls in class_names:
        cls_path = os.path.join(DATA_DIR, cls)
        if os.path.exists(cls_path):
            for f in os.listdir(cls_path):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepaths.append(os.path.join(cls_path, f))
                    labels.append(cls)
    
    image_df = pd.DataFrame({'Filepath': filepaths, 'Label': labels})
    print(f"[INFO] Tong so anh tim thay: {len(image_df)}")

    # Chia Train/Test
    train_df, test_df = train_test_split(image_df, test_size=0.2, shuffle=True, random_state=42)

    # Khoi tao Generators
    train_generator = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        validation_split=0.2
    )

    test_generator = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    ) 

    # Tao data flow
    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=IMG_SIZE,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
        subset='training'
    )

    val_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=IMG_SIZE,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
        subset='validation'
    )

    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=IMG_SIZE,
        color_mode='rgb',
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # 2. Xử lý Class Weight
    counts = image_df['Label'].value_counts()
    print(f"[INFO] Class distribution:\n{counts}")
    
    total = len(image_df)
    n_classes = len(class_names)
    # Tinh weight dua tren class_indices cua generator de dam bao dung ID
    class_indices = train_images.class_indices
    class_weight_dict = {}
    for cls_name, idx in class_indices.items():
        class_weight_dict[idx] = total / (n_classes * counts[cls_name])
    
    print(f"[INFO] Class weights: {class_weight_dict}")

    # 3. Khởi tạo và Huấn luyện Model
    print(f"[INFO] Dang khoi tao mo hinh CNN + MobileNetV2 (Fine-tuning)...")
    model = create_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), learning_rate=LEARNING_RATE)

    # Create checkpoint callback
    checkpoint_path = os.path.join("logs", "smokers_classification_model_checkpoint.weights.h5")
    checkpoint_callback = ModelCheckpoint(
        checkpoint_path,
        save_weights_only=True,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
    
    # Setup EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # TensorBoard callback
    log_dir = os.path.join("logs", "training_logs", "smoker_classification_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir=log_dir)

    print(f"[INFO] Bat dau huan luyen ({EPOCHS} epochs)...")
    history = model.fit(
        train_images,
        steps_per_epoch=len(train_images),
        validation_data=val_images,
        validation_steps=len(val_images),
        epochs=EPOCHS,
        class_weight=class_weight_dict,
        callbacks=[
            early_stopping,
            tensorboard_callback,
            checkpoint_callback,
        ]
    )

    print(f"[DONE] Huan luyen hoan tat! Model luu tai: {MODEL_SAVE_PATH}")
    return history

if __name__ == "__main__":
    # Dam bao folder logs ton tai
    if not os.path.exists("logs"):
        os.makedirs("logs")
        
    train_model()
