import os
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# === CỐ ĐỊNH TÊN CLASS GIỐNG TRAIN.PY ===
CLASS_NAMES = ['not_smoking', 'smoking']

def evaluate_main():
    data_dir = os.path.abspath("data/processed/data")
    model_path = os.path.join("logs", "mobilenetv2_smoker.keras")
    
    if not os.path.exists(model_path):
        print(f"[LỖI] Chưa train xong hoặc không thấy file {model_path}.")
        print("[GỢI Ý] Hãy chạy: python src/train.py")
        return
    
    print(f"[INFO] Load model từ: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Tìm test dataset
    test_path = os.path.join(data_dir, "Testing")
    if os.path.exists(test_path):
        print(f"[INFO] Sử dụng Testing folder: {test_path}")
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_path,
            shuffle=False,
            batch_size=32,
            image_size=(224, 224),
            label_mode='categorical',
            class_names=CLASS_NAMES
        )
    else:
        print("[INFO] Không tìm thấy Testing folder, cắt 20% validation từ root.")
        test_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2, 
            subset="validation",
            seed=42,
            shuffle=False, 
            batch_size=32,
            image_size=(224, 224),
            label_mode='categorical',
            class_names=CLASS_NAMES
        )
        
    print(f"[INFO] Class names: {CLASS_NAMES}")
    
    print("\n[INFO] Đánh giá tổng quan (Evaluate on Test set)")
    loss, accuracy = model.evaluate(test_ds)
    print(f"-> Test Accuracy: {accuracy * 100:.2f}%")
    print(f"-> Test Loss: {loss:.4f}")
    
    print("\n[INFO] Trích xuất Confusion Matrix & Classification Report...")
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        preds_class = np.argmax(preds, axis=1)
        true_class = np.argmax(labels.numpy(), axis=1)
        y_true.extend(true_class)
        y_pred.extend(preds_class)
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Classification Report
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix - Smoker Classification')
    plt.ylabel('True Class (Thực tế)')
    plt.xlabel('Predicted Class (Dự đoán)')
    cm_path = os.path.join("logs", "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    print(f"\n-> Đã xuất ảnh Confusion Matrix ra: {cm_path}")

if __name__ == "__main__":
    evaluate_main()
