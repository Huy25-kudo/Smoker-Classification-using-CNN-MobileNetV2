import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- CẤU HÌNH ---
MODEL_PATH = os.path.join("logs", "mobilenetv2_smoker.h5") # Đường dẫn file model bạn đã train (.h5 cho tính tương thích)
IMAGE_PATH = "du_duong_dan_den_1_buc_anh_test_bat_ky.jpg"     # Sửa lại thành đường dẫn ảnh test của bạn
IMG_SIZE = (224, 224)
CLASS_NAMES = ['not_smoking', 'smoking']

def get_img_array(img_path, size):
    """Load và tiền xử lý ảnh giống hệt generator trong train.py"""
    img = load_img(img_path, target_size=size)
    array = img_to_array(img)
    # Thêm batch dimension: (1, 224, 224, 3)
    array = np.expand_dims(array, axis=0)
    # Tiền xử lý theo chuẩn MobileNetV2 (chuyển pixel về [-1, 1])
    return preprocess_input(array)

def make_gradcam_heatmap(img_array, model, pred_index=None):
    """
    Tạo bản đồ nhiệt Grad-CAM cho mô hình của bạn.
    Vì MobileNetV2 nằm gọn trong 1 layer tên là 'mobilenetv2_base', 
    chúng ta cần trích xuất đầu ra của layer đó.
    """
    # 1. Tạo một mô hình trích xuất đặc trưng và dự đoán
    # Input: Ảnh gốc -> Output: [Ma trận đặc trưng của MobileNetV2, Kết quả dự đoán cuối cùng]
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer('mobilenetv2_base').output, model.output]
    )

    # 2. Tính toán Gradient (Đạo hàm)
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        
        # Nếu không chỉ định class, lấy class có xác suất cao nhất
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
            
        # Lấy giá trị dự đoán của class mục tiêu
        class_channel = preds[:, pred_index]

    # Tính gradient của class mục tiêu đối với bản đồ đặc trưng (feature map)
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 3. Tính trọng số cho từng kênh đặc trưng
    # Pool các gradient (trung bình cộng) qua các trục không gian (chiều rộng, chiều cao)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Nhân từng kênh của feature map với trọng số "quan trọng" của nó
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Đưa heatmap về dải [0, 1] để vẽ
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), pred_index.numpy(), preds[0].numpy()

def save_and_display_gradcam(img_path, heatmap, pred_class, prob, cam_path="cam_output.jpg", alpha=0.6):
    """Đè bản đồ nhiệt (Heatmap) lên ảnh gốc và hiển thị"""
    # Đọc ảnh gốc
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)

    # Đổi kích thước heatmap cho bằng ảnh gốc
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Chuyển heatmap sang dải màu RGB
    heatmap = np.uint8(255 * heatmap)
    colormap = plt.get_cmap("jet") # Dùng dải màu Jet (Xanh -> Đỏ)
    colormap_colors = colormap(np.arange(256))[:, :3]
    heatmap_colored = colormap_colors[heatmap]
    heatmap_colored = np.uint8(255 * heatmap_colored)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR)

    # Trộn ảnh (Overlay)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)

    # Lưu ảnh ra file
    cv2.imwrite(cam_path, superimposed_img)
    print(f"[INFO] Đã lưu ảnh Grad-CAM tại: {cam_path}")

    # Vẽ lên màn hình
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Ảnh gốc")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Dự đoán: {CLASS_NAMES[pred_class]} ({prob[pred_class]*100:.2f}%)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1. Tải mô hình đã được huấn luyện
    print(f"[INFO] Đang tải mô hình từ: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # 2. Tiền xử lý ảnh test
    if os.path.exists(IMAGE_PATH):
        img_array = get_img_array(IMAGE_PATH, IMG_SIZE)
        
        # 3. Tạo Heatmap
        heatmap, pred_class, probabilities = make_gradcam_heatmap(img_array, model)
        
        # 4. Hiển thị kết quả
        print(f"[INFO] Phân tích hoàn tất. Đang vẽ kết quả...")
        save_and_display_gradcam(IMAGE_PATH, heatmap, pred_class, probabilities)
    else:
        print(f"[ERROR] Không tìm thấy ảnh tại: {IMAGE_PATH}")
