import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os

from src.model import create_model

# --- THIẾT LẬP TRANG ---
st.set_page_config(
    page_title="AI Smoking Detection Pro",
    page_icon="🚭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button {
        width: 100%; border-radius: 12px; height: 3em;
        background: linear-gradient(45deg, #ff4b4b, #ff7e5f);
        color: white; border: none; font-weight: bold; transition: 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px); box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
    }
    .result-card {
        padding: 1.5rem; border-radius: 15px; border-left: 5px solid #ff4b4b;
        background: #1a1c24; margin-bottom: 2rem;
    }
    .metric-container {
        display: flex; justify-content: space-around; padding: 1rem;
        background: #262730; border-radius: 10px;
    }
    .stat-box { text-align: center; }
    .stat-val { font-size: 2rem; font-weight: bold; color: #ff4b4b; }
    .stat-label { font-size: 0.8rem; color: #a1a1a1; }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# TÍNH TOÁN GRAD-CAM (Gradient-weighted Class Activation Mapping)
# ============================================================
def get_gradcam_heatmap(img_array, model, last_conv_layer_name="out_relu"):
    """
    Hàm tính toán bản đồ nhiệt Grad-CAM cho mô hình mạng CNN (MobileNetV2).
    Nhận đầu vào là ảnh đã tiền xử lý, mô hình, và tên lớp Convolution cuối cùng.
    """
    try:
        # 1. Tìm base_model (MobileNetV2) linh hoạt (tránh lỗi do cấu trúc .name bị đổi thành mobilenetv2_1.00_224)
        base_model = None
        for layer in model.layers:
            if "mobilenet" in layer.name.lower():
                base_model = layer
                break
                
        if base_model is None:
            raise ValueError("Không tìm thấy layer chứa trúc MobileNetV2.")
            
        last_conv_layer = base_model.get_layer(last_conv_layer_name)
        
        # 2. Tạo mô hình trích xuất trung gian (Gradient Model)
        # Bắt buộc để vừa lấy Feature Map vừa lấy được dữ liệu truyền tiếp lên các lớp phân loại.
        base_grad_model = tf.keras.Model(
            base_model.input, 
            [last_conv_layer.output, base_model.output]
        )
        
        # 3. Mở GradientTape để theo dõi đạo hàm
        with tf.GradientTape() as tape:
            x = img_array
            
            # Cắt qua ảnh gốc vào Data Augmentation nếu có (luôn set training=False)
            if "data_augmentation" in [layer.name for layer in model.layers]:
                aug_layer = model.get_layer("data_augmentation")
                x = aug_layer(x, training=False)
            
            # Giải nén [Bản đồ đặc trưng, Dữ liệu vector max pooling]
            conv_outputs, pooled_features = base_grad_model(x, training=False)
            
            # Đánh dấu theo dõi đạo hàm cho Tensor Feature map (conv_outputs)
            tape.watch(conv_outputs)
            
            # 4. Tiếp tục tính Gradient Forward Pass bằng cách cấp Vector Pool chạy nốt vòng lặp mạng Dense Classifier
            x = pooled_features
            for layer_name in ["dense_1", "dropout_1", "dense_2", "dropout_2", "predictions"]:
                x = model.get_layer(layer_name)(x, training=False)
            predictions = x
            
            # Lấy vector điểm ra class đang thắng (có score cao nhất)
            top_pred_index = tf.argmax(predictions[0])
            top_class_channel = predictions[:, top_pred_index]

        # 5. Lấy đạo hàm (Gradients) ngược của lớp kết quả đối với các kênh hình ảnh của feature map convolution
        grads = tape.gradient(top_class_channel, conv_outputs)
        if grads is None:
            return None
            
        # Tính trọng đạo hàm trung bình toàn cầu
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # 6. Nhân ma trận - áp từng lớp trọng lượng gradient dồn vào kênh kích hoạt mảng features map tương ứng 
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap) 
        
        # Hàm kẹp - chuyển âm xuống 0 và đưa về phân phối max pool (0.0 đến 1.0)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        
        return heatmap.numpy()
        
    except Exception as e:
        print(f"[GRAD-CAM LỖI] {str(e)}")
        return None

def apply_thermal_overlay(img_bgr, heatmap):
    """
    Biến đổi Heatmap Array (0-1) thành bản đồ nhiệt có màu và phủ lên ảnh gốc.
    """
    # Thay đổi kích thước heatmap cho khớp với ảnh gốc (cv2 resize theo Width, Height)
    heatmap = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    
    # Chuyển đổi sang định dạng 8-bit (0-255)
    heatmap = np.uint8(255 * heatmap)
    
    # Phủ màu nhiệt JET (đỏ là vùng nhận diện mạnh, xanh là ít)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.GaussianBlur(heatmap_color, (7, 7), 0)
    
    # Kết hợp ảnh gốc và bản đồ nhiệt với tỉ lệ 60/40 (alpha=0.6, beta=0.4)
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_color, 0.4, 0)
    
    # Vẽ Contour tô viền đậm cho những vùng "rất mạnh" (giá trị điểm nhiệt > 150)
    _, thresh = cv2.threshold(heatmap, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
    
    return overlay

# --- GIAO DIỆN CHÍNH ---
def main():
    st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>🚭 AI SMOKING DETECTION</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Hệ thống nhận diện hành vi hút thuốc chuẩn CNN + MobileNetV2</p>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2855/2855519.png", width=100)
    st.sidebar.title("⚙️ Cấu hình AI")
    
    confidence_threshold = st.sidebar.slider(
        "🎚️ Độ nhạy (Confidence Threshold)", 
        min_value=0.1, max_value=0.99, value=0.5, step=0.05,
        help="Thay đổi ngưỡng dự đoán Hút thuốc (VD: 0.5 là chuẩn)."
    )
    
    # Check model path
    weights_path = os.path.join("logs", "smokers_classification_model_checkpoint.weights.h5")
    if not os.path.exists(weights_path):
        st.error(f"❌ Không tìm thấy trọng số mô hình tại {weights_path}. Vui lòng huấn luyện mô hình (chạy train.py) trước!")
        return

    @st.cache_resource
    def load_ai_model():
        model = create_model(input_shape=(224, 224, 3))
        # Load trọng số vào mô hình
        model.load_weights(weights_path)
        return model

    model = load_ai_model()

    tab1, tab3 = st.tabs(["🖼️ Nhận diện Ảnh tĩnh", "ℹ️ Giới thiệu Grad-CAM"])

    with tab1:
        uploaded_file = st.file_uploader("Tải lên ảnh cần phân tích...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file:
            col1, col2 = st.columns(2)
            
            # Read Image
            img = Image.open(uploaded_file).convert('RGB')
            img_resized = img.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            
            # Preprocessing ĐÚNG chuẩn MobileNetV2 để tránh dự đoán sai
            img_array_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array.copy())
            img_array_batch = tf.expand_dims(img_array_preprocessed, 0)
            
            with col1:
                # Dùng use_container_width vì use_column_width đã bị deprecate
                st.image(img, caption="Ảnh gốc đã tải lên", use_container_width=True)
                
            if st.button("🚀 PHÂN TÍCH VỚI AI + GRAD-CAM"):
                with st.spinner("Đang chạy thuật toán..."):
                    # Dự đoán bằng mô hình đã xử lý MobileNetV2 preprocessing
                    preds = model.predict(img_array_batch, verbose=0)
                    prob_smoking = float(preds[0][1])  # Index 1 là 'smoking'
                    prob_not_smoking = float(preds[0][0])
                    
                    label = "ĐANG HÚT THUỐC" if prob_smoking >= confidence_threshold else "KHÔNG HÚT THUỐC"
                    color = "#ff4b4b" if label == "ĐANG HÚT THUỐC" else "#28a745"
                    
                    heatmap = get_gradcam_heatmap(img_array_batch, model)
                    
                    with col2:
                        st.markdown(f"""
                            <div class="result-card" style="border-left-color: {color};">
                                <h2 style="color: {color}; margin-top:0;">{label}</h2>
                                <hr style="border: 0.5px solid #444;">
                                <div class="metric-container">
                                    <div class="stat-box">
                                        <div class="stat-val">{prob_smoking*100:.1f}%</div>
                                        <div class="stat-label">Xác suất Hút thuốc</div>
                                    </div>
                                    <div class="stat-box">
                                        <div class="stat-val">{prob_not_smoking*100:.1f}%</div>
                                        <div class="stat-label">Không hút</div>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        if heatmap is not None:
                            img_bgr = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
                            overlay = apply_thermal_overlay(img_bgr, heatmap)
                            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Bản đồ nhiệt Grad-CAM", use_container_width=True)
                        else:
                            st.info("Không thể tạo Grad-CAM cho mô hình này.")


if __name__ == "__main__":
    main()
