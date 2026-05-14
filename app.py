import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN STREAMLIT
# ==========================================
st.set_page_config(page_title="Smoker Detection & Explainability", page_icon="🚬", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stApp { max-width: 1200px; margin: 0 auto; }
    .prediction-box {
        padding: 30px; border-radius: 15px; text-align: center;
        margin-top: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .prediction-box:hover { transform: translateY(-5px); }
    .smoking { background-color: #ffebee; border: 2px solid #ef5350; color: #c62828; }
    .not-smoking { background-color: #e8f5e9; border: 2px solid #66bb6a; color: #2e7d32; }
    .title-text { text-align: center; background: -webkit-linear-gradient(#3498db, #2c3e50); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; margin-bottom: 0.5rem; }
    .subtitle-text { text-align: center; color: #7f8c8d; font-size: 1.2rem; margin-bottom: 2rem; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. TẢI MÔ HÌNH VÀ XỬ LÝ GRAD-CAM
# ==========================================
@st.cache_resource
def load_model():
    model_path = "models/smoker_detector_final.keras"
    if not os.path.exists(model_path):
        model_path = "models/smoker_detector_best.keras"
        
    if not os.path.exists(model_path):
        st.error(f"❌ Không tìm thấy mô hình tại {model_path}")
        return None
        
    try:
        model = tf.keras.models.load_model(
            model_path, 
            compile=False,
            safe_mode=False
        )
        return model
    except Exception as e: 
        st.error(f"❌ Lỗi khi tải mô hình: {e}")
        return None 

def make_gradcam_heatmap(img_array, model, branch='deep'):
    try:
        # Lấy đúng layer Pooling của nhánh được chỉ định (mặc định là deep)
        pool_layer_name = 'pool_smoke' if branch == 'deep' else 'pool_cigarette'
        pool_layer = model.get_layer(pool_layer_name)
        
        # Truy ngược lại lớp Multiply/Conv2D ngay trước lớp Pooling
        target_layer_name = pool_layer.input.name.split('/')[0].split(':')[0]
        target_layer = model.get_layer(target_layer_name)

        grad_model = tf.keras.models.Model(
            inputs=model.inputs, 
            outputs=[target_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array, training=False)
            if isinstance(predictions, list): predictions = predictions[-1]
            
            class_channel = predictions[:, 1]

        grads = tape.gradient(class_channel, conv_outputs)
        
        if isinstance(conv_outputs, list):
            conv_outputs = conv_outputs[0]
            grads = grads[0]

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0: 
            heatmap = heatmap / max_val
            
        return heatmap.numpy()
    except Exception as e:
        print(f"Lỗi Grad-CAM: {e}")
        return None

def get_superimposed_img(img_array_rgb, heatmap, alpha=0.4):
    try:
        img_bgr = cv2.cvtColor(img_array_rgb, cv2.COLOR_RGB2BGR)
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_colored = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_colored, alpha, 0)
        return cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    except:
        return img_array_rgb

# ==========================================
# 3. HÀM MAIN - XỬ LÝ LUỒNG GIAO DIỆN
# ==========================================
def main():
    st.markdown("<h1 class='title-text'>🚬 Trợ Lý AI: Phát Hiện Người Hút Thuốc</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Phân tích Toàn cảnh tích hợp Cơ chế Chú ý Kép (CBAM)</p>", unsafe_allow_html=True)

    with st.spinner("Đang tải mô hình AI..."):
        model = load_model()

    if model is not None:
        uploaded_file = st.file_uploader("📂 Kéo thả hoặc chọn ảnh từ thiết bị của bạn...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Thu gọn lại thành 3 cột chuẩn mực
            col1, col2, col3 = st.columns([1, 1, 1], gap="large")
            
            try:
                # 1. ĐỌC VÀ CHUẨN BỊ ẢNH
                image_pil = Image.open(uploaded_file).convert("RGB")
                img_array_original = np.array(image_pil)
                img_resized = cv2.resize(img_array_original, (224, 224))
                
                # Batch input (Chỉ truyền mảng 0-255 vì model đã có lớp Lambda)
                img_batch = img_resized.astype(np.float32)
                img_batch = np.expand_dims(img_batch, axis=0)
                
                with col1:
                    st.markdown("<h3 style='text-align: center;'>Ảnh đầu vào</h3>", unsafe_allow_html=True)
                    st.image(img_resized, use_container_width=True)
                
                # 2. DỰ ĐOÁN
                with col2:
                    st.markdown("<h3 style='text-align: center;'>Dự đoán</h3>", unsafe_allow_html=True)
                    with st.spinner('Đang phân tích...'):
                        preds = model.predict(img_batch, verbose=0)
                        if isinstance(preds, list): preds = preds[-1]
                        
                        p_not_smoking = float(preds[0][0])
                        p_smoking = float(preds[0][1])
                        
                        is_smoking_final = (p_smoking > p_not_smoking)
                        confidence = p_smoking * 100 if is_smoking_final else p_not_smoking * 100
                        
                        if is_smoking_final:
                            st.markdown(f"""
                            <div class="prediction-box smoking">
                                <h2 style="margin: 0; font-size: 2rem;">SMOKING</h2>
                                <p style="margin-top: 10px; font-size: 1.2rem; opacity: 0.9;">Độ tự tin: <b>{confidence:.2f}%</b></p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="prediction-box not-smoking">
                                <h2 style="margin: 0; font-size: 2rem;">NOT SMOKING</h2>
                                <p style="margin-top: 10px; font-size: 1.2rem; opacity: 0.9;">Độ tự tin: <b>{confidence:.2f}%</b></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                # 3. XỬ LÝ GRAD-CAM DUY NHẤT (MẶC ĐỊNH LÀ NHÁNH DEEP GIỐNG NOTEBOOK 3)
                with col3:
                    st.markdown("<h3 style='text-align: center;'>🔍 AI Focus (Grad-CAM)</h3>", unsafe_allow_html=True)
                    with st.spinner('Vẽ biểu đồ...'):
                        heatmap = make_gradcam_heatmap(img_batch, model, branch='deep')
                        if heatmap is not None:
                            gradcam_img = get_superimposed_img(img_resized, heatmap)
                            st.image(gradcam_img, use_container_width=True, caption="Vùng AI tập trung để ra quyết định")
                        else:
                            st.info("Không thể tạo biểu đồ nhiệt cho ảnh này.")
                            
            except Exception as e:
                st.error(f"Đã xảy ra lỗi hệ thống khi xử lý ảnh: {str(e)}")

if __name__ == "__main__":
    main()