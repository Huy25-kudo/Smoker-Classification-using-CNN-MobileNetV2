
HƯỚNG DẪN CÀI ĐẶT & CHẠY DỰ ÁN: NHẬN DIỆN NGƯỜI HÚT THUỐC (SMOKER DETECTION)

I. MÔI TRƯỜNG ẢO (VENV) VÀ CÁC THƯ VIỆN CẦN THIẾT
--------------------------------------------------
Dự án được xây dựng và kiểm thử ổn định nhất trên môi trường Python 3.11 (được thiết lập qua virtual environment - venv).
Link dataset: https://www.kaggle.com/datasets/vitaminc/cigarette-smoker-detection
1. Tên các thư viện và phiên bản cụ thể:
   - tensorflow == 2.15.0 (Quan trọng: Đảm bảo tương thích với chuẩn Keras 2 để chạy code cũ)
   - Các công cụ khác: numpy, pandas, matplotlib, seaborn, opencv-python, Pillow, scikit-learn, scikit-image, albumentations, tqdm, jupyter, streamlit.

2. Trình tự cài đặt môi trường chuẩn:
   Bước 1: Mở Terminal/CMD tại thư mục dự án, tạo môi trường ảo:
     python -m venv venv
   Bước 2: Kích hoạt môi trường ảo:
     - Windows: venv\Scripts\activate
     - Mac/Linux: source venv/bin/activate
   Bước 3: Cài đặt toàn bộ thư viện cơ bản từ requirements:
     pip install -r requirements.txt
 
II. TRÌNH TỰ THỰC THI (LUỒNG CHẠY CODE)
--------------------------------------------------
Dự án được thực hiện tuần tự qua các bước sau. Bạn cần chạy lần lượt từ trên xuống:

1. Notebook 01: 01_EDA_and_DataPreprocessing.ipynb
   - Mục đích: Làm sạch, tiền xử lý và tăng cường (augment) dữ liệu.
   - Đầu vào: Ảnh trong thư mục 'dataset/not_smoking' và 'dataset/smoking'.
   - Đầu ra: Thư mục 'dataset/processed_dataset' chứa ảnh đã được cân bằng sáng, thêm viền (letterbox) và chuẩn hóa kích thước (224x224).

2. Notebook 02: 02_Model_Training_MobileNetV2.ipynb
   - Mục đích: Xây dựng cấu trúc mạng, huấn luyện (Train) và tinh chỉnh (Fine-tune).
   - Đầu vào: Thư mục 'dataset/processed_dataset'.
   - Đầu ra: File 'models/smoker_detector_best.keras' và 'models/smoker_detector_final.keras'. (Cùng file 'test_dataset.csv' để đánh giá).

3. Notebook 03: 03_Model_Evaluation_and_TestReport.ipynb
   - Mục đích: Kiểm thử mô hình trên dữ liệu chưa từng thấy (Test set).
   - Đầu ra: Báo cáo phân loại (Classification Report), ma trận nhầm lẫn (Confusion Matrix) và hình ảnh trực quan Grad-CAM.

4. Ứng dụng Web: app.py
   - Mục đích: Triển khai mô hình lên giao diện Web để dùng thử trực tiếp.
   - Cách chạy: Mở Terminal tại thư mục chứa file app.py và gõ lệnh: 
     streamlit run app.py


III. GIẢI THÍCH CÁC HÀM QUAN TRỌNG
--------------------------------------------------

1. Tại Notebook 01 (Tiền xử lý):
   - advanced_preprocess(): Chịu trách nhiệm Cân bằng sáng bằng thuật toán CLAHE (chống ảnh thiếu/chói sáng) và dùng Letterbox để giữ nguyên tỷ lệ ảnh thật (không bóp méo người) mà vẫn đạt chuẩn 224x224.
   - augment_data(): Sinh ra thêm ảnh để cân bằng dữ liệu giữa nhãn Smoking và Not Smoking.

2. Tại Notebook 02 (Kiến trúc & Huấn luyện):
   - cbam_block() (Convolutional Block Attention Module): Đây là "cặp mắt thần" của mạng. Nó giúp AI biết được "Cần tập trung vào VÙNG NÀO" (Spatial Attention) và "Cần tập trung vào KÊNH ĐẶC TRƯNG NÀO" (Channel Attention) để tìm ra điếu thuốc/khói.
   - build_model(): Dùng MobileNetV2 làm "xương sống" (Backbone) kết hợp với 2 nhánh:
     + Nhánh Shallow (Nông): Lấy đặc trưng từ block thứ 6 để nhìn ra các chi tiết nhỏ như điếu thuốc.
     + Nhánh Deep (Sâu): Lấy đặc trưng ở tầng cuối cùng để nhìn bao quát tư thế đứng, mảng khói lớn.
   - fit() - 2 Giai đoạn: Giai đoạn 1 đóng băng (freeze) xương sống để huấn luyện cái "đầu não" (Head). Giai đoạn 2 rã đông (unfreeze) xương sống với tốc độ học siêu nhỏ (1e-5) để tinh chỉnh tinh tế toàn bộ cơ thể.

3. Tại app.py & Notebook 03 (Dự đoán & Giải thích):
   - make_gradcam_heatmap(): Tính toán đạo hàm (Gradients) từ lớp đặc trưng cuối cùng ngược về bức ảnh. Nó vẽ ra vùng màu đỏ để chứng minh "Tại sao AI lại đoán đây là người hút thuốc".
   - get_superimposed_img(): Trộn bản đồ nhiệt màu (đỏ/vàng/xanh) đè lên bức ảnh gốc để mắt người dễ dàng quan sát.

IV. TÓM TẮT LUỒNG HOẠT ĐỘNG (WORKFLOW)
--------------------------------------------------
[Ảnh Thô] --> (CLAHE + Letterbox) --> [Ảnh Sạch 224x224] --> (Nạp vào Mạng) 
--> [MobileNetV2 trích xuất đặc trưng] --> [CBAM lọc nhiễu, tập trung vùng quan trọng]
--> [Phân loại Softmax] --> (Kết quả Hút thuốc/Không) --> [Grad-CAM quét ngược về ảnh] --> Hiện lên Web.

V. CẤU TRÚC THƯ MỤC (DIRECTORY TREE)
--------------------------------------------------
Smoker Detection Project/
├── dataset/
│   ├── smoking/                    # Dữ liệu ảnh người hút thuốc (gốc)
│   ├── not_smoking/                # Dữ liệu ảnh người không hút thuốc (gốc)
│   └── processed_dataset/          # Dữ liệu ảnh đã qua tiền xử lý (CLAHE, Letterbox, 224x224)                          
├── models/             
│   ├── smoker_detector_best.keras  # File mô hình có độ chính xác cao nhất (được chọn để inference)
│   └── smoker_detector_final.keras # File mô hình lưu lại sau epoch cuối cùng
├── notebooks/
│   ├── 01_EDA_and_DataPreprocessing.ipynb       # Phân tích dữ liệu & Tiền xử lý
│   ├── 02_Model_Training_MobileNetV2.ipynb      # Xây dựng & Huấn luyện kiến trúc mạng
│   └── 03_Model_Evaluation_and_TestReport.ipynb # Kiểm thử & Đánh giá kết quả trên tập Test
│                       
├── app.py                        
├── requirements.txt                # Danh sách thư viện và phiên bản dự án phụ thuộc
└── readme.txt                      # Tài liệu hướng dẫn sử dụng dự án này
