# 🚬 Hệ thống Nhận diện Hút thuốc sử dụng AI (CNN + MobileNetV2)

Dự án phát hiện người hút thuốc theo thời gian thực (Camera) và thông qua ảnh tĩnh (Upload) dựa trên kiến trúc Transfer Learning **MobileNetV2**. Dự án đã loại bỏ sự phụ thuộc vào API của Kaggle, mang đến luồng đi từ Data Prep, Training, Evaluate cho đến Web Dashboard dễ hiểu.

## Công nghệ sử dụng
- **Mô hình Dữ liệu**: `TensorFlow` & `Keras` (MobileNetV2).
- **Computer Vision**: `OpenCV-Python` (Xử lý ảnh thực tế từ Camera).
- **Giao diện Web**: `Streamlit` (Hiển thị kết quả tương tác).
- **Đồ thị, đánh giá hệ số**: `Matplotlib`, `Seaborn`.

---

## 🚀 Hướng Dẫn Cài Đặt và Chạy Dự án

Nếu bạn clone hoặc mở dự án lần đầu tiên, hãy làm theo đúng từng bước sau:

### Bước 1: Khởi chạy và cài đặt môi trường
> **Lưu ý quan trọng**: Rất nhiều máy tính bị lỗi xung đột nhiều phiên bản Python (ví dụ: máy tính có cài MSYS2, Anaconda, Windows Store). Để khắc phục triệt để lỗi "Không nhận diện được thư viện", dự án đã cung cấp file chạy tự động chuẩn hóa môi trường.

1. Hãy click đúp chuột vào file **`setup_and_run.bat`** nằm trong thư mục dự án.
2. File này sẽ tự động tạo Môi trường ảo (Virtual Environment), tự động cài đặt tất cả thư viện (Tensorflow, Streamlit, OpenCV...) và sau cùng là tự động mở Web App lên cho bạn.

*(Nếu bạn muốn tự gõ lệnh thủ công thay vì dùng file .bat, hãy đảm bảo bạn dùng môi trường ảo: `python -m venv .venv`, sau đó kích hoạt `.venv\Scripts\activate` rồi mới chạy `pip install -r requirements.txt`)*

### Bước 2: Chuẩn bị Dữ liệu (Tải rảnh tay qua Kaggle)
1. **[Bắt buộc]** Truy cập vào Kaggle: [Cigarette Smoker Detection Dataset](https://www.kaggle.com/datasets/vitaminc/cigarette-smoker-detection). Mở link và đăng nhập tài khoản của bạn.
2. Nhấn nút "Download" nguyên tập dataset đó (kết quả tải về là 1 file `.zip`).
3. Đổi tên file (nếu dài) thành `archive.zip` (hoặc để nguyên) rồi chép vào thư mục `data/raw/` của thư mục dự án này.
4. Chạy script xử lý giải nén tự động:
```bash
python scripts/prepare_data.py
```
> Lúc này data chuẩn sẽ xuất hiện trong `data/processed/`. Cấu trúc ảnh `Smoker / Non-smoker` sẽ tự xuất hiện.

### Bước 3: Huấn luyện Mô hình
Nếu chưa có sẵn model `.keras`, hãy kích hoạt quá trình Huấn luyện bằng Convolutional Neural Network (CNN):
```bash
python src/train.py
```
Quá trình này tùy thuộc vào GPU của máy, có thể mất một vài phút. Sau khi kết thúc, file `mobilenetv2_smoker.keras` sẽ được lưu ở trong thư mục `logs/` kèm biểu đồ Accuracy/Loss.

### Bước 4: Kiểm tra bằng Code Thuần (Tùy chọn)
Để in Confusion Matrix và đo Metric Report nhanh chóng:
```bash
python src/evaluate.py
```

### Bước 5: Chạy Ứng dụng Web / Camera (Inference)
Đây là WebApp chính mà hệ thống xây dựng. Chạy lệnh:
```bash
streamlit run app.py
```
- Ứng dụng sẽ tự động mở trang web có địa chỉ mặc định `http://localhost:8501`.
- **Tab Nhận Diện Ảnh Tĩnh:** Kéo / thả tệp vào đây và chờ hiển thị '% Hút Thuốc'
- **Tab Nhận Diện Từ Camera:** Bật Checkbox và đưa video hoặc tự di chuyển phía trước Webcam thiết bị để nó gán nhãn ĐỎ (Hút thuốc) hoặc XANH (Không). 
*(Lưu ý: Bạn chọn tắt Checkbox khi muốn ngưng Camera nhé).*

---
🤝 Chúc hoàn thành Đồ án tốt đẹp! Mất thắc mắc có thể xem lại file `task.md` hoặc code cấu trúc bên trong `src`.
