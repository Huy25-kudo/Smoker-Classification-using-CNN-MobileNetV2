import os
import zipfile
from pathlib import Path

def prepare_data():
    """
    Script để tự động quét thư mục data/raw, tìm file .zip được người dùng tải về
    và giải nén vào data/processed.
    """
    raw_dir = Path(os.path.abspath("data/raw"))
    processed_dir = Path(os.path.abspath("data/processed"))
    
    # Tạo thư mục nếu chưa tồn tại
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Đang tìm kiếm file nén .zip trong: {raw_dir}")
    zip_files = list(raw_dir.glob("*.zip"))
    
    if not zip_files:
        print(f"[LỖI] Không tìm thấy file .zip nào trong {raw_dir}.")
        print("Vui lòng tải dataset (archive.zip) từ Kaggle và copy vào thư mục data/raw/ !")
        return
    
    zip_path = zip_files[0]
    print(f"[INFO] Bắt đầu giải nén file: {zip_path.name}... (quá trình này có thể tốn vài phút tùy kích thước)")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Giải nén thẳng vào data/processed
            zip_ref.extractall(processed_dir)
        print(f"[THÀNH CÔNG] Dữ liệu đã được giải nén vào: {processed_dir}")
        print("[INFO] Vui lòng kiểm tra cấu trúc thư mục bên trong để chuẩn bị cho Huấn luyện.")
    except Exception as e:
        print(f"[LỖI] Việc giải nén thất bại: {e}")

if __name__ == "__main__":
    prepare_data()
