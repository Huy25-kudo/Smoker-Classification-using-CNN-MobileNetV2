"""
PHÂN LOẠI & PHÂN TÍCH CÁC TRƯỜNG HỢP NHẬN DIỆN SAI (ERROR ANALYSIS)
+ LƯU ẢNH SAI PHÂN LOẠI THEO NGUYÊN NHÂN
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2, os, shutil
import pandas as pd
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime

# ============================================================
# CẤU HÌNH
# ============================================================
MODEL_PATH = "../models/smoker_detector_best.keras"
TEST_DIR = "../dataset/processed_dataset"
TEST_CSV_PATH = os.path.join(TEST_DIR, "test_dataset.csv")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
HIGH_CONF = 0.80
MID_CONF = 0.60

# Thư mục lưu kết quả
OUTPUT_DIR = "../error_analysis_results"

# ============================================================
# TẢI MÔ HÌNH & DỮ LIỆU
# ============================================================
print("="*70)
print("  PHÂN TÍCH LỖI NHẬN DIỆN - SMOKER DETECTION MODEL")
print("="*70)

print("\n[1/6] Đang tải mô hình...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'preprocess_input': tf.keras.applications.mobilenet_v2.preprocess_input},
    safe_mode=False
)
print("  ✓ Mô hình đã tải thành công!")

print("\n[2/6] Đang nạp dữ liệu test...")
test_df = pd.read_csv(TEST_CSV_PATH)
print(f"  ✓ Đã nạp {len(test_df)} ảnh")

test_gen = ImageDataGenerator().flow_from_dataframe(
    dataframe=test_df, x_col='Filepath', y_col='Label',
    target_size=IMG_SIZE, color_mode='rgb',
    class_mode='categorical', batch_size=BATCH_SIZE, shuffle=False
)
class_names = list(test_gen.class_indices.keys())

print("\n[3/6] Đang chạy dự đoán...")
y_true = np.array(test_gen.classes)
preds = model.predict(test_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
filepaths = test_gen.filepaths
acc = np.mean(y_true == y_pred)
print(f"  📊 Accuracy: {acc*100:.2f}%")

# ============================================================
# PHÂN LOẠI NGUYÊN NHÂN
# ============================================================

# Danh mục nguyên nhân để phân loại thư mục
REASON_CATEGORIES = {
    'do_phan_giai_thap': 'Do phân giải thấp - thiếu chi tiết',
    'khung_hinh_bat_thuong': 'Tỉ lệ khung hình bất thường - resize méo',
    'anh_qua_toi': 'Ảnh quá tối - khó nhận diện',
    'anh_qua_sang': 'Ảnh quá sáng - mất chi tiết',
    'tuong_phan_thap': 'Độ tương phản thấp - đặc trưng mờ',
    'truong_hop_bien': 'Trường hợp biên - confidence sát ngưỡng',
    'conf_cao_sai': 'Confidence cao nhưng sai - lỗi nghiêm trọng',
    'conf_tb_sai': 'Confidence trung bình sai',
    'conf_thap_sai': 'Confidence thấp sai - không chắc chắn',
    'giong_hanh_dong_hut': 'Giống hành động hút thuốc (FP)',
    'thuoc_la_bi_che': 'Thuốc lá bị che khuất (FN)',
    'khoi_khong_ro': 'Khói thuốc không rõ (FN)',
    'khac': 'Nguyên nhân khác'
}

def analyze_image_reasons(filepath, pred_confidence, true_confidence, group_name):
    """Phân tích ảnh và trả về danh sách nguyên nhân (key + mô tả)."""
    reasons = []  # list of (category_key, description)
    h, w = 0, 0
    
    try:
        raw_img = cv2.imread(filepath)
        if raw_img is not None:
            h, w = raw_img.shape[:2]
            gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
            
            if h < 100 or w < 100:
                reasons.append(('do_phan_giai_thap', f'Ảnh gốc {w}x{h} - phân giải thấp'))
            if (w/h if h>0 else 0) > 2.0 or (w/h if h>0 else 1) < 0.5:
                reasons.append(('khung_hinh_bat_thuong', f'Tỉ lệ {w/h:.1f} bất thường'))
            
            brightness = np.mean(gray)
            if brightness < 50:
                reasons.append(('anh_qua_toi', f'Độ sáng TB={brightness:.0f} - quá tối'))
            elif brightness > 220:
                reasons.append(('anh_qua_sang', f'Độ sáng TB={brightness:.0f} - quá sáng'))
            
            if np.std(gray) < 30:
                reasons.append(('tuong_phan_thap', 'Tương phản thấp'))
    except:
        pass
    
    margin = abs(pred_confidence - true_confidence)
    if margin < 20:
        reasons.append(('truong_hop_bien', f'Margin={margin:.1f}% - sát ngưỡng'))
    
    # Phân loại theo confidence
    if pred_confidence >= HIGH_CONF * 100:
        reasons.append(('conf_cao_sai', f'Conf={pred_confidence:.1f}% - rất tự tin nhưng SAI'))
    elif pred_confidence >= MID_CONF * 100:
        reasons.append(('conf_tb_sai', f'Conf={pred_confidence:.1f}% - khá tự tin nhưng sai'))
    else:
        reasons.append(('conf_thap_sai', f'Conf={pred_confidence:.1f}% - không chắc chắn'))
    
    # Nguyên nhân theo loại lỗi
    if group_name == "FP":
        reasons.append(('giong_hanh_dong_hut', 'Có thể giống hành động hút thuốc'))
    else:
        reasons.append(('thuoc_la_bi_che', 'Thuốc lá có thể bị che khuất'))
        reasons.append(('khoi_khong_ro', 'Khói thuốc có thể không rõ'))
    
    if not reasons:
        reasons.append(('khac', 'Không xác định rõ nguyên nhân'))
    
    return reasons, (w, h)

# ============================================================
# PHÂN TÍCH & THU THẬP DỮ LIỆU
# ============================================================
print("\n[4/6] Đang phân loại lỗi...")

smoking_idx = class_names.index('smoking') if 'smoking' in class_names else 1
not_smoking_idx = 1 - smoking_idx
wrong_indices = np.where(y_true != y_pred)[0]
print(f"  ✗ Tổng lỗi: {len(wrong_indices)}/{len(y_true)} ({len(wrong_indices)/len(y_true)*100:.1f}%)")

all_errors = []  # Thu thập tất cả lỗi

for idx in wrong_indices:
    true_idx = y_true[idx]
    pred_idx = y_pred[idx]
    pred_conf = preds[idx][pred_idx] * 100
    true_conf = preds[idx][true_idx] * 100
    
    if true_idx == not_smoking_idx and pred_idx == smoking_idx:
        error_type = "FP"
        error_desc = "False Positive - Báo nhầm có hút thuốc"
    else:
        error_type = "FN"
        error_desc = "False Negative - Bỏ sót người hút thuốc"
    
    # Mức độ nghiêm trọng
    if pred_conf >= HIGH_CONF * 100:
        severity = "NGHIEM_TRONG"
        sev_text = "🔴 NGHIÊM TRỌNG"
    elif pred_conf >= MID_CONF * 100:
        severity = "TRUNG_BINH"
        sev_text = "🟡 TRUNG BÌNH"
    else:
        severity = "NHE"
        sev_text = "🟢 NHẸ"
    
    reasons, img_size = analyze_image_reasons(
        filepaths[idx], pred_conf, true_conf, error_type)
    
    # Xác định nguyên nhân chính (primary reason)
    # Ưu tiên: lỗi ảnh > confidence > loại lỗi
    image_reasons = [r for r in reasons if r[0] in 
        ['do_phan_giai_thap','khung_hinh_bat_thuong','anh_qua_toi','anh_qua_sang','tuong_phan_thap']]
    primary_key = image_reasons[0][0] if image_reasons else reasons[0][0]
    
    all_errors.append({
        'idx': idx, 'filepath': filepaths[idx],
        'true_label': class_names[true_idx], 'pred_label': class_names[pred_idx],
        'pred_confidence': pred_conf, 'true_confidence': true_conf,
        'error_type': error_type, 'error_desc': error_desc,
        'severity': severity, 'severity_text': sev_text,
        'reasons': reasons, 'primary_reason': primary_key,
        'img_size': img_size
    })

fp_errors = [e for e in all_errors if e['error_type'] == 'FP']
fn_errors = [e for e in all_errors if e['error_type'] == 'FN']
print(f"  📌 FP: {len(fp_errors)} | FN: {len(fn_errors)}")

# ============================================================
# LƯU ẢNH SAI PHÂN LOẠI THEO NGUYÊN NHÂN
# ============================================================
print("\n[5/6] Đang lưu ảnh sai phân loại theo nguyên nhân...")

# Tạo thư mục gốc
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
os.makedirs(save_dir, exist_ok=True)

# Cấu trúc: save_dir / FP (hoặc FN) / nguyên_nhân / ảnh
saved_count = 0

for error_type_name, error_list in [("FP_bao_nham", fp_errors), ("FN_bo_sot", fn_errors)]:
    if not error_list:
        continue
    
    type_dir = os.path.join(save_dir, error_type_name)
    os.makedirs(type_dir, exist_ok=True)
    
    # Lưu theo nguyên nhân chính
    for err in error_list:
        reason_key = err['primary_reason']
        reason_dir = os.path.join(type_dir, reason_key)
        os.makedirs(reason_dir, exist_ok=True)
        
        src_path = err['filepath']
        basename = os.path.basename(src_path)
        # Thêm prefix severity + confidence vào tên file
        new_name = f"{err['severity']}_{err['pred_confidence']:.0f}pct_{basename}"
        dst_path = os.path.join(reason_dir, new_name)
        
        try:
            # Copy ảnh gốc
            shutil.copy2(src_path, dst_path)
            
            # Tạo ảnh annotated (có ghi chú nguyên nhân)
            img = cv2.imread(src_path)
            if img is not None:
                h_img, w_img = img.shape[:2]
                # Thêm banner đen ở dưới
                banner_h = 80
                annotated = np.zeros((h_img + banner_h, w_img, 3), dtype=np.uint8)
                annotated[:h_img, :, :] = img
                
                # Ghi thông tin
                font = cv2.FONT_HERSHEY_SIMPLEX
                color = (0, 0, 255) if err['severity'] == 'NGHIEM_TRONG' else \
                        (0, 165, 255) if err['severity'] == 'TRUNG_BINH' else (0, 255, 0)
                
                cv2.putText(annotated, f"True: {err['true_label']} | Pred: {err['pred_label']} ({err['pred_confidence']:.1f}%)",
                           (5, h_img + 25), font, 0.45, (255,255,255), 1)
                cv2.putText(annotated, f"Reason: {REASON_CATEGORIES.get(reason_key, reason_key)}",
                           (5, h_img + 50), font, 0.4, color, 1)
                cv2.putText(annotated, f"Severity: {err['severity']}",
                           (5, h_img + 70), font, 0.4, color, 1)
                
                ann_path = os.path.join(reason_dir, f"ANN_{new_name}")
                cv2.imwrite(ann_path, annotated)
            
            saved_count += 1
        except Exception as e:
            print(f"  ⚠ Lỗi copy {basename}: {e}")

# Lưu cả theo mức độ nghiêm trọng
severity_dir = os.path.join(save_dir, "theo_muc_do")
os.makedirs(severity_dir, exist_ok=True)
for sev_name in ["NGHIEM_TRONG", "TRUNG_BINH", "NHE"]:
    sev_errors = [e for e in all_errors if e['severity'] == sev_name]
    if not sev_errors:
        continue
    sdir = os.path.join(severity_dir, sev_name)
    os.makedirs(sdir, exist_ok=True)
    for err in sev_errors:
        src = err['filepath']
        bn = os.path.basename(src)
        dst = os.path.join(sdir, f"{err['error_type']}_{err['pred_confidence']:.0f}pct_{bn}")
        try:
            shutil.copy2(src, dst)
        except:
            pass

print(f"  ✓ Đã lưu {saved_count} ảnh sai vào: {os.path.abspath(save_dir)}")

# ============================================================
# TẠO BÁO CÁO TỔNG HỢP
# ============================================================
report_path = os.path.join(save_dir, "bao_cao_loi.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("  BÁO CÁO PHÂN TÍCH LỖI NHẬN DIỆN - SMOKER DETECTION\n")
    f.write(f"  Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"Tổng mẫu test: {len(y_true)}\n")
    f.write(f"Accuracy: {acc*100:.2f}%\n")
    f.write(f"Tổng lỗi: {len(all_errors)} ({len(all_errors)/len(y_true)*100:.1f}%)\n")
    f.write(f"  - False Positive (FP): {len(fp_errors)}\n")
    f.write(f"  - False Negative (FN): {len(fn_errors)}\n\n")
    
    # Thống kê theo nguyên nhân
    f.write("="*70 + "\n")
    f.write("THỐNG KÊ THEO NGUYÊN NHÂN CHÍNH\n")
    f.write("="*70 + "\n\n")
    
    reason_stats = {}
    for err in all_errors:
        key = err['primary_reason']
        if key not in reason_stats:
            reason_stats[key] = {'FP': 0, 'FN': 0, 'total': 0, 'confs': []}
        reason_stats[key][err['error_type']] += 1
        reason_stats[key]['total'] += 1
        reason_stats[key]['confs'].append(err['pred_confidence'])
    
    for key, stats in sorted(reason_stats.items(), key=lambda x: -x[1]['total']):
        desc = REASON_CATEGORIES.get(key, key)
        avg_conf = np.mean(stats['confs'])
        f.write(f"[{key}] {desc}\n")
        f.write(f"  Tổng: {stats['total']} (FP={stats['FP']}, FN={stats['FN']})\n")
        f.write(f"  Confidence TB: {avg_conf:.1f}%\n\n")
    
    # Thống kê theo mức độ
    f.write("="*70 + "\n")
    f.write("THỐNG KÊ THEO MỨC ĐỘ NGHIÊM TRỌNG\n")
    f.write("="*70 + "\n\n")
    for sev in ["NGHIEM_TRONG", "TRUNG_BINH", "NHE"]:
        cnt = sum(1 for e in all_errors if e['severity'] == sev)
        f.write(f"  {sev}: {cnt} ({cnt/len(all_errors)*100:.1f}%)\n")
    
    # Chi tiết từng ảnh sai
    f.write("\n" + "="*70 + "\n")
    f.write("CHI TIẾT TỪNG TRƯỜNG HỢP SAI (sắp xếp theo confidence giảm dần)\n")
    f.write("="*70 + "\n\n")
    
    sorted_errors = sorted(all_errors, key=lambda x: -x['pred_confidence'])
    for i, err in enumerate(sorted_errors):
        f.write(f"[{i+1}] {err['severity']} - {err['error_type']}\n")
        f.write(f"  File: {os.path.basename(err['filepath'])}\n")
        f.write(f"  True: {err['true_label']} | Pred: {err['pred_label']} ({err['pred_confidence']:.1f}%)\n")
        f.write(f"  Size: {err['img_size'][0]}x{err['img_size'][1]}\n")
        f.write(f"  Nguyên nhân chính: {REASON_CATEGORIES.get(err['primary_reason'], err['primary_reason'])}\n")
        f.write(f"  Tất cả nguyên nhân:\n")
        for rk, rd in err['reasons']:
            f.write(f"    - [{rk}] {rd}\n")
        f.write("\n")

# Lưu CSV
csv_path = os.path.join(save_dir, "danh_sach_loi.csv")
csv_data = []
for err in all_errors:
    csv_data.append({
        'file': os.path.basename(err['filepath']),
        'filepath': err['filepath'],
        'true_label': err['true_label'],
        'pred_label': err['pred_label'],
        'confidence': f"{err['pred_confidence']:.1f}",
        'error_type': err['error_type'],
        'severity': err['severity'],
        'primary_reason': err['primary_reason'],
        'reason_desc': REASON_CATEGORIES.get(err['primary_reason'], ''),
        'all_reasons': ' | '.join([f"{rk}: {rd}" for rk, rd in err['reasons']])
    })
pd.DataFrame(csv_data).to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"  ✓ Báo cáo: {os.path.basename(report_path)}")
print(f"  ✓ CSV: {os.path.basename(csv_path)}")

# ============================================================
# HIỂN THỊ BIỂU ĐỒ
# ============================================================
print("\n[6/6] Đang tạo biểu đồ...")

def show_error_samples(error_list, title, color, max_show=12):
    if not error_list:
        return
    sorted_list = sorted(error_list, key=lambda x: -x['pred_confidence'])
    n = min(max_show, len(sorted_list))
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5.5*rows))
    fig.suptitle(f'{title} ({n}/{len(error_list)} mẫu)', fontsize=14, fontweight='bold', color=color)
    
    if rows == 1 and cols == 1: axes = np.array([[axes]])
    elif rows == 1: axes = axes.reshape(1, -1)
    elif cols == 1: axes = axes.reshape(-1, 1)
    
    for i in range(n):
        ax = axes[i//cols, i%cols]
        e = sorted_list[i]
        try:
            img = cv2.cvtColor(cv2.imread(e['filepath']), cv2.COLOR_BGR2RGB)
            ax.imshow(img)
        except:
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center')
        
        tc = 'red' if e['severity']=='NGHIEM_TRONG' else 'darkorange' if e['severity']=='TRUNG_BINH' else 'green'
        reason_text = REASON_CATEGORIES.get(e['primary_reason'], e['primary_reason'])
        if len(reason_text) > 35: reason_text = reason_text[:35] + '...'
        ax.set_title(f"[{e['severity']}]\nTrue: {e['true_label']} → Pred: {e['pred_label']}\n"
                    f"Conf: {e['pred_confidence']:.1f}%\n{reason_text}",
                    fontsize=8, color=tc, fontweight='bold')
        ax.axis('off')
    
    for i in range(n, rows*cols):
        axes[i//cols, i%cols].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{title.replace(' ','_')}.png"), dpi=150, bbox_inches='tight')
    plt.show()

show_error_samples(fp_errors, "False Positive - Bao nham", '#e74c3c')
show_error_samples(fn_errors, "False Negative - Bo sot", '#3498db')

# Biểu đồ thống kê nguyên nhân
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Bar chart theo nguyên nhân
reason_names = [REASON_CATEGORIES.get(k, k)[:25] for k in reason_stats.keys()]
reason_counts = [v['total'] for v in reason_stats.values()]
sorted_pairs = sorted(zip(reason_names, reason_counts), key=lambda x: -x[1])
if sorted_pairs:
    names, counts = zip(*sorted_pairs)
    axes[0].barh(range(len(names)), counts, color='#3498db')
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, fontsize=8)
    axes[0].set_xlabel('Số lượng')
    axes[0].set_title('Phân bố theo nguyên nhân', fontweight='bold')
    axes[0].invert_yaxis()

# 2. Pie severity
sev_counts = [sum(1 for e in all_errors if e['severity']==s) for s in ["NGHIEM_TRONG","TRUNG_BINH","NHE"]]
sev_labels = ['Nghiêm trọng','Trung bình','Nhẹ']
sev_colors = ['#ff4444','#ffaa00','#44bb44']
non_zero = [(c,l,cl) for c,l,cl in zip(sev_counts,sev_labels,sev_colors) if c>0]
if non_zero:
    c,l,cl = zip(*non_zero)
    axes[1].pie(c, labels=l, colors=cl, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Phân bố mức độ', fontweight='bold')

# 3. FP vs FN
axes[2].bar(['FP\n(Báo nhầm)', 'FN\n(Bỏ sót)'], [len(fp_errors), len(fn_errors)],
           color=['#e74c3c', '#3498db'])
axes[2].set_ylabel('Số lượng')
axes[2].set_title('FP vs FN', fontweight='bold')
for i, v in enumerate([len(fp_errors), len(fn_errors)]):
    axes[2].text(i, v+0.5, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "thong_ke_tong_hop.png"), dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# TỔNG HỢP
# ============================================================
print(f"\n{'='*70}")
print(f"  HOÀN TẤT PHÂN TÍCH!")
print(f"{'='*70}")
print(f"\n  📁 Thư mục kết quả: {os.path.abspath(save_dir)}")
print(f"  Cấu trúc thư mục:")
print(f"  {save_dir}/")
print(f"  ├── FP_bao_nham/          (ảnh FP phân loại theo nguyên nhân)")
for key in REASON_CATEGORIES:
    d = os.path.join(save_dir, "FP_bao_nham", key)
    if os.path.exists(d) and os.listdir(d):
        print(f"  │   ├── {key}/ ({len(os.listdir(d))} files)")
print(f"  ├── FN_bo_sot/            (ảnh FN phân loại theo nguyên nhân)")
for key in REASON_CATEGORIES:
    d = os.path.join(save_dir, "FN_bo_sot", key)
    if os.path.exists(d) and os.listdir(d):
        print(f"  │   ├── {key}/ ({len(os.listdir(d))} files)")
print(f"  ├── theo_muc_do/          (ảnh phân loại theo mức độ)")
for sev in ["NGHIEM_TRONG", "TRUNG_BINH", "NHE"]:
    d = os.path.join(save_dir, "theo_muc_do", sev)
    if os.path.exists(d) and os.listdir(d):
        print(f"  │   ├── {sev}/ ({len(os.listdir(d))} files)")
print(f"  ├── bao_cao_loi.txt       (báo cáo chi tiết)")
print(f"  ├── danh_sach_loi.csv     (danh sách lỗi dạng bảng)")
print(f"  └── *.png                 (biểu đồ thống kê)")
print(f"\n  Tổng: {saved_count} ảnh sai đã được lưu và phân loại.")
