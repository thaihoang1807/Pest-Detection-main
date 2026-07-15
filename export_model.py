import os
import shutil
from ultralytics import YOLO

# 1. Đường dẫn tới file YOLO11 mới nhất (.pt) bạn vừa tải về và đổi tên
pt_path = r"F:\thực_tập\Pest-Detection-main\backend\models\version\best_yolo11_final.pt"

if not os.path.exists(pt_path):
    print(f"❌ KHÔNG TÌM THẤY file .pt tại: {pt_path}")
    print("Vui lòng đảm bảo bạn đã tải file 'best.pt' từ Google Drive về, đổi tên thành 'best_yolo11_final.pt' và bỏ vào thư mục trên.")
    exit()

# 2. Load mô hình YOLO11 mới tinh
print("⏳ Đang nạp mô hình YOLO11...")
model = YOLO(pt_path)

# 3. Tiến hành xuất (export) sang định dạng ONNX
print("⏳ Đang tiến hành xuất (export) sang ONNX...")
exported_path = model.export(format="onnx") # Thường nó sẽ tự xuất ra 'best_yolo11_final.onnx'

# 4. ÉP ĐỔI TÊN ĐẦU RA THÀNH "best_yolo11_final.onnx" BẰNG CODE (Không làm thủ công)
target_onnx_path = r"F:\thực_tập\Pest-Detection-main\backend\models\version\best_yolo11_final.onnx"

if os.path.exists(exported_path) and exported_path != target_onnx_path:
    # Nếu YOLO lỡ đặt tên khác, code sẽ tự động đổi tên và di chuyển nó về đúng tên mình muốn
    shutil.move(exported_path, target_onnx_path)
    print(f"👉 Đã tự động đổi tên và lưu file tại: {target_onnx_path}")
else:
    print(f"🎉 Xuất file thành công tại: {target_onnx_path}")