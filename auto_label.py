from ultralytics import YOLO

# 1. Đường dẫn tới bộ não AI (vì file chạy đặt ở Pest-Detection-main nên đường dẫn này giữ nguyên)
model = YOLO(r"backend/models/version/best.pt")

# 2. Đường dẫn tuyệt đối tới thư mục 1000 ảnh của bạn (Có chữ 'r' ở đầu cực kỳ quan trọng)
INPUT_FOLDER = r"F:\thực_tập\dataset_raw_1000"

print(" Đang gọi AI càn quét 1000 tấm ảnh để tự động tạo Nhãn...")

# 3. Chạy lệnh càn quét
results = model.predict(
    source=INPUT_FOLDER,
    save_txt=True,       # Bắt buộc True để sinh ra file .txt
    save_conf=False,     # Không lưu % tin cậy vào nhãn
    conf=0.25,           # Ngưỡng tin cậy
    project="auto_labels", 
    name="batch_1"
)

print(f" Xong! Hãy vào F:\\thực_tập\\Pest-Detection-main\\auto_labels\\batch_1\\labels để lấy 1000 file .txt nhé!")