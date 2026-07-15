from ultralytics import YOLO

# 1. Load file PT của bạn
model = YOLO(r"F:\thực_tập\Pest-Detection-main\backend\models\version\best_yolo11_final.pt")

# 2. Xuất ra định dạng ONNX
print("⏳ Đang ép cân sang ONNX...")
success = model.export(format="onnx")

print(f"🎉 Ép cân thành công! File ONNX đã được lưu tại: {success}")