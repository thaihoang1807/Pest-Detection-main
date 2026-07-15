from ultralytics import YOLO

# Load đúng file YOLO11 mới nhất của bạn
model = YOLO(r"F:\thực_tập\Pest-Detection-main\backend\models\version\best_yolo11_final.pt")

# Xuất ra định dạng ONNX
model.export(format="onnx")
print("🎉 Xuất file ONNX thành công!")