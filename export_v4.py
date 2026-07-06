from ultralytics import YOLO
model = YOLO(r"F:\thực_tập\Pest-Detection-main\backend\models\version\best_v4_final.pt")
model.export(format="onnx")