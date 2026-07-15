import time
import os
import uuid
import cv2  # Thêm thư viện xử lý Video
from ultralytics import YOLO
from pathlib import Path
from utils.logger import get_logger
from utils.metrics import (
    INFERENCE_TIME, PEST_DETECTED,
    DETECTION_WITH_PEST, DETECTION_WITHOUT_PEST,
    CONFIDENCE_SCORE
)

logger = get_logger(__name__)

# Bổ sung class số 3 để đồng bộ với YOLO11
PEST_MAPPING = {
    "bo thon": "thin_pest",
    "bo map" : "round_pest",
    "bo to"  : "big_pest",
    "con khac": "big_pest" 
}

# ==========================================================
# HÀM 1: XỬ LÝ ẢNH TĨNH
# ==========================================================
def predict_image(file_path: str, conf: float = 0.25, model=None) -> dict:
    if model is None:
        raise ValueError("Model must be provided.")
    
    start = time.time()
    results = model.predict(source=file_path, conf=conf)
    INFERENCE_TIME.observe(time.time() - start)

    result = results[0]
    
    # Ép AI YOLO dịch từ số sang tên Tiếng Việt (Có class 3)
    result.names = {0: "bo to", 1: "bo map", 2: "bo thon", 3: "con khac"}
    
    input_path = Path(file_path)
    output_img_path = str(input_path.parent / f"output_{input_path.stem}.jpg")
    
    result.save(filename=output_img_path)

    counts = {"thin_pest": 0, "round_pest": 0, "big_pest": 0}
    boxes = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = result.names[class_id]  
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        CONFIDENCE_SCORE.observe(confidence)

        real_name = PEST_MAPPING.get(class_name)
        if real_name is None:
            logger.warning(f"Unknown class: {class_name} !!!")
            continue
        
        counts[real_name] += 1
        PEST_DETECTED.labels(pest_type=real_name).inc()

        boxes.append({
            "class_name": class_name,
            "confidence": round(confidence, 2),
            "x1"        : round(x1, 2),
            "y1"        : round(y1, 2),
            "x2"        : round(x2, 2),
            "y2"        : round(y2, 2),
        })
    
    if len(boxes) > 0:
        DETECTION_WITH_PEST.inc()
    else:
        DETECTION_WITHOUT_PEST.inc()

    logger.info(
        f"Prediction done\n"
        f"Total: {len(boxes)}\n"
        f"thin : {counts['thin_pest']}\n"
        f"round: {counts['round_pest']}\n"
        f"big  : {counts['big_pest']}"
    )

    return {
        "total_count"    : len(boxes),
        "counts"         : counts,
        "boxes"          : boxes,
        "output_img_path": output_img_path
    }


# ==========================================================
# HÀM 2: XỬ LÝ VIDEO (BẢN FINAL: MAX COUNT + KÍNH LÚP 1280)
# ==========================================================
def predict_video_file(input_video_path: str, conf: float = 0.15) -> dict:
    """
    Hàm xử lý video ĐÃ TỐI ƯU HÓA RAM CHO CLOUD:
    - Sử dụng mô hình ONNX siêu nhẹ (11MB) thay vì PT.
    - Giảm imgsz xuống 640 để không bị tràn bộ nhớ 512MB RAM của Render.
    """
    # 1. BẮT BUỘC DÙNG FILE .ONNX ĐỂ TIẾT KIỆM BỘ NHỚ
    model = YOLO("models/version/best_v2_nano.onnx", task="detect")
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise Exception("Không thể đọc được video tải lên!")

    # Cấu hình file Video đầu ra (.webm để xem được trên Web)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    output_filename = f"output_video_{uuid.uuid4().hex[:8]}.webm"
    output_video_path = os.path.join("uploads", output_filename)
    
    # Định dạng vp80 dành cho WebM
    fourcc = cv2.VideoWriter_fourcc(*'vp80')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # THUẬT TOÁN ĐẾM MAX COUNT
    max_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 2. GIẢM IMGSZ XUỐNG 640 ĐỂ TRÁNH BỊ CHẶN RAM SẬP SERVER
        results = model.predict(frame, conf=conf, iou=0.6, imgsz=640, verbose=False)
        result = results[0]
        
        # Đồng bộ từ điển tên (có Class 3)
        result.names = {0: "bo to", 1: "bo map", 2: "bo thon", 3: "con khac"}
        
        annotated_frame = result.plot()
        current_count = len(result.boxes)

        # Cập nhật số lượng kỷ lục (Max Count)
        if current_count > max_count:
            max_count = current_count

        # Ghi chữ lên video xuất ra
        cv2.putText(annotated_frame, f"Dang co tren man hinh: {current_count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"TONG SO THUC TE (MAX): {max_count}", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        out.write(annotated_frame)

    cap.release()
    out.release()

    return {
        "total_count": max_count,
        "output_video_path": output_video_path
    }