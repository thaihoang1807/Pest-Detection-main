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

PEST_MAPPING = {
    "bo thon": "thin_pest",
    "bo map" : "round_pest",
    "bo to"  : "big_pest",
    "con khac": "big_pest"
}

# ==========================================================
# HÀM 1: XỬ LÝ ẢNH TĨNH (Đã chạy hoàn hảo)
# ==========================================================
def predict_image(file_path: str, conf: float = 0.25, model=None) -> dict:
    if model is None:
        raise ValueError("Model must be provided.")
    
    start = time.time()
    results = model.predict(source=file_path, conf=conf)
    INFERENCE_TIME.observe(time.time() - start)

    result = results[0]
    
    # Ép AI YOLO dịch từ số (0, 1, 2) sang tên Tiếng Việt
    result.names = {0: "bo to", 1: "bo map", 2: "bo thon"}
    
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
# HÀM 2: XỬ LÝ VIDEO (Tích hợp thêm cho nhóm Backend)
# ==========================================================
def predict_video_file(input_video_path: str, conf: float = 0.35) -> dict:
    """
    Hàm xử lý video: Đọc video, Tracking, Đếm qua vạch và Xuất video kết quả
    """
    # Load Model (Dùng bản ONNX 6MB cho tốc độ xử lý video mượt)
    # Lưu ý: Đường dẫn này là đường dẫn bên trong Docker Container của Backend
    model = YOLO("models/version/best_v2_nano.onnx", task="detect")
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise Exception("Không thể đọc được video tải lên!")

    # Cấu hình file Video đầu ra (Output)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Tạo tên file output ngẫu nhiên tránh trùng lặp
    output_filename = f"output_video_{uuid.uuid4().hex[:8]}.mp4"
    output_video_path = os.path.join("uploads", output_filename)
    
    # Cấu hình bộ ghi video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Setup Vạch đếm (Vẽ ở giữa màn hình video)
    LINE_Y = height // 2 
    track_history = {}
    total_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, conf=conf, iou=0.5, persist=True, tracker="botsort.yaml", verbose=False)
        annotated_frame = results[0].plot()

        # Vẽ vạch đếm màu vàng
        cv2.line(annotated_frame, (0, LINE_Y), (width, LINE_Y), (0, 255, 255), 2)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                center_y = int((y1 + y2) / 2)
                prev_y = track_history.get(track_id, center_y)

                # Thuật toán đếm qua vạch 2 chiều
                if (prev_y < LINE_Y and center_y >= LINE_Y) or (prev_y > LINE_Y and center_y <= LINE_Y):
                    total_count += 1
                    cv2.line(annotated_frame, (0, LINE_Y), (width, LINE_Y), (0, 0, 255), 6) # Chớp đỏ khi qua vạch

                track_history[track_id] = center_y

        # Ghi chữ lên video
        cv2.putText(annotated_frame, f"Tong so qua vach: {total_count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        out.write(annotated_frame)

    cap.release()
    out.release()

    return {
        "total_count": total_count,
        "output_video_path": output_video_path
    }