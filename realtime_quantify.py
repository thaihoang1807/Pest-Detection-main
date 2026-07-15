import cv2
import numpy as np
import paho.mqtt.client as mqtt
from ultralytics import YOLO

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "nongnghiep_iot/thuctap/dem_sau_ray"

mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT)

model_path = r"backend/models/version/best_yolo11_final.pt"
model = YOLO(model_path)

video_source = "video_test.mp4" 
cap = cv2.VideoCapture(video_source)

max_count = 0
last_sent_count = -1

print("🚀 Bắt đầu chế độ HYBRID AI (Đã tinh chỉnh thông số vàng)...")

while True:
    success, frame = cap.read()
    if not success:
        break

    # 🌟 TINH CHỈNH 1: Trả conf về 0.25 (chắc chắn mới khoanh) và iou=0.45 (xóa khung trùng)
    results = model.predict(frame, conf=0.25, iou=0.45, imgsz=1280, verbose=False)
    result = results[0]
    result.names = {0: "bo to", 1: "bo map", 2: "bo thon", 3: "con khac"}
    
    annotated_frame = result.plot()
    current_count = len(result.boxes)

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_id = int(box.cls[0])
        box_area = (x2 - x1) * (y2 - y1)
        
        # 🌟 TINH CHỈNH 2: Chỉ soi kính lúp với những khung THỰC SỰ BỰ (>2000 pixel)
        if class_id == 0 and box_area > 2000: 
            crop_img = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 🌟 TINH CHỈNH 3: Bỏ qua hạt bụi, đốm đen phải bự (>80) mới tính là bọ
            bugs_in_cluster = len([c for c in contours if cv2.contourArea(c) > 80])
            
            if bugs_in_cluster > 1:
                current_count += (bugs_in_cluster - 1)
                cv2.putText(annotated_frame, f"Chum: {bugs_in_cluster} con", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Cập nhật Max Count
    if current_count > max_count:
        max_count = current_count

    # Bắn MQTT
    if max_count != last_sent_count:
        mqtt_client.publish(MQTT_TOPIC, str(max_count))
        last_sent_count = max_count

    # Ghi chữ lên video
    cv2.putText(annotated_frame, f"Dang co tren man hinh: {current_count}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"TONG SO THUC TE (MAX): {max_count}", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Hiển thị
    h, w = annotated_frame.shape[:2]
    new_h = 800
    new_w = int(w * (new_h / h))
    resized_show = cv2.resize(annotated_frame, (new_w, new_h))
    
    cv2.imshow("He Thong IoT - Tuned Hybrid AI", resized_show)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()