import cv2
import paho.mqtt.client as mqtt
from ultralytics import YOLO

# 1. CẤU HÌNH MQTT (Giữ nguyên)
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "nongnghiep_iot/thuctap/dem_sau_ray"

mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT)

# 2. KHỞI ĐỘNG AI YOLO11 MỚI NHẤT
# Đã sửa đường dẫn tới file YOLO11 và thêm task="detect"
model_path = r"backend/models/version/best_yolo11_final.pt"
model = YOLO(model_path, task="detect") 

video_source = "video_test.mp4" 
cap = cv2.VideoCapture(video_source)

max_count = 0
last_sent_count = -1

print("🚀 Đang chạy TRÙM CUỐI YOLO11 - Chế độ Kính Lúp...")

while True:
    success, frame = cap.read()
    if not success:
        break

    # Ép AI nhìn ảnh phân giải cao 1280 để không bỏ sót bọ nhỏ
    results = model.predict(frame, conf=0.15, iou=0.6, imgsz=1280, verbose=False)
    
    result = results[0]
    
    # ⚠️ LƯU Ý QUAN TRỌNG: 
    # Nếu kết quả hiện tên bị sai (ví dụ bọ to thành bọ thon), 
    # bạn hãy sửa lại thứ tự 0, 1, 2 ở dòng dưới cho khớp với lúc up lên Roboflow nhé.
    result.names = {0: "bo to", 1: "bo map", 2: "bo thon", 3: "con khac"}
    
    annotated_frame = result.plot()
    current_count = len(result.boxes)

    # Cập nhật số lượng kỷ lục (Max Count)
    if current_count > max_count:
        max_count = current_count

    # Bắn MQTT khi số lượng thay đổi
    if max_count != last_sent_count:
        mqtt_client.publish(MQTT_TOPIC, str(max_count))
        last_sent_count = max_count

    # Ghi chữ lên video
    cv2.putText(annotated_frame, f"Dang co tren man hinh: {current_count}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"TONG SO THUC TE (MAX): {max_count}", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Giữ nguyên tỷ lệ dọc để không bị méo hình
    h, w = annotated_frame.shape[:2]
    new_h = 800
    new_w = int(w * (new_h / h))
    resized_show = cv2.resize(annotated_frame, (new_w, new_h))
    
    cv2.imshow("He Thong AIoT YOLO11 - Final Version", resized_show)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()