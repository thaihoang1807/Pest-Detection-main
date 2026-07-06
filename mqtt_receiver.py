import paho.mqtt.client as mqtt

# Phải trùng khớp 100% với cấu hình bên file phát
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "nongnghiep_iot/thuctap/dem_sau_ray"

# Hàm này tự động chạy khi kết nối server thành công
def on_connect(client, userdata, flags, rc):
    print("✅ [APP NÔNG DÂN] Đã kết nối Đám mây thành công! Đang chờ dữ liệu gửi về...")
    client.subscribe(MQTT_TOPIC) # Lắng nghe kênh này

# Hàm này tự động chạy mỗi khi có tin nhắn bay tới
def on_message(client, userdata, msg):
    so_luong = msg.payload.decode()
    print(f"📱 CẢNH BÁO TỪ RUỘNG LÚA: Phát hiện {so_luong} con sâu rầy qua vạch!")

# Khởi tạo và chạy App liên tục
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_forever() # Treo máy ở đây để liên tục hứng dữ liệu