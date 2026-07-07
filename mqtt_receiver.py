import paho.mqtt.client as mqtt

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "nongnghiep_iot/thuctap/dem_sau_ray"

def on_connect(client, userdata, flags, rc):
    print("✅ [APP NÔNG DÂN] Đã kết nối Đám mây thành công! Đang chờ dữ liệu...")
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    so_luong = msg.payload.decode()
    print(f"📱 CẢNH BÁO TỪ RUỘNG LÚA: Phát hiện TỔNG CỘNG {so_luong} con sâu rầy!")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_forever()