from flask import Flask, Response, request, jsonify
from picamera2 import Picamera2
import cv2
import threading
import time
from gpiozero import LED, Buzzer, OutputDevice

app = Flask(__name__)

# ===== Donanım ayarları =====
GREEN_PIN  = 17   # Yeşil LED
RED_PIN    = 27   # Kırmızı LED
BUZZER_PIN = 22   # Buzzer
SOL_PIN    = 23   # Röle IN (solenoid)

green_led = LED(GREEN_PIN)
red_led   = LED(RED_PIN)
buzzer    = Buzzer(BUZZER_PIN)
solenoid  = OutputDevice(SOL_PIN, active_high=False, initial_value=False)

# ===== Picamera2 ayarı =====
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration({"size": (640, 480)}))
picam2.start()
time.sleep(2)

# ===== Video akış üreticisi =====
def generate_mjpeg():
    while True:
        frame = picam2.capture_array()
        # JPEG'e sıkıştır
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        # multipart sınırı ve jalbyte
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

# ===== Video feed endpoint =====
@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ===== Durum ayarlama endpoint =====
lock_timer = None
lock_lock = threading.Lock()

@app.route('/set_status', methods=['POST'])
def set_status():
    global lock_timer
    data = request.get_json() or {}
    status = int(data.get('status', 0))

    # Eğer 1 geldiyse "kilit aç"
    if status == 1:
        # Thread içinde çalıştır, bloklamasın
        def open_lock():
            with lock_lock:
                # LED ve buzzer uyarısı
                green_led.on()
                buzzer.beep(on_time=1.0, off_time=0, n=1, background=True)

                # Solenoid 2 saniye aktif
                solenoid.on()
                time.sleep(2)
                solenoid.off()

                green_led.off()

        # Önce varsa eski timer'ı durdur
        if lock_timer and lock_timer.is_alive():
            lock_timer.cancel()  # yoksa biriken timer'lar olabilir
        lock_timer = threading.Timer(0, open_lock)
        lock_timer.start()

    # 0 geldiyse "kilit kapat" - hemen kapat
    elif status == 0:
        with lock_lock:
            solenoid.off()
            green_led.off()
            red_led.on()
            # Kırmızı 1 saniye sonra sönsün
            threading.Timer(1.0, red_led.off).start()

    return jsonify({'result': 'OK', 'status': status})

# ===== Anasayfa yönlendirme =====
@app.route('/')
def index():
    return """
    <html><body>
    <h1>Video feed:</h1>
    <img src="/video_feed">
    </body></html>
    """

if __name__ == '__main__':
    # 0.0.0.0 → tüm arabirimlerden erişilir
    app.run(host='0.0.0.0', port=5000, threaded=True)
