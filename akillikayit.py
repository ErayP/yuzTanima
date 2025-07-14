import cv2
import os
import time

# === Ayarlar ===
yuz_klasoru = "faces_dataset"
kamera_genislik = 640
kamera_yukseklik = 480
bekleme_suresi = 1.0  # pozlar arası bekleme

# === Giriş al ===
kullanici = input("👤 Kullanıcı adı: ")
kayit_yolu = os.path.join(yuz_klasoru, kullanici)
"""os.makedirs(kayit_yolu, exist_ok=True)"""

# === Kamera ayarları ===
kamera = cv2.VideoCapture(0)
kamera.set(cv2.CAP_PROP_FRAME_WIDTH, kamera_genislik)
kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, kamera_yukseklik)

# === Yüz algılama modeli ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# === Pozlar ve adetleri ===
pozlar = [
    ("Duz bakin", 3),
    ("Saga bakin", 3),
    ("Sola bakin", 3),
    ("Yukari bakin", 2),
    ("Aşagi bakin", 2)
]

foto_id = 0
print("📸 Yönlendirmelere göre poz verin. Yüz algılanınca otomatik kaydedilir.")

for poz_ad, adet in pozlar:
    sayac = 0
    print(f"\n👉 {poz_ad} ({adet} foto)")
    time.sleep(2)

    son_kayit_zamani = time.time()

    while sayac < adet:
        ret, kare = kamera.read()
        if not ret:
            print("🚫 Kamera akışı kesildi.")
            break

        gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
        yuzler = face_cascade.detectMultiScale(gri, scaleFactor=1.1, minNeighbors=5)

        # Sadece en büyük yüz alınır
        if len(yuzler) > 0:
            yuzler = sorted(yuzler, key=lambda r: r[2]*r[3], reverse=True)
            x, y, w, h = yuzler[0]

            # Bekleme süresi kontrolü
            if time.time() - son_kayit_zamani >= bekleme_suresi:
                yuz = kare[y:y+h, x:x+w]
                yuz = cv2.resize(yuz, (224, 224))
                dosya = os.path.join(kayit_yolu, f"{foto_id}.jpg")
                cv2.imwrite(dosya, yuz)
                print(f"✅ {dosya} kaydedildi.")
                sayac += 1
                foto_id += 1
                son_kayit_zamani = time.time()

                # Kaydettikten sonra 1 saniye kare göstermeye devam etsin
                time.sleep(0.5)

        # Görsel yönlendirme
        cv2.putText(kare, poz_ad, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow("Kayıt Ekranı", kare)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

kamera.release()
cv2.destroyAllWindows()
print("\n✅ Tüm pozlar başarıyla kaydedildi.")
