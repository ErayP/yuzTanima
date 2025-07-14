# Yüz Tanıma Sistemi

🎥 [Demo Videosu](https://www.youtube.com/shorts/IONLdlENiBU)

##  Proje Hakkında

Bu proje, **face_recognition** kütüphanesini kullanarak **özellik vektörleri çıkarır** ve bu vektörleri kullanarak **SVM (Support Vector Machine)** ile yüz tanıma işlemi gerçekleştirir. Sistem, kamera üzerinden canlı yüz tanıma ve kayıtlı yüzlerle eşleştirme yapar.

##  Özellikler

- Gerçek zamanlı yüz tanıma (kamera üzerinden)
- Özellik çıkarımı için face_recognition 
- Sınıflandırıcı olarak SVM kullanımı
- Veri artırma (augmentation) desteği
- PyTorch + OpenCV entegrasyonu

## ⚙️ Kurulum

```bash
git clone https://github.com/kullaniciAdi/yuz-tanima-projesi.git
cd yuzTanima
pip install -r requirements.txt
