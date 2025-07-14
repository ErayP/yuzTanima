import cv2
import pickle
import numpy as np
import face_recognition
from sklearn.metrics.pairwise import cosine_distances
from collections import defaultdict

# --- 1) Centroid’leri ve kişileri yükle ---
with open("embeddings_dataset_with_groups.pkl", "rb") as f:
    data = pickle.load(f)
X, y = data["X"], data["y"]
known = defaultdict(list)
for emb, person in zip(X, y):
    known[person].append(emb)
centroids = {p: np.mean(known[p], axis=0) for p in known}
persons = list(centroids.keys())

# --- 2) Eşik (en iyi F1 tepe değeri) ---
threshold = 0.085

# --- 3) Recognize fonksiyonu: isim ve yüzde güven döner ---
def recognize_with_conf(emb, th=threshold):
    dists = cosine_distances([emb], [centroids[p] for p in persons])[0]
    idx   = np.argmin(dists)
    dist  = dists[idx]
    if dist < th:
        # Güven skoru: 1 - (dist / th), yüzdeye çevir
        conf = max(0.0, 1 - (dist / th)) * 100
        return persons[idx], conf
    else:
        return "Tanınmayan kişi", 0.0

# --- 4) Kamera ve canlı döngü ---
cap = cv2.VideoCapture(0)
print("🎥 Başlatıldı. ‘q’ ile çıkın.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")
    encs  = face_recognition.face_encodings(
        rgb, known_face_locations=boxes, num_jitters=1
    )

    for (top, right, bottom, left), emb in zip(boxes, encs):
        name, conf = recognize_with_conf(emb)
        label = f"{name} ({conf:.0f}%)"

        # Çizimler
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, label,
                    (left, bottom + 20),  # kutunun altına yazdırmak için
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255,255,255), 2)

    cv2.imshow("Open-Set Yüz Tanıma", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
