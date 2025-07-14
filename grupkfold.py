import pickle
import numpy as np
from collections import defaultdict
from sklearn.metrics import precision_recall_curve
from sklearn.metrics.pairwise import cosine_distances
import cv2
import face_recognition

# --- 1) Embedding’leri yükle ---
with open("embeddings_dataset_with_groups.pkl", "rb") as f:
    data = pickle.load(f)
X, y = data["X"], data["y"]

# --- 2) Gallery / Probe ayrımı ---
known = defaultdict(list)
for emb, person in zip(X, y):
    known[person].append(emb)

gallery_embs, gallery_labels = [], []
probe_embs, probe_labels     = [], []

for person, embs in known.items():
    split = max(1, int(len(embs) * 0.2))
    gallery_embs += embs[:split]
    gallery_labels += [person] * split
    probe_embs   += embs[split:]
    probe_labels += [person] * (len(embs) - split)

# --- 3) Centroid’leri hesapla ---
gallery_known = defaultdict(list)
for emb, person in zip(gallery_embs, gallery_labels):
    gallery_known[person].append(emb)
centroids = {p: np.mean(embs, axis=0) for p, embs in gallery_known.items()}

# --- 4) Eşik (threshold) optimizasyonu ---
dists, true_labels = [], []
persons = list(centroids.keys())

for emb, true_person in zip(probe_embs, probe_labels):
    cd = cosine_distances([emb], [centroids[p] for p in persons])[0]
    min_idx = np.argmin(cd)
    dists.append(cd[min_idx])
    true_labels.append(1 if persons[min_idx] == true_person else 0)

precision, recall, thresholds = precision_recall_curve(true_labels, -np.array(dists))
f1 = 2 * precision * recall / (precision + recall + 1e-8)
best_idx = np.argmax(f1)
threshold = -thresholds[best_idx]
print(f">>> Optimize edilmiş eşik: {threshold:.3f}")

# --- 5) Canlı tanıma fonksiyonu ---
def recognize(emb):
    cd = cosine_distances([emb], [centroids[p] for p in persons])[0]
    idx = np.argmin(cd)
    return persons[idx] if cd[idx] < threshold else "Unknown"

# --- 6) Kameradan canlı çalışma ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Kamera açılamadı")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = frame[:, :, ::-1]
    # Yüz kutularını tespit et
    boxes = face_recognition.face_locations(rgb)
    # Düzgün call: bounding boxes keyword argümanı olarak veriliyor
    encodings = face_recognition.face_encodings(
        rgb,
        known_face_locations=boxes,
        num_jitters=1
    )

    for (top, right, bottom, left), emb in zip(boxes, encodings):
        name = recognize(emb)
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, name, (left, top-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Open-Set Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
