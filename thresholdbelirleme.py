import os
import pickle
import cv2
import numpy as np
import face_recognition
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_distances
import pandas as pd

# --- 1) Embedding’leri yükle ---
with open("embeddings_dataset.pkl", "rb") as f:
    data = pickle.load(f)
X, y = data["X"], data["y"]

print(f"Toplam embedding sayısı: {len(X)}")
print(f"Benzersiz kişi sayısı: {len(set(y))}")

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

print(f"Gallery örnek sayısı: {len(gallery_embs)}")
print(f"Probe örnek sayısı:   {len(probe_embs)}")

# --- 3) Centroid’leri hesapla ---
gallery_known = defaultdict(list)
for emb, person in zip(gallery_embs, gallery_labels):
    gallery_known[person].append(emb)
centroids = {p: np.mean(embs, axis=0) for p, embs in gallery_known.items()}

print("Örnek centroid boyutu:", list(centroids.values())[0].shape)
print("Centroid sayısı:", len(centroids))

# --- 4) Threshold optimizasyonu ---
persons = list(centroids.keys())
dists, true_labels = [], []
for emb, true_person in zip(probe_embs, probe_labels):
    cd = cosine_distances([emb], [centroids[p] for p in persons])[0]
    min_idx = np.argmin(cd)
    dists.append(cd[min_idx])
    true_labels.append(1 if persons[min_idx] == true_person else 0)

precision, recall, thresholds = precision_recall_curve(true_labels, -np.array(dists))
f1 = 2 * precision * recall / (precision + recall + 1e-8)
best_idx = np.nanargmax(f1)
threshold = -thresholds[best_idx]
print(f"Optimize edilmiş eşik: {threshold:.3f}")

# --- 5) UTKFace’ten “Unknown” embedding’leri hazırla ---
base = "UTKFace"  # Kendi klasör yolunuza göre düzenleyin
unknown_embs = []
for filename in os.listdir(base):
    if not filename.lower().endswith((".jpg", ".png")):
        continue
    img = cv2.imread(os.path.join(base, filename))
    if img is None:
        continue
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")
    encs = face_recognition.face_encodings(
        rgb, known_face_locations=boxes, num_jitters=1
    )
    if encs:
        unknown_embs.append(encs[0])
    if len(unknown_embs) >= 100:
        break
print(f"UTKFace’ten alınan Unknown embedding sayısı: {len(unknown_embs)}")

# --- recognize fonksiyonu (iki argümanlı) ---
def recognize(emb, th):
    cd = cosine_distances([emb], [centroids[p] for p in persons])[0]
    idx = np.argmin(cd)
    return persons[idx] if cd[idx] < th else "Unknown"

# --- 6) Karar metrikleri ---
y_true = probe_labels + ["Unknown"] * len(unknown_embs)
y_pred = [recognize(e, threshold) for e in probe_embs] + [recognize(e, threshold) for e in unknown_embs]

labels = persons + ["Unknown"]
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

cm = confusion_matrix(y_true, y_pred, labels=labels)
print("\n=== Confusion Matrix ===")
print(pd.DataFrame(cm, index=labels, columns=labels))

# --- 7) Threshold sweep ile optimal dengeyi tekrar bulun ---
def eval_threshold(th):
    pred_known = [recognize(e, th) for e in probe_embs]
    pred_unk   = [recognize(e, th) for e in unknown_embs]
    # Bilinen ortalama recall
    recs = []
    for p in persons:
        idxs = [i for i, lab in enumerate(probe_labels) if lab == p]
        recs.append(sum(probe_labels[i]==pred_known[i] for i in idxs)/len(idxs))
    known_rec = np.mean(recs)
    # Unknown recall
    unk_rec = sum(p=="Unknown" for p in pred_unk)/len(pred_unk)
    return known_rec, unk_rec

ths = np.linspace(0.1, 0.6, 51)
results = [eval_threshold(t) for t in ths]
known_recs, unk_recs = zip(*results)
f1s = 2*np.array(known_recs)*np.array(unk_recs)/(np.array(known_recs)+np.array(unk_recs)+1e-8)
best = np.argmax(f1s)
print(f"\nOptimal eşik ≈ {ths[best]:.3f}  (known_recall={known_recs[best]:.3f}, unk_recall={unk_recs[best]:.3f}, F1={f1s[best]:.3f})")
