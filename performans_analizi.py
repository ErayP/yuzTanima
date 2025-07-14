import os
import pickle
import cv2
import numpy as np
import face_recognition
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_distances

# --- 1) Embedding’leri yükle ---
with open("embeddings_dataset_with_groups.pkl", "rb") as f:
    data = pickle.load(f)
X, y = data["X"], data["y"]

# --- 2) Gallery / Probe ayrımı ---
known = defaultdict(list)
for emb, person in zip(X, y):
    known[person].append(emb)

gallery_embs, gallery_labels, probe_embs, probe_labels = [], [], [], []
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
persons = list(centroids.keys())

# --- 4) Bilinmeyen embedding’leri (200 örnek) ---
base = "UTKFace"
all_fns = [fn for fn in os.listdir(base) if fn.lower().endswith((".jpg", ".png"))]
test_fns = all_fns[:200]
unknown_embs = []
for fn in test_fns:
    img = cv2.imread(os.path.join(base, fn))
    if img is None:
        continue
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")
    encs  = face_recognition.face_encodings(
        rgb, known_face_locations=boxes, num_jitters=1
    )
    if encs:
        unknown_embs.append(encs[0])

# --- 5) Fixed threshold at F1 peak ≈0.085 ---
threshold = 0.085
def recognize(emb, th=threshold):
    cd = cosine_distances([emb], [centroids[p] for p in persons])[0]
    idx = np.argmin(cd)
    return persons[idx] if cd[idx] < th else "Unknown"

# --- 6) Son metrikler ---
y_true = probe_labels + ["Unknown"] * len(unknown_embs)
y_pred = [recognize(e) for e in probe_embs] + [recognize(e) for e in unknown_embs]
labels = persons + ["Unknown"]

print("=== Classification Report ===")
print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
print("\n=== Confusion Matrix ===")
print(pd.DataFrame(
    confusion_matrix(y_true, y_pred, labels=labels),
    index=labels, columns=labels
))

# --- 7) Performance vs Threshold grafiği ---
ths = np.linspace(0.0, 0.2, 101)
known_recs, unk_recs, f1_scores = [], [], []

for th in ths:
    # Bilinen recall
    yk = []
    for emb, true_p in zip(probe_embs, probe_labels):
        cd = cosine_distances([emb], [centroids[p] for p in persons])[0]
        yk.append(persons[np.argmin(cd)] if cd.min() < th else "Unknown")
    recs = [sum(probe_labels[i] == yk[i] for i in range(len(probe_labels)) if probe_labels[i]==p)
            / sum(1 for lab in probe_labels if lab==p)
            for p in persons]
    known_recs.append(np.mean(recs))

    # Unknown recall
    yu = []
    for emb in unknown_embs:
        cd = cosine_distances([emb], [centroids[p] for p in persons])[0]
        yu.append(persons[np.argmin(cd)] if cd.min() < th else "Unknown")
    unk_recs.append(sum(l=="Unknown" for l in yu) / len(yu))

    # F1 score
    kr, ur = known_recs[-1], unk_recs[-1]
    f1_scores.append(2 * kr * ur / (kr + ur + 1e-8))

plt.figure(figsize=(10, 5))
plt.plot(ths, known_recs, label="Known Recall", linewidth=2)
plt.plot(ths, unk_recs,   label="Unknown Recall", linewidth=2)
plt.plot(ths, f1_scores,  label="F1 Score", linewidth=2)
best_idx = np.argmax(f1_scores)
plt.axvline(ths[best_idx], color="gray", linestyle="--", label=f"Best Threshold={ths[best_idx]:.3f}")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Performance vs Threshold")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
