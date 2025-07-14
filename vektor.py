import os
import face_recognition
import numpy as np
import pickle

dataset_path = "faces_augmented"
X = []
y = []

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    for filename in os.listdir(person_folder):
        file_path = os.path.join(person_folder, filename)
        image = face_recognition.load_image_file(file_path)
        face_locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image, face_locations)

        if encodings:
            X.append(encodings[0])
            y.append(person_name)
        else:
            print(f"⚠️ Yüz bulunamadı: {file_path}")

# Model için verileri kaydet
with open("embeddings_dataset.pkl", "wb") as f:
    pickle.dump({"X": X, "y": y}, f)

print("✅ Veri seti hazır: embeddings_dataset.pkl")
