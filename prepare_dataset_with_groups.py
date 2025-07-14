import os
import face_recognition
import pickle

dataset_path = "faces_augmented"
X = []
y = []
groups = []

group_id = 0

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
            groups.append(group_id)  # aynı kişi için aynı grup
        else:
            print(f"⚠️ Yüz bulunamadı: {file_path}")

    group_id += 1  # sonraki kişiye geçerken grup ID değişir

# Kaydet
with open("embeddings_dataset_with_groups.pkl", "wb") as f:
    pickle.dump({"X": X, "y": y, "groups": groups}, f)

print("✅ Veri hazır! 'embeddings_dataset_with_groups.pkl' oluşturuldu.")
