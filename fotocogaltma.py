import os
import cv2
import numpy as np

def rotate(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h))

def adjust_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v = v.astype(np.int16)
    v = np.clip(v + value, 0, 255).astype(np.uint8)

    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def augment_image(img):
    variants = []
    variants.append(img) 
    variants.append(cv2.flip(img, 1)) 
    variants.append(rotate(img, 10))
    variants.append(rotate(img, -10))
    variants.append(adjust_brightness(img, 30))
    variants.append(adjust_brightness(img, -30))
    return variants

input_root = "faces_dataset"
output_root = "faces_augmented"

os.makedirs(output_root, exist_ok=True)

for person_name in os.listdir(input_root):
    input_folder = os.path.join(input_root, person_name)
    output_folder = os.path.join(output_root, person_name)
    os.makedirs(output_folder, exist_ok=True)

    for idx, filename in enumerate(os.listdir(input_folder)):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Atlandı: {img_path}")
            continue

        augmented = augment_image(img)

        for i, aug in enumerate(augmented):
            out_path = os.path.join(output_folder, f"{idx}_{i}.jpg")
            cv2.imwrite(out_path, aug)

print("✅ Augmentasyon tamamlandı! Sonuç: faces_augmented klasöründe.")
