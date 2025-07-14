import pickle
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# Veri setini yükle
with open("embeddings_dataset.pkl", "rb") as f:
    data = pickle.load(f)

X = data["X"]
y = data["y"]

# 🎛️ Parametreler — buradan değiştir!
kernel_type = "rbf"       # 'linear', 'rbf', 'poly', 'sigmoid'
test_size_ratio = 0.2      # %20 test verisi
svm_C = 1.0                  # C parametresi (düzenleme gücü)

# Veriyi ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, random_state=42)

# SVM modelini oluştur ve eğit
clf = svm.SVC(kernel=kernel_type, C=svm_C, probability=True)
clf.fit(X_train, y_train)

# Tahmin yap
y_pred = clf.predict(X_test)

# 🎯 Classification Report
print("📋 Classification Report:")
print(classification_report(y_test, y_pred))

# 📊 Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)

# Grafik
disp.plot(cmap="Blues")
plt.title(f"SVM Confusion Matrix (kernel='{kernel_type}', C={svm_C}, test_size={test_size_ratio})")
plt.grid(False)
plt.tight_layout()
plt.show()
