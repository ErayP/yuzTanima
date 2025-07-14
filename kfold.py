import pickle
import numpy as np
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Veri setini yükle
with open("embeddings_dataset.pkl", "rb") as f:
    data = pickle.load(f)

X = np.array(data["X"])
y = np.array(data["y"])

# K-Fold Ayarları
k = 5  # 5 parçaya böl (5-fold cross validation)
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

accuracies = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # SVM modeli
    model = svm.SVC(kernel="linear", probability=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Doğruluk skoru
    accuracy = np.mean(y_pred == y_test)
    accuracies.append(accuracy)

    print(f"\n📊 Fold {fold} - Doğruluk: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))

    # Confusion matrix göster
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap="Blues")
    plt.title(f"Fold {fold} Confusion Matrix")
    plt.grid(False)
    plt.show()

# Ortalama başarı
mean_acc = np.mean(accuracies)
print(f"\n✅ Ortalama Doğruluk (K={k}): {mean_acc:.2f}")
