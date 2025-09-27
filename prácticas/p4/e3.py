import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Dataset sintético (mismo que antes)
X, y = make_moons(n_samples=100, noise=0.35, random_state=42)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Diferentes kernels y valores de C
kernels = ["linear", "poly", "rbf", "sigmoid"]
C_values = [0.1, 1, 10]

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "Accuracy": round(accuracy_score(y_test, y_pred), 2),
        "Precision": round(precision_score(y_test, y_pred), 2),
        "Recall": round(recall_score(y_test, y_pred), 2),
        "F1": round(f1_score(y_test, y_pred), 2),
    }

# Guardar resultados
results = []

for kernel in kernels:
    for C in C_values:
        svm = SVC(kernel=kernel, C=C, gamma="scale")
        svm.fit(X_train_scaled, y_train)
        metrics = evaluate_model(svm, X_test_scaled, y_test)
        results.append((kernel, C, metrics))
        print(f"Kernel={kernel}, C={C} -> {metrics}")

# Función para graficar frontera de decisión
def plot_decision_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.show()

# Ejemplo de visualización con RBF y C=1
svm_rbf = SVC(kernel="rbf", C=1, gamma="scale")
svm_rbf.fit(X_train_scaled, y_train)
plot_decision_boundary(svm_rbf, X_train_scaled, y_train, "SVM con kernel RBF, C=1")
