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

# Graficar todas las combinaciones de kernel y C
for kernel in kernels:
    for C in C_values:
        svm = SVC(kernel=kernel, C=C, gamma="scale")
        svm.fit(X_train_scaled, y_train)
        title = f"SVM kernel={kernel}, C={C}"
        plot_decision_boundary(svm, X_train_scaled, y_train, title)


"""
*noise=0.35*
Kernel=linear, C=0.1 -> {'Accuracy': 0.83, 'Precision': 0.67, 'Recall': 1.0, 'F1': 0.8}
Kernel=linear, C=1 -> {'Accuracy': 0.83, 'Precision': 0.67, 'Recall': 1.0, 'F1': 0.8}
Kernel=linear, C=10 -> {'Accuracy': 0.83, 'Precision': 0.67, 'Recall': 1.0, 'F1': 0.8}
Kernel=poly, C=0.1 -> {'Accuracy': 0.63, 'Precision': 0.48, 'Recall': 1.0, 'F1': 0.65}
Kernel=poly, C=1 -> {'Accuracy': 0.7, 'Precision': 0.53, 'Recall': 1.0, 'F1': 0.69}
Kernel=poly, C=10 -> {'Accuracy': 0.7, 'Precision': 0.53, 'Recall': 1.0, 'F1': 0.69}
Kernel=rbf, C=0.1 -> {'Accuracy': 0.83, 'Precision': 0.67, 'Recall': 1.0, 'F1': 0.8}
Kernel=rbf, C=1 -> {'Accuracy': 0.83, 'Precision': 0.67, 'Recall': 1.0, 'F1': 0.8}
Kernel=rbf, C=10 -> {'Accuracy': 0.8, 'Precision': 0.64, 'Recall': 0.9, 'F1': 0.75}
Kernel=sigmoid, C=0.1 -> {'Accuracy': 0.83, 'Precision': 0.67, 'Recall': 1.0, 'F1': 0.8}
Kernel=sigmoid, C=1 -> {'Accuracy': 0.77, 'Precision': 0.6, 'Recall': 0.9, 'F1': 0.72}
Kernel=sigmoid, C=10 -> {'Accuracy': 0.67, 'Precision': 0.5, 'Recall': 0.9, 'F1': 0.64}


*si incrementamos el ruido: noise=0.60* (parece costarle mas separar bien)
Kernel=linear, C=0.1 -> {'Accuracy': 0.73, 'Precision': 0.56, 'Recall': 1.0, 'F1': 0.71}
Kernel=linear, C=1 -> {'Accuracy': 0.73, 'Precision': 0.56, 'Recall': 1.0, 'F1': 0.71}
Kernel=linear, C=10 -> {'Accuracy': 0.73, 'Precision': 0.56, 'Recall': 1.0, 'F1': 0.71}
Kernel=poly, C=0.1 -> {'Accuracy': 0.47, 'Precision': 0.38, 'Recall': 1.0, 'F1': 0.56}
Kernel=poly, C=1 -> {'Accuracy': 0.53, 'Precision': 0.42, 'Recall': 1.0, 'F1': 0.59}
Kernel=poly, C=10 -> {'Accuracy': 0.57, 'Precision': 0.43, 'Recall': 1.0, 'F1': 0.61}
Kernel=rbf, C=0.1 -> {'Accuracy': 0.57, 'Precision': 0.43, 'Recall': 1.0, 'F1': 0.61}
Kernel=rbf, C=1 -> {'Accuracy': 0.73, 'Precision': 0.56, 'Recall': 1.0, 'F1': 0.71}
Kernel=rbf, C=10 -> {'Accuracy': 0.73, 'Precision': 0.56, 'Recall': 0.9, 'F1': 0.69}
Kernel=sigmoid, C=0.1 -> {'Accuracy': 0.73, 'Precision': 0.56, 'Recall': 1.0, 'F1': 0.71}
Kernel=sigmoid, C=1 -> {'Accuracy': 0.87, 'Precision': 0.75, 'Recall': 0.9, 'F1': 0.82}
Kernel=sigmoid, C=10 -> {'Accuracy': 0.57, 'Precision': 0.42, 'Recall': 0.8, 'F1': 0.55}
"""
