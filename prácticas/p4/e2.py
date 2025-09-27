import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Generamos dataset sintético
X, y = make_moons(n_samples=100, noise=0.35, random_state=42)

# 2. Dividimos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Escalamos las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Probar diferentes k
k_values = [1, 3, 5, 15, 30]
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)

    print(f"Resultados para k={k}:")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
    print("Precision:", round(precision_score(y_test, y_pred), 2))
    print("Recall:", round(recall_score(y_test, y_pred), 2))
    print("F1 Score:", round(f1_score(y_test, y_pred), 2))
    print("-"*30)

# 5. Visualizar decisión
def plot_decision_boundary(clf, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:,0], X[:,1], c=y, s=50, edgecolor='k')
    plt.title(title)
    plt.show()

# Visualizamos para k=1 y k=15 como ejemplo
knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(X_train_scaled, y_train)
plot_decision_boundary(knn1, X_train_scaled, y_train, "k=1")

knn15 = KNeighborsClassifier(n_neighbors=15)
knn15.fit(X_train_scaled, y_train)
plot_decision_boundary(knn15, X_train_scaled, y_train, "k=15")


"""
Conclusiones:
con k=1 es muy preciso pero no se ajusta tan bien a nuevos datos (overfitting)
con k=15,30 tiene un recall de 1.0 osea que baja la precision (underfitting)
"""
