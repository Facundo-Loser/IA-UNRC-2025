import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, hamming_loss

# ------------------------
# Generar dataset sintético
# ------------------------
# Atributos: peso (g), dulzor (0-10), acidez (0-10), tamaño (0-pequeño, 1-medio, 2-grande)
np.random.seed(42)
n_samples = 200

peso = np.random.normal(150, 50, n_samples)
dulzor = np.random.randint(1, 11, n_samples)
acidez = np.random.randint(1, 11, n_samples)
tamano = np.random.randint(0, 3, n_samples)

X = np.column_stack((peso, dulzor, acidez, tamano))

# Etiquetas multi-label: [dulce, ácido, rojo, cítrico, grande]
y = np.zeros((n_samples, 5), dtype=int)

# Reglas simples para generar etiquetas
for i in range(n_samples):
    if dulzor[i] > 6:
        y[i,0] = 1   # dulce
    if acidez[i] > 6:
        y[i,1] = 1   # ácido
    if peso[i] > 180:
        y[i,4] = 1   # grande
    # rojo y cítrico aleatorios según probabilidad
    y[i,2] = np.random.binomial(1, 0.3)  # rojo
    y[i,3] = np.random.binomial(1, 0.2)  # cítrico

# ------------------------
# Separar entrenamiento y validación
# ------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# ------------------------
# Entrenar modelo multi-label
# ------------------------
base_clf = LogisticRegression(max_iter=500)
multi_clf = OneVsRestClassifier(base_clf)
multi_clf.fit(X_train, y_train)

# Predicciones
y_pred = multi_clf.predict(X_val)

# ------------------------
# Evaluación
# ------------------------
# Accuracy exacta (todas las etiquetas correctas)
print("Exact Match Accuracy:", accuracy_score(y_val, y_pred))

# F1 score promedio por etiqueta
print("F1 score (macro):", f1_score(y_val, y_pred, average='macro'))

# Hamming loss: proporción de etiquetas incorrectas
print("Hamming Loss:", hamming_loss(y_val, y_pred))
