import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score, hamming_loss

np.random.seed(42)

# ------------------------
# Generar dataset sintético
# ------------------------
n_samples = 200

# Características: peso (g), color R,G,B (0-255), dulzor (0-10), acidez (0-10), tamaño (0-pequeño, 1-medio, 2-grande)
peso = np.random.normal(150, 50, n_samples)
color_r = np.random.randint(50, 256, n_samples)
color_g = np.random.randint(50, 256, n_samples)
color_b = np.random.randint(50, 256, n_samples)
dulzor = np.random.randint(1, 11, n_samples)
acidez = np.random.randint(1, 11, n_samples)
tamano = np.random.randint(0, 3, n_samples)

X = np.column_stack((peso, color_r, color_g, color_b, dulzor, acidez, tamano))

# Etiquetas multi-label: manzana, naranja, limón, pera
y = np.zeros((n_samples, 4), dtype=int)

for i in range(n_samples):
    # Reglas simples para asignar frutas
    if dulzor[i] > 6 and acidez[i] <= 6 and tamano[i] <= 1:
        y[i,0] = 1  # manzana
    if dulzor[i] > 5 and acidez[i] > 5 and color_r[i] > 150:
        y[i,1] = 1  # naranja
    if acidez[i] > 7 and dulzor[i] <= 5:
        y[i,2] = 1  # limón
    if dulzor[i] > 4 and acidez[i] <= 6 and tamano[i] >= 1:
        y[i,3] = 1  # pera

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
print("Exact Match Accuracy:", accuracy_score(y_val, y_pred))
print("F1 score (macro):", f1_score(y_val, y_pred, average='macro'))
print("Hamming Loss:", hamming_loss(y_val, y_pred))

# ------------------------
# Mostrar algunas predicciones
# ------------------------
for i in range(5):
    print("\nCaracterísticas:", X_val[i])
    print("Etiquetas reales:", y_val[i])
    print("Predicción:", y_pred[i])
