import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------
# Generar dataset sintético
# ------------------------
np.random.seed(42)

# Parámetros: media y desviación estándar de IBU y RMS para cada estilo
# "ibu": (media, desviación estándar)
# "rms": (media, desviación estándar)
# "n": cantidad de muestras de ese estilo
# IBU (International Bitterness Units)  mide cuán amarga es una cerveza
# SRM (Standard Reference Method) mide el color/oscura de la cerveza
parametros = {
    "Lager":     {"ibu": (15, 5), "rms": (20, 5), "n": 100},
    "Stout":     {"ibu": (45, 5), "rms": (60, 10), "n": 100},
    "IPA":       {"ibu": (35, 5), "rms": (50, 5), "n": 100},
    "Scottish":  {"ibu": (20, 5), "rms": (30, 5), "n": 100}
}

X_list = []
y_list = []
clases = {"Lager":0, "Stout":1, "IPA":2, "Scottish":3}

for estilo, param in parametros.items():
    # esto genera datos aleatorios (dist normal) centrados en esas medias
    ibu = np.random.normal(param["ibu"][0], param["ibu"][1], param["n"])
    rms = np.random.normal(param["rms"][0], param["rms"][1], param["n"])
    X_list.append(np.column_stack((ibu, rms)))
    y_list.append(np.full(param["n"], clases[estilo]))

X = np.vstack(X_list)
y = np.hstack(y_list)

# Separar entrenamiento y validación (30% validacion = 120 datos con n=100 cada clase)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ------------------------
# Clasificador base
# ------------------------
base_clf = LogisticRegression(max_iter=500, random_state=42)

# One-vs-Rest
ovr_clf = OneVsRestClassifier(base_clf).fit(X_train, y_train)
y_pred_ovr = ovr_clf.predict(X_val)

# One-vs-One
ovo_clf = OneVsOneClassifier(base_clf).fit(X_train, y_train)
y_pred_ovo = ovo_clf.predict(X_val)

# Softmax (multinomial)
softmax_clf = LogisticRegression(max_iter=500, multi_class='multinomial', solver='lbfgs', random_state=42)
softmax_clf.fit(X_train, y_train)
y_pred_softmax = softmax_clf.predict(X_val)

# ------------------------
# Función de evaluación
# ------------------------
def evaluar(y_true, y_pred, nombre_modelo):
    print(f"\n=== {nombre_modelo} ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision (macro):", precision_score(y_true, y_pred, average='macro'))
    print("Recall (macro):", recall_score(y_true, y_pred, average='macro'))
    print("F1-score (macro):", f1_score(y_true, y_pred, average='macro'))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Matriz de Confusión - {nombre_modelo}")
    plt.show()

# Evaluar todos
evaluar(y_val, y_pred_ovr, "One-vs-Rest")
evaluar(y_val, y_pred_ovo, "One-vs-One")
evaluar(y_val, y_pred_softmax, "Softmax")
