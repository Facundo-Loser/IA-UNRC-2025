from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.ndimage import zoom, shift

def modificar_imagen(img, scale=0.5, desplazamiento=(5, 5)):
    """
    img: vector de 784 (28x28)
    scale: factor de reducción (ej. 0.5 = más chico)
    desplazamiento: (dx, dy) píxeles que se corre
    """
    img = img.reshape(28, 28)

    # 1) Escalar (hacer el número más chico)
    img_resized = zoom(img, scale)

    # 2) Crear un lienzo vacío de 28x28
    canvas = np.zeros((28, 28))
    h, w = img_resized.shape
    canvas[:h, :w] = img_resized  # pego en la esquina superior-izquierda

    # 3) Desplazar en canvas
    img_shifted = shift(canvas, desplazamiento, cval=0.0)

    return img_shifted.flatten()

# obtener el dataset de openml
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"].astype(int)

# Normalizar datos (0-1) (están en esacala de grises)
X = X / 255.0

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar red neuronal (Multilayer perceptron)
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64), # 2 capas ocultas
    activation="relu",
    solver="adam",
    alpha=1e-1, # regularización L2 (Ridge)
    learning_rate_init=0.001,
    max_iter=20, # cantidad de iteraciones
    verbose=True,
    early_stopping=True,   # corta antes si no mejora
    random_state=42
)

mlp.fit(X_train, y_train)

# Evaluación (imagenes modificadas para tener un numero en un extremo de la esquina superior izquierda)
X_test_mod = np.array([
    modificar_imagen(img, scale=0.6, desplazamiento=(6, -4))
    for img in X_test[:200]  # probamos con 200 imágenes para que sea rápido
])

y_test_mod = y_test[:200]

y_pred_mod = mlp.predict(X_test_mod)

print("\nAccuracy en imágenes modificadas:", accuracy_score(y_test_mod, y_pred_mod))
print("\nReporte en imágenes modificadas:\n", classification_report(y_test_mod, y_pred_mod))

# Mostrar algunas imágenes nuevas y sus predicciones
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i, ax in enumerate(axes):
    ax.imshow(X_test_mod[i].reshape(28, 28), cmap="gray")
    ax.set_title(f"Pred: {y_pred_mod[i]}")
    ax.axis("off")
plt.show()

"""
Resultados: (es bastante malo)

Iteration 1, loss = 0.48614256
Validation score: 0.937143
Iteration 2, loss = 0.23490920
Validation score: 0.951071
Iteration 3, loss = 0.18934387
Validation score: 0.956786
Iteration 4, loss = 0.16491265
Validation score: 0.963571
Iteration 5, loss = 0.14850437
Validation score: 0.966964
Iteration 6, loss = 0.13598042
Validation score: 0.968929
Iteration 7, loss = 0.12984044
Validation score: 0.971607
Iteration 8, loss = 0.12200973
Validation score: 0.972679
Iteration 9, loss = 0.11625049
Validation score: 0.970000
Iteration 10, loss = 0.11265775
Validation score: 0.973571
Iteration 11, loss = 0.10845297
Validation score: 0.974107
Iteration 12, loss = 0.10686325
Validation score: 0.974286
Iteration 13, loss = 0.10287735
Validation score: 0.973571
Iteration 14, loss = 0.10079016
Validation score: 0.974286
Iteration 15, loss = 0.09849339
Validation score: 0.973571
Iteration 16, loss = 0.09614414
Validation score: 0.975893
Iteration 17, loss = 0.09557597
Validation score: 0.973929
Iteration 18, loss = 0.09412059
Validation score: 0.974464
Iteration 19, loss = 0.09021109
Validation score: 0.975357
Iteration 20, loss = 0.09152060
Validation score: 0.975357
781: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (20) reached and the optimization hasn't converged yet.
  warnings.warn(

Accuracy en imágenes modificadas: 0.095

Reporte en imágenes modificadas:
               precision    recall  f1-score   support

           0       0.00      0.00      0.00        20
           1       0.00      0.00      0.00        23
           2       0.00      0.00      0.00        17
           3       0.00      0.00      0.00        24
           4       0.14      0.87      0.24        15
           5       0.17      0.15      0.16        20
           6       0.00      0.00      0.00        19
           7       0.07      0.14      0.09        22
           8       0.00      0.00      0.00        20
           9       0.00      0.00      0.00        20

    accuracy                           0.10       200
   macro avg       0.04      0.12      0.05       200
weighted avg       0.03      0.10      0.04       200



"""