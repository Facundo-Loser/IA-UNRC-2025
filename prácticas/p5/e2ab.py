from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# obtener el dataset de openml
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"].astype(int)

# Normalizar datos (0-1) (están en esacala de grises)
X = X / 255.0

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar red neuronal (Multilayer perceptron)
# acalaro: los valores por defecto de MLPClassifier (relu, adam, alpha=0.0001)
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

# Evaluación
y_pred = mlp.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))

# Mostrar algunas predicciones
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i, ax in enumerate(axes):
    ax.imshow(X_test[i].reshape(28, 28), cmap="gray")
    ax.set_title(f"Pred: {y_pred[i]}")
    ax.axis("off")
plt.show()


"""
2)a) Hiperparametros:
solver="adam",
alpha=1e-4, # regularización L2 (Ridge)

Iteration 1, loss = 0.39286260
Iteration 2, loss = 0.15239755
Iteration 3, loss = 0.10608706
Iteration 4, loss = 0.08038778
Iteration 5, loss = 0.06403363
Iteration 6, loss = 0.05088564
Iteration 7, loss = 0.04088828
Iteration 8, loss = 0.03441392
Iteration 9, loss = 0.02911492
Iteration 10, loss = 0.02490929
Iteration 11, loss = 0.01988026
Iteration 12, loss = 0.01600589
Iteration 13, loss = 0.01437448
Iteration 14, loss = 0.01047791
Iteration 15, loss = 0.01086533
Iteration 16, loss = 0.00985182
Iteration 17, loss = 0.00891190
Iteration 18, loss = 0.00688038
Iteration 19, loss = 0.00833706
Iteration 20, loss = 0.00814995
: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (20) reached and the optimization hasn't converged yet.
  warnings.warn(

Accuracy: 0.9724285714285714

Reporte de clasificación:
               precision    recall  f1-score   support

           0       0.99      0.99      0.99      1343
           1       0.98      0.99      0.98      1600
           2       0.93      0.99      0.96      1380
           3       0.98      0.97      0.97      1433
           4       0.97      0.98      0.97      1295
           5       0.98      0.97      0.97      1273
           6       0.97      0.99      0.98      1396
           7       0.99      0.94      0.96      1503
           8       0.97      0.95      0.96      1357
           9       0.97      0.95      0.96      1420

    accuracy                           0.97     14000
   macro avg       0.97      0.97      0.97     14000
weighted avg       0.97      0.97      0.97     14000
"""

"""
2)b)
Ahora cambiando los hiperparametros:
solver="sgd"
early_stopping=True,   # corta antes si no mejora

(Es peor)

Iteration 1, loss = 1.71052727
Validation score: 0.774286
Iteration 2, loss = 0.78359852
Validation score: 0.842500
Iteration 3, loss = 0.53345190
Validation score: 0.866964
Iteration 4, loss = 0.44373221
Validation score: 0.879107
Iteration 5, loss = 0.39616590
Validation score: 0.888393
Iteration 6, loss = 0.36542392
Validation score: 0.894286
Iteration 7, loss = 0.34329059
Validation score: 0.898036
Iteration 8, loss = 0.32589678
Validation score: 0.902321
Iteration 9, loss = 0.31165570
Validation score: 0.904107
Iteration 10, loss = 0.29965119
Validation score: 0.906429
Iteration 11, loss = 0.28914457
Validation score: 0.909286
Iteration 12, loss = 0.27968897
Validation score: 0.911071
Iteration 13, loss = 0.27104885
Validation score: 0.913750
Iteration 14, loss = 0.26310552
Validation score: 0.916250
Iteration 15, loss = 0.25596976
Validation score: 0.916964
Iteration 16, loss = 0.24929963
Validation score: 0.920714
Iteration 17, loss = 0.24284590
Validation score: 0.921786
Iteration 18, loss = 0.23689002
Validation score: 0.923571
Iteration 19, loss = 0.23118204
Validation score: 0.926250
Iteration 20, loss = 0.22559318
Validation score: 0.928036
781: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (20) reached and the optimization hasn't converged yet.
  warnings.warn(

Accuracy: 0.9325

Reporte de clasificación:
               precision    recall  f1-score   support

           0       0.97      0.97      0.97      1343
           1       0.95      0.98      0.96      1600
           2       0.93      0.92      0.92      1380
           3       0.91      0.91      0.91      1433
           4       0.92      0.94      0.93      1295
           5       0.92      0.92      0.92      1273
           6       0.94      0.95      0.95      1396
           7       0.94      0.94      0.94      1503
           8       0.93      0.89      0.91      1357
           9       0.92      0.91      0.92      1420

    accuracy                           0.93     14000
   macro avg       0.93      0.93      0.93     14000
weighted avg       0.93      0.93      0.93     14000
"""

"""
Ahora usando una regularización fuerte:
solver="adam",
alpha=1e-1,

(Mejoro)

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

Accuracy: 0.9750714285714286

Reporte de clasificación:
               precision    recall  f1-score   support

           0       0.99      0.98      0.99      1343
           1       0.98      0.99      0.98      1600
           2       0.97      0.97      0.97      1380
           3       0.97      0.97      0.97      1433
           4       0.99      0.97      0.98      1295
           5       0.97      0.97      0.97      1273
           6       0.98      0.98      0.98      1396
           7       0.98      0.98      0.98      1503
           8       0.98      0.96      0.97      1357
           9       0.95      0.97      0.96      1420

    accuracy                           0.98     14000
   macro avg       0.98      0.97      0.97     14000
weighted avg       0.98      0.98      0.98     14000
"""

"""
Ahora usando:
alpha=1e-1, # regularización L2 (Ridge)
learning_rate="adaptive" (si despues de 2 epocas no mejora lo achica)

(Sigue siendo bastante bueno, no cambio anda respecto al anterior)

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
C:\Users\facun\AppData\Roaming\Python\Python312\site-packages\sklearn\neural_network\_multilayer_perceptron.py:781: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (20) reached and the optimization hasn't converged yet.
  warnings.warn(

Accuracy: 0.9750714285714286

Reporte de clasificación:
               precision    recall  f1-score   support

           0       0.99      0.98      0.99      1343
           1       0.98      0.99      0.98      1600
           2       0.97      0.97      0.97      1380
           3       0.97      0.97      0.97      1433
           4       0.99      0.97      0.98      1295
           5       0.97      0.97      0.97      1273
           6       0.98      0.98      0.98      1396
           7       0.98      0.98      0.98      1503
           8       0.98      0.96      0.97      1357
           9       0.95      0.97      0.96      1420

    accuracy                           0.98     14000
   macro avg       0.98      0.97      0.97     14000
weighted avg       0.98      0.98      0.98     14000
"""