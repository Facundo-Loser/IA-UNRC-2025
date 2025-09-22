# 5)a)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Datos: (IBU, RMS)
X = np.array([
    [15, 20], # Lager
    [12, 15], # Lager
    [28, 39], # Lager
    [21, 30], # Lager
    [18, 22], # Lager
    [25, 35], # Lager
    [30, 28], # Lager

    [45, 20], # Stout
    [40, 61], # Stout
    [42, 70], # Stout
    [55, 65], # Stout
    [48, 72], # Stout
    [52, 60], # Stout
    [60, 80]  # Stout
])

# 0 = Lager, 1 = Stout
y = np.array([
    0,0,0,0,0,0,0,  # Lagers
    1,1,1,1,1,1,1   # Stouts
])


# Funcion sigmoid
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# Modelo
def modelo(X, Y, learning_rate, iterations):
    X = X.T # trasponemos para realizar la multiplicación de matrices
    n = X.shape[0]  # cantidad de características
    m = X.shape[1]  # cantidad de casos
    W = np.zeros((n,1)) # vector de pesos para cada característica
    B = 0
    Y = Y.reshape(1, m)

    for i in range(iterations):
        Z = np.dot(W.T, X) + B # mult pesos por casos
        A = sigmoid(Z) # tenemos el vector de resultados de cada caso

        # evitar log(0)
        A = np.clip(A, 1e-10, 1 - 1e-10)

        # Función costo
        costo = -(1/m)*np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))

        # Aplicación de la técnica Gradient Descent
        dW = (1/m) * np.dot(X, (A - Y).T)
        dB = (1/m) * np.sum(A - Y)

        # Ajuste de pesos
        W -= learning_rate * dW
        B -= learning_rate * dB

        if i % (iterations//10) == 0:
            print(f"costo luego de iteración {i}: {costo}")

    return W, B

def prediccion(X, W, B):
    Z = np.dot(W.T, X.T) + B
    A = sigmoid(Z)
    return (A >= 0.5).astype(int).flatten() # retorna 1 si es mayor o igual; 0 cc

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # separar los datos

W, B = modelo(X_train, y_train, learning_rate=0.01, iterations=10000) # entrenar el modelo

y_pred = prediccion(X_test, W, B)

# metricas
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))

# --- graficos ---
plt.figure(figsize=(6,6))

# puntos de entrenamiento
plt.scatter(X_train[y_train==0][:,0], X_train[y_train==0][:,1], color="red", label="Lager (train)")
plt.scatter(X_train[y_train==1][:,0], X_train[y_train==1][:,1], color="blue", label="Stout (train)")

# puntos de validación
plt.scatter(X_test[y_test==0][:,0], X_test[y_test==0][:,1], color="orange", marker="x", s=100, label="Lager (test)")
plt.scatter(X_test[y_test==1][:,0], X_test[y_test==1][:,1], color="cyan", marker="x", s=100, label="Stout (test)")

# frontera de decisión
x1_vals = np.linspace(min(X[:,0])-2, max(X[:,0])+2, 100)
x2_vals = -(W[0,0]*x1_vals + B)/W[1,0]
plt.plot(x1_vals, x2_vals, "k--", label="Frontera de decisión")

plt.xlabel("IBU")
plt.ylabel("RMS")
plt.legend()
plt.title("Clasificación Stout vs No-Stout con Regresión Logística")
plt.grid(True)
plt.show()
