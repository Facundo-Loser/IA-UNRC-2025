
# 3)
import numpy as np
import matplotlib.pyplot as plt

# Dataset
X = np.array([[0,0],
              [0,1],
              [1,0],
              [2,2],
              [2,3],
              [3,2]])

y = np.array([0,0,0,1,1,1]) # Clase

# Hiperparámetros
eta = 1.0  # learning rate
epochs = 4 # cantidad de épocas para el entrenamiento
w = np.array([0, 0]) # pesos iniciales
b = 0 # bias

# mi función de predicción
def step(z):
    if (z >= 0):
        return 1
    else:
        return 0

# Entrenamiento
for epoch in range(epochs):
    for i in range(len(X)):
        xi = X[i]
        target = y[i]
        z = np.dot(w, xi) + b
        y_pred = step(z) # mi predicción
        error = target - y_pred

        # Actualización
        w = w + eta * error * xi
        b = b + eta * error

print("pesos finales:", w)
print("bias final:", b)


# graficar:
plt.figure(figsize=(6,6))

# separar puntos por clase
X0 = X[y==0]
X1 = X[y==1]

plt.scatter(X0[:,0], X0[:,1], color='red', label='Clase 0')
plt.scatter(X1[:,0], X1[:,1], color='blue', label='Clase 1')

# Graficar la frontera de decisión w1*x1 + w2*x2 + b = 0
# x2 = (-b - w1*x1)/w2
x1_vals = np.linspace(-1, 4, 100)
if w[1] != 0:
    x2_vals = (-b - w[0]*x1_vals)/w[1]
    plt.plot(x1_vals, x2_vals, 'k--', label='Frontera de decisión')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Perceptrón - Clasificación')
plt.legend()
plt.xlim(-1,4)
plt.ylim(-1,4)
plt.grid(True)
plt.show()