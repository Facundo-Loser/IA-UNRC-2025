import numpy as np
import matplotlib.pyplot as plt

# generar datos sintéticos
np.random.seed(0)  # para reproducibilidad
n = 100
x = np.random.rand(n, 1) * 10  # valores entre 0 y 10
e = np.random.randn(n, 1)      # N(0,1)
y = 3 * x + 2 + e              # y = 3x + 2 + e

# agrego la columna de 1s (termino independiente)
X = np.hstack([np.ones((n, 1)), x])  # matriz de diseño

# (a) ecuación normal
theta_normal = np.linalg.inv(X.T @ X) @ X.T @ y
print("Coeficientes (Ecuación normal):", theta_normal.ravel()) # ravel aplana la matriz para quequede como un array

# (b) gradiente descendente

# inicializamos en cero (bias y pendiente) 2 filas y 1 columna
# [theta0]
# [theta1]
theta_gd = np.zeros((2, 1))
alpha = 0.01                 # tasa de aprendizaje
iterations = 500
mse_history = []

for i in range(iterations):
    y_pred = X @ theta_gd
    error = y_pred - y
    grad = (2/n) * (X.T @ error)         # gradiente de MSE
    theta_gd = theta_gd - alpha * grad   # actualización
    mse = np.mean(error**2)
    mse_history.append(mse)

print("Coeficientes (Gradiente Descendente):", theta_gd.ravel())

# (c) comparar resultados
print(f"MSE (Normal): {np.mean((X @ theta_normal - y)**2):.4f}")
print(f"MSE (GD final): {mse_history[-1]:.4f}")

# (d) gráficos

# gráfico de datos + rectas
plt.scatter(x, y, color="blue", label="Datos")
plt.plot(x, X @ theta_normal, color="red", label="Ecuación normal")
plt.plot(x, X @ theta_gd, color="green", linestyle="--", label="Gradiente descendente")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Ajuste lineal: comparación de métodos")
plt.show()

# gráfico de convergencia del error
plt.plot(range(iterations), mse_history, color="purple")
plt.xlabel("Iteración")
plt.ylabel("MSE")
plt.title("Convergencia del error (Gradiente descendente)")
plt.show()
