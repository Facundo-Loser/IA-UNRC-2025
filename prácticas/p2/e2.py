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
# (b) gradiente descendente

theta_gd = np.zeros((2, 1))   # θ0 y θ1 inicializados en 0
eta = 0.01                    # tasa de aprendizaje
n_iter = 1000
m = len(y)                    # cantidad de muestras
mse_history = []

for iteration in range(n_iter):
    gradients = (1/m) * X.T @ (X @ theta_gd - y)
    theta_gd = theta_gd - eta * gradients
    mse_history.append(np.mean((X @ theta_gd - y) ** 2))

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
plt.plot(range(n_iter), mse_history, color="purple")
plt.xlabel("Iteración")
plt.ylabel("MSE")
plt.title("Convergencia del error (Gradiente descendente)")
plt.show()
