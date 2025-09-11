import numpy as np

# X_b es la martiz con todos los datos

# Parámetros iniciales
theta = np.zeros((2, 1)) # θ0 y θ1
eta = 0.01 # tasa de aprendizaje
n_iter = 1000 # número de iteraciones

for iteration in range(n_iter):
    gradients = (1/m) * X_b.T @ (X_b @ theta - y) # derivadas parciales
    theta = theta - eta * gradients

"""
Arrancás con
- 𝜃 = (0,0)
- Calculás los gradientes (qué tan mal ajusta tu recta).
- Ajustás un poquito los parámetros en la dirección que reduce el error.
- Repetís muchas veces → la recta se va acomodando hasta acercarse al mínimo error cuadrático (MSE mínimo).
"""
