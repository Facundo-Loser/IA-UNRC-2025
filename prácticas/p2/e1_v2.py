import numpy as np
import matplotlib.pyplot as plt

# datos
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 5, 7, 8])

# minimos cuadrados
x_mean = np.mean(x)
y_mean = np.mean(y)

w = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
b = y_mean - w * x_mean

print(f"Recta ajustada: y = {w:.2f}x + {b:.2f}")

# gráfico
plt.scatter(x, y, color="blue", label="Datos")
plt.plot(x, w*x + b, color="red", label=f"Recta: y={w:.2f}x+{b:.2f}")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
