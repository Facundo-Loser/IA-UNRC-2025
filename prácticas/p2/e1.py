import numpy as np
import matplotlib.pyplot as plt

# Datos originales
x = np.array([1,2,3,4,5], dtype=float)
y = np.array([2,3,5,7,8], dtype=float)

# Matriz de diseño
X = np.column_stack((np.ones_like(x), x))

# Ecuación normal
XT = X.T
XTX = XT.dot(X)
XTX_inv = np.linalg.inv(XTX)
XTy = XT.dot(y)
theta = XTX_inv.dot(XTy)   # [b, w]
b, w = theta[0], theta[1]

print("X^T X =\n", XTX)
print("(X^T X)^-1 =\n", XTX_inv)
print("X^T y =\n", XTy)
print("theta (b, w) = ", theta)
print(f"Recta: y = {w:.4f}x + {b:.4f}")

# Predicción y MSE
y_hat = X.dot(theta)
residuals = y - y_hat
mse = (residuals**2).mean()
print("Predicciones:", y_hat)
print("Residuals:", residuals)
print("MSE:", mse)

# Gráfico de la recta y puntos
xs = np.linspace(x.min()-1, x.max()+1, 200)
plt.figure()
plt.scatter(x, y, label="Datos")
plt.plot(xs, w*xs + b, label=f"Recta: y={w:.2f}x+{b:.2f}")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Ajuste por mínimos cuadrados (ecuación normal)")
plt.show()

# Añadir un outlier para ver el efecto
x2 = np.append(x, 10.0)
y2 = np.append(y, 30.0)   # outlier extremo
X2 = np.column_stack((np.ones_like(x2), x2))
theta2 = np.linalg.inv(X2.T.dot(X2)).dot(X2.T.dot(y2))
b2, w2 = theta2[0], theta2[1]
print("Con outlier -> theta:", theta2)
plt.figure()
plt.scatter(x2, y2, label="Datos + outlier")
plt.plot(xs, w*xs + b, label="Recta original", linestyle='--')
plt.plot(xs, w2*xs + b2, label=f"Recta con outlier: y={w2:.2f}x+{b2:.2f}")
plt.legend()
plt.title("Efecto de outlier en mínimos cuadrados")
plt.show()
