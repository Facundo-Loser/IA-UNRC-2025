import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# a)

# dataset
np.random.seed(0)
X = np.linspace(-5, 5, 20).reshape(-1, 1)  # 20 puntos entre -5 y 5
y = X**2 + np.random.normal(0, 2, size=X.shape)  # y = x^2 + ruido

# ajustar regresión lineal
lin_reg = LinearRegression()
lin_reg.fit(X, y)               # fit calcula los coeficientes de la recta (esto entrena el modelo de regresión lineal)
y_pred_lin = lin_reg.predict(X) # predict para estima los valores de salida y dados los mismos X

# calcular MSE
mse_lin = mean_squared_error(y, y_pred_lin) # calcula el MSE (distancia entre los puntos reales y los de la func hipotesis)
print("MSE de la regresión lineal:", mse_lin)

# 4. Graficar
plt.scatter(X, y, color='blue', label='Datos reales')
plt.plot(X, y_pred_lin, color='red', label='Ajuste lineal')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regresión lineal sobre datos cuadráticos')
plt.legend()
plt.show()
