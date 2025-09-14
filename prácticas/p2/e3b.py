import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

# a)
np.random.seed(0)
X = np.linspace(-5, 5, 20).reshape(-1, 1)  # 20 puntos entre -5 y 5
y = X**2 + np.random.normal(0, 2, size=X.shape)  # y = x^2 + ruido

# Ajuste lineal
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

# MSE lineal
mse_lin = mean_squared_error(y, y_pred_lin)
print("MSE de la regresión lineal:", mse_lin)

# b) regresión polinomial
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)  # columnas: [1, X, X^2] (agrega las nevas caract)
print("\nDataset transformado (grado 2):")
print(X_poly)

# Ajuste polinomial
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly, y)
y_pred_poly = lin_reg_poly.predict(X_poly)

# MSE polinomial
mse_poly = mean_squared_error(y, y_pred_poly)
print("\nMSE de la regresión polinomial grado 2:", mse_poly)

# gráfica comparativa
plt.scatter(X, y, color='blue', label='Datos reales')
plt.plot(X, y_pred_lin, color='red', label='Regresión lineal')
plt.plot(X, y_pred_poly, color='green', label='Regresión polinomial grado 2')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Comparación: lineal vs polinomial')
plt.legend()
plt.show()
