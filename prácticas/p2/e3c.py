import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

np.random.seed(0)

# 100 puntos entre -5 y 5
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = X**2 + np.random.normal(0, 4, size=X.shape)  # más ruido para simular datos reales

# Separar en entrenamiento y validación (80% - 20%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

grados = [1, 2, 6, 10, 80]  # distintos grados a probar
mse_train = []
mse_val = []

plt.figure(figsize=(10,6))
plt.scatter(X, y, color='blue', label='Datos reales', alpha=0.5)

for grado in grados:
    # Transformar datos a polinomiales
    poly = PolynomialFeatures(degree=grado)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)

    # Ajustar regresión
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_poly, y_train)

    # Predicciones
    y_train_pred = lin_reg.predict(X_train_poly)
    y_val_pred = lin_reg.predict(X_val_poly)

    # Calcular MSE
    mse_train.append(mean_squared_error(y_train, y_train_pred))
    mse_val.append(mean_squared_error(y_val, y_val_pred))

    # Graficar curva polinomial (sobre todos los X)
    X_plot_poly = poly.transform(X)
    y_plot = lin_reg.predict(X_plot_poly)
    plt.plot(X, y_plot, label=f'Grado {grado}')

plt.xlabel('X')
plt.ylabel('y')
plt.title('Regresión polinomial: comparación de grados')
plt.legend()
plt.show()

# Imprimir MSEs
for i, grado in enumerate(grados):
    print(f"Grado {grado} -> MSE train: {mse_train[i]:.2f}, MSE val: {mse_val[i]:.2f}")
