import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree

# Cargar CSV
df = pd.read_csv("Fish.csv")
print(df.head())

X = df.drop("Weight", axis=1)
y = df["Weight"]

# One-hot encoding para Species
X = pd.get_dummies(X, drop_first=True)

# Split en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def evaluar_modelo(modelo, X_train, X_test, y_train, y_test):
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

# Modelo sin restricciones
dt_default = DecisionTreeRegressor(random_state=42)
mse_def, r2_def = evaluar_modelo(dt_default, X_train, X_test, y_train, y_test)

# Modelo con max_depth=3
dt_depth3 = DecisionTreeRegressor(max_depth=3, random_state=42)
mse_d3, r2_d3 = evaluar_modelo(dt_depth3, X_train, X_test, y_train, y_test)

# Modelo con min_samples_leaf=5
dt_leaf5 = DecisionTreeRegressor(min_samples_leaf=5, random_state=42)
mse_l5, r2_l5 = evaluar_modelo(dt_leaf5, X_train, X_test, y_train, y_test)

# Modelo con pruning por ccp_alpha
dt_ccp = DecisionTreeRegressor(ccp_alpha=10, random_state=42)
mse_ccp, r2_ccp = evaluar_modelo(dt_ccp, X_train, X_test, y_train, y_test)

print("Modelo default:   MSE=%.2f, R2=%.2f" % (mse_def, r2_def))
print("Max depth=3:      MSE=%.2f, R2=%.2f" % (mse_d3, r2_d3))
print("Min leaf=5:       MSE=%.2f, R2=%.2f" % (mse_l5, r2_l5))
print("CCP alpha=10:     MSE=%.2f, R2=%.2f" % (mse_ccp, r2_ccp))

# Elegimos un modelo entrenado (ej: profundidad=3)
dt_depth3.fit(X_train, y_train)

# Caso del test
sample = X_test.iloc[[0]]
print("Caso de prueba:\n", sample)
print("Peso real:", y_test.iloc[0])
print("Predicción:", dt_depth3.predict(sample)[0])

# Visualizar árbol
plt.figure(figsize=(16,8))
tree.plot_tree(dt_depth3, feature_names=X.columns, filled=True, rounded=True)
plt.show()
