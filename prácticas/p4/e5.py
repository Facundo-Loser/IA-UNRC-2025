import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import export_text

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
    r2 = r2_score(y_test, y_pred) # R² mide la calidad relativa del modelo.
    return mse, r2

# Modelo sin restricciones
dt_default = DecisionTreeRegressor(random_state=42)
mse_def, r2_def = evaluar_modelo(dt_default, X_train, X_test, y_train, y_test)

# Modelo con max_depth=3
dt_depth3 = DecisionTreeRegressor(max_depth=3, random_state=42)
mse_d3, r2_d3 = evaluar_modelo(dt_depth3, X_train, X_test, y_train, y_test)

# Modelo con min_samples_leaf=5 (requiere que cada hoja tenga al menos 5 muestras)
dt_leaf5 = DecisionTreeRegressor(min_samples_leaf=5, random_state=42)
mse_l5, r2_l5 = evaluar_modelo(dt_leaf5, X_train, X_test, y_train, y_test)

# Modelo con pruning por ccp_alpha
# El parámetro ccp_alpha activa la poda de costo-complejidad (Cost Complexity Pruning),
# que elimina ramas que no reducen significativamente el error total del árbol penalizado por su tamaño.
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

# Visualizar árboles
# Lista de modelos y nombres
# Lista de modelos y nombres
modelos = [
    ("Árbol sin poda (default)", dt_default),
    ("Poda temprana: max_depth=3", dt_depth3),
    ("Poda temprana: min_samples_leaf=5", dt_leaf5),
    ("Poda posterior: ccp_alpha=10", dt_ccp)
]

# Mostrar cada árbol en una figura distinta
for titulo, modelo in modelos:
    plt.figure(figsize=(18, 9))
    tree.plot_tree(
        modelo,
        feature_names=X.columns,
        filled=True,
        rounded=True,
        fontsize=10,
        impurity=True,     # muestra el MSE
        precision=1
    )
    plt.title(titulo, fontsize=14)
    plt.show()


# =============================================================================
# CONSULTA DE CASO Y ANÁLISIS DE LAS CONDICIONES DEL ÁRBOL
# =============================================================================

# b)
# Caso de prueba (ya definido antes, pero lo volvemos a usar)
sample = X_test.iloc[[0]]
print("\n================= CONSULTA DE CASO =================")
print("Caso de prueba:\n", sample)
print("Peso real:", y_test.iloc[0])

# Recorremos todos los modelos y analizamos las condiciones
for nombre, modelo in modelos:
    print("\n" + "=" * 80)
    print(nombre)
    print("=" * 80)

    # Mostrar estructura del árbol en texto (reglas)
    reglas = export_text(modelo, feature_names=list(X.columns), decimals=2)
    print("\nEstructura del árbol:\n")
    print(reglas)

    # Predicción del modelo
    pred = modelo.predict(sample)[0]
    print(f"\nPredicción para el caso de prueba: {pred:.2f}")
    print(f"Valor real: {y_test.iloc[0]:.2f}")

    print("------------------------------------------------------------")
