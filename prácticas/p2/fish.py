from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- Cargar datos ---
def load_fish_data():
    return pd.read_csv(Path("Fish.csv"))

fish = load_fish_data()
print(fish.head())

# --- graficos y busqueda de datos faltantes ---
fish.hist(bins=50, figsize=(12,8))
plt.show()

fish.plot(kind="scatter", x="Length2", y="Weight", grid=True)
plt.show()

attributes = ["Weight", "Length1", "Length2", "Length3", "Height", "Width"]
scatter_matrix(fish[attributes], figsize=(12, 8))
plt.show()

print(fish.isnull().sum())  # ver si faltan datos

# --- Separar conjunto entrenamiento y prueba (estratificado por Species) ---
strat_train_set, strat_test_set = train_test_split(
    fish, test_size=0.2, stratify=fish["Species"], random_state=42
)

# --- Separar variables y target ---
X_train = strat_train_set.drop("Weight", axis=1)
y_train = strat_train_set["Weight"].copy()

X_test = strat_test_set.drop("Weight", axis=1)
y_test = strat_test_set["Weight"].copy()

# --- Preparar pipelines ---
num_attribs = ["Length1", "Length2", "Length3", "Height", "Width"] # columnas numéricas
cat_attribs = ["Species"]                                          # columna categórica

# pipeline para columnas numéricas
num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

# pipeline para columnas categóricas
cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

# combinar los pipelines
preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

# --- Crear pipeline completo con regresión ---
lin_reg = make_pipeline(preprocessing, LinearRegression())

# --- Entrenamiento ---
lin_reg.fit(X_train, y_train)

# --- Predicciones y evaluación ---
y_pred = lin_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) # sqrt para obtener RMSE en gramos

print("Primeras 5 predicciones:", y_pred[:5])
print("Valores reales:", y_test.values[:5])
print("RMSE:", rmse)

# En promedio los pesos de la prediccion difieren: 117.4256083608108 gramos del valor real (RMSE)
