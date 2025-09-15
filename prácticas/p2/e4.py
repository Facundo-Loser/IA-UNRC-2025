from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

def load_fish_data():
    return pd.read_csv(Path("Fish.csv"))

fish = load_fish_data()
print(fish)

# caract de los peces:
# Species (Especie)
# Weight  (Peso en gramos)
# Length1 (Longitud vertical cm)
# Length2 (Longitud diagonal cm)
# Length3 (Longitud transversal)
# Height  (Altura del pez)
# Width   (Ancho del pez cm)

# -> varaible objetivo a predecir: Weigth (peso del pez)

# histograma
fish.hist(bins=50, figsize=(12,8))
plt.show()

# separar datos para aprendizaje y para evaluación (muestreo aleatorio)
#train_set, test_set = train_test_split(fish, test_size=0.2, random_state=42)
#print(len(train_set), "train +", len(test_set), "test")

# alternativamente se puede usar muestreo estratificado:
# muestreo estratificado aleatorio
# n_splits=1 (cantidad de particiones)
strat_train_set, strat_test_set = train_test_split(fish, test_size=0.2, stratify=fish["Species"], random_state=42)

# al hacer los histogramas de nuevo se puede observar que tiene una forma de dist normal
strat_train_set.hist(bins=50, figsize=(12,8))
strat_test_set.hist(bins=50, figsize=(12,8)) # aca no se nota tanto pq son pocos datos
plt.show()

# grafico la info
# podemos buscar correlacion entre 2 variables
fish.plot(kind="scatter", x="Length2", y="Weight", grid=True) # es mas o menos lineal
plt.show()

# buscamos correlación entre datos
attributes = ["Weight", "Length1", "Length2", "Length3", "Height", "Width"]
scatter_matrix(fish[attributes], figsize=(12, 8))
plt.show()

# observar la correlación entre variables
corr_matrix = fish.corr(numeric_only=True)
corr_matrix["Weight"].sort_values(ascending=False)
print(pd.DataFrame(corr_matrix).rename(columns={"Weight": "Correlation"}))

# con esto chequeamos que no falte ningun dato (en este caso no falta nada)
print(fish.isnull().sum())

# mi variable categorica es Species (crea una columna por cada especie con valores 0-1 para evitar suponer un orden)
cat_encoder = OneHotEncoder()
species_1hot = cat_encoder.fit_transform(fish[["Species"]]).toarray()
print(species_1hot[:5])

# escalamos las variables
num_features = ["Weight", "Length1", "Length2", "Length3", "Height", "Width"]
scaler = StandardScaler()
fish[num_features] = scaler.fit_transform(fish[num_features])

# pipeline
num_attribs = ["Weight", "Length1", "Length2", "Length3", "Height", "Width"]
cat_attribs = ["Species"]

num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),  # en caso de faltar algún valor
    StandardScaler()                   # estandariza media 0 y desviación 1
)

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

# combinar los pipelines
preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

# aplicar pipeline a tu dataset Fish
fish_prepared = preprocessing.fit_transform(fish)

print(fish_prepared.shape)  # para ver cuántas columnas resultaron

# finalmente:
X_train = strat_train_set.copy()
y_train = X_train.pop("Weight")  # variable objetivo
X_test = strat_test_set.copy()
y_test = X_test.pop("Weight")

# pipeline
lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)
