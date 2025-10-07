import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# -------------------------
# 1. Cargar CSV y limpiar
# -------------------------
df = pd.read_csv("ScreenTime vs MentalWellness.csv")

# Eliminar columna basura y user_id
df = df.drop(columns=["Unnamed: 15", "user_id"])

# Eliminar filas con valores faltantes
df = df.dropna()

# -------------------------
# 2. Codificar variables categóricas
# -------------------------
categorical_cols = ["gender", "occupation", "work_mode"]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)  # drop_first evita multicolinealidad

# -------------------------
# 3. Separar X e y
# -------------------------
X = df_encoded.drop("sleep_quality_1_5", axis=1)
y = df_encoded["sleep_quality_1_5"]

# -------------------------
# 4. Partición train/test
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Train:", X_train.shape, " Test:", X_test.shape)

# -------------------------
# 5. Exploración gráfica
# -------------------------
sns.countplot(x=y_train)
plt.title("Distribución de calidad de sueño en Train")
plt.show()

sns.boxplot(x=y_train, y=X_train["screen_time_hours"])
plt.title("Tiempo de pantalla según calidad de sueño")
plt.show()

# Heatmap de correlación solo con variables numéricas
sns.heatmap(X_train.corr(), annot=True, cmap="coolwarm")
plt.title("Correlación entre variables")
plt.show()

# -------------------------
# 6. Modelos de clasificación
# -------------------------
modelos = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="rbf", random_state=42)
}

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    print(f"\nModelo: {nombre}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# -------------------------
# 7. Características de baja calidad de sueño (sleep_quality <=2)
# -------------------------
baja = df[df["sleep_quality_1_5"] <= 2]

print("Promedio de horas de pantalla (baja calidad de sueño):", baja["screen_time_hours"].mean())
print("Promedio general:", df["screen_time_hours"].mean())
print("\nEstadísticas de las personas con baja calidad de sueño:\n", baja.describe())


"""
Promedio (mean)

screen_time_hours = 9.15 : en promedio estas personas pasan más de 9 horas frente a pantallas.
age = 29.98 : la mayoría son jóvenes adultos (~30 años).
mental_wellness_index_0_100 = 16.96 : bajo bienestar mental promedio.
social_hours_per_week = 7.83 : poco tiempo de socialización semanal.
El promedio te da una idea central de cómo suelen ser estas personas.

Las personas con baja calidad de sueño tienden a ser jóvenes adultos (~30 años), con mayor tiempo frente a pantallas (promedio 9.15 h, mediana 9.14 h), bajo bienestar mental
(promedio 16.96, mediana 12.2), poca socialización semanal (~7.8 h) y variabilidad en ejercicio y otras actividades. La mayoría cae entre 7 y 11 horas de pantalla, y el bienestar
mental es bajo para la mitad de ellos.
"""
