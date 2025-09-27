import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Cargar el CSV
df = pd.read_csv("ScreenTime vs MentalWellness.csv")

# Eliminar columna basura (Unnamed: 15)
df = df.drop(columns=["Unnamed: 15"])

# X = todas menos la variable target
X = df.drop("sleep_quality_1_5", axis=1)

# y = variable objetivo
y = df["sleep_quality_1_5"]

# Partición train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Train:", X_train.shape, " Test:", X_test.shape)

# Exploración gráfica
sns.countplot(x=y_train)
plt.title("Distribución de calidad de sueño en Train")
plt.show()

sns.boxplot(x=y_train, y=X_train["screen_time_hours"])
plt.title("Tiempo de pantalla según calidad de sueño")
plt.show()

sns.heatmap(X_train.corr(), annot=True, cmap="coolwarm")
plt.title("Correlación entre variables")
plt.show()

# Modelos
modelos = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="rbf", random_state=42)
}

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    print(f"\nModelo: {nombre}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# Características de baja calidad de sueño (ejemplo: <=2)
baja = df[df["sleep_quality_1_5"] <= 2]

print("Promedio de horas de pantalla (baja calidad de sueño):", baja["screen_time_hours"].mean())
print("Promedio general:", df["screen_time_hours"].mean())
print("\nEstadísticas de los de baja calidad de sueño:\n", baja.describe())
