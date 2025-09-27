import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report

# Dataset original
data = {
    "Horas": ["Baja", "Baja", "Media", "Alta", "Media", "Alta", "Baja", "Alta", "Media", "Alta"],
    "Asistencia": ["Baja", "Alta", "Alta", "Alta", "Baja", "Alta", "Baja", "Baja", "Alta", "Alta"],
    "Tareas": ["No", "No", "No", "Si", "Si", "No", "Si", "Si", "Si", "Si"],
    "Resultado": ["Reprobado", "Reprobado", "Aprobado", "Promoción",
                  "Aprobado", "Aprobado", "Reprobado", "Aprobado", "Promoción", "Promoción"]
}

df = pd.DataFrame(data)
print(df)

# Variables independientes (X) y dependiente (y)
X = df[["Horas", "Asistencia", "Tareas"]]
y = df["Resultado"]

# One-hot encoding para X
X = pd.get_dummies(X)

# Codificación para y (Reprobado=0, Aprobado=1, Promoción=2)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print(X)
print(y)

# Creamos el modelo
clf = DecisionTreeClassifier(criterion="gini", random_state=0)

# Entrenamos
clf.fit(X, y)

plt.figure(figsize=(12,6))
tree.plot_tree(clf,
               feature_names=X.columns,
               class_names=label_encoder.classes_,
               filled=True, rounded=True)
plt.show()

# Predicciones sobre el mismo dataset
y_pred = clf.predict(X)

# Métricas
print("Accuracy:", accuracy_score(y, y_pred))
print("\nReporte de clasificación:\n", classification_report(y, y_pred, target_names=label_encoder.classes_))