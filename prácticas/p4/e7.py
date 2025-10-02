import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# 1) Generar dataset ruidoso
X, y = make_classification(
    n_samples=500,
    n_features=20,
    n_informative=5,      # solo 5 realmente aportan info
    n_redundant=2,        # 2 redundantes
    n_classes=2,          # binario
    flip_y=0.1,           # 10% de ruido en las etiquetas
    random_state=42
)

# 2) Partición train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3) Entrenar un Árbol de Decisión
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# 4) Entrenar un Random Forest (bagging)
rf = RandomForestClassifier(
    n_estimators=100,
    max_samples=0.8,
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# 5) Evaluar con matrices de confusión
print("Árbol de decisión")
print(classification_report(y_test, y_pred_dt))
cm_dt = confusion_matrix(y_test, y_pred_dt)
ConfusionMatrixDisplay(cm_dt).plot()
plt.title("Matriz de confusión - Decision Tree")
plt.show()

print("Random Forest")
print(classification_report(y_test, y_pred_rf))
cm_rf = confusion_matrix(y_test, y_pred_rf)
ConfusionMatrixDisplay(cm_rf).plot()
plt.title("Matriz de confusión - Random Forest")
plt.show()
