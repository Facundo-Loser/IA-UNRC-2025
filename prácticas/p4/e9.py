import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("Iris.csv")

# La columna "Id" no sirve como predictor
X = df.drop(["Id", "Species"], axis=1)
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Modelos base
base_estimators = [
    ("dt", DecisionTreeClassifier(max_depth=5, random_state=42)),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
    ("lr", LogisticRegression(max_iter=1000, random_state=42))
]

# Probar distintos meta-modelos
meta_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42)
}

# Entrenar y evaluar
for nombre, meta in meta_models.items():
    print(f"\n*** Stacking con meta-modelo: {nombre} ***")

    stack_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta,
        cv=5
    )

    stack_clf.fit(X_train, y_train)
    y_pred = stack_clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))


"""
*** Stacking con meta-modelo: Logistic Regression ***
Accuracy: 0.9555555555555556
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        15
Iris-versicolor       0.93      0.93      0.93        15
 Iris-virginica       0.93      0.93      0.93        15

       accuracy                           0.96        45
      macro avg       0.96      0.96      0.96        45
   weighted avg       0.96      0.96      0.96        45


*** Stacking con meta-modelo: Random Forest ***
Accuracy: 0.8888888888888888
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        15
Iris-versicolor       0.78      0.93      0.85        15
 Iris-virginica       0.92      0.73      0.81        15

       accuracy                           0.89        45
      macro avg       0.90      0.89      0.89        45
   weighted avg       0.90      0.89      0.89        45


*** Stacking con meta-modelo: SVM (RBF) ***
Accuracy: 0.9333333333333333
                 precision    recall  f1-score   support

    Iris-setosa       1.00      1.00      1.00        15
Iris-versicolor       1.00      0.80      0.89        15
 Iris-virginica       0.83      1.00      0.91        15

       accuracy                           0.93        45
      macro avg       0.94      0.93      0.93        45
   weighted avg       0.94      0.93      0.93        45
"""