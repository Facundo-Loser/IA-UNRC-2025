from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Dataset sintético
X, y = make_classification(
    n_samples=500, n_features=20, n_informative=5,
    n_redundant=2, n_classes=2, flip_y=0.1, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --- AdaBoost con DecisionTree ---
base_tree = DecisionTreeClassifier(max_depth=1, random_state=42)
ada_tree = AdaBoostClassifier(
    estimator=base_tree,
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
ada_tree.fit(X_train, y_train)
y_pred_tree = ada_tree.predict(X_test)

print("=== AdaBoost con Árbol ===")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_tree)).plot()
plt.title("AdaBoost + DecisionTree")
plt.show()

# --- AdaBoost con Logistic Regression ---
base_logreg = LogisticRegression(solver="saga", max_iter=1000, random_state=42)
ada_logreg = AdaBoostClassifier(
    estimator=base_logreg,
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
ada_logreg.fit(X_train, y_train)
y_pred_logreg = ada_logreg.predict(X_test)

print("=== AdaBoost con Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, y_pred_logreg))
print(classification_report(y_test, y_pred_logreg))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_logreg)).plot()
plt.title("AdaBoost + LogisticRegression")
plt.show()
