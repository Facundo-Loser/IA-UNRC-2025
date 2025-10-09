import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Cargar datos
iris = load_iris()
X = iris.data          # 150 muestras, 4 características
y = iris.target        # 3 clases

# Normalizar
X = StandardScaler().fit_transform(X)

# Reconvertir cada muestra en "imagen" 2x2
X = X.reshape(-1, 2, 2, 1)

# One-hot encoding
y = to_categorical(y, 3)

# División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Configuración 1: Modelo simple
model1 = Sequential([
    Conv2D(4, (2, 2), activation='relu', input_shape=(2, 2, 1)),
    Flatten(),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])

model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist1 = model1.fit(X_train, y_train, epochs=30, batch_size=8, verbose=0, validation_split=0.2)
loss1, acc1 = model1.evaluate(X_test, y_test, verbose=0)

# Configuración 2: Modelo intermedio
model2 = Sequential([
    Conv2D(8, (2, 2), activation='relu', input_shape=(2, 2, 1)),
    Dropout(0.2),
    Flatten(),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')
])

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist2 = model2.fit(X_train, y_train, epochs=40, batch_size=8, verbose=0, validation_split=0.2)
loss2, acc2 = model2.evaluate(X_test, y_test, verbose=0)

# Configuración 3: Modelo más profundo
model3 = Sequential([
    Conv2D(8, (2, 2), activation='relu', input_shape=(2, 2, 1)),
    Dropout(0.2),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')
])

model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist3 = model3.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0, validation_split=0.2)
loss3, acc3 = model3.evaluate(X_test, y_test, verbose=0)

# Comparar resultados
results = pd.DataFrame({
    'Modelo': ['CNN Simple', 'CNN Intermedia', 'CNN Profunda'],
    'Accuracy': [acc1, acc2, acc3],
    'Loss': [loss1, loss2, loss3]
})

print("\nResultados comparativos:")
print(results)

"""
Resultados comparativos:
           Modelo  Accuracy      Loss
0      CNN Simple  0.866667  0.472668
1  CNN Intermedia  0.933333  0.204229
2    CNN Profunda  1.000000  0.086649
"""

# Graficar evolución del accuracy
plt.figure(figsize=(8, 5))
plt.plot(hist1.history['val_accuracy'], label='Simple')
plt.plot(hist2.history['val_accuracy'], label='Intermedia')
plt.plot(hist3.history['val_accuracy'], label='Profunda')
plt.title('Comparación de accuracy de validación')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
