from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import pandas as pd
import matplotlib.pyplot as plt

# Cargar dataset MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalizar imágenes (de 0–255 a 0–1)
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Redimensionar para que tenga formato (altura, ancho, canales)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# One-hot encoding de etiquetas
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    epochs=5,
                    batch_size=128,
                    validation_split=0.1,
                    verbose=1)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy en test: {acc:.4f}")

"""
Accuracy en test: 0.9853
   accuracy  val_accuracy
0  0.926889      0.976833
1  0.975833      0.985000
2  0.983389      0.983167
3  0.986796      0.985833
4  0.989611      0.988167
"""


# pruebo con otras configuraciones:
model2 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])


results = pd.DataFrame(history.history)
print(results[['accuracy', 'val_accuracy']].tail())

plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.legend()
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.title('Evolución del Accuracy')
plt.show()