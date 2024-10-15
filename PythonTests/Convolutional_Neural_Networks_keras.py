import numpy as np
import pandas as pd
import arff  # Para manejar archivos ARFF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Cargar el archivo ARFF y convertirlo en un DataFrame
with open("C:/DB_Covid19Arg/csv_archivos_limpios/Amazon_test/phpKo8OWT.arff") as f:
    dataset = arff.load(f)

# Convertir el dataset ARFF en un DataFrame de pandas
df = pd.DataFrame(dataset['data'], columns=[attr[0] for attr in dataset['attributes']])

# Convertir los datos en formato numérico
df = df.apply(pd.to_numeric, errors='coerce')

# Eliminar filas con valores NaN que pudieran quedar tras la conversión
df = df.dropna()

# Separar las características y la variable objetivo (Class)
X = df.drop(columns=['Class'])
y = df['Class']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos para mejorar el rendimiento del modelo
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Redimensionar los datos para adaptarlos a la entrada de Conv1D (Se espera una entrada 3D: muestras, pasos de tiempo, características)
X_train_scaled = np.expand_dims(X_train_scaled, axis=2)
X_test_scaled = np.expand_dims(X_test_scaled, axis=2)

# Crear el modelo de CNN
model = Sequential()

# Agregar una capa Conv1D
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_scaled.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))

# Añadir más capas convolucionales si es necesario
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# Aplanar los resultados y pasar a capas densas
model.add(Flatten())
model.add(Dense(128, activation='relu'))

# Agregar Dropout para evitar el sobreajuste
model.add(Dropout(0.5))

# Capa de salida para clasificación binaria
model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=50, batch_size=32)

# Evaluar el modelo en el conjunto de prueba
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Exactitud en el conjunto de prueba: {accuracy * 100:.2f}%")

# Graficar el desempeño durante el entrenamiento
plt.plot(history.history['accuracy'], label='Exactitud en entrenamiento')
plt.plot(history.history['val_accuracy'], label='Exactitud en validación')
plt.title('Exactitud del Modelo CNN')
plt.xlabel('Época')
plt.ylabel('Exactitud')
plt.legend()
plt.show()

# Graficar la función de pérdida durante el entrenamiento
plt.plot(history.history['loss'], label='Pérdida en entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida en validación')
plt.title('Pérdida del Modelo CNN')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# Generar la matriz de confusión y el reporte de clasificación
y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))
