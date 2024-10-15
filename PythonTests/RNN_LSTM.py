import numpy as np
import pandas as pd
import arff  # Para manejar archivos ARFF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Cargar el archivo ARFF y convertirlo en un DataFrame
with open("C:/DB_Covid19Arg/csv_archivos_limpios/Amazon_test/phpKo8OWT.arff") as f:
    dataset = arff.load(f)

# Convertir el dataset ARFF en un DataFrame de pandas
df = pd.DataFrame(dataset['data'], columns=[attr[0] for attr in dataset['attributes']])

# Convertir los datos en formato numérico
df = df.apply(pd.to_numeric, errors='coerce')

# Eliminar filas con valores NaN que pudieran quedar tras la conversión
df = df.dropna()

# Verificar las primeras filas del DataFrame para asegurar que se cargó correctamente
print(df.head())

# Separar las características y la variable objetivo (Class)
X = df.drop(columns=['Class'])
y = df['Class']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos para mejorar el rendimiento del modelo
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Redimensionar los datos para adaptarlos a la entrada de LSTM (Se espera una entrada 3D: muestras, pasos de tiempo, características)
X_train_lstm = np.expand_dims(X_train_scaled, axis=2)
X_test_lstm = np.expand_dims(X_test_scaled, axis=2)

# Crear el modelo LSTM
model = Sequential()

# Añadir la primera capa LSTM
model.add(LSTM(64, input_shape=(X_train_lstm.shape[1], 1), activation='tanh', return_sequences=True))
model.add(Dropout(0.5))  # Dropout para evitar el sobreajuste

# Añadir otra capa LSTM si es necesario
model.add(LSTM(32, activation='tanh'))
model.add(Dropout(0.5))

# Capa de salida para clasificación binaria
model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo usando Adam como optimizador y binary_crossentropy como función de pérdida
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train_lstm, y_train, validation_data=(X_test_lstm, y_test), epochs=50, batch_size=32)

# Evaluar el modelo en el conjunto de prueba
loss, accuracy = model.evaluate(X_test_lstm, y_test)
print(f"Exactitud en el conjunto de prueba: {accuracy * 100:.2f}%")

# Matriz de confusión y reporte de clasificación
y_pred = (model.predict(X_test_lstm) > 0.5).astype("int32")
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Matriz de Confusión:")
print(conf_matrix)
print("\nReporte de Clasificación:")
print(class_report)

# Graficar el desempeño durante el entrenamiento
plt.plot(history.history['accuracy'], label='Exactitud en entrenamiento')
plt.plot(history.history['val_accuracy'], label='Exactitud en validación')
plt.title('Exactitud del Modelo LSTM')
plt.xlabel('Época')
plt.ylabel('Exactitud')
plt.legend()
plt.show()

# También podemos graficar la función de pérdida durante el entrenamiento
plt.plot(history.history['loss'], label='Pérdida en entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida en validación')
plt.title('Pérdida del Modelo LSTM')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.show()
