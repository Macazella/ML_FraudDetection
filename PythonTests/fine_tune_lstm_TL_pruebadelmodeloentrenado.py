import numpy as np
import pandas as pd
import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
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

# Separar las características y la variable objetivo (Class)
X = df.drop(columns=['Class'])
y = df['Class']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Redimensionar los datos para que sean compatibles con LSTM
X_train_scaled = np.expand_dims(X_train_scaled, axis=2)
X_test_scaled = np.expand_dims(X_test_scaled, axis=2)

# Cargar el modelo LSTM previamente entrenado
pretrained_model = load_model('LSTM_fraud_detection_model.h5')

# Congelar las primeras capas del modelo preentrenado para que no se ajusten durante el nuevo entrenamiento
for layer in pretrained_model.layers[:-3]:
    layer.trainable = False

# Crear un nuevo modelo reutilizando el preentrenado y agregar nuevas capas densas para fine-tuning
model = Sequential(pretrained_model.layers)

# Añadir capas densas adicionales para el ajuste a los nuevos datos
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Capa de salida para clasificación binaria

# Compilar el modelo ajustado
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo ajustado (fine-tuning)
history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=20, batch_size=32)

# Evaluar el modelo en el conjunto de prueba
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Exactitud en el conjunto de prueba: {accuracy * 100:.2f}%")

# Matriz de confusión y reporte de clasificación
y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Matriz de Confusión:")
print(conf_matrix)
print("\nReporte de Clasificación:")
print(class_report)

# Graficar el desempeño durante el entrenamiento
plt.plot(history.history['accuracy'], label='Exactitud en entrenamiento')
plt.plot(history.history['val_accuracy'], label='Exactitud en validación')
plt.title('Exactitud del Modelo Transfer Learning')
plt.xlabel('Época')
plt.ylabel('Exactitud')
plt.legend()
plt.show()

# Graficar la función de pérdida durante el entrenamiento
plt.plot(history.history['loss'], label='Pérdida en entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida en validación')
plt.title('Pérdida del Modelo Transfer Learning')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# Guardar el nuevo modelo ajustado
model.save('LSTM_fine_tuned_fraud_detection_model.h5')
print("Modelo ajustado guardado exitosamente.")
