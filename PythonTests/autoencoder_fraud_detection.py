import numpy as np
import pandas as pd
import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
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

# Crear el modelo Autoencoder
input_dim = X_train_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation='relu')(input_layer)  # Capa de codificación
decoded = Dense(input_dim, activation='sigmoid')(encoded)  # Capa de decodificación

# Modelo completo del Autoencoder
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Entrenar el Autoencoder (Autoentrenado)
history = autoencoder.fit(X_train_scaled, X_train_scaled, 
                          epochs=50, batch_size=32, 
                          validation_data=(X_test_scaled, X_test_scaled))

# Graficar la función de pérdida (loss) durante el entrenamiento
plt.plot(history.history['loss'], label='Pérdida en entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida en validación')
plt.title('Pérdida del Autoencoder')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# Paso 1: Reconstruir los datos de test usando el autoencoder
X_test_pred = autoencoder.predict(X_test_scaled)

# Paso 2: Calcular el error de reconstrucción
mse = np.mean(np.power(X_test_scaled - X_test_pred, 2), axis=1)

# Paso 3: Establecer un umbral para detectar anomalías (ajustar después)
threshold = np.percentile(mse, 95)  # Se puede ajustar este valor

# Paso 4: Detectar anomalías (fraudes) basándose en el umbral
y_pred = (mse > threshold).astype(int)

# Paso 5: Evaluación del modelo usando la matriz de confusión y el reporte de clasificación
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Paso 6: Graficar el histograma del error de reconstrucción
plt.hist(mse, bins=50)
plt.axvline(threshold, color='r', linestyle='--', label=f'Umbral: {threshold:.4f}')
plt.title('Distribución del Error de Reconstrucción')
plt.xlabel('Error de Reconstrucción (MSE)')
plt.ylabel('Frecuencia')
plt.legend()
plt.show()
