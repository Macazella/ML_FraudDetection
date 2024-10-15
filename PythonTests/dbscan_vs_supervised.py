import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.io import arff

# Cargar el archivo ARFF (asume que ya tienes el DataFrame `df` con clusters y outliers)
file_path = r'C:\DB_Covid19Arg\csv_archivos_limpios\Amazon_test\phpKo8OWT.arff'
data, meta = arff.loadarff(file_path)

# Convertir los datos en un DataFrame de pandas
df = pd.DataFrame(data)

# Convertir la variable 'Class' a entero (eliminar los caracteres b'')
df['Class'] = df['Class'].apply(lambda x: int(x.decode('utf-8')))

# Preprocesar los datos (escalado)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(columns=['Class']))

# Aplicar DBSCAN (con los parámetros ajustados que elegiste)
dbscan = DBSCAN(eps=1.0, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Añadir la etiqueta de DBSCAN (outliers = -1)
df['DBSCAN_Cluster'] = dbscan_labels

# Filtrar los outliers detectados por DBSCAN
outliers = df[df['DBSCAN_Cluster'] == -1]

# Comparar con el modelo supervisado: ver cuántos outliers son fraudes
outliers_frauds = outliers[outliers['Class'] == 1]

# Mostrar el número de outliers que son fraudes
num_outliers_frauds = len(outliers_frauds)
print(f"Número de outliers detectados como fraudes por DBSCAN: {num_outliers_frauds}")

# Mostrar algunos ejemplos de outliers que son fraudes
print("Ejemplos de outliers detectados como fraudes:")
print(outliers_frauds.head())

# Comparar con el modelo supervisado (si tienes las predicciones del modelo supervisado en una columna, por ejemplo 'Pred_Fraude')
if 'Pred_Fraude' in df.columns:  # Supón que tienes una columna 'Pred_Fraude' con las predicciones del modelo
    # Verificar cuántos outliers también fueron predichos como fraudes por el modelo supervisado
    outliers_pred_fraud = outliers_frauds[outliers_frauds['Pred_Fraude'] == 1]
    num_outliers_pred_fraud = len(outliers_pred_fraud)
    print(f"Número de outliers detectados como fraudes tanto por DBSCAN como por el modelo supervisado: {num_outliers_pred_fraud}")

    # Mostrar algunos ejemplos
    print("Ejemplos de outliers detectados como fraudes tanto por DBSCAN como por el modelo supervisado:")
    print(outliers_pred_fraud.head())
