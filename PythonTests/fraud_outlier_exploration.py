import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff

# Cargar el archivo ARFF
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
dbscan = DBSCAN(eps=1.0, min_samples=10)  # Ajusta los parámetros como sea necesario
dbscan_labels = dbscan.fit_predict(X_scaled)

# Añadir la etiqueta de DBSCAN (outliers = -1)
df['DBSCAN_Cluster'] = dbscan_labels

# 1. Filtrar los outliers detectados por DBSCAN
outliers = df[df['DBSCAN_Cluster'] == -1]

# 2. Calcular estadísticas descriptivas de los outliers
outlier_stats = outliers[['Amount', 'V1', 'V2', 'V3', 'V4', 'V5']].describe()
print("Estadísticas descriptivas de los outliers:")
print(outlier_stats)

# 3. Visualizar la distribución de montos de transacción para los outliers
plt.figure(figsize=(10, 6))
sns.histplot(outliers['Amount'], kde=True, color='red', label='Outliers')
plt.title('Distribución de Monto de Transacciones para los Outliers')
plt.xlabel('Monto de Transacción (Amount)')
plt.ylabel('Frecuencia')
plt.legend()
plt.show()

# Comparar con las transacciones que no son outliers
non_outliers = df[df['DBSCAN_Cluster'] != -1]

plt.figure(figsize=(10, 6))
sns.histplot(non_outliers['Amount'], kde=True, color='blue', label='No Outliers')
plt.title('Distribución de Monto de Transacciones para No Outliers')
plt.xlabel('Monto de Transacción (Amount)')
plt.ylabel('Frecuencia')
plt.legend()
plt.show()

# 4. Comparar otras características transformadas como V1, V4, etc.
plt.figure(figsize=(10, 6))
sns.boxplot(data=[outliers['V1'], non_outliers['V1']], palette='Set2', orient='h')
plt.title('Comparación de V1 entre Outliers y No Outliers')
plt.yticks([0, 1], ['Outliers', 'No Outliers'])
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=[outliers['V4'], non_outliers['V4']], palette='Set2', orient='h')
plt.title('Comparación de V4 entre Outliers y No Outliers')
plt.yticks([0, 1], ['Outliers', 'No Outliers'])
plt.show()
