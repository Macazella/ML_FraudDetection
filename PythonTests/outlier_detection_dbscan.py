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

# Ajuste de DBSCAN con parámetros modificados
# Aumenta `eps` y `min_samples` para reducir el número de outliers
dbscan = DBSCAN(eps=1.0, min_samples=10)  # Aumentar eps y min_samples
dbscan_labels = dbscan.fit_predict(X_scaled)

# Añadir la etiqueta de DBSCAN (outliers = -1)
df['DBSCAN_Cluster'] = dbscan_labels

# Contar cuántos puntos fueron identificados como outliers
outliers_count = len(df[df['DBSCAN_Cluster'] == -1])
print(f"Número de outliers detectados: {outliers_count}")

# Visualizar los clusters y los outliers
plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=df['DBSCAN_Cluster'], palette='viridis', legend="full")
plt.title('Clusters y Outliers detectados con DBSCAN (Parámetros Ajustados)')
plt.show()
