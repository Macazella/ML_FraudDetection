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

# Aplicar DBSCAN
dbscan = DBSCAN(eps=1.0, min_samples=10)  # Ajusta eps y min_samples según lo necesario
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

# Definir bins para los montos de transacción
bins = [0, 10, 50, 100, 500, 1000, 5000, 10000, 15000, 20000, 25000]
df['Amount_bins'] = pd.cut(df['Amount'], bins=bins)

# Comparar la distribución de los bins en outliers y no outliers
outliers = df[df['DBSCAN_Cluster'] == -1]
no_outliers = df[df['DBSCAN_Cluster'] != -1]

# Contar la frecuencia en cada bin para outliers y no outliers
outliers_amount_dist = outliers['Amount_bins'].value_counts().sort_index()
no_outliers_amount_dist = no_outliers['Amount_bins'].value_counts().sort_index()

# Visualizar las distribuciones
plt.figure(figsize=(10, 6))
outliers_amount_dist.plot(kind='bar', color='red', alpha=0.7, label='Outliers')
no_outliers_amount_dist.plot(kind='bar', color='blue', alpha=0.7, label='No Outliers')
plt.title('Distribución de Montos de Transacción en Outliers vs No Outliers')
plt.xlabel('Rangos de Montos (Amount)')
plt.ylabel('Frecuencia')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.show()
