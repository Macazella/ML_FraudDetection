import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.io import arff

# Cargar el archivo ARFF
file_path = r'C:\DB_Covid19Arg\csv_archivos_limpios\Amazon_test\phpKo8OWT.arff'
data, meta = arff.loadarff(file_path)
df = pd.DataFrame(data)

# Convertir la variable 'Class' a entero (eliminar los caracteres b'')
df['Class'] = df['Class'].apply(lambda x: int(x.decode('utf-8')))

# Preprocesar los datos para DBSCAN (escalado)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(columns=['Class']))

# Aplicar DBSCAN para detectar outliers
dbscan = DBSCAN(eps=1.0, min_samples=10)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

# Verificar si se aplicó correctamente el clustering DBSCAN
if 'DBSCAN_Cluster' not in df.columns:
    raise KeyError("La columna 'DBSCAN_Cluster' no existe en el DataFrame. Asegúrate de haber aplicado DBSCAN correctamente.")

# Separar los outliers y no outliers
outliers = df[df['DBSCAN_Cluster'] == -1]
no_outliers = df[df['DBSCAN_Cluster'] != -1]

# Elegir las variables críticas para la comparación
variables_criticas = ['V1', 'V4', 'Amount']

# Comparación de distribución entre outliers y no outliers para cada variable crítica
for var in variables_criticas:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[outliers[var], no_outliers[var]], palette="Set2")
    plt.xticks([0, 1], ['Outliers', 'No Outliers'])
    plt.title(f'Comparación de {var} entre Outliers y No Outliers')
    plt.show()

# Descripción estadística de los outliers
print("Estadísticas descriptivas de los outliers:")
print(outliers[variables_criticas].describe())

# Descripción estadística de los no outliers
print("\nEstadísticas descriptivas de los no outliers:")
print(no_outliers[variables_criticas].describe())
