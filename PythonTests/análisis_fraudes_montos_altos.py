import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.io import arff
import matplotlib.pyplot as plt
import seaborn as sns

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

# Aplicar DBSCAN si no lo has hecho antes
dbscan = DBSCAN(eps=1.0, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Añadir la etiqueta de DBSCAN (outliers = -1)
df['DBSCAN_Cluster'] = dbscan_labels

# Filtrar las transacciones en el rango de montos altos (5000-25000)
monto_alto_outliers = df[(df['Amount'] >= 5000) & (df['Amount'] <= 25000) & (df['DBSCAN_Cluster'] == -1)]

# Ver cuántas de estas transacciones están etiquetadas como fraudulentas
fraudes_monto_alto = monto_alto_outliers[monto_alto_outliers['Class'] == 1]
no_fraudes_monto_alto = monto_alto_outliers[monto_alto_outliers['Class'] == 0]

# Estadísticas descriptivas para fraudes y no fraudes en montos altos
print("Estadísticas descriptivas de las transacciones fraudulentas en montos altos:")
print(fraudes_monto_alto.describe())

print("\nEstadísticas descriptivas de las transacciones no fraudulentas en montos altos:")
print(no_fraudes_monto_alto.describe())

# Visualizar la distribución de algunas características (Amount, V1, V4) para fraudes y no fraudes
variables_criticas = ['Amount', 'V1', 'V4']

for var in variables_criticas:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[fraudes_monto_alto[var], no_fraudes_monto_alto[var]], palette="Set2")
    plt.xticks([0, 1], ['Fraudes', 'No Fraudes'])
    plt.title(f'Comparación de {var} entre Fraudes y No Fraudes en Montos Altos')
    plt.show()

# Ver la proporción de fraudes en montos altos
fraudes_proporcion = len(fraudes_monto_alto) / len(monto_alto_outliers) * 100
print(f"Proporción de fraudes en montos altos: {fraudes_proporcion:.2f}%")
