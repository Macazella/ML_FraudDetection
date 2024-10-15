import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.io import arff
import seaborn as sns

# Cargar el archivo ARFF
file_path = r'C:\DB_Covid19Arg\csv_archivos_limpios\Amazon_test\phpKo8OWT.arff'
data, meta = arff.loadarff(file_path)

# Convertir los datos en un DataFrame de pandas
df = pd.DataFrame(data)

# Convertir la variable 'Class' a entero (eliminar los caracteres b'')
df['Class'] = df['Class'].apply(lambda x: int(x.decode('utf-8')))

# Definir las características (X) - Usamos solo las variables PCA y Amount para clustering
X = df.drop(columns=['Class'])

# Reducción de dimensionalidad con PCA para visualizar en 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Aplicar K-Means con 3 clusters (puedes ajustar el número de clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# Añadir los clusters al DataFrame
df['Cluster'] = clusters

# Visualizar los clusters con PCA
plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster'], palette='viridis')
plt.title('Clusters visualizados con PCA')
plt.show()

# Visualización con t-SNE para mejor separación
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Visualizar los clusters con t-SNE
plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=df['Cluster'], palette='viridis')
plt.title('Clusters visualizados con t-SNE')
plt.show()
