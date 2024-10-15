import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from scipy.io import arff
from sklearn.metrics import mean_squared_error, r2_score

# Cargar el archivo ARFF
file_path = r'C:\DB_Covid19Arg\csv_archivos_limpios\Amazon_test\phpKo8OWT.arff'
data, meta = arff.loadarff(file_path)

# Convertir los datos en un DataFrame de pandas
df = pd.DataFrame(data)

# Convertir la variable 'Class' a entero (eliminar los caracteres b'')
df['Class'] = df['Class'].apply(lambda x: int(x.decode('utf-8')))

# Definir las características (X) y el valor continuo a predecir (y) - En este caso 'Amount'
X = df.drop(columns=['Amount'])  # Usaremos el resto de las variables para predecir el 'Amount'
y = df['Amount']

# Dividir los datos en conjuntos de entrenamiento y prueba (80% para entrenamiento, 20% para prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar el modelo CatBoostRegressor
catboost_regressor = CatBoostRegressor(
    iterations=200,
    depth=5,
    learning_rate=0.1,
    subsample=1.0,
    verbose=0
)

# Entrenar el modelo en el conjunto de entrenamiento
catboost_regressor.fit(X_train, y_train)

# Predecir los valores en el conjunto de prueba
y_pred = catboost_regressor.predict(X_test)

# Calcular el RMSE (Root Mean Squared Error)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Calcular el R2 score
r2 = r2_score(y_test, y_pred)

# Mostrar los resultados
print(f'RMSE en conjunto de prueba: {rmse}')
print(f'R2 score en conjunto de prueba: {r2}')
