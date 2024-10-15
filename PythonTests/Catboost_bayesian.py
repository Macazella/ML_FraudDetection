import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier
from skopt import BayesSearchCV
import shap

# Cargar el dataset desde el archivo ARFF
from scipy.io import arff

file_path = 'C:/DB_Covid19Arg/csv_archivos_limpios/Amazon_test/phpKo8OWT.arff'
data, meta = arff.loadarff(file_path)
df = pd.DataFrame(data)

# Convertir la columna 'Class' de bytes a enteros
df['Class'] = df['Class'].apply(lambda x: int(x.decode('utf-8')))

# Dividir los datos en variables independientes (X) y dependiente (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Definir el modelo CatBoost
catboost_model = CatBoostClassifier(verbose=0)

# Espacio de búsqueda para Bayesian Optimization
search_spaces = {
    'depth': (3, 10),
    'learning_rate': (0.01, 0.2),
    'iterations': (100, 500),
    'colsample_bylevel': (0.5, 1.0),
    'subsample': (0.5, 1.0),
    'scale_pos_weight': (1, 5)
}

# Configurar BayesSearchCV
opt = BayesSearchCV(
    estimator=catboost_model,
    search_spaces=search_spaces,
    n_iter=30,
    cv=3,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=3,
    random_state=42
)

# Entrenar el modelo con BayesSearchCV
opt.fit(X_train, y_train)

# Mejor modelo encontrado
best_model = opt.best_estimator_
print(f"Mejores Hiperparámetros: {opt.best_params_}")

# Evaluación en el conjunto de prueba
y_pred = best_model.predict(X_test)
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# Interpretación del modelo con SHAP
print("Generando gráficos SHAP...")

# Crear el explainer de SHAP
explainer = shap.TreeExplainer(best_model)

# Generar los valores SHAP (asegurarse de que shap_values sea una matriz y no una lista)
shap_values = explainer.shap_values(X_test)

# Si los valores SHAP son una lista, seleccionamos la matriz correspondiente a la clase positiva
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# Gráfico resumen SHAP
shap.summary_plot(shap_values, X_test)

# Gráfico de dependencia SHAP para una característica específica (puedes cambiar el nombre de la característica)
shap.dependence_plot("V14", shap_values, X_test)

print("Gráficos SHAP generados.")

