import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
import tensorflow as tf
import time
import shap  # Asegúrate de que SHAP esté instalado: pip install shap

# Cargar el dataset desde el archivo ARFF
from scipy.io import arff

file_path = 'C:/DB_Covid19Arg/csv_archivos_limpios/Amazon_test/phpKo8OWT.arff'
data, meta = arff.loadarff(file_path)
df = pd.DataFrame(data)

# Convertir la columna 'Class' de bytes a enteros
df['Class'] = df['Class'].apply(lambda x: int(x.decode('utf-8')))

# Configurar TensorBoard
log_dir = "logs/catboost/" + time.strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
tensorboard_writer = tf.summary.create_file_writer(log_dir)

# Dividir los datos en variables independientes (X) y dependiente (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Definir el modelo CatBoost
cat_model = CatBoostClassifier(verbose=0, scale_pos_weight=1)

# Definir el espacio de hiperparámetros para RandomizedSearchCV
param_dist = {
    'iterations': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'depth': [3, 4, 5],
    'subsample': [0.8, 1.0],
    'colsample_bylevel': [0.8, 1.0],
    'scale_pos_weight': [1, 2, 3]  # Peso para el desbalance de clases
}

# Aplicar RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=cat_model, param_distributions=param_dist, n_iter=10, cv=3, random_state=42, n_jobs=-1, verbose=2)
random_search.fit(X_train, y_train)

# Mejor modelo encontrado
best_cat_model = random_search.best_estimator_
print(f"Mejores Hiperparámetros encontrados: {random_search.best_params_}")

# Realizar predicciones
y_pred = best_cat_model.predict(X_test)

# Evaluación del modelo
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# Registrar las métricas en TensorBoard
report_dict = classification_report(y_test, y_pred, output_dict=True)
with tensorboard_writer.as_default():
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            tf.summary.scalar(f"Precision/{label}", metrics["precision"], step=0)
            tf.summary.scalar(f"Recall/{label}", metrics["recall"], step=0)
            tf.summary.scalar(f"F1-Score/{label}", metrics["f1-score"], step=0)

# Forzar la escritura en TensorBoard
tensorboard_writer.flush()

# Mostrar el mejor modelo y su rendimiento
print(f"Mejor modelo: {best_cat_model}")

# Interpretación del modelo con SHAP
print("Generando gráficos SHAP...")

# Crear el explainer de SHAP
explainer = shap.TreeExplainer(best_cat_model)  # Usar el modelo CatBoost encontrado

# Generar los valores SHAP
shap_values = explainer.shap_values(X_test)

# Gráfico resumen SHAP
shap.summary_plot(shap_values, X_test)

# Gráfico de dependencia SHAP para una característica específica (puedes cambiar el nombre de la característica)
shap.dependence_plot("V14", shap_values, X_test)

print("Gráficos SHAP generados.")

