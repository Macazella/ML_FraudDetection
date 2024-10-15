import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report
import lightgbm as lgb
import tensorflow as tf
import time
import shap  # Asegúrate de tener SHAP instalado: pip install shap
import matplotlib.pyplot as plt  # Importar matplotlib para guardar gráficos

# Cargar el dataset desde el archivo ARFF
from scipy.io import arff

file_path = 'C:/DB_Covid19Arg/csv_archivos_limpios/Amazon_test/phpKo8OWT.arff'
data, meta = arff.loadarff(file_path)
df = pd.DataFrame(data)

# Convertir la columna 'Class' de bytes a enteros
df['Class'] = df['Class'].apply(lambda x: int(x.decode('utf-8')))

# Configurar TensorBoard
log_dir = "logs/lightgbm/" + time.strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
tensorboard_writer = tf.summary.create_file_writer(log_dir)

# Dividir los datos en variables independientes (X) y dependiente (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Definir el modelo LightGBM
lgb_model = lgb.LGBMClassifier()

# Definir el espacio de hiperparámetros para RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [-1, 3, 4, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 0.1, 0.5],  # Regularización L1
    'reg_lambda': [0, 0.1, 0.5]  # Regularización L2
}

# Aplicar RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=lgb_model, param_distributions=param_dist, n_iter=10, cv=3, random_state=42, n_jobs=-1, verbose=2)
random_search.fit(X_train, y_train)

# Mejor modelo encontrado
best_lgb_model = random_search.best_estimator_
print(f"Mejores Hiperparámetros encontrados: {random_search.best_params_}")

# Realizar predicciones
y_pred = best_lgb_model.predict(X_test)

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
print(f"Mejor modelo: {best_lgb_model}")

# Interpretación del modelo con SHAP
print("Generando gráficos SHAP...")

# Crear el explainer de SHAP
explainer = shap.TreeExplainer(best_lgb_model)

# Generar los valores SHAP (debido a un cambio en SHAP para LightGBM, obtenemos los valores para la clase positiva)
shap_values = explainer.shap_values(X_test)[1]  # Extraemos los valores para la clase positiva (1)

# Gráfico resumen SHAP
shap.summary_plot(shap_values, X_test, show=False)  # Evitar mostrar el gráfico
plt.savefig("shap_summary_plot_lightgbm.png")  # Guardar el gráfico resumen

# Gráfico de dependencia SHAP para una característica específica
shap.dependence_plot("V14", shap_values, X_test, show=False)  # Evitar mostrar el gráfico
plt.savefig("shap_dependence_plot_V14_lightgbm.png")  # Guardar el gráfico de dependencia

print("Gráficos SHAP generados y guardados como imágenes.")
