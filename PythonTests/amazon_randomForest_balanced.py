import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
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
log_dir = "logs/random_forest/" + time.strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
tensorboard_writer = tf.summary.create_file_writer(log_dir)

# Dividir los datos en variables independientes (X) y dependiente (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Entrenar un modelo Random Forest con ajuste para clases desbalanceadas
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

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
print(f"Mejor modelo: {model}")

# Interpretación del modelo con SHAP
print("Generando gráficos SHAP...")

# Crear el explainer de SHAP
explainer = shap.TreeExplainer(model)  # Usamos el modelo entrenado (RandomForestClassifier)

# Generar los valores SHAP
shap_values = explainer.shap_values(X_test)

# Gráfico resumen SHAP
shap.summary_plot(shap_values, X_test, show=False)  # Evitar mostrar el gráfico
plt.savefig("shap_summary_plot_balanced.png")  # Guardar el gráfico resumen

# Gráfico de dependencia SHAP para una característica específica
shap.dependence_plot("V14", shap_values, X_test, show=False)  # Evitar mostrar el gráfico
plt.savefig("shap_dependence_plot_V14_balanced.png")  # Guardar el gráfico de dependencia

print("Gráficos SHAP generados y guardados como imágenes.")

