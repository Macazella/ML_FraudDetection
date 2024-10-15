import pandas as pd
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from scipy.io import arff
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Cargar el archivo ARFF
file_path = r'C:\DB_Covid19Arg\csv_archivos_limpios\Amazon_test\phpKo8OWT.arff'  # Asegúrate de que esta ruta sea correcta
data, meta = arff.loadarff(file_path)

# Convertir los datos en un DataFrame de pandas
df = pd.DataFrame(data)

# Convertir la variable 'Class' a entero (eliminar los caracteres b'')
df['Class'] = df['Class'].apply(lambda x: int(x.decode('utf-8')))

# Definir las características (X) y la variable objetivo (y)
X = df.drop(columns=['Class'])
y = df['Class']

# Inicializar el modelo CatBoost
catboost_model = CatBoostClassifier(
    iterations=200, 
    depth=5, 
    learning_rate=0.1, 
    subsample=1.0, 
    colsample_bylevel=0.8, 
    scale_pos_weight=3, 
    verbose=0
)

# Realizar validación cruzada estratificada
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

precision_scores = []
recall_scores = []
f1_scores = []
roc_auc_scores = []

for train_index, test_index in skf.split(X, y):
    # Dividir los datos
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Entrenar el modelo
    catboost_model.fit(X_train, y_train)
    
    # Realizar predicciones
    y_pred = catboost_model.predict(X_test)
    y_proba = catboost_model.predict_proba(X_test)[:,1]
    
    # Calcular métricas
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    precision_scores.append(report['1']['precision'])
    recall_scores.append(report['1']['recall'])
    f1_scores.append(report['1']['f1-score'])
    roc_auc_scores.append(roc_auc_score(y_test, y_proba))
    
# Mostrar las métricas promedio
print(f'Precisión promedio (Clase 1 - Fraude): {sum(precision_scores)/len(precision_scores):.4f}')
print(f'Recall promedio (Clase 1 - Fraude): {sum(recall_scores)/len(recall_scores):.4f}')
print(f'F1-Score promedio (Clase 1 - Fraude): {sum(f1_scores)/len(f1_scores):.4f}')
print(f'ROC AUC promedio: {sum(roc_auc_scores)/len(roc_auc_scores):.4f}')

# Generar la matriz de confusión para el último fold
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Actual')
plt.show()

# Generar valores SHAP para el último fold
explainer = shap.TreeExplainer(catboost_model)
shap_values = explainer.shap_values(X_test)

# Gráfico de resumen SHAP
shap.summary_plot(shap_values, X_test)
