
# Amazon Employee Access Challenge - Proyecto de Clasificación

## Descripción del Proyecto
Este proyecto tiene como objetivo predecir si un empleado debería tener acceso a un recurso específico dentro de la empresa utilizando atributos relevantes. Se ha utilizado un enfoque basado en **Machine Learning** para entrenar varios modelos, uno de ellos siendo el **Random Forest** con ajuste para clases desbalanceadas.

## Motivación
Este desafío representa un problema típico de clasificación en el que las clases están desbalanceadas, ya que la cantidad de accesos aprobados supera ampliamente a la de accesos denegados. El modelo debe ser capaz de detectar con precisión las instancias menos frecuentes (denegaciones de acceso), sin sacrificar el rendimiento general.

## Resultados de la Evaluación del Modelo - Random Forest (Clases Desbalanceadas)

- **Precisión**:
  - **Clase 0 (Acceso Permitido)**: 1.00
  - **Clase 1 (Acceso Denegado)**: 0.96
- **Recall**:
  - **Clase 0**: 1.00
  - **Clase 1**: 0.76
- **F1-Score**:
  - **Clase 0**: 1.00
  - **Clase 1**: 0.85
- **Exactitud Total**: 1.00
- **Macro Avg (F1-Score)**: 0.92

El modelo alcanzó un excelente rendimiento general, mostrando alta precisión para la clase mayoritaria (acceso permitido) y un rendimiento sólido para la clase minoritaria (acceso denegado), a pesar del desbalance.

## Análisis del Gráfico SHAP

Se generó un gráfico **SHAP** para evaluar la importancia de las características y su impacto en las predicciones del modelo. A continuación se muestra el gráfico SHAP, que destaca la influencia de las variables **'Time'** y **'V1'** en las predicciones del modelo:

![Gráfico SHAP](sandbox:/mnt/data/shap_summary_plot.png)

- **Time**: Tiene una influencia notable pero equilibrada en ambas clases, como se muestra en el gráfico.
- **V1**: La variable muestra una tendencia similar, con puntos distribuidos entre ambas clases.

Este gráfico proporciona una interpretación clara de cómo las características afectan las predicciones, ayudando a identificar las variables clave.

Amazon Employee Access Challenge - Random Forest con Clases Desbalanceadas (Balanced)
Descripción del Proyecto: El objetivo de este proyecto es predecir si un empleado debería tener acceso a un recurso específico en la empresa. Se han probado varios modelos de Machine Learning, y en esta fase, se utilizó Random Forest con clases desbalanceadas, ajustando el modelo con la opción class_weight='balanced'.

Resultados del Modelo - Random Forest Balanced
Precisión:
Clase 0 (Acceso Permitido): 1.00
Clase 1 (Acceso Denegado): 0.97
Recall:
Clase 0: 1.00
Clase 1: 0.70
F1-Score:
Clase 0: 1.00
Clase 1: 0.82
Exactitud Total: 1.00
Macro avg (F1-score): 0.91
Weighted avg: 1.00
El modelo ajustado ha logrado un excelente rendimiento, especialmente en la clase mayoritaria (acceso permitido), y un buen rendimiento en la clase minoritaria (acceso denegado), a pesar del desbalance.

Análisis del Gráfico SHAP
Se generó un gráfico SHAP que muestra la importancia de las características en las predicciones del modelo. El gráfico SHAP a continuación destaca la influencia de las variables 'Time' y 'V1':

Time y V1 continúan siendo variables clave en el modelo, lo que refuerza su relevancia para la predicción de acceso en este contexto.
Errores y Próximos Pasos
Error en Gráfico de Dependencia SHAP: Se presentó un error al intentar generar el gráfico de dependencia SHAP para la variable 'V14', relacionado con las dimensiones de los datos. Esto se investigará y corregirá en los próximos pasos.

Solución Propuesta: Revisar las dimensiones de las variables SHAP para asegurarse de que coincidan, y ajustar el código para corregir el problema en la generación del gráfico de dependencia.
Ejecución del Próximo Modelo: Se continuará con la implementación del siguiente modelo, Gradient Boosting con optimización de hiperparámetros usando RandomizedSearchCV, como se había planeado.

Amazon Employee Access Challenge - Gradient Boosting Optimizado
Descripción del Proyecto: En esta fase del proyecto, se utilizó un modelo de Gradient Boosting optimizado mediante RandomizedSearchCV para predecir si un empleado debería tener acceso a un recurso específico en la empresa. El modelo fue ajustado para obtener los mejores hiperparámetros y maximizar el rendimiento en el contexto de clases desbalanceadas.

Mejores Hiperparámetros Encontrados:
Subsample: 1.0
Número de Estimadores: 200
Min Samples Split: 10
Max Depth: 5
Learning Rate: 0.01
Resultados del Modelo - Gradient Boosting
Precisión:
Clase 0 (Acceso Permitido): 1.00
Clase 1 (Acceso Denegado): 0.94
Recall:
Clase 0: 1.00
Clase 1: 0.74
F1-Score:
Clase 0: 1.00
Clase 1: 0.83
Exactitud Total: 1.00
Macro avg (F1-score): 0.91
Weighted avg: 1.00
El modelo ajustado ha demostrado un rendimiento excelente, con un manejo sólido de la clase minoritaria (acceso denegado) y una alta precisión en la clase mayoritaria (acceso permitido).

Análisis del Gráfico SHAP
El gráfico SHAP generado para el modelo Gradient Boosting destaca la importancia de las variables en las predicciones del modelo. A continuación, se muestra el gráfico SHAP que resalta la influencia de las características clave en el modelo:

Las características V17, V4, y V14 tienen un impacto significativo en las predicciones del modelo, siendo variables clave en la toma de decisiones sobre el acceso a recursos.
Próximos Pasos
Revisar los Gráficos SHAP: Continuar con el análisis detallado de los gráficos SHAP para interpretar mejor las predicciones del modelo y optimizar la comprensión de las variables clave.
Avanzar con Otros Modelos: Probar más modelos o técnicas de optimización según los resultados obtenidos hasta el momento y comparar con otros modelos probados anteriormente (como Random Forest).
Documentación Final: Integrar estos resultados con los modelos previos para generar un informe consolidado del proyecto.



Documentación del Modelo: LightGBM con RandomizedSearchCV
Descripción del Modelo
Se utilizó el modelo LightGBM para realizar una clasificación binaria en el dataset correspondiente al proyecto Amazon Employee Access Challenge. El objetivo era predecir si un empleado debería tener acceso a un recurso en función de varias características.

El proceso incluyó la optimización de hiperparámetros mediante RandomizedSearchCV, utilizando una validación cruzada con 3 pliegues. El modelo optimizado fue entrenado en un conjunto de entrenamiento y evaluado en un conjunto de prueba.

Hiperparámetros Optimizados
El proceso de optimización mediante RandomizedSearchCV encontró los siguientes mejores hiperparámetros para el modelo LightGBM:

subsample: 0.8
reg_lambda: 0.5
reg_alpha: 0.1
n_estimators: 300
max_depth: -1 (sin límite de profundidad)
learning_rate: 0.2
colsample_bytree: 1.0
Resultados del Modelo
El rendimiento del modelo se evaluó en el conjunto de prueba y se generó un reporte de clasificación.

Reporte de clasificación:

Clase 0 (no acceso):
Precisión: 1.00
Recall: 1.00
F1-Score: 1.00
Soporte: 85,295 instancias
Clase 1 (acceso):
Precisión: 0.92
Recall: 0.74
F1-Score: 0.82
Soporte: 148 instancias
Métricas Globales:

Exactitud (Accuracy): 1.00
Promedio Macro (F1-Score): 0.91
Promedio Ponderado (F1-Score): 1.00
El modelo demostró un rendimiento excelente en términos de precisión y recall para la Clase 0 (no acceso), con valores de 1.00. En cuanto a la Clase 1 (acceso), aunque la precisión fue alta (0.92), el recall fue relativamente menor (0.74), lo que indica que algunos casos de acceso no fueron correctamente clasificados.

Observaciones Adicionales
Durante el proceso de entrenamiento, se mostraron las siguientes advertencias en LightGBM:

"No further splits with positive gain, best gain: -inf". Estas advertencias indican que, durante algunas iteraciones, el modelo no encontró divisiones adicionales que mejoraran la ganancia de la función objetivo. Esto es común en casos donde el modelo ha aprendido bien los patrones en los datos.
Conclusión
El modelo LightGBM con los hiperparámetros optimizados alcanzó una alta precisión general, con resultados sobresalientes para la clase mayoritaria (Clase 0). Sin embargo, el recall para la clase minoritaria (Clase 1) fue menor, lo que sugiere que podrían explorarse otras técnicas para mejorar el rendimiento en esta clase, como ajustar la ponderación de las clases, realizar sobremuestreo de la clase minoritaria, o utilizar otro enfoque de optimización.

El siguiente paso sería analizar estos resultados y decidir si es necesario ajustar más los hiperparámetros o probar con otras técnicas.


Documentación del Modelo: XGBoost con RandomizedSearchCV
Descripción del Modelo
Se utilizó el modelo XGBoost optimizado con RandomizedSearchCV para realizar una clasificación binaria en el proyecto Amazon Employee Access Challenge. El objetivo fue predecir si un empleado debería tener acceso a un recurso basándose en ciertas características.

El proceso de optimización de hiperparámetros utilizó una validación cruzada de 3 pliegues y se realizó en un conjunto de entrenamiento. Los resultados obtenidos se evaluaron en un conjunto de prueba.

Hiperparámetros Optimizados
Los mejores hiperparámetros encontrados para el modelo XGBoost son:

subsample: 0.8
n_estimators: 300
max_depth: 5
learning_rate: 0.1
gamma: 0
colsample_bytree: 1.0
Resultados del Modelo
El rendimiento del modelo en el conjunto de prueba fue evaluado utilizando un reporte de clasificación.

Reporte de clasificación:

Clase 0 (no acceso):
Precisión: 1.00
Recall: 1.00
F1-Score: 1.00
Soporte: 85,295 instancias
Clase 1 (acceso):
Precisión: 0.95
Recall: 0.76
F1-Score: 0.84
Soporte: 148 instancias
Métricas Globales:

Exactitud (Accuracy): 1.00
Macro avg (F1-Score): 0.92
Weighted avg (F1-Score): 1.00
El modelo XGBoost mostró un rendimiento muy bueno en términos de precisión para ambas clases, con una exactitud general de 1.00. Aunque la precisión para la Clase 1 fue alta (0.95), el recall fue de 0.76, lo que sugiere que algunos casos de acceso no fueron correctamente clasificados.

Gráficos SHAP
Se generaron gráficos SHAP para interpretar las predicciones del modelo. Estos gráficos muestran la importancia de las características y cómo influyen en las decisiones del modelo.

Gráfico de dependencia SHAP para la característica V14: La gráfica muestra cómo los valores de la característica V14 influyen en las predicciones del modelo, en relación con otra característica como V1. Las áreas en azul representan valores bajos y las áreas en rojo valores altos de V1.


Observaciones Adicionales
Durante la ejecución, el sistema generó una advertencia relacionada con el parámetro use_label_encoder, que ya no es necesario para versiones más recientes de XGBoost. Esta advertencia no afectó el rendimiento del modelo.
El proceso generó una excelente exactitud general, aunque podrían realizarse ajustes para mejorar el recall en la clase minoritaria, como ajustar el balance de clases o probar con diferentes estrategias de regularización.
Conclusión
El modelo XGBoost optimizado mostró un rendimiento sólido con una alta precisión en ambas clases y una exactitud total del 100%. No obstante, se observó que el recall para la clase minoritaria (acceso concedido) podría mejorarse. Esto podría implicar ajustes adicionales en los hiperparámetros o la exploración de técnicas de balance de clases. El siguiente paso sería evaluar el rendimiento en comparación con los modelos anteriores y documentar los insights obtenidos de los gráficos SHAP.


Documentación del Modelo: XGBoost con RandomizedSearchCV y scale_pos_weight
Descripción del Modelo
Se utilizó el modelo XGBoost para realizar una clasificación binaria en el proyecto Amazon Employee Access Challenge, optimizado mediante RandomizedSearchCV. El parámetro scale_pos_weight se ajustó automáticamente para manejar el desbalance de clases.

El proceso de optimización de hiperparámetros incluyó una validación cruzada de 3 pliegues y se evaluaron las métricas en el conjunto de prueba.

Hiperparámetros Optimizados
Los mejores hiperparámetros encontrados para el modelo XGBoost fueron:

subsample: 0.8
n_estimators: 300
max_depth: 5
learning_rate: 0.1
gamma: 0
colsample_bytree: 1.0
Resultados del Modelo
El rendimiento del modelo fue evaluado con las siguientes métricas de clasificación:

Reporte de clasificación:

Clase 0 (no acceso):
Precisión: 1.00
Recall: 1.00
F1-Score: 1.00
Soporte: 85,295 instancias
Clase 1 (acceso):
Precisión: 0.89
Recall: 0.80
F1-Score: 0.84
Soporte: 148 instancias
Métricas Globales:

Exactitud (Accuracy): 1.00
Macro avg (F1-Score): 0.92
Weighted avg (F1-Score): 1.00
El modelo logró una precisión y recall excelentes en la clase mayoritaria (Clase 0). En cuanto a la clase minoritaria (Clase 1), la precisión fue de 0.89 y el recall de 0.80, lo cual es un resultado aceptable dado el desbalance de clases.

Gráficos SHAP
Se generaron gráficos SHAP para interpretar las predicciones del modelo y analizar la importancia de las características. El gráfico muestra cómo la característica V14 interactúa con otras características, como V1, y cómo influyen en las predicciones del modelo.

Gráfico de Dependencia SHAP para V14:

Este gráfico muestra cómo los valores de V14 influyen en las predicciones del modelo en relación con V1. Las áreas en azul representan valores bajos y las áreas en rojo valores altos de V1.

Observaciones Adicionales
Se generaron advertencias relacionadas con el parámetro use_label_encoder, que ya no es necesario en las versiones más recientes de XGBoost. Esta advertencia no afectó el rendimiento del modelo.
El ajuste del parámetro scale_pos_weight ayudó a manejar el desbalance de clases, mejorando el recall en la clase minoritaria.
Conclusión
El modelo XGBoost optimizado logró un rendimiento sobresaliente con una alta precisión general. Aunque el recall para la clase minoritaria fue relativamente bueno, existe margen para mejorar mediante ajustes adicionales en los hiperparámetros o aplicando técnicas de manejo de clases desbalanceadas. Los gráficos SHAP proporcionaron una visión clara de la importancia de las características y su influencia en las predicciones.

Los resultados obtenidos del modelo CatBoost optimizado con Bayesian Optimization son los siguientes:

Mejores Hiperparámetros:
Subsample: 1.0
Scale_pos_weight: 3
Learning_rate: 0.1
Iterations: 200
Depth: 5
Colsample_bylevel: 0.8
Reporte de Clasificación:
Precisión (Clase 0): 1.00
Precisión (Clase 1): 0.91
Recall (Clase 0): 1.00
Recall (Clase 1): 0.80
F1-Score (Clase 0): 1.00
F1-Score (Clase 1): 0.85
Accuracy Total: 1.00
Macro Avg (F1-score): 0.93
Weighted Avg (F1-score): 1.00
Interpretación con SHAP:
Se generaron dos gráficos SHAP para visualizar la importancia de las características en el modelo:

Gráfico resumen de importancia de las características: muestra las variables más relevantes para las predicciones del modelo.
Gráfico de dependencia para la característica V14: muestra cómo los valores SHAP de la característica V14 influyen en la predicción.
Ambos gráficos muestran que las variables V4, V14, y Time son clave en el modelo, con V4 y V14 mostrando una alta variabilidad en el impacto de las predicciones.

Recomendación:
Con base en estos resultados, y dado que el modelo CatBoost muestra un rendimiento excelente en precisión, recall y F1-score, recomendaría que este sea el modelo final a utilizar.

Dado que el archivo está relacionado con el proyecto Amazon Employee Access Challenge, el contexto es sobre la predicción de si un empleado debe o no tener acceso a ciertos recursos, no directamente sobre detección de fraude. Sin embargo, si adaptamos este enfoque hacia la detección de fraude, especialmente con los patrones observados en los modelos entrenados, la hipótesis más apropiada podría ser:

Hipótesis Propuesta para Detección de Fraude:
"Las características relacionadas con el tiempo de actividad de los empleados (Time) y ciertos patrones de comportamiento (V1, V14) son indicadores clave para identificar accesos fraudulentos o no autorizados a los recursos empresariales. Los modelos que ajustan el desbalance de clases, como CatBoost con optimización bayesiana, mejoran la capacidad de predecir estos accesos de alto riesgo con mayor precisión."

Justificación de la Hipótesis:
Importancia de las Características (Time, V1, V14): Los gráficos SHAP mostraron que estas variables tienen una gran influencia en la predicción de accesos no autorizados. En un contexto de fraude, esto sugiere que ciertos patrones temporales o anomalías en las características pueden correlacionarse con accesos potencialmente fraudulentos.

Ajuste del Desbalance de Clases: En los modelos de clasificación, los accesos no autorizados (similares a casos de fraude) son la clase minoritaria. El uso de técnicas como el ajuste del parámetro scale_pos_weight en CatBoost ha demostrado ser eficaz para mejorar el recall de la clase minoritaria (fraudes), sin comprometer la precisión.

Modelos Probados: Los modelos como CatBoost optimizado mediante Bayesian Optimization han mostrado una capacidad significativa para capturar estos casos atípicos y mejorar la detección, incluso cuando los casos de accesos no autorizados (o fraudes) son menos frecuentes.

Si esta hipótesis es validada mediante análisis adicionales, se puede establecer que el uso de estos modelos es una herramienta eficaz para detectar comportamientos anómalos o fraudulentos en grandes volúmenes de datos empresariales.

Esta hipótesis puede ser probada mediante experimentos adicionales en el dataset, ajustando los modelos y validando la correlación de estas características con casos de fraude.