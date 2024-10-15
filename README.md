# Machine Learning para Detección de Fraude y Accesos No Autorizados

Este proyecto abarca múltiples enfoques para la detección de fraudes y predicción de accesos no autorizados, utilizando tanto técnicas supervisadas como no supervisadas de machine learning. A lo largo del proyecto se implementaron modelos para predecir el acceso a recursos por empleados de Amazon, así como algoritmos avanzados de deep learning y clustering para la detección de patrones anómalos en datos financieros.

## 1. Descripción del Proyecto

El **Amazon Employee Access Challenge** se centra en predecir si un empleado debería tener acceso a un recurso específico, un problema de clasificación binaria con clases desbalanceadas. Este proyecto también extiende las técnicas de machine learning aplicadas a la detección de fraudes en transacciones financieras usando redes neuronales profundas y clustering para identificar comportamientos anómalos.

## 2. Estructura del Proyecto

### a. **Modelos Supervisados**

Se evaluaron varios modelos de machine learning para la clasificación binaria:

- **Random Forest con Clases Desbalanceadas**: Ajustado para mejorar la precisión de la clase minoritaria (accesos denegados).
- **Gradient Boosting**: Optimizado con `RandomizedSearchCV`, se logró un buen balance entre precisión y recall.
- **XGBoost**: Utilizando optimización de hiperparámetros, demostró ser efectivo en datos desbalanceados.
- **CatBoost con Bayesian Optimization**: Este modelo ofreció el mejor rendimiento general, logrando altos valores de precisión y recall en la clase minoritaria.

#### Métricas Clave:
- **Precisión**: Hasta 1.00 en clase 0, y hasta 0.96 en clase 1.
- **Recall**: Hasta 0.80 en la clase minoritaria (accesos denegados).
- **F1-Score**: Valores balanceados, destacando la clase minoritaria.

### b. **Análisis SHAP**

Se utilizaron gráficos SHAP para interpretar las características más influyentes en las predicciones. Las variables clave fueron:
- **V14** y **V4**: Presentaron alta variabilidad e impacto en las predicciones de los modelos.
- **Time**: Mostró una influencia crítica en la detección de accesos no autorizados.

### c. **Técnicas de Deep Learning**

Se implementaron varios modelos de deep learning utilizando **Keras** y **TensorFlow**:

- **Red Neuronal Profunda (DNN)**: Se logró una exactitud de hasta el 99.94% en la detección de fraudes en datos financieros.
- **Redes Convolucionales (CNN)**: Aplicadas a datos tabulares para mejorar la extracción de características.
- **Redes Recurrentes (RNN)**: Exploración de dependencias temporales en los datos para la predicción de fraudes a lo largo del tiempo.
- **Autoencoders**: Utilizados para la detección de anomalías y reducción de dimensionalidad.

### d. **Técnicas No Supervisadas**

- **K-Means Clustering**: Se aplicó a los datos transformados por PCA para identificar grupos de transacciones con comportamientos similares, lo que permitió segmentar transacciones y posibles fraudes.
- **DBSCAN**: Se utilizó para detectar outliers o transacciones potencialmente fraudulentas que no encajaban en ninguno de los clusters principales.
- **t-SNE**: Ayudó a visualizar las agrupaciones de datos en dimensiones reducidas, facilitando la interpretación de los resultados de clustering.

## 3. Conclusiones y Recomendaciones

- **Mejor Modelo**: El modelo CatBoost optimizado con técnicas bayesianas fue el que mostró mejor rendimiento para el problema de accesos no autorizados.
- **Detección de Anomalías**: DBSCAN demostró ser útil para identificar transacciones fuera de lo común, sugiriendo potenciales fraudes.
- **Monitoreo de Variables Críticas**: Se recomienda implementar un sistema de monitoreo en tiempo real para las variables más influyentes (como V14, V4 y Time) para mejorar la seguridad en los accesos y la detección de fraudes.

## 4. Próximos Pasos

- Ajustar el modelo CatBoost con nuevos datos de acceso para mantener un sistema de predicción eficiente.
- Explorar nuevas técnicas de **Transfer Learning** y **Reinforcement Learning** para mejorar la detección de fraudes en entornos dinámicos.
- Ampliar el uso de **Deep Learning** para la identificación de patrones más complejos y detección proactiva de fraudes.

## 5. Estructura del Repositorio

- **Notebooks**: Contiene los análisis exploratorios y los entrenamientos de modelos supervisados y no supervisados.
- **Documentación**: Informes detallados de los avances del proyecto y las metodologías utilizadas.
- **Imágenes**: Visualizaciones generadas por SHAP, t-SNE y otros análisis visuales.

---

Este README proporciona una descripción clara y detallada de los objetivos, técnicas y resultados clave del proyecto.
