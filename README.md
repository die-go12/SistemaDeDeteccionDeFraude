# Optimización Numérica y Modelos de Ensamble para la Detección de Fraude Financiero

Proyecto académico de **Métodos Numéricos** — Universidad Peruana Cayetano Heredia.  
Análisis comparativo de alta capacidad y optimización del umbral crítico sobre un escenario masivo y desbalanceado.

---

## Sobre el Proyecto

Este proyecto aborda el problema de la detección de fraudes en transacciones financieras digitales utilizando el dataset masivo **IEEE-CIS Fraud Detection**. El núcleo metodológico de la investigación se centra en resolver el colapso de los modelos lineales tradicionales ante desbalances críticos mediante dos estrategias de la ingeniería de datos y métodos numéricos:

1. **Suavizado Geométrico de Fronteras de Decisión:** Uso de Técnicas de Remuestreo Sintético (SMOTE) para balancear la carga informativa en el espacio de entrenamiento.
2. **Optimización Numérica Unidimensional No Restringida:** Formulación y programación de un algoritmo iterativo de búsqueda lineal para encontrar el umbral de decisión óptimo ($t^*$) que maximiza la función objetivo del negocio ($F_1\text{-score}$ y $MCC$).

Se contrastan modelos base de la industria contra estructuras de ensamble por *Bagging* (**Random Forest de Alta Capacidad**) y *Gradient Boosting* (**XGBoost**), demostrando de forma matemática cómo se comporta el *trade-off* dinámico entre *Precision* y *Recall*.

---

## El Dataset y Gestión de Memoria (RAM)

Se utilizó el dataset relacional **IEEE-CIS Fraud Detection** (Vesta Corporation / Kaggle), compuesto por:
* `train_transaction.csv`: 590,540 filas y 394 columnas.
* `train_identity.csv`: 144,233 filas y 41 columnas.

### Reto del Desbalance y Cuidado de la RAM
* **Desbalance Crítico:** La variable objetivo `isFraud` presenta un desbalance severo donde solo el **3.50%** de las transacciones son fraudulentas.
* **Optimización Computacional:** Para evitar el desbordamiento de memoria en Colab Pro debido a las 425 columnas resultantes del `LEFT JOIN`, se ejecutó una limpieza estricta (eliminación de columnas con $>90\%$ de nulos), downcasting de tipos numéricos a `float32` y codificación categórica eficiente mediante *Frequency Encoding*.

---

##  Pipeline de Ejecución (Estructura del Notebook)

El flujo de trabajo en el archivo `VERSION_FINAL_DETECCION_DE_FRAUDE.ipynb` está estructurado en las siguientes etapas limpias:

* **ETAPA 1:** Carga, Integración Relacional y Reducción Eficiente de Memoria (RAM).
* **ETAPA 2:** Partición Temporal Estricta (80% pasado para entrenamiento, 20% futuro para evaluación), evitando la fuga de datos (*Data Leakage*).
* **ETAPA 3:** Ingeniería de Características y *Frequency Encoding*.
* **ETAPA 4:** Balanceo del Espacio de Características mediante SMOTE Ligero.
* **ETAPA 5:** Evaluación de Modelos Base e Inestabilidad Lineal (Análisis del colapso del Gradiente Descendente SAGA en Regresión Logística).
* **ETAPA 6:** Modelos de Ensamble y Optimización Numérica del Umbral ($t^*$).
* **ETAPA 7:** Análisis de Sensibilidad Multiumbral y Modelos Avanzados (XGBoost).

---

##  Resultados Finales Actualizados

Tras someter los modelos al pipeline optimizado y al algoritmo de búsqueda de umbral, los resultados finales consolidados en la **Etapa 7** son:

| Modelo / Enfoque | Precision | Recall | F1-Score | MCC | AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (SAGA) | 0.0000 | 0.0000 | 0.0000 | -0.0192 | 0.4813 |
| **KNN** (BallTree) | 0.3780 | 0.0763 | 0.1269 | 0.1576 | 0.6452 |
| **Decision Tree** | 0.4076 | 0.2562 | 0.3146 | 0.3044 | 0.7341 |
| **Random Forest** (Base $t=0.50$) | **0.8473** | 0.2485 | 0.3843 | 0.4503 | 0.8594 |
| **Random Forest** (Optimizado $t^*=0.38$) | 0.6886 | 0.3819 | 0.4913 | 0.5006 | 0.8949 |
| **XGBoost** (Optimizado $t^*=0.27$) | 0.6495 | **0.4865** | **0.5563** | **0.5488** | **0.9174** |

### 🔬 Conclusiones Clave del Análisis de Sensibilidad:
1. **Colapso Lineal:** La Regresión Logística arroja métricas nulas debido a que la función de costo de entropía cruzada se vuelve plana ante el desbalance del 96.5%. El Gradiente Descendente converge a un mínimo global trivial (predecir siempre clase 0).
2. **Ley del Intercambio (Trade-off):** Al evaluar umbrales propuestos ($t = 0.10, 0.20, 0.30$), se demostró experimentalmente que bajar el umbral maximiza el *Recall* (captura de fraudes) pero destruye la *Precision* debido al incremento exponencial de falsos positivos.
3. **El Máximo Global Real:** **XGBoost con $t^* = 0.27$** se consolida como el modelo campeón absoluto. Al usar un enfoque iterativo secuencial (*Gradient Boosting*), optimiza el espacio métrico alcanzando un **AUC de 0.9174** y un **MCC de 0.5488**, el balance financiero ideal para la institución bancaria.

---

## 🛠️ Cómo Ejecutar en Google Colab

1. Descarga los archivos `train_transaction.csv` y `train_identity.csv` desde Kaggle.
2. Súbelos a tu Google Drive en la ruta: `MyDrive/Archivos_CSV_Deteccion_Fraude/`.
3. Abre el archivo `VERSION_FINAL_DETECCION_DE_FRAUDE.ipynb` en Google Colab.
4. Selecciona un entorno de ejecución con alta prioridad de RAM (Colab Pro recomendado para acelerar el procesamiento paralelo con `n_jobs=-1`) y ejecuta todas las celdas en orden.

---

## Autores

- **Lopez Vega, Juan Diego**
- **Valenzuela Valer, Luis Martin**
- **Roldán Montalvan, Jorge Esteban**

Curso: Métodos Numéricos
Docentes: Josue Angel Mauricio Salazar · Jeremy Karsen Holguin Mori
Universidad Peruana Cayetano Heredia

---

## Referencias clave

- IEEE Computational Intelligence Society. (2019). *IEEE-CIS Fraud Detection*. Kaggle.
- Sinap, V. (2024). *Comparative analysis of machine learning techniques for credit card fraud detection*. Turkish Journal of Engineering, 8(2), 196–208.
- Le Borgne, Y.-A., Siblini, W., Lebichot, B., & Bontempi, G. (2022). *Reproducible Machine Learning for Credit Card Fraud Detection — Practical Handbook*. Université Libre de Bruxelles.
- Hafez, I. Y., et al. (2025). *A systematic review of AI-enhanced techniques in credit card fraud detection*. Journal of Big Data, 12.

> Para la bibliografía completa, consultar el documento PDF del informe oficial.
