# Optimización Numérica y Modelos de Ensamble para la Detección de Fraude Financiero

Proyecto académico de **Métodos Numéricos para Machine Learning** — Universidad Peruana Cayetano Heredia.
Análisis comparativo de modelos supervisados y optimización numérica del umbral de decisión sobre un escenario masivo y fuertemente desbalanceado.

---

## Sobre el Proyecto

Este proyecto aborda la detección de fraude en transacciones financieras digitales usando el dataset masivo **IEEE-CIS Fraud Detection**. El eje metodológico aplica los temas del curso a un problema real:

1. **Unidad 1 — Optimización Numérica.** Implementación explícita del **Gradiente Descendente** para la regresión logística (con análisis del efecto de la tasa de aprendizaje η), y formulación de la **selección del umbral de decisión `t*`** como un problema de optimización unidimensional que maximiza el F1-score.
2. **Unidad 2 — Álgebra Lineal Numérica.** **PCA vía Descomposición en Valores Singulares (SVD)**: cálculo de autovalores/autovectores, varianza explicada y reducción dimensional.

Adicionalmente se aplica **SMOTE** para tratar el desbalance y se contrastan modelos base contra ensambles por *Bagging* (**Random Forest**) y *Gradient Boosting* (**XGBoost**), mostrando de forma cuantitativa el *trade-off* entre *Precision* y *Recall*.

---

## El Dataset

Se utilizó el dataset relacional **IEEE-CIS Fraud Detection** (Vesta Corporation / Kaggle):

* `train_transaction.csv`: 590,540 filas × 394 columnas.
* `train_identity.csv`: 144,233 filas × 41 columnas.

Ambas tablas se integran con un **LEFT JOIN** sobre `TransactionID`, dando una matriz de **590,540 × 434**. La información de identidad solo está presente en ~24% de las transacciones, por lo que esas columnas quedan en su mayoría vacías.

### Reto del desbalance y eficiencia de memoria
* **Desbalance crítico:** la variable objetivo `isFraud` es fraude en solo el **3.50%** de los casos.
* **Eficiencia computacional:** limpieza de columnas con >90% de nulos, *downcasting* a `float32` y codificación categórica mediante *Frequency Encoding* para mantener manejable el dataset en Colab.

---

## Pipeline de ejecución (estructura del notebook)

Archivo principal: **`Proyecto_deteccion_fraude_MNPOML_ParteII.ipynb`**

* **ETAPA 1 — Datos:** carga, integración relacional (Left Join), ingeniería de características *row-wise* y filtrado de columnas casi vacías.
* **ETAPA 2 — Partición temporal:** división cronológica en **train (64%) / validación (16%) / test (20%)** para una evaluación rigurosa.
* **ETAPA 3 — Preprocesamiento:** imputación por mediana, *Frequency Encoding* y optimización de memoria (`float32`). Todos los estadísticos se ajustan solo en train.
* **ETAPA 4 — Gradiente Descendente (U1):** implementación explícita del descenso de gradiente para la regresión logística y análisis del efecto de la tasa de aprendizaje η en la convergencia.
* **ETAPA 5 — SVD / PCA (U2):** análisis de correlación, descomposición en valores singulares, autovalores, varianza explicada y proyección del espacio reducido.
* **ETAPA 6 — Modelos base:** Regresión Logística, KNN, Árbol de Decisión y Random Forest (con estandarización donde corresponde).
* **ETAPA 7 — SMOTE + optimización del umbral:** balanceo sintético sobre el train y búsqueda numérica del umbral óptimo `t*` sobre el conjunto de validación.
* **ETAPA 8 — Resultados:** comparación final y análisis de sensibilidad multiumbral.

---

## Resultados finales

Tras aplicar el pipeline completo sobre las 590,540 transacciones y el algoritmo de búsqueda del umbral óptimo, los resultados consolidados son:

| Modelo / Enfoque | Precision | Recall | F1-Score | MCC | AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Regresión Logística | 0.3552 | 0.2188 | 0.2707 | 0.2589 | 0.8119 |
| KNN | 0.6572 | 0.2382 | 0.3496 | 0.3839 | 0.7062 |
| Árbol de Decisión | 0.5459 | 0.2852 | 0.3747 | 0.3797 | 0.7832 |
| Random Forest (base, $t=0.50$) | **0.8291** | 0.2436 | 0.3766 | 0.4406 | 0.8546 |
| Random Forest + SMOTE ($t^*=0.30$) | 0.3815 | 0.4094 | 0.3950 | 0.3729 | 0.8793 |
| **XGBoost ($t^*=0.30$)** | 0.5534 | **0.4294** | **0.4836** | **0.4716** | **0.8874** |

### Conclusiones clave
1. **La optimización del umbral es tan importante como la elección del modelo.** Al reemplazar el corte estático de 0.5 por un umbral óptimo `t*` obtenido numéricamente sobre validación, se incrementa de forma notable la captura de fraudes reales.
2. **Trade-off Precisión–Recall.** Bajar el umbral maximiza el *Recall* (atrapar más fraudes) a costa de la *Precisión* (más falsas alarmas); el equilibrio se cuantifica con F1 y MCC.
3. **Caso del Random Forest base.** Logra una precisión altísima (0.8291) pero un *Recall* muy bajo (0.2436): deja pasar demasiados fraudes. Con SMOTE y la calibración del umbral el *Recall* casi se duplica, dando un modelo mucho más útil para la entidad financiera.
4. **Modelo de mejor desempeño: XGBoost.** Combinando *Gradient Boosting* con el umbral optimizado (`t*=0.30`) alcanza el mejor equilibrio global, liderando las métricas más exigentes para datos desbalanceados: **F1-Score 0.4836, MCC 0.4716 y AUC 0.8874.**

---

## Cómo ejecutar en Google Colab

1. Descarga `train_transaction.csv` y `train_identity.csv` desde Kaggle (IEEE-CIS Fraud Detection).
2. Súbelos a tu Google Drive en la ruta: `MyDrive/Archivos_CSV_Deteccion_Fraude/`.
3. Abre **`Proyecto_deteccion_fraude_MNPOML_ParteII.ipynb`** en Google Colab.
4. Selecciona un entorno con alta prioridad de RAM (Colab Pro recomendado para `n_jobs=-1`) y ejecuta todas las celdas en orden.

---

## Autores

- **Lopez Vega, Juan Diego**
- **Valenzuela Valer, Luis Martin**
- **Roldán Montalvan, Jorge Esteban**

Curso: Métodos Numéricos para Machine Learning
Docentes: Josue Angel Mauricio Salazar · Jeremy Karsen Holguin Mori
Universidad Peruana Cayetano Heredia

---

## Referencias clave

- IEEE Computational Intelligence Society. (2019). *IEEE-CIS Fraud Detection*. Kaggle.
- Sinap, V. (2024). *Comparative analysis of machine learning techniques for credit card fraud detection*. Turkish Journal of Engineering, 8(2), 196–208.
- Le Borgne, Y.-A., Siblini, W., Lebichot, B., & Bontempi, G. (2022). *Reproducible Machine Learning for Credit Card Fraud Detection — Practical Handbook*. Université Libre de Bruxelles.
- Hafez, I. Y., et al. (2025). *A systematic review of AI-enhanced techniques in credit card fraud detection*. Journal of Big Data, 12.

> Para la bibliografía completa, consultar el documento PDF del informe oficial.
