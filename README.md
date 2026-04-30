# Detección de Fraude en Transacciones en Línea

Proyecto académico de **Métodos Numéricos** — Universidad Peruana Cayetano Heredia.
Comparación experimental de cuatro algoritmos de aprendizaje supervisado aplicados a la detección de fraude en transacciones financieras digitales.

---

## Sobre el proyecto

Este repositorio contiene la implementación práctica de un análisis comparativo entre **Regresión Logística**, **K-Nearest Neighbors (KNN)**, **Árbol de Decisión** y **Random Forest** sobre un escenario realista de fraude en línea con desbalance severo de clases (~3.5% de transacciones fraudulentas).

Los dos primeros modelos están implementados **manualmente desde cero** usando únicamente NumPy, aplicando los principios de optimización vistos en el curso (gradiente descendente, distancia euclidiana, optimización convexa). Los dos últimos se incorporan como **rama referencial externa** mediante `scikit-learn`, sirviendo como punto de comparación contra implementaciones estándar de la industria.

El informe metodológico completo, hipótesis, marco teórico y discusión de resultados se entregan como documento aparte (PDF).

---

## Dataset

Se utilizó el dataset **IEEE-CIS Fraud Detection** publicado por Vesta Corporation en Kaggle (2019), compuesto por dos archivos relacionales:

| Archivo | Filas | Columnas | Descripción |
|---|---|---|---|
| `train_transaction.csv` | 590,540 | 394 | Información de cada transacción (monto, tarjeta, dirección, conteos, deltas temporales y features anonimizadas V1–V339) |
| `train_identity.csv` | 144,233 | 41 | Información de identidad digital del comprador (dispositivo, navegador, conexión) |

La variable objetivo es `isFraud` (binaria). El dataset original presenta un desbalance severo: solo el **3.50%** de las transacciones corresponden a fraude.

Ambos archivos vienen incluidos en este repositorio dentro de la carpeta local `ARCHIVOS CSV/`, lista para que puedas descargarlos junto con el resto del proyecto. Para ejecutar el notebook desde Google Colab, deberás subirlos a tu propio Google Drive (ver siguiente sección).

**Fuente original del dataset:** https://www.kaggle.com/c/ieee-fraud-detection

---

## Estructura del proyecto (descarga local)

```
Proyecto_deteccion_de_fraude/
├── README.md                                  # Este archivo
├── ARCHIVOS CSV/                              # Datasets para descarga local
│   ├── train_identity.csv                     # Dataset complementario de identidad
│   └── train_transaction.csv                  # Dataset principal con la variable objetivo
├── VERSION_FINAL_DETECCION_DE_FRAUDE.ipynb    # Notebook principal (Google Colab)
└── version_final_deteccion_de_fraude.py       # Versión exportada como script Python
```

> La carpeta `ARCHIVOS CSV/` está pensada únicamente para que los archivos viajen junto al proyecto al descargarlo. Para ejecutar el notebook en Google Colab los CSV deben subirse luego a Google Drive en la ubicación esperada por el código (ver instrucciones más abajo).

---

## Cómo ejecutar el proyecto

El código está preparado para correr en **Google Colab**. Se ofrece también como script `.py` por si se desea ejecutar en un entorno local con Python.

### Opción A — Google Colab (recomendada)

**1.** Descarga este repositorio a tu computador. Dentro encontrarás la carpeta `ARCHIVOS CSV/` con los dos datasets.

**2.** Sube los dos CSV a tu **Google Drive** dentro de una carpeta llamada exactamente `Archivos_CSV_Deteccion_Fraude` en la raíz de tu Drive:

```
MyDrive/
└── Archivos_CSV_Deteccion_Fraude/
    ├── train_identity.csv
    └── train_transaction.csv
```

Esta es la ruta que el notebook espera por defecto. Si prefieres otra ubicación, modifica la constante `CARPETA` en la celda de configuración inicial.

**3.** Sube el notebook `VERSION_FINAL_DETECCION_DE_FRAUDE.ipynb` a Colab (o ábrelo desde Drive con clic derecho → *Abrir con* → *Google Colaboratory*).

**4.** Ejecuta todo en orden con *Runtime → Run all*. La primera celda monta tu Google Drive y solicitará autenticación.

**Tiempo estimado de ejecución:** 5–7 minutos en Colab Free.

### Opción B — Entorno local con Python

Si prefieres correr el script `.py` localmente, asegúrate de tener instalado:

```
python >= 3.10
numpy
pandas
scikit-learn
```

Modifica la constante `CARPETA` para que apunte a la ruta local de la subcarpeta `ARCHIVOS CSV`, comenta las líneas de montaje de Google Drive y ejecuta:

```bash
python version_final_deteccion_de_fraude.py
```

---

## Pipeline implementado

```
1.  Carga e integración (LEFT JOIN sobre TransactionID)
2.  Optimización de tipos de datos
3.  Ordenamiento temporal por TransactionDT
4.  Partición temporal 80/20 (train / test)
5.  Submuestreo estratificado del train (100,000 filas)
6.  Eliminación de columnas con >90% de nulos
7.  Filtro de correlación (>0.90) sobre numéricas
8.  Imputación (mediana / "Missing")
9.  Codificación categórica (one-hot + frequency encoding)
10. División en rama manual (escalada) y rama referencial
11. Validación temporal interna (subtrain / val)
12. Entrenamiento de los 4 modelos
13. Evaluación con 6 métricas
```

---

## Modelos y configuración

| Modelo | Implementación | Tratamiento del desbalance |
|---|---|---|
| Regresión Logística | Manual (gradiente descendente + L2) | Cost-sensitive learning (pesos w₀, w₁) |
| KNN | Manual (distancia euclidiana, predicción por chunks) | Selección de umbral |
| Árbol de Decisión | scikit-learn (rama referencial) | `class_weight='balanced'` |
| Random Forest | scikit-learn (rama referencial) | `class_weight='balanced'` |

**Métricas calculadas (todas implementadas manualmente):**
Precision, Recall, F1-score, MCC, AUC-ROC, AUPRC.

---

## Reproducibilidad

Todos los pasos que involucran aleatoriedad (submuestreo estratificado, partición interna, bootstrap de Random Forest) usan `random_state=42`. Ejecutar el notebook múltiples veces produce los mismos resultados.

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
