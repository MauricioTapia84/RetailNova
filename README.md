# RetailNova: predicción de abandono y fidelización

Prototipo académico de comercio minorista desarrollado para la EP5 de Gestión de Proyectos de Datos. El objetivo es explorar señales de comportamiento que podrían ayudar a priorizar acciones de fidelización. **Todos los registros son sintéticos**: no representan clientes, compras ni resultados comerciales reales.

## Resumen ejecutivo

- Se generó una cohorte sintética de **185.000 clientes recurrentes** y un historial mensual agregado de **36 meses**, desde abril de 2023 hasta marzo de 2026.
- El abandono (`churn_6m`) se define como **cero compras desde el 1 de abril de 2026 inclusive hasta el 1 de octubre de 2026 exclusive**.
- El mejor clasificador seleccionado en validación es una regresión logística con pesos de clase, calibrada con sigmoide CV3. El umbral binario final, elegido en validación, es **0,2067**.
- En el test sintético de 37.000 clientes, el modelo obtuvo **average precision 0,4170**, **ROC-AUC 0,6748**, **recall 70,08%**, **precision 32,33%** y **F1 44,25%**.
- Estos resultados son una **reevaluación del prototipo** sobre datos sintéticos, no evidencia de desempeño con clientes reales ni una validación confirmatoria independiente.

## Objetivo y definición temporal

El objetivo de negocio académico es explorar cómo priorizar clientes para posibles acciones de fidelización. El objetivo analítico es estimar abandono en una ventana de seis meses posterior al corte.

- **Corte de predictores:** antes del 1 de abril de 2026.
- **Ventana objetivo:** del 1 de abril de 2026 inclusive al 1 de octubre de 2026 exclusive.
- **Etiqueta:** `churn_6m = 1` cuando `compras_periodo_objetivo` es cero; en caso contrario, `0`. Se puede verificar con la fecha de última compra al cierre.
- **Cohorte:** se simulan clientes con al menos dos compras en los 12 meses previos al corte. Es un supuesto académico; el caso de negocio informa clientes registrados, no que todos sean recurrentes.
- **Historial:** 36 meses agregados por cliente y mes (abril de 2023–marzo de 2026); los meses sin fila representan ausencia de actividad.
- **Generación:** semilla 42; prevalencia esperada de abandono cercana al 24%, sin imponer una tasa observada exacta ni crear predictores a partir de la etiqueta.

## Ejecución reproducible

Requiere Python 3.11 o superior. Desde la raíz del proyecto:

```bash
python -m pip install -r requirements.txt
python src/validate_notebook_execution.py
python src/verify_project.py
```

El validador ejecuta en orden los tres notebooks: genera la fuente sintética, crea la partición de desarrollo/test, explora y prepara desarrollo, y ejecuta el modelado. El verificador comprueba contratos de datos, ventanas temporales, particiones, artefactos, métricas e inferencia. Los gráficos se guardan en `outputs/graphs/` y los artefactos del modelo en `outputs/model/`.

## Qué se hizo en los notebooks

### 1. Generación (`notebooks/01_generacion_datos.ipynb`)

Se creó `data/raw/dataset.csv` con 185.000 registros sintéticos, junto con `data/raw/historial_mensual_36m.csv.gz` y sus metadatos. El historial contiene **3.705.677 filas con actividad**. Se incorporaron variables de compras, gasto, promociones, devoluciones, reclamos, interacciones con campañas, canal y recencia. El conjunto simula datos demográficos incompletos; la edad tiene 8% de faltantes por diseño.

### 2. Exploración y preparación (`notebooks/02_analisis_exploratorio.ipynb`)

Se documentó el diccionario de variables, la temporalidad y los faltantes, y se realizó una partición estratificada 80/20 con semilla 42:

| Partición | Clientes | Uso |
| --- | ---: | --- |
| Desarrollo / train | 148.000 | EDA, selección y ajuste |
| Test reservado | 37.000 | Evaluación final del prototipo |

En train se observaron 11.770 edades faltantes (7,95%) y 5.900 comunas faltantes (3,99%); no se detectaron filas duplicadas. Las imputaciones y transformaciones se ajustan dentro de la pipeline, no usando el holdout.

### 3. Modelado (`notebooks/03_modelado_abandono.ipynb`)

Se subdividieron los 148.000 registros de train en 118.400 de ajuste y 29.600 de validación estratificada. El test no se usa para escoger familia, parámetros ni umbral. El notebook incluye:

- Comparación de regresión logística sin balanceo y con pesos de clase, Random Forest con pesos y Balanced Random Forest.
- Clustering exploratorio con K-Means, Gaussian Mixture y clustering aglomerativo, evaluado con silhouette, Davies–Bouldin y Calinski–Harabasz. K-Means con `k=3` obtuvo el silhouette mayor observado (**0,1347**); la separación es moderada y los segmentos son descriptivos, no una predicción de abandono.
- Búsqueda acotada con Optuna para regresión logística, Extra Trees, Random Forest y Balanced Random Forest (cinco pruebas por familia), optimizando Average Precision en validación.
- Comparación supervisada, selección de umbral por F1 en validación, calibración sigmoide CV3 sobre los datos de ajuste y evaluación final.
- Exportación de predicciones, curvas, matriz de confusión, ganancia/lift, auditoría regional y explicaciones SHAP.

## Variables y preparación del modelo

La lista `FEATURES` en `src/modeling.py` define explícitamente los predictores:

`edad`, `canal_compra`, `antiguedad_meses`, `cantidad_compras_12m`, `monto_promedio_compra`, `categoria_principal`, `proporcion_compras_promocion`, `devoluciones_12m`, `reclamos_12m`, `interacciones_campanas_12m`, `recencia_dias` y `tasa_devoluciones`.

La última variable se deriva como devoluciones de los 12 meses divididas por compras de los 12 meses (con denominador mínimo 1). La pipeline imputa valores numéricos con la mediana y agrega indicadores de faltantes; imputa categóricos con la categoría más frecuente y aplica one-hot encoding. Las variables numéricas se escalan con `RobustScaler`.

Se excluyen del modelo el ID, la etiqueta, las columnas de información posterior al corte y las variables geográficas (`region` y `comuna`). La edad se mantiene como variable de demostración; su pertinencia y posibles diferencias entre grupos requerirían revisión antes de cualquier uso real. La auditoría regional es descriptiva y no demuestra ausencia de sesgo.

## Resultados finales vigentes

La fuente única de métricas es `outputs/model/metricas_modelado_avanzado_test.json`. `outputs/model/metricas_test.json` es un alias idéntico.

| Métrica en test sintético | Resultado |
| --- | ---: |
| Clientes evaluados | 37.000 |
| Tasa de abandono observada | 23,9378% |
| Probabilidad media predicha | 23,9865% |
| Umbral binario seleccionado en validación | 0,2067 |
| Accuracy | 57,73% |
| Precision | 32,33% |
| Recall | 70,08% |
| F1 | 44,25% |
| ROC-AUC | 0,6748 |
| Average Precision | 0,4170 |
| Brier | 0,1673 |
| Log loss | 0,5120 |

Matriz de confusión en test (filas: clase real `0`, `1`; columnas: predicción `0`, `1`):

|  | Predicción sin abandono | Predicción abandono |
| --- | ---: | ---: |
| Real sin abandono | 15.154 | 12.989 |
| Real abandono | 2.650 | 6.207 |

El umbral refleja una decisión de clasificación optimizada por F1 en validación, no una regla económica validada. El recall debe leerse junto con la precision: en este resultado hay un volumen importante de falsos positivos. La tasa observada y la probabilidad media predicha son cantidades distintas.

### Candidatos comparados en validación

La siguiente tabla corresponde a la comparación de modelos en la validación interna, a umbral 0,5; **no** son las métricas finales del test ni las del umbral elegido finalmente.

| Modelo | Average Precision | ROC-AUC | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Regresión logística | 0,4103 | 0,6756 | 0,3504 | 0,5917 | 0,4402 |
| Extra Trees | 0,4053 | 0,6721 | 0,3576 | 0,5559 | 0,4352 |
| Balanced Random Forest | 0,4008 | 0,6695 | 0,3572 | 0,5607 | 0,4364 |
| Random Forest | 0,4008 | 0,6706 | 0,3543 | 0,5680 | 0,4364 |

## Interpretación y límites

1. **Datos sintéticos:** las métricas describen la recuperación de patrones incorporados al generador; no demuestran capacidad predictiva en clientes reales, causalidad, ahorro, fidelización efectiva ni resultados de un piloto.
2. **Test previamente inspeccionado:** el holdout se observó en versiones anteriores. Esta actualización es una reevaluación del prototipo, no una validación confirmatoria independiente. Para confirmación se requiere una cohorte temporal nueva e independiente, sin selección de modelos sobre ella.
3. **Umbral y operación:** el umbral 0,2067 se eligió por F1 en validación; no incorpora costos de contacto, capacidad de campañas ni valor de cliente. Los cortes de riesgo Bajo/Medio/Alto son ilustrativos y distintos del umbral binario.
4. **SHAP:** explica el score del clasificador base antes de calibración. No representa cambios porcentuales en la probabilidad calibrada ni relaciones causales. `top5_shap_final.csv` sirve como resumen para demostración.
5. **Segmentación y región:** los clústeres y métricas regionales son análisis exploratorios/descriptivos; no acreditan estabilidad de segmentos ni ausencia de sesgo.

Antes de cualquier uso comercial se requieren datos reales autorizados, definición operacional validada, evaluación temporal independiente, análisis de sesgo, revisión de privacidad y monitoreo. También debería evaluarse la utilidad de la intervención mediante un piloto controlado.

## Salidas y uso en demostración

- `outputs/model/metricas_modelado_avanzado_test.json`: métricas oficiales del modelo final; `metricas_test.json` es alias idéntico.
- `outputs/model/modelo_retailnova_avanzado.joblib` y `modelo_retailnova.joblib`: aliases del pipeline final calibrado.
- `data/processed/test/clientes_riesgo_test.csv` y `predicciones_modelo_final_test.csv`: puntuaciones de demostración del test, según las salidas existentes en la entrega.
- `data/processed/test/auditoria_regional.csv`: resumen descriptivo por región.
- `data/processed/test/top5_shap_final.csv`: cinco factores SHAP principales del score base.
- `data/processed/train/dashboard_train.csv`: dataset inicial para el dashboard, sin predicciones finales.
- `outputs/graphs/`: gráficos de EDA, comparación, calibración, clustering, curvas, matriz de confusión, ganancia/lift y SHAP.

Para actualizar el dashboard, seguir [`docs/GUIA_POWER_BI_FINAL.md`](docs/GUIA_POWER_BI_FINAL.md). La probabilidad media estimada no debe presentarse como la tasa observada de abandono.

### Puntuar nuevos datos

```bash
python src/predict.py --input ruta/a/clientes.csv --output data/processed/test/nuevas_puntuaciones.csv
```

El CSV de entrada debe tener un `id_cliente` único y las variables previas al corte que requiere el modelo. No necesita `churn_6m` ni datos posteriores. El ejemplo de comando es ilustrativo; para inferencia real, sustituir el archivo por una fuente autorizada y revisar que las definiciones y ventanas coincidan con las del entrenamiento.

## Archivos principales

- `src/generate_data.py`: generador sintético, supuestos y metadatos.
- `src/paths.py`: rutas canónicas de datos y artefactos.
- `src/modeling.py`: preparación, balanceo, clustering, Optuna, comparación supervisada, calibración y evaluación final.
- `src/predict.py`: puntuación de filas nuevas sin requerir etiqueta futura.
- `src/analyze.py`: compatibilidad de la lista de variables.
- `src/validate_notebook_execution.py`, `src/check_notebook_execution.py` y `src/verify_project.py`: ejecución controlada y verificaciones.
- `notebooks/01_generacion_datos.ipynb`: generación reproducible.
- `notebooks/02_analisis_exploratorio.ipynb`: perfilado, EDA y preparación.
- `notebooks/03_modelado_abandono.ipynb`: experimentos, selección, calibración y evaluación.
- `docs/CORRECCIONES_EP5.md`: registro de correcciones y límites de la entrega.
- `docs/GUIA_POWER_BI_FINAL.md`: guía para conectar las salidas al dashboard.
- `requirements-lock.txt`: versiones del entorno verificado.

## Estado de entrega EP5

Los documentos de factibilidad y planificación originales se conservan como línea base; no se modifican retroactivamente. El registro de correcciones indica que el tablero Jira y su historial no fueron consultados ni modificados en esta revisión. La guía de Power BI y los CSV facilitan la demostración, pero no acreditan que el PBIX final esté completado. Según ese registro, siguen pendientes confirmar/entregar el dashboard Power BI, el informe final de 5–8 páginas y la presentación, además de incorporar en Jira evidencia, responsables, fechas y esfuerzo reales. No se afirma un porcentaje de avance global ni cumplimiento de presupuesto.

## Verificación y entorno

`requirements-lock.txt` registra versiones del entorno de la corrección. Según el registro, se conservaron los notebooks 01 y 02 de la entrega del equipo, se reejecutó el notebook 03 y se ejecutó el verificador integral. Para comprobar la versión del repositorio local:

```bash
python src/validate_notebook_execution.py
python src/verify_project.py
```

La generación reproduce los datos sintéticos con semilla 42; el historial y los artefactos existentes corresponden a la ejecución documentada en el repositorio.
