# RetailNova: predicción de abandono y fidelización

Prototipo académico de comercio minorista para la EP5 de Gestión de Proyectos de Datos. Todos los registros son sintéticos. No se reportan resultados de clientes reales.

## Ejecución

Python 3.11 o superior. Desde la carpeta raíz:

```bash
python -m pip install -r requirements.txt
python src/validate_notebook_execution.py
python src/verify_project.py
```

El primer comando instala las dependencias. El segundo ejecuta en orden los tres notebooks: genera la fuente raw, separa train/test, explora solo train, prepara artefactos y ejecuta el modelado. Los gráficos se guardan sin abrir ventanas en `outputs/graphs/`; el modelo y métricas en `outputs/model/`. El tercero comprueba los contratos del dataset, temporalidad, particiones y evidencias. La generación reemplaza los datos sintéticos de forma reproducible con semilla 42.

## Datos y definición temporal

Se simulan 185.000 clientes previamente recurrentes (mínimo dos compras en los 12 meses anteriores al 1 de abril de 2026). Esta cohorte es un supuesto académico: el caso informa 185.000 registrados, sin afirmar que todos sean recurrentes. `data/raw/historial_mensual_36m.csv.gz` contiene el historial sintético agregado por cliente y mes entre abril de 2023 y marzo de 2026; los meses sin actividad se omiten y equivalen a cero compras, reclamos e interacciones. `data/raw/dataset.csv` contiene el resumen por cliente derivado de ese historial, con variables predictoras calculadas sobre los últimos 12 meses y totales de los 36 meses para trazabilidad. El EDA hace un split estratificado 80/20 y guarda `data/raw/train/train.csv` y `data/raw/test/test.csv`.

El objetivo es cero compras entre el 1 de abril de 2026 inclusive y el 1 de octubre de 2026 exclusive. Se calcula desde `compras_periodo_objetivo` y se puede contrastar con la fecha de última compra al cierre. Esas dos columnas de auditoría son posteriores al corte y quedan excluidas del modelo. La recencia predictora corresponde exclusivamente a compras anteriores al corte.

La probabilidad de abandono se calibra a un 24% esperado y las etiquetas se sortean, sin forzar el porcentaje observado ni construir predictores a partir de ellas. El 8% exacto tiene edad faltante; parte de ese grupo también tiene comuna faltante. Se conserva la región para auditar representación: Metropolitana 65%, Valparaíso 20% y Biobío 15% como proporciones esperadas de simulación.

## Entrenamiento y evaluación

El EDA separa clientes en 80% desarrollo (148.000) y 20% test (37.000). El desarrollo se divide en 118.400 clientes de ajuste y 29.600 de validación. Se comparan familias, balanceo y parámetros con Optuna en validación. El modelo seleccionado se calibra con sigmoide y CV3 dentro de los 118.400 clientes: el preprocesamiento se ajusta dentro de cada fold y la calibración usa scores fuera de fold. Se fija el umbral por F1 en los 29.600 de validación y no se reajusta incorporando esa validación. Se mide Brier, log loss y curva de fiabilidad. Las matrices procesadas se exportan con el preprocesador final.

El test ya fue inspeccionado en versiones anteriores. La actualización es una **reevaluación del prototipo**, no una nueva validación confirmatoria independiente. Se necesita una cohorte temporal nueva para evaluar desempeño futuro. No se debe repetir selección de modelos mirando test. El modelo usa una cohorte de clientes sintéticos; no acredita utilidad comercial real.

La lista `FEATURES` en `src/modeling.py` limita explícitamente las entradas al modelo. Se excluyen ID, etiqueta, información futura, región y comuna. Edad se conserva como variable de demostración; su pertinencia y posibles diferencias por edad requieren revisión antes de uso real. La auditoría regional es descriptiva; no acredita ausencia de sesgos. Los niveles de riesgo usan cortes ilustrativos de 0,20 y 0,50, distintos del umbral binario optimizado; deben ajustarse a capacidad y costos de campañas.

## Archivos

- `src/generate_data.py`: generador con supuestos y metadatos.
- `src/paths.py`: rutas absolutas del proyecto y rutas canónicas de datos y artefactos; se importan en scripts y notebooks.
- `src/analyze.py`: compatibilidad de la lista de variables; no sobrescribe el modelo final.
- `src/predict.py`: puntuación de nuevas filas sin requerir etiqueta futura.
- `docs/GUIA_POWER_BI_FINAL.md`: conexión del dashboard a las salidas finales.
- `src/modeling.py`: experimentos reutilizables de balanceo, clustering, Optuna, comparación supervisada y evaluación final.
- `src/validate_notebook_execution.py` y `src/verify_project.py`: ejecución controlada y verificación del proyecto.
- `src/check_notebook_execution.py`: acceso alternativo al validador de notebooks.
- `notebooks/01_generacion_datos.ipynb`: generación reproducible del dataset sintético.
- `notebooks/02_analisis_exploratorio.ipynb`: perfil y exploración descriptiva de la cohorte.
- `notebooks/03_modelado_abandono.ipynb`: selección, entrenamiento y evaluación del modelo.
- `data/raw/`: dataset sintético por cliente, historial mensual agregado de 36 meses y metadatos; los splits se separan en `train/` y `test/`.
- `data/processed/`: CSV derivados organizados en `train/` y `test/`; el manifiesto de particiones queda en la raíz.
- `outputs/graphs/`: histogramas, comparaciones y matrices de confusión exportados sin abrir ventanas.
- `outputs/model/`: modelo serializado, métricas JSON y resumen de calidad.
- `docs/`: documentos originales de planificación y registro de correcciones.

### Convención narrativa para notebooks

Cada sección de los notebooks debe comenzar indicando qué se hará y por qué ese paso es pertinente para el problema de retail de RetailNova. Después del análisis debe cerrar con una conclusión basada en los hallazgos observados y sus límites. Mantener esta estructura al ampliar o modificar los notebooks; distinguir siempre los resultados sintéticos de evidencia comercial real.

El EDA exporta a `data/processed/train/` la fuente de train, su versión limpia y el dataset inicial del dashboard; el holdout permanece en `data/raw/test/test.csv`. El procesamiento genera `data/processed/train/train.csv` y `data/processed/test/test.csv` con el mismo preprocesador ajustado sobre train. Las comparaciones y artefactos de entrenamiento se guardan en la subcarpeta `train/`; puntuaciones y auditorías del holdout, en `test/`. El dataset de dashboard es una base inicial sin predicciones.

Para una demostración se pueden importar `data/processed/train/dashboard_train.csv`, `data/processed/test/predicciones_modelo_final_test.csv` y `data/processed/test/auditoria_regional.csv` en Power BI. Este paquete no incluye el archivo PBIX del equipo: debe actualizarse siguiendo docs/GUIA_POWER_BI_FINAL.md. Los modelos joblib están en `outputs/model/` y son para el entorno de confianza del proyecto; solo cargar modelos de origen conocido.

## Alcance y pendientes EP5

Los documentos originales de factibilidad y planificación se conservan como línea base, sin modificarlos retroactivamente. Revisar `docs/CORRECCIONES_EP5.md`. El tablero Jira y su historial no fueron consultados ni modificados: faltan las evidencias reales de responsables, fechas, avance y comparación contra EP3. No se afirma un porcentaje de avance global ni cumplimiento del presupuesto. El informe final de 5–8 páginas y la presentación tampoco están incluidos.

Las métricas miden recuperación de patrones diseñados por el generador. No prueban capacidad predictiva real, causalidad, ahorro, fidelización efectiva ni resultados de un piloto. Antes de uso comercial se necesitan datos reales autorizados, cohortes temporales independientes, validación comercial, revisión de variables y monitoreo.

## Fuente única de resultados finales

`outputs/model/metricas_modelado_avanzado_test.json` y `metricas_test.json` son alias idénticos. Los modelos `modelo_retailnova_avanzado.joblib` y `modelo_retailnova.joblib` corresponden al mismo pipeline calibrado. `data/processed/test/predicciones_modelo_final_test.csv` y `clientes_riesgo_test.csv` contienen exactamente las mismas puntuaciones. La auditoría regional y los gráficos de evaluación se recalculan desde esa fuente final. Las comparaciones de candidatos en validación se conservan como experimentos previos, no son las métricas del modelo calibrado final.

SHAP explica el score del clasificador base antes de calibración. El CSV `top5_shap_final.csv` permite mostrar cinco factores en Power BI. Sus magnitudes no son cambios porcentuales en la probabilidad calibrada. Los cortes Bajo/Medio/Alto son ilustrativos y no se confunden con el umbral binario optimizado.

### Puntuar datos nuevos

```bash
python src/predict.py --input data/raw/test/test.csv --output data/processed/test/nuevas_puntuaciones.csv
```

El comando es ilustrativo; para puntuar futuros clientes, sustituir input por una tabla con ID único y las variables previas al corte. No requiere `churn_6m` ni información posterior. Antes de inferencia revisar que las ventanas y definiciones coincidan.

### Entorno verificado

`requirements-lock.txt` registra versiones del entorno de esta corrección. Los notebooks 01 y 02 se conservaron de la entrega del equipo; se reejecutó el notebook 03 y el verificador integral con los nuevos resultados. No se reconstruyó nuevamente el historial de 36 meses al aplicar esta corrección.
