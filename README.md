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

El EDA separa una vez los clientes en 80% train y 20% holdout estratificado; el holdout se guarda y no se consulta durante la exploración, selección ni ingeniería de variables. El notebook de modelado carga solo train, compara pesos de clase y Balanced Random Forest sin sobremuestrear por defecto, evalúa clustering en una muestra común, y usa Optuna para ajustar regresión logística, Random Forest, Extra Trees y Balanced Random Forest con una validación interna. Después compara los modelos optimizados con métricas y tiempos en esa misma validación. El modelo y el umbral se eligen antes de cargar test, se reajustan con todo train y finalmente se evalúan una sola vez en holdout. Las matrices procesadas se guardan en `data/processed/train/train.csv` y `data/processed/test/test.csv`, con transformaciones ajustadas exclusivamente con train. Esta división es por clientes de una misma cohorte; no reemplaza una futura validación temporal con varias cohortes.

La lista `FEATURES` en `src/analyze.py` limita explícitamente las entradas al modelo. Se excluyen ID, etiqueta, información futura, región y comuna. Edad se conserva como variable de demostración; su pertinencia y posibles diferencias por edad requieren revisión antes de uso real. La auditoría regional es descriptiva; no acredita ausencia de sesgos. Los niveles de riesgo usan cortes ilustrativos de 0,20 y 0,50, distintos del umbral binario optimizado; deben ajustarse a capacidad y costos de campañas.

## Archivos

- `src/generate_data.py`: generador con supuestos y metadatos.
- `src/paths.py`: rutas absolutas del proyecto y rutas canónicas de datos y artefactos; se importan en scripts y notebooks.
- `src/analyze.py`: pipeline inicial de preparación, entrenamiento y exportaciones.
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

Para una demostración se pueden importar `data/processed/train/dashboard_train.csv`, `data/processed/test/predicciones_modelo_final_test.csv` y `data/processed/test/auditoria_regional.csv` en Power BI. Este paquete no incluye un dashboard Power BI terminado. Los modelos joblib están en `outputs/model/` y son para el entorno de confianza del proyecto; solo cargar modelos de origen conocido.

## Alcance y pendientes EP5

Los documentos originales de factibilidad y planificación se conservan como línea base, sin modificarlos retroactivamente. Revisar `docs/CORRECCIONES_EP5.md`. El tablero Jira y su historial no fueron consultados ni modificados: faltan las evidencias reales de responsables, fechas, avance y comparación contra EP3. No se afirma un porcentaje de avance global ni cumplimiento del presupuesto. El informe final de 5–8 páginas y la presentación tampoco están incluidos.

Las métricas miden recuperación de patrones diseñados por el generador. No prueban capacidad predictiva real, causalidad, ahorro, fidelización efectiva ni resultados de un piloto. Antes de uso comercial se necesitan datos reales autorizados, cohortes temporales independientes, validación comercial, revisión de variables y monitoreo.
