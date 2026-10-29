# Registro de correcciones técnicas EP5

Fecha de revisión: 28 de octubre de 2026. Evidencia: contenido del ZIP proporcionado y ejecución local de la versión corregida. Este registro no representa el historial del tablero Jira ni acredita fechas anteriores de trabajo.

| Hallazgo | Corrección aplicada | Evidencia |
| --- | --- | --- |
| README y variables de telecomunicaciones | Contexto de retail, compras, promociones, devoluciones, reclamos y campañas | README y src/generate_data.py |
| Regiones genéricas | Metropolitana, Valparaíso y Biobío, con mayor representación metropolitana | data/raw/dataset.csv y outputs/graphs/resumen_regional.png |
| Tasa de abandono 2,14% sin alineación | Simulación probabilística calibrada a 24% esperado; 23,995% observado | Metadatos del dataset |
| Ausencia de nulos | 14.800 clientes (8%) con demografía incompleta | Metadatos y resumen_calidad.json |
| Generador en orden inverso | Flujo secuencial encapsulado en función reproducible | Notebook 01 y script generador |
| Historial de 36 meses declarado pero no materializado | Historial mensual sintético agregado desde abril de 2023 hasta marzo de 2026; resumen por cliente derivado de la actividad | data/raw/historial_mensual_36m.csv.gz, data/raw/dataset.csv y metadatos |
| Rutas de importación dependientes de scripts/ | Rutas centralizadas e importaciones desde src/ para scripts y notebooks | src/paths.py y notebooks/*.ipynb |
| Prioridad derivada de la etiqueta usada como predictor | Eliminación de prioridad y lista explícita de predictores anteriores al corte | FEATURES en src/analyze.py |
| Etiqueta sintética sin compras verificables | Cero compras en período posterior de seis meses, contrastable con última compra al cierre | CSV y verify_project.py |
| Comparación y ajuste sobre test | Selección de modelo y umbral en validación; test reservado | data/processed/train/comparacion_validacion.csv, outputs/model/metricas_test.json y data/processed/particiones.csv |
| Celdas sin resultados guardados | Ejecución de los notebooks con salidas y gráficos embebidos | notebooks/*.ipynb |
| Ruta absoluta a un computador | Rutas relativas a la raíz del proyecto | scripts y notebooks |
| Sin artefactos exportados para demostración | Modelo, clientes puntuados de test, métricas, gráficos y auditoría regional | outputs/model/, outputs/graphs/ y data/processed/ |

## Validación realizada

Se ejecutaron los notebooks en orden mediante el ejecutor en proceso de IPython. En este entorno no fue posible iniciar el kernel de Jupyter por una restricción de sockets; se utilizó ejecución en proceso sin modificar el cálculo, y se guardaron salidas de texto, tablas e imágenes en los archivos ipynb. La ejecución interactiva en la interfaz de Jupyter no se validó aquí.

El control src/verify_project.py pasó: tamaño de cohorte, unicidad, regiones, proporción de abandono, 8% de demografía incompleta, fechas, definición del objetivo, exclusión de fugas conocidas, particiones disjuntas, aislamiento del test y notebooks sin errores. No es una prueba exhaustiva de ausencia de sesgos o de fuga en datos reales.

## Resultados de test sintético

Regresión logística con pesos de clase seleccionada por Optuna en validación. El EDA reserva 20% (37.000 clientes) como holdout; la selección se hace con una validación interna del 80% restante y el modelo final se reajusta con todo train. En la última ejecución sintética: ROC-AUC 0,675; average precision 0,417; precision 0,324; recall 0,698; F1 0,443. Umbral elegido en validación: 0,453. K-Means con k=3 obtuvo el mayor silhouette (0,135) entre las técnicas evaluadas sobre una muestra común. Estos resultados no implican cumplimiento de criterios de aceptación de EP3; deben contrastarse con esos criterios y con una validación comercial. No se presentan como resultados reales ni como un modelo listo para producción.

El notebook EDA cubre diccionario y traducción, inspección, holdout estratificado, análisis de train, ingeniería de características, diagnósticos de selección, gráficos guardados y exportaciones para modelado y dashboard. El notebook de modelado carga primero solo train, compara estrategias de balanceo sin sobremuestreo por defecto, explora clustering con una muestra común, optimiza cuatro familias con Optuna, compara las métricas de los modelos optimizados y reporta curvas ROC/precision–recall, lift, ganancia acumulada y SHAP. Los splits CSV raw y procesados se organizan en subcarpetas `train/` y `test/`; la fuente sintética completa y el historial mensual de 36 meses permanecen en la raíz de `data/raw/`. Las figuras y artefactos del modelo están separados en `outputs/graphs/` y `outputs/model/`. La selección del modelo se realiza con validación interna y el holdout solo se carga para la evaluación final.

## Pendientes de gestión y entrega

El informe de EP3 se conserva en docs/Encargo 2.docx como línea base. No se alteraron sus compromisos. Se debe registrar en el tablero real el trabajo efectuado, responsables, fechas, esfuerzo, estado y desviaciones; adjuntar enlaces o capturas verificables y calcular el avance contra tareas ponderadas del plan. El enlace a Jira presente en el documento no fue consultado. No se inventaron responsables, costos, aprobaciones ni porcentajes de avance.

Se mantiene pendiente elaborar el dashboard comprometido en Power BI, el informe de ejecución de 5–8 páginas y la presentación. El CSV de riesgo cubre solo clientes de test para demostración; no constituye despliegue comercial sobre todos los clientes.
