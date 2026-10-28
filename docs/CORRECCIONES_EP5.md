# Registro de correcciones técnicas EP5

Fecha de revisión: 28 de octubre de 2026. Evidencia: contenido del ZIP proporcionado y ejecución local de la versión corregida. Este registro no representa el historial del tablero Jira ni acredita fechas anteriores de trabajo.

| Hallazgo | Corrección aplicada | Evidencia |
| --- | --- | --- |
| README y variables de telecomunicaciones | Contexto de retail, compras, promociones, devoluciones, reclamos y campañas | README y scripts/generate_data.py |
| Regiones genéricas | Metropolitana, Valparaíso y Biobío, con mayor representación metropolitana | CSV y outputs/resumen_datos.png |
| Tasa de abandono 2,14% sin alineación | Simulación probabilística calibrada a 24% esperado; 23,995% observado | Metadatos del dataset |
| Ausencia de nulos | 14.800 clientes (8%) con demografía incompleta | Metadatos y resumen_calidad.json |
| Generador en orden inverso | Flujo secuencial encapsulado en función reproducible | Notebook 01 y script generador |
| Prioridad derivada de la etiqueta usada como predictor | Eliminación de prioridad y lista explícita de predictores anteriores al corte | FEATURES en scripts/analyze.py |
| Etiqueta sintética sin compras verificables | Cero compras en período posterior de seis meses, contrastable con última compra al cierre | CSV y verify_project.py |
| Comparación y ajuste sobre test | Selección de modelo y umbral en validación; test reservado | comparacion_validacion.csv, metricas_test.json y particiones.csv |
| Celdas sin resultados guardados | Ejecución de ambos notebooks con salidas y gráficos embebidos | notebooks/*.ipynb |
| Ruta absoluta a un computador | Rutas relativas a la raíz del proyecto | scripts y notebooks |
| Sin artefactos exportados para demostración | Modelo, clientes puntuados de test, métricas, gráficos y auditoría regional | outputs/ |

## Validación realizada

Se ejecutaron ambos notebooks en orden mediante el ejecutor en proceso de IPython. En este entorno no fue posible iniciar el kernel de Jupyter por una restricción de sockets; se utilizó ejecución en proceso sin modificar el cálculo, y se guardaron salidas de texto, tablas e imágenes en los archivos ipynb. La ejecución interactiva en la interfaz de Jupyter no se validó aquí.

El control verify_project.py pasó: tamaño de cohorte, unicidad, regiones, proporción de abandono, 8% de demografía incompleta, fechas, definición del objetivo, exclusión de fugas conocidas, particiones disjuntas y notebooks sin errores. No es una prueba exhaustiva de ausencia de sesgos o de fuga en datos reales.

## Resultados de test sintético

Regresión logística seleccionada en validación. Test: 37.000 clientes. ROC-AUC: 0,675; average precision: 0,412; precision: 0,354; recall: 0,588; F1: 0,442. Umbral seleccionado en validación: 0,24. Estos resultados no implican cumplimiento de criterios de aceptación de EP3; deben contrastarse con esos criterios y con una validación comercial. No se presentan como resultados reales ni como un modelo listo para producción.

## Pendientes de gestión y entrega

El informe de EP3 se conserva en docs/Encargo 2.docx como línea base. No se alteraron sus compromisos. Se debe registrar en el tablero real el trabajo efectuado, responsables, fechas, esfuerzo, estado y desviaciones; adjuntar enlaces o capturas verificables y calcular el avance contra tareas ponderadas del plan. El enlace a Jira presente en el documento no fue consultado. No se inventaron responsables, costos, aprobaciones ni porcentajes de avance.

Se mantiene pendiente elaborar el dashboard comprometido en Power BI, el informe de ejecución de 5–8 páginas y la presentación. El CSV de riesgo cubre solo clientes de test para demostración; no constituye despliegue comercial sobre todos los clientes.
