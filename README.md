# RetailNova: predicción de abandono y fidelización

Prototipo académico de comercio minorista para la EP5 de Gestión de Proyectos de Datos. Todos los registros son sintéticos. No se reportan resultados de clientes reales.

## Ejecución

Python 3.11 o superior. Desde la carpeta raíz:

```bash
python -m pip install -r requirements.txt
python scripts/validate_notebook_execution.py
python scripts/verify_project.py
```

El primer comando instala las dependencias. El segundo ejecuta en orden ambos notebooks y guarda sus salidas (incluidos gráficos). El tercero comprueba los contratos del dataset, temporalidad, particiones y evidencias. También se pueden abrir los notebooks en Jupyter y ejecutar todas sus celdas. La primera ejecución reemplaza los datos y resultados generados de forma reproducible con semilla 42.

## Datos y definición temporal

Se simulan 185.000 clientes previamente recurrentes (mínimo dos compras en los 12 meses anteriores al 1 de abril de 2026). Esta cohorte es un supuesto académico: el caso informa 185.000 registrados, sin afirmar que todos sean recurrentes. Las variables predictoras resumen el comportamiento de los 12 meses previos al corte. El caso dispone de 36 meses históricos, pero este prototipo no reconstruye cada transacción de ese período.

El objetivo es cero compras entre el 1 de abril de 2026 inclusive y el 1 de octubre de 2026 exclusive. Se calcula desde `compras_periodo_objetivo` y se puede contrastar con la fecha de última compra al cierre. Esas dos columnas de auditoría son posteriores al corte y quedan excluidas del modelo. La recencia predictora corresponde exclusivamente a compras anteriores al corte.

La probabilidad de abandono se calibra a un 24% esperado y las etiquetas se sortean, sin forzar el porcentaje observado ni construir predictores a partir de ellas. El 8% exacto tiene edad faltante; parte de ese grupo también tiene comuna faltante. Se conserva la región para auditar representación: Metropolitana 65%, Valparaíso 20% y Biobío 15% como proporciones esperadas de simulación.

## Entrenamiento y evaluación

Se comparan baseline, regresión logística y Random Forest con particiones estratificadas 60/20/20. La imputación, escalado y codificación se ajustan únicamente con entrenamiento. El modelo se selecciona por average precision en validación y el umbral por F1 en validación. Solo después se evalúa test. Esta división es por clientes de una misma cohorte; no reemplaza una futura validación temporal con varias cohortes.

La lista `FEATURES` en `scripts/analyze.py` limita explícitamente las entradas al modelo. Se excluyen ID, etiqueta, información futura, región y comuna. Edad se conserva como variable de demostración; su pertinencia y posibles diferencias por edad requieren revisión antes de uso real. La auditoría regional es descriptiva; no acredita ausencia de sesgos. Los niveles de riesgo usan cortes ilustrativos de 0,20 y 0,50, distintos del umbral binario optimizado; deben ajustarse a capacidad y costos de campañas.

## Archivos

- `scripts/generate_data.py`: generador con supuestos y metadatos.
- `scripts/analyze.py`: preparación, modelos, métricas y exportaciones.
- `notebooks/`: ejecución documentada con resultados guardados; la lógica reutilizable reside en scripts.
- `data/generated/`: CSV sintético y metadatos.
- `outputs/`: comparación de validación, métricas de test, auditoría regional, particiones, clientes puntuados de test, gráficos y modelo serializado.
- `docs/`: documentos originales de planificación y registro de correcciones.

Para una demostración se pueden importar `outputs/clientes_riesgo_test.csv` y `outputs/auditoria_regional.csv` en Power BI. Este paquete no incluye un dashboard Power BI terminado. El archivo joblib es para el entorno de confianza del proyecto; solo cargar modelos de origen conocido.

## Alcance y pendientes EP5

Los documentos originales de factibilidad y planificación se conservan como línea base, sin modificarlos retroactivamente. Revisar `docs/CORRECCIONES_EP5.md`. El tablero Jira y su historial no fueron consultados ni modificados: faltan las evidencias reales de responsables, fechas, avance y comparación contra EP3. No se afirma un porcentaje de avance global ni cumplimiento del presupuesto. El informe final de 5–8 páginas y la presentación tampoco están incluidos.

Las métricas miden recuperación de patrones diseñados por el generador. No prueban capacidad predictiva real, causalidad, ahorro, fidelización efectiva ni resultados de un piloto. Antes de uso comercial se necesitan datos reales autorizados, cohortes temporales independientes, validación comercial, revisión de variables y monitoreo.
