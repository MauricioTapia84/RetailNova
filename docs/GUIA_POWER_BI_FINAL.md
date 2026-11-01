# Actualizar el dashboard existente

1. Guardar una copia del PBIX actual y sustituir los archivos del proyecto por esta entrega.
2. En Power BI: Transformar datos, seleccionar ClientesRiesgo y editar el paso Origen para apuntar a `data/processed/test/clientes_riesgo_test.csv`. Se mantienen los nombres de columnas.
3. Conservar probabilidad_abandono como número decimal usando configuración regional Inglés (Estados Unidos). Si el paso automático Tipo cambiado interpretó mal el punto, eliminarlo y aplicar el tipo desde los valores originales de texto.
4. Cerrar y aplicar. Con filtros limpios deben verse 37.000 clientes y aproximadamente 24,0% de probabilidad media. Los recuentos por riesgo pueden cambiar respecto de la versión anterior.
5. Importar `data/processed/test/top5_shap_final.csv`. Crear barras con variable en eje Y e importancia_shap en X. Título: Cinco factores más influyentes en el score base. Nota: SHAP de muestra sintética del clasificador antes de calibración; no representa causalidad ni puntos porcentuales.
6. Para desempeño regional usar `data/processed/test/auditoria_regional.csv`, ya sincronizada con el modelo final. Top5 SHAP es una explicación global y puede mantenerse desconectada; indicar que no cambia con el filtro regional. No relacionar tablas sin una clave única apropiada.
7. Mantener las tarjetas de clientes y probabilidad media, barras por riesgo y región y segmentación por región. La media estimada no equivale al porcentaje observado de abandono.
8. Probar las tres regiones y limpiar filtros. Medir carga con el Analizador de rendimiento y guardar la evidencia si se pretende demostrar el criterio E05 de menos de tres segundos. No declararlo aprobado sin medición.
9. Guardar RetailNova_EP5.pbix y adjuntar archivo y captura a KAN-97 con fecha, responsable y prueba realizada.

## Medidas

```dax
Clientes analizados = COUNTROWS(ClientesRiesgo)
Probabilidad promedio = AVERAGE(ClientesRiesgo[probabilidad_abandono])
Abandono observado = AVERAGE(ClientesRiesgo[abandono_observado])
```

Formatear las dos últimas como porcentaje con un decimal. Usar Clientes analizados como valor del gráfico por nivel_riesgo.

## Fuente oficial y límites

Métricas: outputs/model/metricas_modelado_avanzado_test.json. Predicciones finales: data/processed/test/clientes_riesgo_test.csv. El umbral binario seleccionado en validación es distinto de los cortes ilustrativos de riesgo 0,20 y 0,50. Todos los clientes son sintéticos y pertenecen al test de reevaluación. El PBIX del equipo debe actualizarse localmente; no está incluido en este paquete.
