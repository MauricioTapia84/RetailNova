# Actualizar el dashboard existente

1. Trabajar sobre una copia de [RetailNova_EP6_Dashboard_Corregido.pbix](./RetailNova_EP6_Dashboard_Corregido.pbix). El archivo contiene un modelo embebido; por eso no basta con copiar el CSV al proyecto: hay que actualizar la consulta o reemplazar la tabla dentro de Power BI.
2. En Power BI: Transformar datos, seleccionar `clientesRiesgo` y editar el paso Origen para apuntar a `data/processed/test/clientes_riesgo_test.csv`. Se mantienen los nombres de columnas. Si el PBIX no muestra una consulta editable, eliminar/reimportar la tabla embebida con ese nombre y volver a asociar las visualizaciones.
3. Conservar probabilidad_abandono como número decimal usando configuración regional Inglés (Estados Unidos). Si el paso automático Tipo cambiado interpretó mal el punto, eliminarlo y aplicar el tipo desde los valores originales de texto.
4. Cerrar y aplicar. Con filtros limpios deben verse 37.000 clientes y aproximadamente 24,0% de probabilidad media. Los recuentos por riesgo pueden cambiar respecto de la versión anterior.
5. Importar `data/processed/test/top5_shap_final.csv`. Crear barras con variable en eje Y e importancia_shap en X. Título: Cinco factores más influyentes en el score base. Nota: SHAP de muestra sintética del clasificador antes de calibración; no representa causalidad ni puntos porcentuales.
6. Para desempeño regional usar `data/processed/test/auditoria_regional.csv`, ya sincronizada con el modelo final. Top5 SHAP es una explicación global y puede mantenerse desconectada; indicar que no cambia con el filtro regional. No relacionar tablas sin una clave única apropiada.
7. Mantener las tarjetas de clientes y probabilidad media, barras por riesgo y región y segmentación por región. La media estimada no equivale al porcentaje observado de abandono.
8. Probar las tres regiones y limpiar filtros. Medir carga con el Analizador de rendimiento y guardar la evidencia si se pretende demostrar el criterio E05 de menos de tres segundos. No declararlo aprobado sin medición.
9. Guardar `RetailNova_EP6_Dashboard_Corregido.pbix` y adjuntar archivo y captura a KAN-97 con fecha, responsable y prueba realizada.

## Medidas

```dax
Clientes analizados = COUNTROWS(ClientesRiesgo)
Probabilidad promedio = AVERAGE(ClientesRiesgo[probabilidad_abandono])
Abandono observado = AVERAGE(ClientesRiesgo[abandono_observado])
```

Formatear las dos últimas como porcentaje con un decimal. Usar Clientes analizados como valor del gráfico por nivel_riesgo.

## Fuente oficial y límites

Métricas: `outputs/model/metricas_modelado_avanzado_test.json`. Predicciones finales: `data/processed/test/clientes_riesgo_test.csv`. El umbral binario seleccionado en validación es distinto de los cortes ilustrativos de riesgo 0,20 y 0,50. Todos los clientes son sintéticos y pertenecen al test de reevaluación.

No usar `data/processed/test/metricas_dashboard.csv`: corresponde a una exportación anterior y no coincide exactamente con las métricas oficiales actuales. Tampoco usar `data/processed/train/dashboard_train.csv` como tabla de riesgo: contiene solamente train y no tiene las predicciones finales.

## Control específico del PBIX EP6

El PBIX revisado contiene dos páginas (`01 | Riesgo y priorización` y `02 | Validación del modelo`) y una tabla de modelo llamada `clientesRiesgo`. Las visualizaciones principales referencian correctamente `region`, `nivel_riesgo`, `probabilidad_abandono`, `prediccion` y `abandono_observado`.

Antes de darlo por corregido, verificar dentro de Power BI que la tabla tenga exactamente 37.000 filas y que la media de `probabilidad_abandono` sea aproximadamente 23,99%. Las tarjetas globales de la página de validación son cifras de referencia y no deben interpretarse como medidas dinámicas filtradas. `abandono_observado` solo es válido para esta cohorte de test; no debe mantenerse en un dashboard operativo de clientes nuevos.

## Corrección detectada en las capturas del dashboard

Las capturas revisadas muestran una versión anterior de las predicciones. La matriz visible presenta `18.608 / 9.514 / 3.654 / 5.224`, mientras que el archivo oficial actual debe mostrar:

| Abandono observado | Predicción 0 | Predicción 1 | Total |
| --- | ---: | ---: | ---: |
| 0 | 15.154 | 12.989 | 28.143 |
| 1 | 2.650 | 6.207 | 8.857 |
| Total | 17.804 | 19.196 | 37.000 |

Los conteos actuales por nivel de riesgo son: Bajo 16.768, Medio 18.697 y Alto 1.535. Si Power BI muestra otros valores, la consulta sigue apuntando a un CSV antiguo o la tabla embebida no fue reemplazada.

Las tarjetas de la página comercial deben mostrar, con filtros limpios:

- Clientes analizados: 37.000.
- Probabilidad promedio: aproximadamente 23,99%.
- Clientes en alerta (`prediccion = 1`): 19.196.
- Abandonos observados (`abandono_observado = 1`): 8.857.

Si las dos últimas tarjetas muestran nuevamente 37.000, no deben reutilizar solamente `Clientes analizados` con un filtro visual que no se está aplicando. Crear medidas explícitas:

```dax
Clientes en alerta =
CALCULATE(
    COUNTROWS(clientesRiesgo),
    clientesRiesgo[prediccion] = 1
)

Abandonos observados =
CALCULATE(
    COUNTROWS(clientesRiesgo),
    clientesRiesgo[abandono_observado] = 1
)
```

Después de reemplazar la tabla, ejecutar `Actualizar` y confirmar estos conteos antes de publicar el PBIX.
