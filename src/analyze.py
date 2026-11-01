"""Compatibilidad del pipeline y lista de predictores.

El entrenamiento oficial reside en modeling.py y notebook 03.
"""
FEATURES = ["edad", "canal_compra", "antiguedad_meses", "cantidad_compras_12m",
"monto_promedio_compra", "categoria_principal", "proporcion_compras_promocion",
"devoluciones_12m", "reclamos_12m", "interacciones_campanas_12m",
"recencia_dias", "tasa_devoluciones"]
if __name__ == "__main__":
    print("Ejecute python src/validate_notebook_execution.py para el flujo oficial.")
