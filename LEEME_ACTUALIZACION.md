# Cómo aplicar esta entrega

1. Extraer el ZIP en una carpeta temporal.
2. Copiar el contenido de RetailNova dentro de la carpeta local RetailNova-git, reemplazando archivos. Conservar la carpeta .git del repositorio local.
3. En la terminal de RetailNova-git ejecutar:

```powershell
python -m pip install -r requirements-lock.txt
python src/verify_project.py
```

Las versiones fijadas corresponden al entorno probado con Python 3.12. Si el Python local no admite esas versiones, usar un entorno Python 3.12; requirements.txt mantiene rangos para desarrollo. No regenerar datos solo para aplicar las correcciones: los artefactos actualizados ya se incluyen.

4. Si los controles muestran PASS, abrir el PBIX existente y seguir docs/GUIA_POWER_BI_FINAL.md. El CSV definitivo es data/processed/test/clientes_riesgo_test.csv.
5. Incorporar al informe docs/Informe_EP5_RetailNova.docx la evidencia actual de Jira, integrantes, fechas y PBIX. El informe no se presenta como entrega final cerrada sin esos datos.
6. Subir los cambios desde la terminal:

```powershell
git add src/ notebooks/ data/processed/ outputs/ docs/ README.md requirements-lock.txt LEEME_ACTUALIZACION.md
git commit -m "Calibra modelo y unifica evidencias EP5"
git push
```

El contenido completo de data/raw se conserva de la entrega anterior. Los resultados siguen siendo exclusivamente sintéticos; no se ha ejecutado un piloto real.
