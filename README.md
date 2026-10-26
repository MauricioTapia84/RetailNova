# RetailNova | Informe ejecutivo del caso de churn y riesgo comercial

## Resumen ejecutivo
RetailNova es un proyecto de análisis de datos y modelado predictivo orientado a anticipar el riesgo de abandono de clientes en un contexto de telecomunicaciones y servicios digitales. El objetivo principal es transformar una base sintética, alineada con un caso de negocio realista, en una evidencia analítica útil para la toma de decisiones de retención, soporte operativo y segmentación comercial.

El caso se estructura como una iniciativa de ciencia de datos con enfoque ejecutivo: validar la calidad del dato, diseñar un dataset sintético representativo, explorar sesgos estructurales, preparar un flujo de modelado robusto y entregar una narrativa analítica clara para stakeholders no técnicos.

## Contexto del negocio
La empresa ficticia RetailNova opera un modelo de servicios con clientes distribuidos por región, zona, segmento y tipo de plan. El desafío no es solo estimar quien podría abandonar, sino identificar dónde, para quién y bajo qué condiciones el riesgo es mayor. Esto permite actuar antes de que el problema se materialice, reduciendo la pérdida de ingresos y mejorando la experiencia del cliente.

En la práctica, el análisis combina:
- diagnóstico de churn y riesgo de retención,
- revisión de calidad operativa y de servicio,
- segmentación por perfil del cliente,
- modelado predictivo con validación prudente,
- interpretación de resultados en lenguaje de negocio.

## Objetivos
1. Generar un dataset sintético que refleje el caso de negocio de RetailNova con sesgos y condiciones realistas.
2. Revisar el comportamiento del churn por segmentación, región, plan, calidad de servicio y riesgo crediticio.
3. Construir una base analítica reproducible para EDA, preparación de datos y modelado.
4. Diseñar una estructura de trabajo tipo proyecto de churn con narrativa ejecutiva y validación técnica.
5. Crear un repositorio listo para uso local y para replicación en un entorno de ciencia de datos.

## Alcance analítico
El proyecto contempla cuatro fases principales:

### 1. Generación del dataset
Se crea un dataset sintético con variables demográficas, comerciales, operativas y de riesgo. El objetivo es reflejar patrones plausibles de negocio, evitando ruido aleatorio y priorizando relaciones causales realistas.

### 2. Exploración y análisis descriptivo
Se revisan indicadores clave como:
- prevalencia de churn,
- comportamiento por región y segmento,
- distribución de métricas de experiencia y servicio,
- correlación entre variables operativas y abandono,
- identificación de señales de riesgo discriminantes.

### 3. Preparación para modelado
Se ejecuta una separación train/test sin fuga de información, se identifican variables numéricas y categóricas, y se aplican transformaciones de forma consistente para preparar la base de entrenamiento y evaluación.

### 4. Evaluación y narrativa de negocio
Se comparan modelos base y supervisados para entender qué nivel de capacidad predictiva tiene el proyecto y cuál es la decisión más valiosa para la organización desde un punto de vista operacional y comercial.

## Estructura del repositorio
- `data/generated/`: datasets sintéticos generados.
- `docs/`: documentación del caso, requisitos y matriz de riesgos.
- `notebooks/`: cuadernos de generación del dataset y análisis exploratorio.
- `environment.yml`: entorno de conda para reproducibilidad.
- `requirements.txt`: dependencias de Python para instalación rápida.

## Indicadores clave del caso
Los resultados esperados del proyecto apuntan a identificar los clientes con mayor probabilidad de abandono según variables como:
- región y zona de operación,
- segmento del cliente,
- plan contratado,
- antigüedad del cliente,
- incidencias técnicas y calidad de experiencia,
- nivel de riesgo crediticio,
- NPS y percepción de servicio.

## Requisitos del entorno
El proyecto se ejecuta con Python 3.12 y usa bibliotecas estándares de análisis de datos y machine learning, especialmente:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyterlab / notebook
- ipykernel
- openpyxl

## Instalación del entorno con conda
```bash
conda env create -f environment.yml
conda activate retailnova
```

## Instalación alternativa con pip
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Cómo usar el proyecto
1. Revisar la carpeta `docs/` para comprender el caso de negocio y los requisitos.
2. Ejecutar el notebook de generación del dataset en `notebooks/01_generador_dataset_retailnova_case1.ipynb`.
3. Abrir el notebook de EDA y storytelling en `notebooks/02_eda_retailnova_storytelling.ipynb`.
4. Validar resultados, interpretar sesgos y comunicar hallazgos de negocio.

## Conclusión
RetailNova combina analítica de datos, modelado predictivo y narrativa ejecutiva con un enfoque claro: detectar riesgo de churn antes de que se materialice. El valor del proyecto no está solo en la precisión del modelo, sino en la capacidad de convertir evidencia analítica en decisiones comerciales concretas, con prioridad en retención, servicio y sostenibilidad del negocio.
