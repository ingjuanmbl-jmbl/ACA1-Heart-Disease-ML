# ACA1 — Clasificación de enfermedad cardíaca

Aplicación académica desarrollada para el proyecto ACA 1 de la asignatura **Aprendizaje Automático** de la CUN. La interfaz está construida con Streamlit y utiliza un pipeline de scikit-learn entrenado previamente para clasificar registros del conjunto de datos Heart Disease.

> **Aviso:** este proyecto tiene fines exclusivamente educativos. No es una herramienta clínica validada, no diagnostica enfermedades y no sustituye la valoración de un profesional de salud. Las probabilidades mostradas son salidas del modelo y no deben interpretarse como probabilidades clínicas calibradas.

## Funcionalidades

- Formulario interactivo para ingresar las 13 variables utilizadas por el modelo.
- Predicción de la clase asignada por el clasificador.
- Visualización de las probabilidades estimadas para las clases 0 y 1.
- Visualización de los datos enviados al modelo.
- Carga del pipeline completo guardado en formato `.joblib`; la aplicación no vuelve a entrenar el modelo ni ajusta transformadores nuevos.

## Tecnologías

- Python 3.11 para el entorno local
- Streamlit 1.66.0
- scikit-learn 1.6.1
- NumPy 2.1.3
- SciPy 1.16.3
- pandas 2.2.3
- joblib 1.6.0

Se fijan las versiones en `requirements.txt` para hacer el entorno más reproducible. Los artefactos serializados de scikit-learn deben cargarse con cuidado y preferiblemente con las mismas versiones de dependencias usadas durante el entrenamiento.

## Estructura del proyecto

```text
ACA1-Heart-Disease-ML/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── verificar_modelo.py
├── modelo/
│   └── modelo_regresion_logistica.joblib
└── notebooks/                     # Opcional: notebooks del proceso analítico
```

La carpeta `notebooks/` puede incluir los notebooks de extracción, EDA/limpieza y modelado para documentar el proceso académico. No se debe subir el entorno virtual `.venv/` ni archivos de datos personales o credenciales.

## Instalación y ejecución local

Requiere Python instalado y Git solo si se desea clonar el repositorio.

### 1. Crear y activar el entorno virtual (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Instalar dependencias

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Verificar el modelo (opcional)

```powershell
python verificar_modelo.py
```

El archivo `modelo/modelo_regresion_logistica.joblib` debe existir antes de ejecutar esta comprobación o iniciar la aplicación.

### 4. Ejecutar Streamlit

```powershell
python -m streamlit run app.py
```

Streamlit mostrará una dirección local, normalmente `http://localhost:8501`, que se puede abrir en el navegador.

## Modelo y evaluación

La aplicación carga un `Pipeline` que contiene un `ColumnTransformer` para el preprocesamiento y un clasificador de regresión logística. El preprocesamiento guardado incluye las transformaciones ajustadas durante el entrenamiento, por lo que la aplicación recibe las variables originales y llama a `predict()` y `predict_proba()`.

Métricas reportadas para el conjunto de prueba en el experimento académico:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.8500 |
| Precisión | 0.8800 |
| Recall | 0.7857 |
| F1-score | 0.8302 |
| ROC-AUC | 0.9442 |

Estas métricas describen el experimento y no constituyen evidencia de validez clínica. Conviene revisarlas contra el notebook final antes de la entrega académica.

## Fuente de datos

UCI Machine Learning Repository — Heart Disease: <https://archive.ics.uci.edu/dataset/45/heart+disease>

## Aplicación desplegada

Accede a la aplicación interactiva:

https://aca1-heart-disease-ml-j4wqqgj9aqfucblg34avy6.streamlit.app/


## Entregables académicos

Este repositorio documenta principalmente la aplicación interactiva y el modelo utilizado. Verifica también las indicaciones del curso para entregar el notebook reproducible, el póster/infografía en PDF y el video de sustentación de 3 a 5 minutos, si no se incluyen aquí.
