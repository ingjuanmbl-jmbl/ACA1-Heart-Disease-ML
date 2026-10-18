from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# Configuración general
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ACA1 | Predicción de enfermedad cardíaca",
    page_icon="🫀",
    layout="wide",
)

RUTA_MODELO = (
    Path(__file__).resolve().parent
    / "modelo"
    / "modelo_regresion_logistica.joblib"
)


@st.cache_resource
def cargar_modelo():
    """Carga y mantiene en memoria el pipeline entrenado."""
    return joblib.load(RUTA_MODELO)


st.title("Clasificación de enfermedad cardíaca")
st.subheader("Proyecto ACA 1 · Aprendizaje Automático · CUN")
st.write(
    "Ingresa las variables del registro para obtener la clasificación y las "
    "probabilidades estimadas por el modelo de regresión logística."
)
st.warning(
    "Uso exclusivamente académico. Esta aplicación es una demostración basada "
    "en un conjunto de datos limitado; no constituye una herramienta clínica "
    "validada ni reemplaza la evaluación de profesionales de salud."
)


# -----------------------------------------------------------------------------
# Cargar el pipeline completo (preprocesador + regresión logística)
# -----------------------------------------------------------------------------
try:
    modelo = cargar_modelo()
except FileNotFoundError:
    st.error(
        "No se encontró el archivo del modelo. Verifica que exista en: "
        "modelo/modelo_regresion_logistica.joblib"
    )
    st.stop()
except Exception as exc:
    st.error(
        "No fue posible cargar el modelo. Verifica las versiones de las "
        "librerías y que el archivo .joblib corresponda al pipeline entrenado."
    )
    st.exception(exc)
    st.stop()


# -----------------------------------------------------------------------------
# Formulario. Los nombres de las columnas coinciden con el entrenamiento.
# -----------------------------------------------------------------------------
st.markdown("### 1. Datos demográficos y mediciones")

with st.form("formulario_prediccion"):
    col1, col2, col3 = st.columns(3)

    with col1:
        edad = st.number_input(
            "Edad (años)", min_value=1, max_value=120, value=55, step=1
        )
        sexo = st.selectbox("Sexo", ["Mujer", "Hombre"])
        presion_arterial_reposo = st.number_input(
            "Presión arterial en reposo (mm Hg)",
            min_value=40,
            max_value=300,
            value=130,
            step=1,
        )
        colesterol_serico = st.number_input(
            "Colesterol sérico (mg/dl)",
            min_value=50,
            max_value=800,
            value=240,
            step=1,
        )

    with col2:
        frecuencia_cardiaca_max = st.number_input(
            "Frecuencia cardíaca máxima (latidos/min)",
            min_value=30,
            max_value=250,
            value=150,
            step=1,
        )
        depresion_ST = st.number_input(
            "Depresión del segmento ST",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            format="%.1f",
        )
        glucemia_ayunas_alta = st.selectbox(
            "Glucemia en ayunas superior a 120 mg/dl", ["No", "Si"]
        )
        angina_inducida_ejercicio = st.selectbox(
            "Angina inducida por ejercicio", ["No", "Si"]
        )

    with col3:
        tipo_dolor_toracico = st.selectbox(
            "Tipo de dolor torácico",
            [
                "Angina típica",
                "Angina atípica",
                "Dolor no anginoso",
                "Asintomático",
            ],
        )
        electrocardiograma_reposo = st.selectbox(
            "Electrocardiograma en reposo",
            [
                "Normal",
                "Anormalidad de la onda ST-T",
                "Hipertrofia ventricular izquierda probable o definida",
            ],
        )
        pendiente_segmento_ST = st.selectbox(
            "Pendiente del segmento ST",
            ["Ascendente", "Plana", "Descendente"],
        )
        numero_vasos_principales = st.selectbox(
            "Número de vasos principales",
            options=[0.0, 1.0, 2.0, 3.0],
            format_func=lambda valor: str(int(valor)),
        )

    resultado_talasemia = st.selectbox(
        "Resultado de talasemia",
        ["Normal", "Defecto fijo", "Defecto reversible"],
    )

    enviar = st.form_submit_button(
        "Realizar predicción", type="primary", use_container_width=True
    )


if enviar:
    # El DataFrame conserva los nombres y tipos de variables del entrenamiento.
    datos_usuario = pd.DataFrame(
        [
            {
                "edad": int(edad),
                "sexo": sexo,
                "tipo_dolor_toracico": tipo_dolor_toracico,
                "presion_arterial_reposo": int(presion_arterial_reposo),
                "colesterol_serico": int(colesterol_serico),
                "glucemia_ayunas_alta": glucemia_ayunas_alta,
                "electrocardiograma_reposo": electrocardiograma_reposo,
                "frecuencia_cardiaca_max": int(frecuencia_cardiaca_max),
                "angina_inducida_ejercicio": angina_inducida_ejercicio,
                "depresion_ST": float(depresion_ST),
                "pendiente_segmento_ST": pendiente_segmento_ST,
                "numero_vasos_principales": float(numero_vasos_principales),
                "resultado_talasemia": resultado_talasemia,
            }
        ]
    )

    try:
        clase = int(modelo.predict(datos_usuario)[0])
        probabilidades = modelo.predict_proba(datos_usuario)[0]
        clases_modelo = list(modelo.classes_)
        prob_clase_0 = float(probabilidades[clases_modelo.index(0)])
        prob_clase_1 = float(probabilidades[clases_modelo.index(1)])

        st.markdown("---")
        st.markdown("### 2. Resultado de la clasificación")

        if clase == 1:
            st.error(
                "El modelo asigna este registro a la clase 1: "
                "registros etiquetados con enfermedad cardíaca en el dataset."
            )
        else:
            st.success(
                "El modelo asigna este registro a la clase 0: "
                "registros etiquetados sin enfermedad cardíaca en el dataset."
            )

        met1, met2 = st.columns(2)
        with met1:
            st.metric("Probabilidad estimada · Clase 0", f"{prob_clase_0:.1%}")
        with met2:
            st.metric("Probabilidad estimada · Clase 1", f"{prob_clase_1:.1%}")

        grafico = pd.DataFrame(
            {
                "Clase": ["0 · Sin enfermedad en el dataset", "1 · Con enfermedad en el dataset"],
                "Probabilidad estimada": [prob_clase_0, prob_clase_1],
            }
        ).set_index("Clase")
        st.bar_chart(grafico, x_label="Clase", y_label="Probabilidad")

        with st.expander("Ver datos enviados al modelo"):
            st.dataframe(datos_usuario, use_container_width=True)

        st.caption(
            "Las probabilidades son salidas del modelo entrenado y no deben "
            "interpretarse como probabilidades clínicas calibradas. La clase "
            "predicha utiliza el umbral predeterminado del clasificador."
        )

    except Exception as exc:
        st.error("No fue posible generar la predicción con los datos ingresados.")
        st.exception(exc)


with st.expander("Sobre el modelo"):
    st.write(
        "Se carga el pipeline completo entrenado en el notebook: "
        "preprocesador (imputación, RobustScaler y OneHotEncoder) y regresión "
        "logística. No se ajustan transformadores nuevos al realizar una "
        "predicción."
    )
    st.write(
        "En la evaluación reportada del conjunto de prueba, el modelo obtuvo "
        "accuracy de 0.85, precisión de 0.88, recall de 0.7857, F1 de 0.8302 "
        "y ROC-AUC de 0.9442. Estas métricas corresponden al experimento "
        "académico y no representan validación clínica."
    )
