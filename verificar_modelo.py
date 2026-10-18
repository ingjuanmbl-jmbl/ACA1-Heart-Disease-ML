from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

# Ruta relativa a la raíz del proyecto
RUTA_MODELO = (
    Path(__file__).parent
    / "modelo"
    / "modelo_regresion_logistica.joblib"
)


def main():
    if not RUTA_MODELO.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo: {RUTA_MODELO}"
        )

    modelo = joblib.load(RUTA_MODELO)

    print("Modelo cargado correctamente.")
    print("Tipo:", type(modelo).__name__)
    print("Clases:", modelo.classes_)

    if isinstance(modelo, Pipeline):
        print("Pasos del pipeline:")
        for nombre, transformador in modelo.steps:
            print(f"  - {nombre}: {type(transformador).__name__}")

    if hasattr(modelo, "feature_names_in_"):
        print("\nVariables esperadas:")
        for variable in modelo.feature_names_in_:
            print(f"  - {variable}")
    else:
        print(
            "\nEl modelo no expone feature_names_in_; "
            "revisaremos las variables desde el pipeline."
        )


if __name__ == "__main__":
    main()