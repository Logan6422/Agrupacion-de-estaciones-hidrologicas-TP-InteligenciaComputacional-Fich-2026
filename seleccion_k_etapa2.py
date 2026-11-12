from pathlib import Path

import numpy as np

import seleccion_k as _base
from seleccion_k import *


def cargar_datos():
    from analisis_rios import cargar_caudales, cargar_coordenadas, normalizar

    base = Path(__file__).resolve().parent
    datos = base / "csv" / "csv_etapa2"
    caudales, qmod = cargar_caudales(datos, "DatosEtapa2*.csv")
    X = normalizar(caudales, qmod, True, False, False)
    archivos_coords = [
        (datos / "coordsetapa2.csv", True),
        (datos / "DatosEtapa2 - Buenos Aires.csv", True),
    ]
    disponibles = [ruta.exists() for ruta, _ in archivos_coords]
    if not any(disponibles):
        return X.to_numpy(dtype=float), None
    coordenadas = cargar_coordenadas([
        par for par, disponible in zip(archivos_coords, disponibles) if disponible
    ]).set_index("Codigo")
    coordenadas = coordenadas.reindex(X.index)
    validas = coordenadas[["Latitud", "Longitud"]].notna().all(axis=1)
    X = X.loc[validas]
    coords = coordenadas.loc[validas, ["Latitud", "Longitud"]]
    return X.to_numpy(dtype=float), np.asarray(coords, dtype=float)


_base.cargar_datos = cargar_datos
_base.RUTA_RESULTADOS = Path("resultados_k_etapa2.csv")
_base.RUTA_GRAFICO = Path("seleccion_k_etapa2.png")


if __name__ == "__main__":
    _base.ejecutar()
