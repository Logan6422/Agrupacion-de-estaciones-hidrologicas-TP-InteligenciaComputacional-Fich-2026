from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

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
    coords_array = np.asarray(coords, dtype=float)
    X = np.column_stack((X.to_numpy(dtype=float), coords_array))
    # Igual que en etapa 1: se estandariza cada columna para que Lat/Lon
    # (en grados) no dominen la distancia frente a los caudales Q/Qmod.
    return StandardScaler().fit_transform(X), coords_array


_base.cargar_datos = cargar_datos
_base.RUTA_RESULTADOS = Path("resultados_k_etapa2_posicion.csv")
# Etapa 2: la coherencia espacial NO entra en el criterio de elección de k.
# Se sigue calculando y graficando sólo como información.
_base.COLUMNAS_SCORE = ["ari_boot", "silueta"]
# 12 caudales Q/Qmod + Latitud + Longitud.
_base.N_COLUMNAS = 14
_base.TITULO = "Selección de k (etapa 2, sin coherencia espacial en el criterio)"
_base.RUTA_GRAFICO = Path("seleccion_k_etapa2_posicion.png")


if __name__ == "__main__":
    _base.ejecutar()
