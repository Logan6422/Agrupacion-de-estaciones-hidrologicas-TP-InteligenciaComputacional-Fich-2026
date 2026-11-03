"""Ejecuta el análisis completo del dataset de la segunda etapa."""

from pathlib import Path

import pandas as pd

from analisis_rios import (
    analizar,
    generar_histogramas,
    graficar_espacio_solucion,
    graficar_mapa_geopandas,
    graficar_mapa_clusters,
    graficar_resultados,
)


CARPETA_BASE = Path(__file__).parent
CARPETA_DATOS = CARPETA_BASE / "csv" / "datosetapa2"
CARPETA_SALIDA = CARPETA_BASE / "datosetapa2"
CARPETA_EXTRA = CARPETA_BASE / "extra"
COORDENADAS = pd.read_csv(CARPETA_EXTRA / "coordsetapa2.csv")
COORDENADAS_BA = pd.read_csv(
    CARPETA_BASE / "csv" / "datosetapa2" / "DatosEtapa2 - Buenos Aires.csv",
    encoding="latin1",
)
COORDENADAS_BA["Latitud"] = -COORDENADAS_BA["Latitud"].abs()
COORDENADAS_BA["Longitud"] = -COORDENADAS_BA["Longitud"].abs()
COORDENADAS = pd.concat(
    [COORDENADAS, COORDENADAS_BA],
    ignore_index=True,
)
COORDENADAS["Codigo"] = pd.to_numeric(COORDENADAS["Codigo"], errors="coerce")
COORDENADAS["Codigo"] = COORDENADAS["Codigo"].astype("Int64")
# El archivo usa magnitudes decimales para Argentina; el mapa necesita
# latitudes sur y longitudes oeste expresadas con signo negativo.
COORDENADAS["Latitud"] = -COORDENADAS["Latitud"].abs()
COORDENADAS["Longitud"] = -COORDENADAS["Longitud"].abs()
NOMBRES_POR_CODIGO = (
    COORDENADAS.dropna(subset=["Codigo"])
    .set_index("Codigo")["Rio"]
    .fillna("Río")
    .to_dict()
)


def ejecutar_analisis(cantidad_clusters):
    """Genera todos los resultados de etapa 2 para una cantidad de clusters."""
    carpeta = CARPETA_SALIDA / f"{cantidad_clusters}_cluster"
    carpeta_resultados = carpeta / "resultados"
    carpeta_histogramas = carpeta / "histogramas"
    carpeta_resultados.mkdir(parents=True, exist_ok=True)
    carpeta_histogramas.mkdir(parents=True, exist_ok=True)

    resultados = analizar(
        k_seleccionado=cantidad_clusters,
        carpeta_datos=CARPETA_DATOS,
        patron="DatosEtapa2*.csv",
        k_maximo=80,
    )
    graficar_resultados(resultados, carpeta_resultados / "metricas_kmeans.png")
    resultados["asignaciones"].to_csv(
        carpeta_resultados / "clusters_rios.csv",
        index=False,
    )
    resultados["metricas"].to_csv(
        carpeta_resultados / "metricas_kmeans.csv",
        index=False,
    )
    codigos_datos = set(resultados["asignaciones"]["Codigo"])
    codigos_coordenadas = set(
        COORDENADAS["Codigo"].dropna().astype(int)
    )
    codigos_sin_coordenadas = sorted(codigos_datos - codigos_coordenadas)
    pd.DataFrame({"Codigo": codigos_sin_coordenadas}).to_csv(
        carpeta_resultados / "rios_sin_coordenadas.csv",
        index=False,
    )
    graficar_espacio_solucion(
        resultados,
        carpeta_resultados / "espacio_solucion_kmeans.png",
    )
    graficar_mapa_clusters(
        resultados,
        COORDENADAS,
        carpeta_resultados / "mapa_clusters_argentina.png",
    )
    graficar_mapa_geopandas(
        resultados,
        COORDENADAS,
        carpeta_resultados / "mapa_clusters_argentina_geopandas.png",
        carpeta_resultados / "referencia_mapa_geopandas.csv",
    )
    generar_histogramas(
        resultados,
        carpeta_histogramas,
        NOMBRES_POR_CODIGO,
    )
    print(f"\n===== Dataset etapa 2: k={cantidad_clusters} =====")
    print(f"Ríos procesados: {len(resultados['codigos'])}")
    print(f"Tamaño del SOM: {resultados['tamano_som']}")
    print(
        "Ríos con coordenadas: "
        f"{len(codigos_datos) - len(codigos_sin_coordenadas)}; "
        f"sin coordenadas: {len(codigos_sin_coordenadas)}"
    )
    print(resultados["metricas"].head().to_string(index=False))
    print(f"k seleccionado: {resultados['k_optimo']}")
    print(f"k del codo: {resultados['k_codo']}")


for cantidad_clusters in (8, 18, 40, 60, 80):
    ejecutar_analisis(cantidad_clusters)
