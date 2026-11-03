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
CARPETA_ANALISIS = CARPETA_BASE
CARPETA_EXTRA = Path(__file__).parent / "extra"

coordenadas = pd.read_csv(CARPETA_EXTRA / "coordenadas.csv")
coordenadas["Codigo"] = pd.to_numeric(coordenadas["Codigo"], errors="coerce")
coordenadas["Codigo"] = coordenadas["Codigo"].astype("Int64")
nombres_por_codigo = (
    coordenadas.dropna(subset=["Codigo"])
    .set_index("Codigo")["Rio"]
    .fillna("Río")
    .to_dict()
)

def ejecutar_analisis(cantidad_clusters):
    """Ejecuta un análisis y guarda todos sus archivos en su propia carpeta."""
    carpeta = CARPETA_ANALISIS / f"{cantidad_clusters}_cluster"
    carpeta_resultados = carpeta / "resultados"
    carpeta_histogramas = carpeta / "histogramas"
    carpeta_resultados.mkdir(parents=True, exist_ok=True)
    carpeta_histogramas.mkdir(parents=True, exist_ok=True)

    resultados = analizar(k_seleccionado=cantidad_clusters)
    graficar_resultados(
        resultados,
        carpeta_resultados / "metricas_kmeans.png",
    )
    resultados["asignaciones"].to_csv(
        carpeta_resultados / "clusters_rios.csv",
        index=False,
    )
    resultados["metricas"].to_csv(
        carpeta_resultados / "metricas_kmeans.csv",
        index=False,
    )

    referencia = pd.read_csv(CARPETA_EXTRA / "comparacion_referencia.csv")
    referencia["Codigo"] = pd.to_numeric(referencia["Lugar"], errors="coerce")
    referencia["Region_Congreso"] = pd.to_numeric(
        referencia["Region_Congreso"],
        errors="coerce",
    )
    comparacion = resultados["asignaciones"].merge(
        referencia[["Codigo", "Region_Congreso"]],
        on="Codigo",
        how="left",
    )
    comparacion.to_csv(
        carpeta_resultados / "comparacion_clusters_region_congreso.csv",
        index=False,
    )
    graficar_espacio_solucion(
        resultados,
        carpeta_resultados / "espacio_solucion_kmeans.png",
    )
    graficar_mapa_clusters(
        resultados,
        coordenadas,
        carpeta_resultados / "mapa_clusters_argentina.png",
    )
    graficar_mapa_geopandas(
        resultados,
        coordenadas,
        carpeta_resultados / "mapa_clusters_argentina_geopandas.png",
        carpeta_resultados / "referencia_mapa_geopandas.csv",
    )
    generar_histogramas(resultados, carpeta_histogramas, nombres_por_codigo)

    print(f"\n===== Análisis con k={cantidad_clusters} =====")
    print(resultados["metricas"].head().to_string(index=False))
    print(f"k seleccionado: {resultados['k_optimo']}")
    print(f"k del codo: {resultados['k_codo']}")


for cantidad_clusters in (8, 18, 40):
    ejecutar_analisis(cantidad_clusters)
