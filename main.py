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


CARPETA_RESULTADOS = Path(__file__).parent / "resultados"
CARPETA_RESULTADOS.mkdir(exist_ok=True)
CARPETA_HISTOGRAMAS = Path(__file__).parent / "histogramas"
CARPETA_HISTOGRAMAS.mkdir(exist_ok=True)
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

# Ejecuta todas las etapas: carga, normalización, SOM, K-Means y métricas.
resultados = analizar()

# Guarda y muestra los gráficos del método del codo y de la silueta.
graficar_resultados(
    resultados,
    CARPETA_RESULTADOS / "metricas_kmeans.png",
)

# Guarda los códigos y su cluster óptimo.
resultados["asignaciones"].to_csv(
    CARPETA_RESULTADOS / "clusters_rios.csv",
    index=False,
)

# Guarda las métricas ordenadas de mejor a peor según la silueta.
resultados["metricas"].to_csv(
    CARPETA_RESULTADOS / "metricas_kmeans.csv",
    index=False,
)

# Une la clasificación automática con la clasificación manual previa.
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
    CARPETA_RESULTADOS / "comparacion_clusters_region_congreso.csv",
    index=False,
)

# Guarda una imagen final con los datos coloreados y los centroides.
graficar_espacio_solucion(
    resultados,
    CARPETA_RESULTADOS / "espacio_solucion_kmeans.png",
)
graficar_mapa_clusters(
    resultados,
    coordenadas,
    CARPETA_RESULTADOS / "mapa_clusters_argentina.png",
)
graficar_mapa_geopandas(
    resultados,
    coordenadas,
    CARPETA_RESULTADOS / "mapa_clusters_argentina_geopandas.png",
)
generar_histogramas(resultados, CARPETA_HISTOGRAMAS, nombres_por_codigo)

# Muestra todas las métricas calculadas para los valores de k evaluados.
print(resultados["metricas"].to_string(index=False))

# Muestra la comparación contra la clasificación manual.
print("\nComparación con Region_Congreso:")
print(comparacion.to_string(index=False))

# Informa el óptimo por silueta y el punto estimado por el método del codo.
print(f"\nk óptimo por silueta: {resultados['k_optimo']}")
print(f"k del codo: {resultados['k_codo']}")
