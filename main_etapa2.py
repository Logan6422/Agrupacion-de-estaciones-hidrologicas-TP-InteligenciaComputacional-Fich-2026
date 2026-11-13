"""Análisis de la segunda etapa."""

from pathlib import Path
import pandas as pd
from analisis_rios import (
    analizar,
    cargar_coordenadas,
    guardar_histogramas,
    guardar_mapa,
    guardar_metricas,
    guardar_rios_por_cluster,
)
from animacion_y_graficos import generar_graficos, generar_gif

BASE = Path(__file__).parent
DATOS = BASE / "csv" / "csv_etapa2"
SALIDA = BASE / "resultados_etapa2"
COORDENADAS = cargar_coordenadas([
    (DATOS / "coordsetapa2.csv", True),
    (DATOS / "DatosEtapa2 - Buenos Aires.csv", True),
])

# Estas tres variables pueden cambiarse a False sin romper las demás etapas.
USAR_QMOD, USAR_LOG, USAR_STANDARSCALER = True, False, False

analisis = analizar(
    DATOS, "DatosEtapa2*.csv", 15,
    USAR_QMOD, USAR_LOG, USAR_STANDARSCALER,
)
analisis["metricas"].to_csv(SALIDA / "metricas_kmeans.csv", index=False)
guardar_metricas(analisis, SALIDA / "metricas_kmeans.png")
for k in (10, 15):
    if k in locals().get("_generados", set()):
        continue
    _generados = locals().get("_generados", set()) | {k}
    destino = SALIDA / f"{k}_cluster"
    (destino / "histogramas").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "Codigo": analisis["codigos"],
        "Cluster": analisis["modelos"][k].labels_,
    }).to_csv(destino / "clusters_rios.csv", index=False)
    guardar_rios_por_cluster(
        analisis["caudales"],
        pd.DataFrame({
            "Codigo": analisis["codigos"],
            "Cluster": analisis["modelos"][k].labels_,
        }),
        COORDENADAS.set_index("Codigo")["Rio"].to_dict(),
        BASE / "rios_por_cluster_etapa2" / f"{k}_cluster",
    )
    generar_graficos(analisis, destino, k)
    generar_gif(analisis, destino, k)
    guardar_mapa(analisis, COORDENADAS, destino / "mapa_geopandas.png", k,
                 destino / "referencia_mapa.csv")
    guardar_mapa(
        analisis, COORDENADAS,
        destino / "mapa_geopandas_con_nombres.png", k,
        destino / "referencia_mapa.csv", mostrar_nombres=True,
    )
    guardar_histogramas(analisis, destino / "histogramas", k)
