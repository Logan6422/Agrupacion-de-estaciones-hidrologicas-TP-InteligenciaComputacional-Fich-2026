"""Análisis de la primera etapa."""

from pathlib import Path
import pandas as pd
from analisis_rios import (
    analizar, cargar_coordenadas, guardar_histogramas,
    guardar_rios_por_cluster,
    guardar_mapa, guardar_metricas,
)
from animacion_y_graficos import generar_graficos, generar_gif

BASE = Path(__file__).parent
DATOS = BASE / "csv" / "csv_etapa1"
SALIDA = BASE / "resultados_etapa1"
COORDENADAS = cargar_coordenadas([
    (BASE / "csv" / "csv_etapa1" / "coordenadas.csv", False),
    (BASE / "csv" / "csv_etapa1" / "estaciones_final - Buenos Aires.csv", True),
])

# Ejemplo: 10 m3/s pasa a 0.5 al dividirlo por Qmod=20.
USAR_QMOD = True
# Ejemplo: log1p(99)=4.60 reduce el peso de un extremo de 99.
USAR_LOG = False
# Ejemplo: transforma una columna con media 10 y desvío 2 en z=(x-10)/2.
USAR_STANDARSCALER = False


analisis = analizar(
    DATOS, "estaciones_final*.csv", 15,
    USAR_QMOD, USAR_LOG, USAR_STANDARSCALER,
)
analisis["metricas"].to_csv(SALIDA / "metricas_kmeans.csv", index=False)
guardar_metricas(analisis, SALIDA / "metricas_kmeans.png")
for k in (9, 15):
    if k in locals().get("_generados", set()):
        continue
    _generados = locals().get("_generados", set()) | {k}
    destino = SALIDA / f"{k}_cluster"
    (destino / "histogramas").mkdir(parents=True, exist_ok=True)
    analisis["metricas"].to_csv(destino / "metricas_kmeans.csv", index=False)
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
        BASE / "rios_por_cluster_etapa1" / f"{k}_cluster",
    )
    generar_graficos(analisis, destino, k)
    generar_gif(analisis, destino, k)
    guardar_mapa(analisis, COORDENADAS, destino / "mapa_geopandas.png", k,
                 destino / "referencia_mapa.csv")
    guardar_histogramas(analisis, destino / "histogramas", k)
