"""Análisis de la primera etapa.

Vector de 15 datos por río: 12 caudales mensuales normalizados (Q/Qmod)
+ Latitud + Longitud + Altitud. Sólo se usan los CSV de regiones de la
etapa 1 (no se usa "estaciones_final - Buenos Aires.csv").
Luego el vector completo se estandariza con StandardScaler
(z = (x - media) / desvío por columna) para que todo quede en la misma escala.
"""

import shutil
from pathlib import Path
import pandas as pd
from analisis_rios import (
    analizar, cargar_caracteristicas_etapa1, guardar_histogramas,
    guardar_rios_por_cluster,
    guardar_mapa, guardar_metricas,
)
from animacion_y_graficos import generar_graficos, generar_gif

BASE = Path(__file__).parent
DATOS = BASE / "csv" / "csv_etapa1"
SALIDA = BASE / "resultados_etapa1"
COORDENADAS, CARACTERISTICAS = cargar_caracteristicas_etapa1(BASE)
NOMBRES = COORDENADAS.set_index("Codigo")["Rio"].to_dict()

# 15: resolución fina previa; 26: k fino óptimo de seleccion_k.py; 27: pedido.
KS = (15, 26, 27)

# Ejemplo: 10 m3/s pasa a 0.5 al dividirlo por Qmod=20.
USAR_QMOD = True
# Ejemplo: log1p(99)=4.60 reduce el peso de un extremo de 99.
USAR_LOG = False
# Estandarización sólo de los caudales (antes de agregar lat/lon/altitud).
USAR_STANDARSCALER = False
# Estandariza el vector completo de 15 datos (caudales + lat/lon/altitud).
ESCALAR_VECTOR = True


analisis = analizar(
    DATOS, "estaciones_final - Region_*.csv", max(KS),
    USAR_QMOD, USAR_LOG, USAR_STANDARSCALER,
    CARACTERISTICAS,
    ESCALAR_VECTOR,
)
SALIDA.mkdir(exist_ok=True)
analisis["metricas"].to_csv(SALIDA / "metricas_kmeans.csv", index=False)
guardar_metricas(analisis, SALIDA / "metricas_kmeans.png")
for k in KS:
    destino = SALIDA / f"{k}_cluster"
    if destino.exists():
        shutil.rmtree(destino)
    (destino / "histogramas").mkdir(parents=True, exist_ok=True)
    analisis["metricas"].to_csv(destino / "metricas_kmeans.csv", index=False)
    asignaciones = pd.DataFrame({
        "Codigo": analisis["codigos"],
        "Rio": [NOMBRES.get(c, "") for c in analisis["codigos"]],
        "Cluster": analisis["modelos"][k].labels_,
    })
    asignaciones.to_csv(destino / "clusters_rios.csv", index=False)
    carpeta_rios = BASE / "rios_por_cluster_etapa1" / f"{k}_cluster"
    if carpeta_rios.exists():
        shutil.rmtree(carpeta_rios)
    guardar_rios_por_cluster(
        analisis["caudales"], asignaciones, NOMBRES, carpeta_rios,
    )
    generar_graficos(analisis, destino, k)
    generar_gif(analisis, destino, k)
    guardar_mapa(analisis, COORDENADAS, destino / "mapa_geopandas.png", k,
                 destino / "referencia_mapa.csv")
    guardar_mapa(
        analisis, COORDENADAS,
        destino / "mapa_geopandas_numerado.png", k,
        destino / "referencia_mapa.csv", numerar=True,
    )
    guardar_histogramas(analisis, destino / "histogramas", k, NOMBRES)
