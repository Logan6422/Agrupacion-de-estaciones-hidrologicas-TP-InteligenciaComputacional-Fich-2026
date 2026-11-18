"""Métricas completas del K-Means de una carpeta resultados_etapa1/<k>_cluster.

Uso:  python metricas_cluster.py 37

Se evalúa el agrupamiento real de la carpeta (las etiquetas de
clusters_rios.csv, el mismo modelo de main_etapa1.py) sobre el mismo vector de
15 datos estandarizado (12 Q/Qmod + Latitud + Longitud + Altitud).

Genera en la carpeta del k:
- metricas_k<k>.csv: una fila con todas las métricas globales (incluye ARI).
- metricas_por_cluster_k<k>.csv: una fila por cluster.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)

import seleccion_k as sk

BASE = Path(__file__).resolve().parent


def ari_y_jaccard_bootstrap(X, etiquetas, k, n_boot, frac, seed):
    """Mismo bootstrap que seleccion_k.py, pero con referencia = etiquetas dadas.

    Devuelve los ARI de cada repetición y, por cluster, la estabilidad de
    Jaccard media (Hennig 2007): para cada cluster de referencia, el mayor
    Jaccard con algún cluster de la repetición.
    """
    n = len(X)
    tamano = max(k, int(round(n * frac)))
    aris, jaccard = [], np.zeros((n_boot, k))
    for rep in range(n_boot):
        rng = np.random.default_rng(seed + rep + 1)
        incluidos = np.sort(rng.choice(n, tamano, replace=False))
        excluidos = np.setdiff1d(np.arange(n), incluidos, assume_unique=True)
        modelo = KMeans(n_clusters=k, n_init=3, random_state=seed + rep + 1)
        modelo.fit(X[incluidos])
        nuevas = np.empty(n, dtype=int)
        nuevas[incluidos] = modelo.labels_
        if len(excluidos):
            nuevas[excluidos] = modelo.predict(X[excluidos])
        aris.append(adjusted_rand_score(etiquetas, nuevas))
        for c in range(k):
            a = etiquetas == c
            jaccard[rep, c] = max(
                np.sum(a & (nuevas == d)) / np.sum(a | (nuevas == d))
                for d in range(k)
            )
    return np.asarray(aris), jaccard.mean(axis=0)


def main(k):
    carpeta = BASE / "resultados_etapa1" / f"{k}_cluster"
    asignaciones = pd.read_csv(carpeta / "clusters_rios.csv")
    X, coords = sk.cargar_datos()
    X = np.asarray(X, dtype=float)
    etiquetas = asignaciones["Cluster"].to_numpy()
    if len(etiquetas) != len(X):
        raise ValueError("clusters_rios.csv no coincide con los datos.")

    centroides = np.vstack([X[etiquetas == c].mean(axis=0) for c in range(k)])
    sse_cluster = np.array([
        ((X[etiquetas == c] - centroides[c]) ** 2).sum() for c in range(k)
    ])
    distancias = np.linalg.norm(
        centroides[:, None, :] - centroides[None, :, :], axis=2
    )
    np.fill_diagonal(distancias, np.inf)
    correlaciones = np.corrcoef(centroides)
    np.fill_diagonal(correlaciones, -np.inf)
    vecinos = sk.calcular_vecinos(coords, sk.K_VECINOS)
    coherencia, coherencia_base = sk.coherencia_espacial(
        etiquetas, vecinos, sk.SEED + k
    )
    aris, jaccard = ari_y_jaccard_bootstrap(
        X, etiquetas, k, sk.N_BOOT, sk.FRAC, sk.SEED
    )

    # Valores de la tabla de selección de k (otra corrida de K-Means, n_init=10)
    # para ubicar este k dentro del criterio de elección.
    tabla = pd.read_csv(BASE / sk.RUTA_RESULTADOS)
    candidatos = tabla[
        (tabla["k"] >= sk.K_MIN)
        & (tabla["ari_boot"] >= sk.ARI_PISO)
        & (tabla["pares_centroides_redundantes"] == 0)
    ].copy()
    for columna in sk.COLUMNAS_SCORE:
        candidatos[f"rango_{columna}"] = candidatos[columna].rank(pct=True)
    candidatos["score"] = candidatos[
        [f"rango_{c}" for c in sk.COLUMNAS_SCORE]
    ].mean(axis=1)
    fila_tabla = tabla.loc[tabla["k"] == k].iloc[0]
    es_candidato = k in set(candidatos["k"])
    fila_cand = candidatos.loc[candidatos["k"] == k].iloc[0] \
        if es_candidato else None

    globales = {
        "k": k,
        "n_rios": len(X),
        "inercia": float(sse_cluster.sum()),
        "silueta": silhouette_score(X, etiquetas),
        "calinski_harabasz": calinski_harabasz_score(X, etiquetas),
        "davies_bouldin": davies_bouldin_score(X, etiquetas),
        "ari_boot": aris.mean(),
        "ari_std": aris.std(ddof=1),
        "ari_min": aris.min(),
        "ari_max": aris.max(),
        "coherencia_espacial": coherencia,
        "coherencia_base": coherencia_base,
        "exceso_coherencia": coherencia - coherencia_base,
        "cociente_coherencia": coherencia / max(coherencia_base, 1e-3),
        "distancia_min_centroides": float(distancias.min()),
        "correlacion_max_centroides": float(correlaciones.max()),
        "pares_centroides_redundantes": int(
            np.sum(np.triu(correlaciones > sk.CORRELACION_REDUNDANTE, 1))
        ),
        "clusters_de_1_rio": int(np.sum(np.bincount(etiquetas) == 1)),
        "gap": fila_tabla["gap"],
        "gap_std": fila_tabla["gap_std"],
        "criterio_es_candidato": es_candidato,
        "criterio_rango_ari_boot":
            fila_cand["rango_ari_boot"] if es_candidato else np.nan,
        "criterio_rango_silueta":
            fila_cand["rango_silueta"] if es_candidato else np.nan,
        "criterio_score": fila_cand["score"] if es_candidato else np.nan,
        "criterio_puesto_score": int(
            (candidatos["score"] > fila_cand["score"]).sum() + 1
        ) if es_candidato else np.nan,
        "criterio_n_candidatos": len(candidatos),
    }
    pd.DataFrame([globales]).to_csv(carpeta / f"metricas_k{k}.csv", index=False)

    silueta_rio = silhouette_samples(X, etiquetas)
    filas = []
    for c in range(k):
        miembros = etiquetas == c
        filas.append({
            "Cluster": c,
            "n_rios": int(miembros.sum()),
            "silueta_media": silueta_rio[miembros].mean(),
            "silueta_min": silueta_rio[miembros].min(),
            "estabilidad_jaccard": jaccard[c],
            "inercia_intra": sse_cluster[c],
            "dispersion_media": float(np.sqrt(
                sse_cluster[c] / miembros.sum()
            )),
            "distancia_centroide_mas_cercano": distancias[c].min(),
            "cluster_mas_cercano": int(distancias[c].argmin()),
            "correlacion_max_otro_centroide": correlaciones[c].max(),
            "Latitud_media": coords[miembros, 0].mean(),
            "Longitud_media": coords[miembros, 1].mean(),
            "Rios": " | ".join(
                asignaciones.loc[miembros, "Rio"].fillna("").astype(str)
            ),
        })
    por_cluster = pd.DataFrame(filas)
    por_cluster.to_csv(carpeta / f"metricas_por_cluster_k{k}.csv", index=False)
    return globales, por_cluster


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 37
    globales, por_cluster = main(k)
    for nombre, valor in globales.items():
        print(f"{nombre:32s} {valor}")
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(por_cluster.round(3).to_string(index=False))
