from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import BallTree


# Configuración editable.
KS = range(2, 51)
B_GAP = 20
N_BOOT = 50
FRAC = 0.8
K_VECINOS = 8
SEED = 0
N_JOBS = -1
K_MIN = 6
K_REF = 37
ARI_PISO = 0.55
COHERENCIA_TOL = 0.03
CORRELACION_REDUNDANTE = 0.98

RUTA_RESULTADOS = Path("resultados_k.csv")
RUTA_GRAFICO = Path("seleccion_k.png")


def cargar_datos():
    from analisis_rios import cargar_caudales, cargar_coordenadas, normalizar

    base = Path(__file__).resolve().parent
    datos = base / "csv" / "csv_etapa1"
    caudales, qmod = cargar_caudales(datos, "estaciones_final*.csv")
    X = normalizar(caudales, qmod, True, False, False)

    archivos_coords = [
        (datos / "coordenadas.csv", False),
        (datos / "estaciones_final - Buenos Aires.csv", True),
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
    return X.to_numpy(dtype=float), coords.to_numpy(dtype=float)


def _modelo(X, k, n_init, seed):
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X)


def _bootstrap_ari(X, k, n_boot, frac, seed):
    referencia = _modelo(X, k, 10, seed)
    n = len(X)
    tamano = max(k, int(round(n * frac)))

    def una_repeticion(rep):
        rng = np.random.default_rng(seed + rep + 1)
        incluidos = np.sort(rng.choice(n, tamano, replace=False))
        excluidos = np.setdiff1d(np.arange(n), incluidos, assume_unique=True)
        modelo = _modelo(X[incluidos], k, 3, seed + rep + 1)
        etiquetas = np.empty(n, dtype=int)
        etiquetas[incluidos] = modelo.labels_
        if len(excluidos):
            etiquetas[excluidos] = modelo.predict(X[excluidos])
        return adjusted_rand_score(referencia.labels_, etiquetas)

    valores = Parallel(n_jobs=N_JOBS)(
        delayed(una_repeticion)(rep) for rep in range(n_boot)
    )
    return float(np.mean(valores)), float(np.std(valores, ddof=1))


def _gap_una_repeticion(X, k, minimo, amplitud, seed):
    rng = np.random.default_rng(seed)
    referencia = rng.uniform(size=X.shape) * amplitud + minimo
    return np.log(_modelo(referencia, k, 3, seed).inertia_)


def calcular_gap(X, ks, b_gap, seed):
    log_inercias = {}
    for k in ks:
        log_inercias[k] = np.log(_modelo(X, k, 3, seed + k).inertia_)
    minimo = X.min(axis=0)
    amplitud = X.max(axis=0) - minimo

    def para_k(k):
        valores = Parallel(n_jobs=N_JOBS)(
            delayed(_gap_una_repeticion)(
                X, k, minimo, amplitud, seed + 10000 * k + b
            )
            for b in range(b_gap)
        )
        referencia = np.asarray(valores)
        esperado = float(np.mean(referencia))
        desvio = float(np.std(referencia, ddof=1))
        return k, (esperado - log_inercias[k], desvio * np.sqrt(1 + 1 / b_gap))

    return dict(Parallel(n_jobs=N_JOBS)(delayed(para_k)(k) for k in ks))


def calcular_vecinos(coords, k_vecinos):
    if coords is None:
        return None
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords debe tener forma (n_rios, 2).")
    if len(coords) <= k_vecinos:
        raise ValueError("K_VECINOS debe ser menor que la cantidad de ríos.")
    radianes = np.radians(coords)
    return BallTree(radianes, metric="haversine").query(
        radianes, k=k_vecinos + 1, return_distance=False
    )[:, 1:]


def coherencia_espacial(labels, vecinos, seed):
    if vecinos is None:
        return np.nan, np.nan
    real = np.mean(labels[vecinos] == labels[:, None])
    permutados = np.random.default_rng(seed).permutation(labels)
    base = np.mean(permutados[vecinos] == permutados[:, None])
    return float(real), float(base)


def metricas_k(X, k, vecinos, seed):
    modelo = _modelo(X, k, 10, seed + k)
    etiquetas = modelo.labels_
    coherencia, base = coherencia_espacial(etiquetas, vecinos, seed + k)
    centroides = modelo.cluster_centers_
    distancias = np.linalg.norm(
        centroides[:, None, :] - centroides[None, :, :], axis=2
    )
    distancias[np.diag_indices(k)] = np.inf
    correlaciones = np.corrcoef(centroides)
    pares_redundantes = int(
        np.sum(np.triu(correlaciones > CORRELACION_REDUNDANTE, 1))
    )
    return {
        "k": k,
        "silueta": silhouette_score(X, etiquetas),
        "calinski_harabasz": calinski_harabasz_score(X, etiquetas),
        "davies_bouldin": davies_bouldin_score(X, etiquetas),
        "coherencia_espacial": coherencia,
        "coherencia_base": base,
        "distancia_min_centroides": float(np.min(distancias)),
        "pares_centroides_redundantes": pares_redundantes,
    }


def decidir_k(tabla, gap):
    candidatos = tabla[
        (tabla["k"] >= K_MIN)
        & (tabla["ari_boot"] >= ARI_PISO)
        & (tabla["pares_centroides_redundantes"] == 0)
    ].copy()
    columnas_score = [
        "ari_boot", "exceso_coherencia", "cociente_coherencia", "silueta"
    ]
    for columna in columnas_score:
        candidatos[f"rango_{columna}"] = candidatos[columna].rank(pct=True)
    candidatos["score"] = candidatos[
        [f"rango_{columna}" for columna in columnas_score]
    ].mean(axis=1)
    top5 = candidatos.sort_values("score", ascending=False).head(5)
    gruesos = candidatos[candidatos["k"].between(K_MIN, 10)]
    finos = candidatos[candidatos["k"].between(15, 40)]
    k_grueso = int(gruesos.loc[gruesos["ari_boot"].idxmax(), "k"]) \
        if len(gruesos) else None
    k_fino = int(finos.loc[finos["score"].idxmax(), "k"]) \
        if len(finos) else None
    fila_ref = tabla[tabla["k"] == K_REF].iloc[0] \
        if (tabla["k"] == K_REF).any() else None
    ks = sorted(gap)
    gap_maximo = max(ks, key=lambda k: gap[k][0])
    gap_regla = next(
        (k for k, sig in zip(ks[:-1], ks[1:])
        if gap[k][0] >= gap[sig][0] - gap[sig][1]),
        ks[-1],
    )
    return k_grueso, k_fino, top5, fila_ref, gap_regla, gap_maximo


def graficar(tabla, gap, k_grueso, k_fino):
    ks = np.array(sorted(gap))
    gaps = np.array([gap[k][0] for k in ks])
    errores = np.array([gap[k][1] for k in ks]) 
    fig, ejes = plt.subplots(3, 2, figsize=(14, 14), sharex=True)
    ejes[0, 0].plot(tabla.k, tabla.ari_boot)
    ejes[0, 0].fill_between(
        tabla.k, tabla.ari_boot - tabla.ari_std, tabla.ari_boot + tabla.ari_std,
        alpha=.2,
    )
    ejes[0, 0].set_ylabel("ARI bootstrap")
    ejes[0, 1].errorbar(ks, gaps, yerr=errores, fmt="o-")
    ejes[0, 1].set_ylabel("Gap")
    ejes[1, 0].plot(tabla.k, tabla.silueta)
    ejes[1, 0].set_ylabel("Silueta")
    ejes[1, 1].plot(tabla.k, tabla.coherencia_espacial, label="real")
    ejes[1, 1].plot(tabla.k, tabla.coherencia_base, "--", label="azar")
    ejes[1, 1].set_ylabel("Coherencia espacial")
    exceso = ejes[1, 1].twinx()
    exceso.plot(tabla.k, tabla.exceso_coherencia, color="purple",
                label="exceso")
    exceso.set_ylabel("Exceso de coherencia")
    ejes[1, 1].legend(loc="upper left")
    exceso.legend(loc="upper right")
    ejes[2, 0].plot(tabla.k, tabla.calinski_harabasz)
    ejes[2, 0].set_ylabel("Calinski-Harabasz")
    ejes[2, 1].plot(tabla.k, tabla.davies_bouldin)
    ejes[2, 1].set_ylabel("Davies-Bouldin")
    for eje in ejes.flat:
        eje.grid(alpha=.25)
        eje.set_xlabel("k")
        if k_grueso is not None:
            eje.axvline(k_grueso, color="blue", linestyle="--")
        if k_fino is not None:
            eje.axvline(k_fino, color="green", linestyle="--")
        eje.axvline(K_REF, color="gray", linestyle=":")
    fig.tight_layout()
    fig.savefig(RUTA_GRAFICO, dpi=150)
    plt.close(fig)


def ejecutar():
    X, coords = cargar_datos()
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != 12:
        raise ValueError("X debe tener forma (n_rios, 12).")
    ks = list(KS)
    columnas_cache = {
        "k", "ari_boot", "ari_std", "gap", "gap_std",
        "silueta", "coherencia_espacial", "coherencia_base",
        "exceso_coherencia", "cociente_coherencia",
        "calinski_harabasz", "davies_bouldin",
        "distancia_min_centroides", "pares_centroides_redundantes",
    }
    if RUTA_RESULTADOS.exists():
        tabla = pd.read_csv(RUTA_RESULTADOS)
        if set(tabla.columns) >= columnas_cache and set(tabla["k"]) == set(ks):
            tabla = tabla.sort_values("k").reset_index(drop=True)
            gap = {
                int(fila.k): (fila.gap, fila.gap_std)
                for fila in tabla.itertuples()
            }
        else:
            tabla = None
    else:
        tabla = None
    if tabla is None:
        vecinos = calcular_vecinos(coords, K_VECINOS)
        boot = Parallel(n_jobs=N_JOBS)(
            delayed(_bootstrap_ari)(X, k, N_BOOT, FRAC, SEED) for k in ks
        )
        metricas = Parallel(n_jobs=N_JOBS)(
            delayed(metricas_k)(X, k, vecinos, SEED) for k in ks
        )
        tabla = pd.DataFrame(metricas)
        tabla["ari_boot"] = [valor[0] for valor in boot]
        tabla["ari_std"] = [valor[1] for valor in boot]
        gap = calcular_gap(X, ks, B_GAP, SEED)
        tabla["gap"] = tabla["k"].map(lambda k: gap[k][0])
        tabla["gap_std"] = tabla["k"].map(lambda k: gap[k][1])
    tabla["exceso_coherencia"] = (
        tabla["coherencia_espacial"] - tabla["coherencia_base"]
    )
    tabla["cociente_coherencia"] = (
        tabla["coherencia_espacial"]
        / tabla["coherencia_base"].clip(lower=1e-3)
    )
    tabla.to_csv(RUTA_RESULTADOS, index=False)
    k_grueso, k_fino, top5, fila_ref, gap_regla, gap_maximo = decidir_k(
        tabla, gap
    )
    graficar(tabla.sort_values("k"), gap, k_grueso, k_fino)
    print(tabla.sort_values("k").to_string(index=False))
    print("\nK grueso:", k_grueso)
    print("K fino:", k_fino)
    print("Top 5 por score:\n", top5)
    if fila_ref is not None and k_fino is not None:
        columnas_comparacion = [
            "ari_boot", "coherencia_espacial", "exceso_coherencia",
            "cociente_coherencia", "silueta", "pares_centroides_redundantes",
        ]
        comparacion = pd.DataFrame(
            {"K_REF": fila_ref[columnas_comparacion],
             "K_FIN": tabla.loc[tabla["k"] == k_fino,
                                columnas_comparacion].iloc[0]}
        )
        print("\nK_REF vs K fino:\n", comparacion)
    print("K por regla Gap:", gap_regla)
    print("K de Gap máximo:", gap_maximo)
    print("No hay k natural si el gap crece monótonamente; elegir k por resolución.")
    return tabla, gap, k_fino


if __name__ == "__main__":
    ejecutar()
