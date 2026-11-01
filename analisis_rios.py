from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


MESES = [
    "Sep", "Oct", "Nov", "Dic", "Ene", "Feb",
    "Mar", "Abr", "May", "Jun", "Jul", "Ago",
]
K_TEORICO = 23
K_MAXIMO = 40
TAMANO_SOM = (10, 10)
SEMILLA = 42


def cargar_caudales():
    carpeta = Path(__file__).parent / "csv"
    archivos = sorted(carpeta.glob("estaciones_final*.csv"))
    estaciones = pd.concat(
        (pd.read_csv(archivo, encoding="latin1") for archivo in archivos),
        ignore_index=True,
    )
    estaciones["Codigo"] = pd.to_numeric(estaciones["Codigo"], errors="coerce")
    estaciones["Qmod"] = pd.to_numeric(estaciones["Qmod"], errors="coerce")
    estaciones = estaciones.dropna(subset=["Codigo", "Qmod"]).copy()
    estaciones["Codigo"] = estaciones["Codigo"].astype(int)
    caudales = estaciones.groupby("Codigo", sort=True)[MESES].mean()
    qmod = estaciones.groupby("Codigo", sort=True)["Qmod"].mean()
    return caudales.div(qmod, axis=0)


def preparar_datos(caudales):
    logaritmos = np.log1p(caudales.clip(lower=0))
    return StandardScaler().fit_transform(logaritmos)


def entrenar_som(datos, filas, columnas, epocas=200):
    rng = np.random.default_rng(SEMILLA)
    prototipos = rng.normal(
        datos.mean(axis=0),
        datos.std(axis=0) + 1e-12,
        size=(filas, columnas, datos.shape[1]),
    )
    posiciones = np.indices((filas, columnas)).transpose(1, 2, 0)
    radio_inicial = max(filas, columnas) / 2

    for epoca in range(epocas):
        progreso = epoca / max(epocas - 1, 1)
        tasa = 0.5 * (0.05 / 0.5) ** progreso
        radio = radio_inicial * (1 / radio_inicial) ** progreso
        for indice in rng.permutation(len(datos)):
            vector = datos[indice]
            distancias = np.linalg.norm(prototipos - vector, axis=2)
            ganador = np.unravel_index(np.argmin(distancias), distancias.shape)
            distancia_mapa = np.sum((posiciones - ganador) ** 2, axis=2)
            vecindad = np.exp(-distancia_mapa / (2 * radio**2))
            prototipos += tasa * vecindad[:, :, None] * (vector - prototipos)
    return prototipos


def suavizar_con_som(datos):
    prototipos = entrenar_som(datos, *TAMANO_SOM)
    distancias = np.linalg.norm(
        datos[:, None, None, :] - prototipos[None, :, :, :],
        axis=3,
    )
    ganadores = np.argmin(distancias.reshape(len(datos), -1), axis=1)
    return prototipos.reshape(-1, datos.shape[1])[ganadores], prototipos


def calcular_codo(inercias, valores_k):
    puntos = np.column_stack((valores_k, inercias))
    recta = puntos[-1] - puntos[0]
    numerador = np.abs(
        recta[0] * (puntos[:, 1] - puntos[0, 1])
        - recta[1] * (puntos[:, 0] - puntos[0, 0])
    )
    return int(puntos[np.argmax(numerador / np.linalg.norm(recta)), 0])


def evaluar_clusters(datos, valores_k):
    filas = []
    modelos = {}
    for k in valores_k:
        modelo = KMeans(n_clusters=k, n_init=50, random_state=SEMILLA)
        etiquetas = modelo.fit_predict(datos)
        modelos[k] = modelo
        filas.append(
            {
                "k": k,
                "inercia": modelo.inertia_,
                "silueta": silhouette_score(datos, etiquetas),
                "calinski_harabasz": calinski_harabasz_score(datos, etiquetas),
                "davies_bouldin": davies_bouldin_score(datos, etiquetas),
            }
        )
    return pd.DataFrame(filas), modelos


def analizar():
    caudales = cargar_caudales()
    datos = preparar_datos(caudales)
    datos_suavizados, prototipos_som = suavizar_con_som(datos)
    valores_k = np.arange(2, min(K_MAXIMO, len(datos) - 1) + 1)
    metricas, modelos = evaluar_clusters(datos_suavizados, valores_k)
    k_optimo = int(metricas.loc[metricas["silueta"].idxmax(), "k"])
    k_codo = calcular_codo(metricas["inercia"].to_numpy(), valores_k)
    comparacion = metricas[metricas["k"].isin([k_optimo, K_TEORICO])].sort_values("k")
    asignaciones = pd.DataFrame({"Codigo": caudales.index})
    for k, nombre in [
        (k_optimo, "Cluster_optimo"),
        (K_TEORICO, "Cluster_teorico"),
    ]:
        asignaciones[nombre] = modelos[k].labels_
    return {
        "codigos": caudales.index.to_numpy(),
        "datos": datos_suavizados,
        "prototipos_som": prototipos_som,
        "metricas": metricas.sort_values("silueta", ascending=False),
        "modelos": modelos,
        "k_optimo": k_optimo,
        "k_codo": k_codo,
        "k_teorico": K_TEORICO,
        "comparacion": comparacion,
        "asignaciones": asignaciones,
    }


def graficar_resultados(resultados):
    metricas = resultados["metricas"].sort_values("k")
    k_optimo = resultados["k_optimo"]
    k_teorico = resultados["k_teorico"]
    fig, ejes = plt.subplots(1, 2, figsize=(14, 5))

    ejes[0].plot(metricas["k"], metricas["inercia"], marker="o")
    ejes[0].axvline(k_optimo, color="red", linestyle="--", label=f"Óptimo: {k_optimo}")
    ejes[0].axvline(k_teorico, color="green", linestyle=":", label=f"Teórico: {k_teorico}")
    ejes[0].set(title="Método del codo", xlabel="k", ylabel="Inercia")

    ejes[1].plot(metricas["k"], metricas["silueta"], marker="o")
    ejes[1].axvline(k_optimo, color="red", linestyle="--", label=f"Óptimo: {k_optimo}")
    ejes[1].axvline(k_teorico, color="green", linestyle=":", label=f"Teórico: {k_teorico}")
    ejes[1].set(title="Coeficiente de silueta", xlabel="k", ylabel="Silueta")

    for eje in ejes:
        eje.grid(alpha=0.3)
        eje.legend()
    plt.tight_layout()
    plt.show()
