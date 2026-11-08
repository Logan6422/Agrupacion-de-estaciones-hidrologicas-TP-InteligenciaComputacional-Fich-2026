"""Genera gráficos complementarios para un análisis ya calculado."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from analisis_rios import guardar_espacio


def generar_graficos(analisis, carpeta, k):
    """Guarda el espacio de solución, codo y silueta."""
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    guardar_espacio(analisis, carpeta / f"espacio_solucion_k{k}.png", k)


def generar_gif(analisis, carpeta, k):
    """Guarda un GIF simple de la asignación final."""
    carpeta = Path(carpeta)
    puntos = PCA(2).fit_transform(analisis["datos"])
    modelo = analisis["modelos"][k]
    colores = plt.cm.nipy_spectral(np.linspace(0, 1, k, endpoint=False))
    fig, eje = plt.subplots(figsize=(8, 6))
    eje.scatter(puntos[:, 0], puntos[:, 1], c=colores[modelo.labels_])
    eje.set(title=f"K-Means final (k={k})", xlabel="PCA 1", ylabel="PCA 2")
    fig.savefig(carpeta / f"kmeans_k{k}.png", dpi=150)
    plt.close(fig)
