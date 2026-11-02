"""Genera un GIF con la convergencia de dos ejecuciones de K-Means.

El panel izquierdo usa el k óptimo según la silueta y el derecho usa la
referencia teórica de 23 clusters. PCA se utiliza únicamente para visualizar
los datos en dos dimensiones; K-Means sigue trabajando con todos los atributos.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA

from analisis_rios import analizar


# Reutiliza el mismo pipeline, incluyendo normalización, logaritmo y SOM.
analisis = analizar()
datos = analisis["datos"]
k_optimo = analisis["k_optimo"]
k_teorico = analisis["k_teorico"]
salida = Path(__file__).parent / "kmeans_comparacion.gif"
SEMILLA = 42  # Hace reproducible la selección de centros iniciales.
EPOCAS = 30  # Límite máximo de iteraciones que se muestran.


def pasos_kmeans(datos, k, epocas):
    """Devuelve los estados de K-Means para poder animar su convergencia."""
    rng = np.random.default_rng(SEMILLA)
    # Se eligen k observaciones como centros iniciales.
    centros = datos[rng.choice(len(datos), size=k, replace=False)].copy()
    pasos = []

    for _ in range(epocas):
        # Cada observación se asigna al centroide más cercano.
        distancias = np.linalg.norm(datos[:, None, :] - centros[None, :, :], axis=2)
        etiquetas = np.argmin(distancias, axis=1)
        pasos.append((etiquetas.copy(), centros.copy()))

        # Cada centroide se reemplaza por la media de sus observaciones.
        nuevos_centros = centros.copy()
        for cluster in range(k):
            puntos = datos[etiquetas == cluster]
            if len(puntos):
                nuevos_centros[cluster] = puntos.mean(axis=0)
        # Si los centros dejaron de cambiar, el algoritmo convergió.
        if np.allclose(centros, nuevos_centros):
            pasos.append((etiquetas.copy(), nuevos_centros.copy()))
            break
        centros = nuevos_centros
    return pasos


def completar_pasos(pasos, cantidad):
    """Repite el último estado para igualar la duración de ambos paneles."""
    return pasos + [pasos[-1]] * (cantidad - len(pasos))


# PCA reduce la dimensión solo para graficar; no modifica los datos de K-Means.
pca = PCA(n_components=2)
datos_2d = pca.fit_transform(datos)
pasos_optimo = pasos_kmeans(datos, k_optimo, EPOCAS)
pasos_teorico = pasos_kmeans(datos, k_teorico, EPOCAS)
cantidad_frames = max(len(pasos_optimo), len(pasos_teorico))
pasos_optimo = completar_pasos(pasos_optimo, cantidad_frames)
pasos_teorico = completar_pasos(pasos_teorico, cantidad_frames)

fig, ejes = plt.subplots(1, 2, figsize=(14, 6))
limites = np.ptp(datos_2d, axis=0)
margen = np.maximum(limites * 0.1, 1)
for eje in ejes:
    eje.set_xlim(datos_2d[:, 0].min() - margen[0], datos_2d[:, 0].max() + margen[0])
    eje.set_ylim(datos_2d[:, 1].min() - margen[1], datos_2d[:, 1].max() + margen[1])
    eje.set_xlabel("Componente principal 1")
    eje.set_ylabel("Componente principal 2")

# Colección gráfica de los ríos para cada panel.
puntos = [eje.scatter([], [], s=45) for eje in ejes]
centros = [
    eje.scatter([], [], marker="X", s=180, edgecolors="black")
    for eje in ejes
]


def actualizar(frame):
    """Actualiza ríos, colores y centroides en un frame del GIF."""
    for eje, puntos_eje, centros_eje, pasos, k, nombre in zip(
        ejes,
        puntos,
        centros,
        [pasos_optimo, pasos_teorico],
        [k_optimo, k_teorico],
        ["Óptimo matemático", "Referencia teórica"],
    ):
        etiquetas, centros_originales = pasos[frame]
        # Se genera un color diferente por cluster.
        colores = plt.cm.nipy_spectral(
            np.linspace(0, 1, max(k, 2), endpoint=False)
        )
        puntos_eje.set_offsets(datos_2d)
        puntos_eje.set_color(colores[etiquetas])
        centros_eje.set_offsets(pca.transform(centros_originales))
        centros_eje.set_color(colores[:k])
        eje.set_title(f"{nombre}: k={k}\nIteración {frame + 1}/{cantidad_frames}")
    return [*puntos, *centros]


# FuncAnimation llama a actualizar una vez por frame.
animacion = FuncAnimation(
    fig,
    actualizar,
    frames=cantidad_frames,
    interval=500,
    blit=True,
    repeat=False,
)
# PillowWriter guarda todos los frames como archivo GIF.
animacion.save(salida, writer=PillowWriter(fps=2))
plt.show()
print(f"Animación guardada en: {salida}")
