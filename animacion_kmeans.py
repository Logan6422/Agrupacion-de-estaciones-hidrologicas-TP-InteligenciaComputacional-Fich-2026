from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA

from analisis_rios import analizar


analisis = analizar()
datos = analisis["datos"]
k_optimo = analisis["k_optimo"]
k_teorico = analisis["k_teorico"]
salida = Path(__file__).parent / "kmeans_comparacion.gif"
SEMILLA = 42
EPOCAS = 30


def pasos_kmeans(datos, k, epocas):
    rng = np.random.default_rng(SEMILLA)
    centros = datos[rng.choice(len(datos), size=k, replace=False)].copy()
    pasos = []

    for _ in range(epocas):
        distancias = np.linalg.norm(datos[:, None, :] - centros[None, :, :], axis=2)
        etiquetas = np.argmin(distancias, axis=1)
        pasos.append((etiquetas.copy(), centros.copy()))

        nuevos_centros = centros.copy()
        for cluster in range(k):
            puntos = datos[etiquetas == cluster]
            if len(puntos):
                nuevos_centros[cluster] = puntos.mean(axis=0)
        if np.allclose(centros, nuevos_centros):
            pasos.append((etiquetas.copy(), nuevos_centros.copy()))
            break
        centros = nuevos_centros
    return pasos


def completar_pasos(pasos, cantidad):
    return pasos + [pasos[-1]] * (cantidad - len(pasos))


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

puntos = [eje.scatter([], [], s=45) for eje in ejes]
centros = [
    eje.scatter([], [], marker="X", s=180, edgecolors="black")
    for eje in ejes
]


def actualizar(frame):
    for eje, puntos_eje, centros_eje, pasos, k, nombre in zip(
        ejes,
        puntos,
        centros,
        [pasos_optimo, pasos_teorico],
        [k_optimo, k_teorico],
        ["Óptimo matemático", "Referencia teórica"],
    ):
        etiquetas, centros_originales = pasos[frame]
        colores = plt.cm.nipy_spectral(
            np.linspace(0, 1, max(k, 2), endpoint=False)
        )
        puntos_eje.set_offsets(datos_2d)
        puntos_eje.set_color(colores[etiquetas])
        centros_eje.set_offsets(pca.transform(centros_originales))
        centros_eje.set_color(colores[:k])
        eje.set_title(f"{nombre}: k={k}\nIteración {frame + 1}/{cantidad_frames}")
    return [*puntos, *centros]


animacion = FuncAnimation(
    fig,
    actualizar,
    frames=cantidad_frames,
    interval=500,
    blit=True,
    repeat=False,
)
animacion.save(salida, writer=PillowWriter(fps=2))
plt.show()
print(f"Animación guardada en: {salida}")
