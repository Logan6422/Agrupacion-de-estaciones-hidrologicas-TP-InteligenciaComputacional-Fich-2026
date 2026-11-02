"""Pipeline de preparación, reducción de ruido y clasificación de ríos.
El módulo trabaja únicamente con los doce caudales mensuales. Primero los
normaliza por Qmod, luego reduce valores extremos con log1p, estandariza cada
mes, suaviza los perfiles con un SOM y finalmente evalúa K-Means.
"""

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


# Orden de las doce columnas mensuales del año hidrológico.
MESES = [
    "Sep", "Oct", "Nov", "Dic", "Ene", "Feb",
    "Mar", "Abr", "May", "Jun", "Jul", "Ago",
]
# Cantidad de clusters definida como referencia por el análisis manual.
K_TEORICO = 23
# Máximo k utilizado en las curvas de codo y silueta.
K_MAXIMO = 40
# El SOM tiene 100 neuronas organizadas en diez filas y diez columnas.
TAMANO_SOM = (10, 10)
# Semilla para que la inicialización aleatoria sea reproducible.
SEMILLA = 42


def cargar_caudales():
    """Carga los CSV y devuelve un perfil normalizado por cada código.
    Si un código aparece en varios archivos, se calcula la media de sus
    caudales mensuales y de su Qmod antes de normalizar.
    """
    carpeta = Path(__file__).parent
    # Se buscan los archivos relativos a este módulo para que el programa
    # funcione aunque se ejecute desde otro directorio. / "csv"
    archivos = sorted(carpeta.glob("estaciones_final*.csv"))
    estaciones = pd.concat(
        (pd.read_csv(archivo, encoding="latin1") for archivo in archivos),
        ignore_index=True,
    )
    # Convierte texto numérico a números; los valores inválidos quedan como NaN.
    estaciones["Codigo"] = pd.to_numeric(estaciones["Codigo"], errors="coerce")
    estaciones["Qmod"] = pd.to_numeric(estaciones["Qmod"], errors="coerce")
    # Sin código no se puede identificar el río y sin Qmod no se puede
    # normalizarlo, por eso esas filas se eliminan.
    estaciones = estaciones.dropna(subset=["Codigo", "Qmod"]).copy()
    estaciones["Codigo"] = estaciones["Codigo"].astype(int)

    # Une filas con el mismo código y obtiene un único perfil mensual.
    caudales = estaciones.groupby("Codigo", sort=True)[MESES].mean()
    qmod = estaciones.groupby("Codigo", sort=True)["Qmod"].mean()

    # Divide cada mes por el caudal medio Qmod. Así se compara la forma
    # relativa del hidrograma y no solamente el tamaño del río.
    return caudales.div(qmod, axis=0)


def preparar_datos(caudales):
    """Aplica logaritmo y estandarización a los perfiles mensuales."""
    # log1p(x) calcula log(1+x), reduciendo el peso de caudales muy grandes.
    # clip evita valores negativos inesperados dentro del logaritmo.
    logaritmos = np.log1p(caudales.clip(lower=0))
    # Estandariza cada mes: media cero y desviación estándar uno.
    return StandardScaler().fit_transform(logaritmos)


def entrenar_som(datos, filas, columnas, epocas=200):
    """Entrena un mapa autoorganizado y devuelve sus prototipos.
    La neurona más cercana a cada vector se llama ganadora. La ganadora y sus
    vecinas se actualizan juntas, por lo que perfiles similares quedan cerca
    dentro de la grilla.
    """
    rng = np.random.default_rng(SEMILLA)
    # Cada prototipo tiene la misma cantidad de atributos que un río.
    prototipos = rng.normal(
        datos.mean(axis=0),
        datos.std(axis=0) + 1e-12,
        size=(filas, columnas, datos.shape[1]),
    )
    # Coordenadas fila/columna de cada neurona para calcular vecindades.
    posiciones = np.indices((filas, columnas)).transpose(1, 2, 0)
    radio_inicial = max(filas, columnas) / 2

    for epoca in range(epocas):
        # La tasa de aprendizaje y el radio disminuyen gradualmente:
        # primero se ordena el mapa globalmente y luego se ajusta localmente.
        progreso = epoca / max(epocas - 1, 1)
        tasa = 0.5 * (0.05 / 0.5) ** progreso
        radio = radio_inicial * (1 / radio_inicial) ** progreso
        for indice in rng.permutation(len(datos)):
            vector = datos[indice]
            # Encuentra la neurona cuyo prototipo está más cerca del vector.
            distancias = np.linalg.norm(prototipos - vector, axis=2)
            ganador = np.unravel_index(np.argmin(distancias), distancias.shape)
            # Calcula la distancia de cada neurona a la neurona ganadora.
            distancia_mapa = np.sum((posiciones - ganador) ** 2, axis=2)
            # Las neuronas cercanas reciben una actualización más grande.
            vecindad = np.exp(-distancia_mapa / (2 * radio**2))
            prototipos += tasa * vecindad[:, :, None] * (vector - prototipos)
    return prototipos


def suavizar_con_som(datos):
    """Reemplaza cada río por el prototipo de su neurona ganadora."""
    prototipos = entrenar_som(datos, *TAMANO_SOM)
    # Distancia de cada río contra todas las neuronas del mapa.
    distancias = np.linalg.norm(
        datos[:, None, None, :] - prototipos[None, :, :, :],
        axis=3,
    )
    # Selecciona la neurona más cercana para cada río.
    ganadores = np.argmin(distancias.reshape(len(datos), -1), axis=1)
    # Aplana la grilla y conserva una fila por río: no se elimina ningún río.
    return prototipos.reshape(-1, datos.shape[1])[ganadores], prototipos


def calcular_codo(inercias, valores_k):
    """Estima el codo como el punto más alejado de la recta de extremos."""
    # Cada punto es (k, inercia). La mayor distancia indica una caída
    # especialmente pronunciada antes de que la curva se estabilice.
    puntos = np.column_stack((valores_k, inercias))
    recta = puntos[-1] - puntos[0]
    numerador = np.abs(
        recta[0] * (puntos[:, 1] - puntos[0, 1])
        - recta[1] * (puntos[:, 0] - puntos[0, 0])
    )
    return int(puntos[np.argmax(numerador / np.linalg.norm(recta)), 0])


def evaluar_clusters(datos, valores_k):
    """Entrena K-Means y calcula sus métricas para cada k."""
    filas = []
    modelos = {}
    for k in valores_k:
        # n_init prueba varias posiciones iniciales y conserva la mejor.
        modelo = KMeans(n_clusters=k, n_init=50, random_state=SEMILLA)
        etiquetas = modelo.fit_predict(datos)
        modelos[k] = modelo
        filas.append(
            {
                "k": k,
                # Suma de distancias cuadráticas de cada punto a su centro.
                "inercia": modelo.inertia_,
                # Mayor es mejor: mide cohesión interna y separación externa.
                "silueta": silhouette_score(datos, etiquetas),
                # Mayor es mejor.
                "calinski_harabasz": calinski_harabasz_score(datos, etiquetas),
                # Menor es mejor.
                "davies_bouldin": davies_bouldin_score(datos, etiquetas),
            }
        )
    return pd.DataFrame(filas), modelos


def analizar():
    """Ejecuta el pipeline completo y devuelve todos los resultados."""
    caudales = cargar_caudales()
    datos = preparar_datos(caudales)
    # El SOM reemplaza cada perfil por un prototipo cercano para reducir ruido.
    datos_suavizados, prototipos_som = suavizar_con_som(datos)
    # Se empieza en 2 porque las métricas de separación no tienen sentido con
    # un único cluster.
    valores_k = np.arange(2, min(K_MAXIMO, len(datos) - 1) + 1)
    metricas, modelos = evaluar_clusters(datos_suavizados, valores_k)
    # El óptimo matemático se define como la silueta máxima.
    k_optimo = int(metricas.loc[metricas["silueta"].idxmax(), "k"])
    k_codo = calcular_codo(metricas["inercia"].to_numpy(), valores_k)
    # Conserva la comparación entre el óptimo y los 23 grupos teóricos.
    comparacion = metricas[metricas["k"].isin([k_optimo, K_TEORICO])].sort_values("k")
    asignaciones = pd.DataFrame({"Codigo": caudales.index})
    for k, nombre in [
        (k_optimo, "Cluster_optimo"),
        (K_TEORICO, "Cluster_teorico"),
    ]:
        # Guarda la etiqueta de cluster correspondiente a cada código.
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
    """Dibuja el método del codo y la curva de silueta."""
    metricas = resultados["metricas"].sort_values("k")
    k_optimo = resultados["k_optimo"]
    k_teorico = resultados["k_teorico"]
    fig, ejes = plt.subplots(1, 2, figsize=(14, 5))

    # La inercia baja al aumentar k; el codo marca una mejora decreciente.
    ejes[0].plot(metricas["k"], metricas["inercia"], marker="o")
    ejes[0].axvline(k_optimo, color="red", linestyle="--", label=f"Óptimo: {k_optimo}")
    ejes[0].axvline(k_teorico, color="green", linestyle=":", label=f"Teórico: {k_teorico}")
    ejes[0].set(title="Método del codo", xlabel="k", ylabel="Inercia")

    # La silueta compara simultáneamente cohesión y separación de clusters.
    ejes[1].plot(metricas["k"], metricas["silueta"], marker="o")
    ejes[1].axvline(k_optimo, color="red", linestyle="--", label=f"Óptimo: {k_optimo}")
    ejes[1].axvline(k_teorico, color="green", linestyle=":", label=f"Teórico: {k_teorico}")
    ejes[1].set(title="Coeficiente de silueta", xlabel="k", ylabel="Silueta")

    for eje in ejes:
        eje.grid(alpha=0.3)
        eje.legend()
    plt.tight_layout()
    plt.show()
