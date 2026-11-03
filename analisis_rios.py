"Preparacion, reduccion de ruido y clasificacion de ruidos"
#Se utilizan vectores de 12 valores de caudales mensuales normalizados por Qmod
#Qmod = caudal medio anual del río
#Reduccion de valores extremos con log1p
#Estandarizacion de cada mes con StandardScaler (media 0 y desviación estándar 1)
#Suavizado de perfiles con un SOM para reducir ruido (reemplazo de cada perfil por el prototipo de su neurona ganadora)
#Evaluacion con kmeans


from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from minisom import MiniSom
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
# Máximo k utilizado en las curvas de codo y silueta.
K_MAXIMO = 40
# Semilla para que la inicialización aleatoria sea reproducible.
SEMILLA = 42


def cargar_caudales(carpeta_datos=None, patron="estaciones_final*.csv"):
    """Carga los CSV y devuelve los 12 valores normalizados para cada código.
    Si un código aparece varias veces, se calcula la media de sus caudales
    mensuales y de su Qmod antes de normalizar.
    """
    carpeta = Path(carpeta_datos) if carpeta_datos else Path(__file__).parent / "csv"
    # Se buscan los archivos relativos a este módulo para que el programa
    # funcione aunque se ejecute desde otro directorio.
    archivos = sorted(carpeta.glob(patron))
    estaciones = pd.concat(
        (pd.read_csv(archivo, encoding="latin1") for archivo in archivos),
        ignore_index=True,
    )
    #Concatena los archivos y elimina duplicados, dejando la primera ocurrencia de cada código.
    # Convierte texto numérico a números; los valores inválidos quedan como NaN.
    estaciones["Codigo"] = pd.to_numeric(estaciones["Codigo"], errors="coerce")
    estaciones["Qmod"] = pd.to_numeric(estaciones["Qmod"], errors="coerce")
    estaciones = estaciones.dropna(subset=["Codigo", "Qmod"]).copy()
    #Si la estacion no tiene codigo o Qmod, no se puede normalizar y se elimina del análisis.
    estaciones["Codigo"] = estaciones["Codigo"].astype(int)
    # Elimina estaciones con caudales negativos, que no tienen sentido físico.

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
    """Entrena un SOM de MiniSom y devuelve sus prototipos."""
    # sigma controla el radio inicial de vecindad y learning_rate la magnitud
    # de la actualización. MiniSom reduce ambos valores durante el entrenamiento.
    som = MiniSom(
        x=filas,
        y=columnas,
        input_len=datos.shape[1],
        sigma=max(filas, columnas) / 2,
        learning_rate=0.5,
        neighborhood_function="gaussian",
        random_seed=SEMILLA,
    )
    som.random_weights_init(datos)
    som.train_random(datos, num_iteration=epocas, verbose=False)
    return som.get_weights()


def calcular_tamano_som(cantidad_datos):
    """Calcula una grilla proporcional a la cantidad de ríos analizados."""
    # La configuración original de 10x10 se toma como referencia para 87 ríos.
    # La grilla crece con la raíz cuadrada del aumento de observaciones para
    # mantener una cantidad razonable de ríos por neurona.
    lado = round(10 * np.sqrt(cantidad_datos / 87))
    lado = max(2, lado)
    return lado, lado


def suavizar_con_som(datos):
    """Reemplaza cada río por el prototipo de su neurona ganadora."""
    tamano_som = calcular_tamano_som(len(datos))
    prototipos = entrenar_som(datos, *tamano_som)
    # Distancia de cada río contra todas las neuronas del mapa.
    distancias = np.linalg.norm(
        datos[:, None, None, :] - prototipos[None, :, :, :],
        axis=3,
    )
    # Selecciona la neurona más cercana para cada río.
    ganadores = np.argmin(distancias.reshape(len(datos), -1), axis=1)
    # Aplana la grilla y conserva una fila por río: no se elimina ningún río.
    return prototipos.reshape(-1, datos.shape[1])[ganadores], prototipos, tamano_som


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


def analizar(k_seleccionado=None, carpeta_datos=None, patron="estaciones_final*.csv"):
    """Ejecuta el pipeline completo y permite seleccionar el número de clusters."""
    caudales = cargar_caudales(carpeta_datos, patron)
    datos = preparar_datos(caudales)
    # El SOM reemplaza cada perfil por un prototipo cercano para reducir ruido.
    datos_suavizados, prototipos_som, tamano_som = suavizar_con_som(datos)
    # Se empieza en 2 porque las métricas de separación no tienen sentido con
    # un único cluster.
    valores_k = np.arange(2, min(K_MAXIMO, len(datos) - 1) + 1)
    metricas, modelos = evaluar_clusters(datos_suavizados, valores_k)
    # El óptimo matemático se define como la silueta máxima.
    k_optimo = int(metricas.loc[metricas["silueta"].idxmax(), "k"])
    if k_seleccionado is not None:
        if k_seleccionado not in valores_k:
            raise ValueError(
                f"k_seleccionado debe estar entre 2 y {valores_k[-1]}."
            )
        k_optimo = int(k_seleccionado)
    k_codo = calcular_codo(metricas["inercia"].to_numpy(), valores_k)
    asignaciones = pd.DataFrame({"Codigo": caudales.index})
    # Guarda la etiqueta del cluster óptimo para cada código.
    asignaciones["Cluster_optimo"] = modelos[k_optimo].labels_
    return {
        "codigos": caudales.index.to_numpy(),
        "caudales": caudales,
        "datos": datos_suavizados,
        "prototipos_som": prototipos_som,
        "tamano_som": tamano_som,
        "metricas": metricas.sort_values("silueta", ascending=False),
        "modelos": modelos,
        "k_optimo": k_optimo,
        "k_codo": k_codo,
        "asignaciones": asignaciones,
    }


def graficar_resultados(resultados, ruta=None):
    """Dibuja y opcionalmente guarda el método del codo y la silueta."""
    metricas = resultados["metricas"].sort_values("k")
    k_optimo = resultados["k_optimo"]
    fig, ejes = plt.subplots(1, 2, figsize=(14, 5))

    # La inercia baja al aumentar k; el codo marca una mejora decreciente.
    ejes[0].plot(metricas["k"], metricas["inercia"], marker="o")
    ejes[0].axvline(k_optimo, color="red", linestyle="--", label=f"Óptimo: {k_optimo}")
    ejes[0].set(title="Método del codo", xlabel="k", ylabel="Inercia")

    # La silueta compara simultáneamente cohesión y separación de clusters.
    ejes[1].plot(metricas["k"], metricas["silueta"], marker="o")
    ejes[1].axvline(k_optimo, color="red", linestyle="--", label=f"Óptimo: {k_optimo}")
    ejes[1].set(title="Coeficiente de silueta", xlabel="k", ylabel="Silueta")

    for eje in ejes:
        eje.grid(alpha=0.3)
        eje.legend()
    plt.tight_layout()
    if ruta is not None:
        fig.savefig(ruta, dpi=150)
    plt.show()


def graficar_espacio_solucion(resultados, ruta):
    """Guarda una proyección PCA con los ríos y centroides del k óptimo."""
    from sklearn.decomposition import PCA

    modelo = resultados["modelos"][resultados["k_optimo"]]
    datos = resultados["datos"]
    pca = PCA(n_components=2)
    datos_2d = pca.fit_transform(datos)
    centroides_2d = pca.transform(modelo.cluster_centers_)
    etiquetas = modelo.labels_
    colores = plt.cm.nipy_spectral(
        np.linspace(0, 1, resultados["k_optimo"], endpoint=False)
    )
    # El SOM puede asignar varios ríos al mismo prototipo. Se agrega un
    # desplazamiento visual pequeño para que los 87 códigos sean visibles.
    datos_2d = desplazar_puntos_superpuestos(datos_2d)

    figura, eje = plt.subplots(figsize=(10, 8))
    eje.scatter(
        datos_2d[:, 0],
        datos_2d[:, 1],
        c=colores[etiquetas],
        s=55,
        edgecolors="black",
        linewidths=0.3,
    )
    for codigo, punto in zip(resultados["codigos"], datos_2d):
        eje.annotate(str(codigo), punto, xytext=(3, 3), textcoords="offset points", fontsize=6)
    eje.scatter(
        centroides_2d[:, 0],
        centroides_2d[:, 1],
        c=colores,
        marker="X",
        s=180,
        edgecolors="black",
        linewidths=0.8,
    )
    eje.set_title(f"Espacio de solución K-Means (k={resultados['k_optimo']})")
    eje.set_xlabel("Componente principal 1")
    eje.set_ylabel("Componente principal 2")
    eje.grid(alpha=0.3)
    figura.tight_layout()
    figura.savefig(ruta, dpi=150)
    plt.close(figura)


def coordenadas_a_decimal(coordenada):
    """Convierte coordenadas DMS con orientación S/O a grados decimales."""
    numericas = pd.to_numeric(coordenada, errors="coerce")
    if numericas.notna().all():
        return numericas
    partes = coordenada.astype(str).str.extract(
        r"(?P<grados>\d+).*?(?P<minutos>\d+).*?"
        r"(?P<segundos>\d+(?:\.\d+)?).*?(?P<sentido>[SO])",
        expand=True,
    )
    valor = (
        pd.to_numeric(partes["grados"], errors="coerce")
        + pd.to_numeric(partes["minutos"], errors="coerce") / 60
        + pd.to_numeric(partes["segundos"], errors="coerce") / 3600
    )
    return valor.where(~partes["sentido"].isin(["S", "O"]), -valor)


def graficar_mapa_clusters(resultados, coordenadas, ruta):
    """Guarda la distribución geográfica de los ríos coloreada por cluster."""
    import geopandas as gpd

    modelo = resultados["modelos"][resultados["k_optimo"]]
    url_provincias = (
        "https://naturalearth.s3.amazonaws.com/10m_cultural/"
        "ne_10m_admin_1_states_provinces.zip"
    )
    provincias = gpd.read_file(url_provincias)
    argentina = provincias[provincias["admin"] == "Argentina"]

    mapa = coordenadas.copy()
    mapa["Codigo"] = pd.to_numeric(mapa["Codigo"], errors="coerce")
    mapa["Latitud"] = coordenadas_a_decimal(mapa["Latitud"])
    mapa["Longitud"] = coordenadas_a_decimal(mapa["Longitud"])
    mapa = mapa.dropna(subset=["Codigo", "Latitud", "Longitud"]).copy()
    mapa["Codigo"] = mapa["Codigo"].astype(int)
    mapa = mapa.merge(resultados["asignaciones"], on="Codigo", how="inner")
    cantidad_clusters = resultados["k_optimo"]
    colores = np.array([
        plt.cm.tab20(indice % 20)
        if indice < 20
        else plt.cm.tab20b(indice - 20)
        for indice in range(cantidad_clusters)
    ])

    figura, eje = plt.subplots(figsize=(10, 12))
    argentina.plot(
        ax=eje,
        facecolor="#f2efe6",
        edgecolor="gray",
        linewidth=0.6,
        zorder=0,
    )
    eje.scatter(
        mapa["Longitud"],
        mapa["Latitud"],
        c=colores[mapa["Cluster_optimo"]],
        s=65,
        edgecolors="black",
        linewidths=0.4,
        zorder=3,
    )
    eje.set_aspect("equal", adjustable="box")
    eje.set_xlim(-73.5, -53.5)
    eje.set_ylim(-55.5, -21.5)
    eje.set_title(f"Distribución geográfica de clusters (k={resultados['k_optimo']})")
    eje.set_xlabel("Longitud")
    eje.set_ylabel("Latitud")
    eje.grid(alpha=0.3)
    leyenda = [
        Patch(facecolor=colores[cluster], edgecolor="black", label=f"Cluster {cluster}")
        for cluster in range(cantidad_clusters)
    ]
    eje.legend(
        handles=leyenda,
        title="Clusters",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=7,
        title_fontsize=8,
        ncol=2,
    )
    figura.tight_layout()
    figura.savefig(ruta, dpi=150)
    plt.close(figura)


def graficar_mapa_geopandas(resultados, coordenadas, ruta, ruta_referencia):
    """Guarda el mapa y una tabla que relaciona referencias con nombres."""
    import geopandas as gpd

    url_provincias = (
        "https://naturalearth.s3.amazonaws.com/10m_cultural/"
        "ne_10m_admin_1_states_provinces.zip"
    )
    provincias = gpd.read_file(url_provincias)
    argentina = provincias[provincias["admin"] == "Argentina"]

    mapa = coordenadas.copy()
    mapa["Codigo"] = pd.to_numeric(mapa["Codigo"], errors="coerce")
    mapa["Latitud"] = coordenadas_a_decimal(mapa["Latitud"])
    mapa["Longitud"] = coordenadas_a_decimal(mapa["Longitud"])
    mapa = mapa.dropna(subset=["Codigo", "Latitud", "Longitud"]).copy()
    mapa["Codigo"] = mapa["Codigo"].astype(int)
    mapa = mapa.merge(resultados["asignaciones"], on="Codigo", how="inner")
    mapa = mapa.sort_values("Codigo").reset_index(drop=True)
    mapa["Referencia"] = [
        f"R{indice:03d}" for indice in range(1, len(mapa) + 1)
    ]
    pd.DataFrame(
        {
            "Referencia": mapa["Referencia"],
            "Codigo": mapa["Codigo"],
            "Rio": mapa["Rio"].fillna("Río"),
            "Cluster_optimo": mapa["Cluster_optimo"],
        }
    ).to_csv(ruta_referencia, index=False)
    coordenadas_mapa = desplazar_puntos_superpuestos(
        mapa[["Longitud", "Latitud"]].to_numpy(),
        escala_relativa=0.02,
    )
    puntos = gpd.GeoDataFrame(
        mapa,
        geometry=gpd.points_from_xy(
            coordenadas_mapa[:, 0],
            coordenadas_mapa[:, 1],
        ),
        crs="EPSG:4326",
    )
    cantidad_clusters = resultados["k_optimo"]
    colores = np.array([
        plt.cm.tab20(indice % 20)
        if indice < 20
        else plt.cm.tab20b(indice - 20)
        for indice in range(cantidad_clusters)
    ])

    figura, eje = plt.subplots(figsize=(10, 14))
    argentina.plot(
        ax=eje,
        facecolor="#f2efe6",
        edgecolor="gray",
        linewidth=0.6,
        zorder=0,
    )
    puntos.plot(
        ax=eje,
        color=[colores[cluster] for cluster in puntos["Cluster_optimo"]],
        markersize=60,
        edgecolor="black",
        linewidth=0.4,
        zorder=3,
    )
    for _, rio in puntos.iterrows():
        eje.annotate(
            rio["Referencia"],
            (rio.geometry.x, rio.geometry.y),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6,
            zorder=4,
        )
    eje.set_xlim(-73.5, -53.5)
    eje.set_ylim(-55.5, -21.5)
    eje.set_aspect("equal")
    eje.set_xlabel("Longitud")
    eje.set_ylabel("Latitud")
    eje.set_title(f"Distribución geográfica de clusters (k={resultados['k_optimo']})")
    leyenda = [
        Patch(facecolor=colores[cluster], edgecolor="black", label=f"Cluster {cluster}")
        for cluster in range(cantidad_clusters)
    ]
    eje.legend(
        handles=leyenda,
        title="Clusters",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=7,
        title_fontsize=8,
        ncol=2,
    )
    figura.tight_layout()
    figura.savefig(ruta, dpi=150)
    plt.close(figura)


def generar_histogramas(resultados, carpeta, nombres_por_codigo=None):
    """Genera perfiles medios por cluster y perfiles individuales por río."""
    carpeta_por_cluster = carpeta / "por_cluster"
    carpeta_por_rio = carpeta / "por rio"
    carpeta_por_cluster.mkdir(parents=True, exist_ok=True)
    carpeta_por_rio.mkdir(parents=True, exist_ok=True)

    modelo = resultados["modelos"][resultados["k_optimo"]]
    asignaciones = pd.DataFrame(
        {
            "Codigo": resultados["codigos"],
            "Cluster": modelo.labels_,
        }
    )
    caudales = resultados["caudales"].copy()
    caudales["Cluster"] = modelo.labels_
    medias = caudales.groupby("Cluster")[MESES].mean()

    # Se agrupan como máximo nueve clusters por imagen para que sean legibles.
    clusters = sorted(medias.index)
    for inicio in range(0, len(clusters), 9):
        grupo_clusters = clusters[inicio:inicio + 9]
        figura, ejes = plt.subplots(3, 3, figsize=(15, 12), squeeze=False)
        for eje, cluster in zip(ejes.flat, grupo_clusters):
            eje.bar(MESES, medias.loc[cluster])
            eje.set_title(f"Cluster {cluster} - perfil medio")
            eje.set_xlabel("Mes")
            eje.set_ylabel("Caudal normalizado")
            eje.grid(alpha=0.3)
        for eje in ejes.flat[len(grupo_clusters):]:
            eje.set_visible(False)
        figura.tight_layout()
        figura.savefig(
            carpeta_por_cluster / f"clusters_{inicio + 1:02d}_{inicio + len(grupo_clusters):02d}.png",
            dpi=150,
        )
        plt.close(figura)

    # Cada archivo contiene todos los perfiles mensuales de los ríos del cluster.
    for cluster in clusters:
        codigos = asignaciones.loc[
            asignaciones["Cluster"] == cluster,
            "Codigo",
        ].tolist()
        cantidad_columnas = 3
        cantidad_filas = max(1, int(np.ceil(len(codigos) / cantidad_columnas)))
        figura, ejes = plt.subplots(
            cantidad_filas,
            cantidad_columnas,
            figsize=(15, 4 * cantidad_filas),
            squeeze=False,
        )
        for eje, codigo in zip(ejes.flat, codigos):
            eje.bar(MESES, caudales.loc[codigo, MESES])
            nombre = (
                nombres_por_codigo.get(codigo, f"Río {codigo}")
                if nombres_por_codigo
                else f"Río {codigo}"
            )
            eje.set_title(f"{nombre} ({codigo}) - cluster {cluster}")
            eje.set_xlabel("Mes")
            eje.set_ylabel("Caudal normalizado")
            eje.grid(alpha=0.3)
        for eje in ejes.flat[len(codigos):]:
            eje.set_visible(False)
        figura.tight_layout()
        figura.savefig(carpeta_por_rio / f"cluster_{cluster:02d}.png", dpi=150)
        plt.close(figura)


def desplazar_puntos_superpuestos(puntos, escala_relativa=0.008):
    """Separa visualmente puntos idénticos sin modificar sus datos."""
    resultado = puntos.copy()
    rng = np.random.default_rng(SEMILLA)
    for punto in np.unique(puntos, axis=0):
        indices = np.flatnonzero(np.all(puntos == punto, axis=1))
        if len(indices) > 1:
            escala = max(np.ptp(puntos, axis=0).max() * escala_relativa, 0.01)
            resultado[indices] += rng.normal(
                0,
                escala,
                size=(len(indices), puntos.shape[1]),
            )
    return resultado
