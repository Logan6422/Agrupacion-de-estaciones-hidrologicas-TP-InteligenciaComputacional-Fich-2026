"""Funciones compartidas para los anÃ¡lisis hidrolÃ³gicos."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

MESES = ["Sep", "Oct", "Nov", "Dic", "Ene", "Feb",
         "Mar", "Abr", "May", "Jun", "Jul", "Ago"]
SEMILLA = 67


def cargar_caudales(carpeta, patron):
    """Promedia codigos repetidos y devuelve caudales mensuales y Qmod."""
    archivos = sorted(Path(carpeta).glob(patron))
    datos = pd.concat(
        [pd.read_csv(archivo, encoding="latin1") for archivo in archivos],
        ignore_index=True,
    )
    datos["Codigo"] = pd.to_numeric(datos["Codigo"], errors="coerce")
    datos["Qmod"] = pd.to_numeric(datos["Qmod"], errors="coerce")
    for mes in MESES:
        datos[mes] = pd.to_numeric(datos[mes], errors="coerce")
    datos = datos.dropna(subset=["Codigo", "Qmod", *MESES]).copy()
    datos["Codigo"] = datos["Codigo"].astype(int)
    caudales = datos.groupby("Codigo")[MESES].mean().sort_index()
    qmod = datos.groupby("Codigo")["Qmod"].mean().reindex(caudales.index)
    return caudales, qmod


def normalizar(caudales, qmod, usar_qmod=True, usar_log=True, usar_scaler=True):
    """Aplica opcionalmente las tres normalizaciones del pipeline."""
    datos = caudales.copy()
    if usar_qmod:
        datos = datos.div(qmod, axis=0)
    if usar_log:
        datos = np.log1p(datos.clip(lower=0))
    if usar_scaler:
        datos = pd.DataFrame(
            StandardScaler().fit_transform(datos),
            index=datos.index,
            columns=datos.columns,
        )
    return datos


def evaluar_clusters(datos, k_maximo):
    """Calcula metricas y conserva un K-Means entrenado para cada k."""
    valores_k = range(2, min(k_maximo, len(datos) - 1) + 1)
    metricas, modelos = [], {}
    for k in valores_k:
        modelo = KMeans(n_clusters=k, n_init=50, random_state=SEMILLA)
        etiquetas = modelo.fit_predict(datos)
        modelos[k] = modelo
        metricas.append({
            "k": k,
            "inercia": modelo.inertia_,
            "silueta": silhouette_score(datos, etiquetas),
            "calinski_harabasz": calinski_harabasz_score(datos, etiquetas),
            "davies_bouldin": davies_bouldin_score(datos, etiquetas),
        })
    metricas = pd.DataFrame(metricas)
    optimo = int(metricas.loc[metricas["silueta"].idxmax(), "k"])
    return metricas.sort_values("silueta", ascending=False), modelos, optimo


def analizar(carpeta, patron, k_maximo, usar_qmod=True, usar_log=True, usar_scaler=True):
    """Carga, normaliza y evalua los perfiles mensuales."""
    caudales, qmod = cargar_caudales(carpeta, patron)
    datos = normalizar(caudales, qmod, usar_qmod, usar_log, usar_scaler)
    metricas, modelos, optimo = evaluar_clusters(datos, k_maximo)
    return {
        "caudales": caudales,
        "datos": datos,
        "codigos": caudales.index.to_numpy(),
        "metricas": metricas,
        "modelos": modelos,
        "optimo": optimo,
    }


def guardar_metricas(analisis, ruta):
    fig, ejes = plt.subplots(1, 2, figsize=(13, 5))
    metricas = analisis["metricas"].sort_values("k")
    limite_x = max(45, int(metricas["k"].max()))
    ejes[0].plot(metricas["k"], metricas["inercia"], marker=".")
    ejes[0].set(title="MÃ©todo del codo", xlabel="k", ylabel="Inercia")
    ejes[1].plot(metricas["k"], metricas["silueta"], marker=".")
    ejes[1].axvline(analisis["optimo"], color="red", linestyle="--")
    ejes[1].set(title="Coeficiente de silueta", xlabel="k", ylabel="Silueta")
    for eje in ejes:
        eje.set_xlim(1, limite_x)
        eje.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def guardar_espacio(analisis, ruta, k):
    modelo = analisis["modelos"][k]
    datos = np.asarray(analisis["datos"])
    pca = PCA(2).fit(datos)
    puntos = pca.transform(datos)
    centros = pca.transform(modelo.cluster_centers_)
    colores = plt.cm.nipy_spectral(np.linspace(0, 1, k, endpoint=False))
    fig, eje = plt.subplots(figsize=(10, 7))
    eje.scatter(puntos[:, 0], puntos[:, 1], c=colores[modelo.labels_],
                edgecolors="black", linewidths=.3)
    eje.scatter(centros[:, 0], centros[:, 1], c=colores, marker="X",
                s=160, edgecolors="black")
    eje.set(title=f"Espacio de soluciÃ³n K-Means (k={k})",
            xlabel="Componente principal 1", ylabel="Componente principal 2")
    eje.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def guardar_rios_por_cluster(caudales, asignaciones, nombres, carpeta):
    """Guarda un CSV por cluster con nombre, código y caudales mensuales."""
    carpeta.mkdir(parents=True, exist_ok=True)
    tabla = caudales.copy()
    tabla.index.name = "Codigo"
    tabla = tabla.reset_index()
    tabla["Nombre"] = tabla["Codigo"].map(nombres).fillna("Río")
    tabla = tabla.merge(asignaciones[["Codigo", "Cluster"]], on="Codigo")
    columnas = ["Nombre", "Codigo", "Cluster", *MESES]
    for cluster, grupo in tabla.groupby("Cluster", sort=True):
        grupo[columnas].sort_values("Codigo").to_csv(
            carpeta / f"cluster_{int(cluster):02d}.csv",
            index=False,
        )


def _decimal(serie, positivo=False):
    numerica = pd.to_numeric(serie, errors="coerce")
    if numerica.notna().all():
        return -numerica.abs() if positivo else numerica
    partes = serie.astype(str).str.extract(
        r"(?P<g>\d+).*?(?P<m>\d+).*?(?P<s>\d+(?:\.\d+)?).*?(?P<o>[SO])"
    )
    valor = (pd.to_numeric(partes["g"], errors="coerce")
             + pd.to_numeric(partes["m"], errors="coerce") / 60
             + pd.to_numeric(partes["s"], errors="coerce") / 3600)
    return valor.where(~partes["o"].isin(["S", "O"]), -valor)


def cargar_coordenadas(archivos):
    """Une coordenadas; cada tupla es (ruta, coordenadas_decimales_positivas)."""
    tablas = []
    for ruta, positivas in archivos:
        tabla = pd.read_csv(ruta, encoding="latin1")
        tabla["Codigo"] = pd.to_numeric(tabla["Codigo"], errors="coerce")
        tabla["Latitud"] = _decimal(tabla["Latitud"], positivas)
        tabla["Longitud"] = _decimal(tabla["Longitud"], positivas)
        tablas.append(tabla)
    coordenadas = pd.concat(tablas, ignore_index=True).dropna(
        subset=["Codigo", "Latitud", "Longitud"]
    )
    return coordenadas.drop_duplicates("Codigo", keep="last").copy()


def guardar_mapa(analisis, coordenadas, ruta, k, referencia, mostrar_nombres=False):
    mapa = coordenadas.merge(
        pd.DataFrame({"Codigo": analisis["codigos"],
                      "Cluster": analisis["modelos"][k].labels_}),
        on="Codigo",
        how="inner",
    ).reset_index(drop=True)
    mapa["Referencia"] = [f"R{i:03d}" for i in range(1, len(mapa) + 1)]
    mapa[["Referencia", "Codigo", "Rio", "Cluster"]].to_csv(
        referencia, index=False
    )
    colores = plt.cm.nipy_spectral(np.linspace(0, 1, k, endpoint=False))
    fig, eje = plt.subplots(figsize=(10, 14))
    try:
        import geopandas as gpd
        provincias = gpd.read_file(
            "https://naturalearth.s3.amazonaws.com/10m_cultural/"
            "ne_10m_admin_1_states_provinces.zip"
        )
        provincias[provincias["admin"] == "Argentina"].plot(
            ax=eje, facecolor="#f2efe6", edgecolor="gray", linewidth=.6
        )
    except Exception as error:
        raise RuntimeError("No se pudo cargar el mapa de Argentina.") from error
    eje.scatter(mapa["Longitud"], mapa["Latitud"],
                c=colores[mapa["Cluster"]], edgecolors="black", zorder=3)
    if mostrar_nombres:
        for _, rio in mapa.iterrows():
            eje.annotate(
                rio["Rio"],
                (rio["Longitud"], rio["Latitud"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6,
                zorder=4,
            )
    eje.set(xlim=(-73.5, -53.5), ylim=(-55.5, -21.5),
            title=f"DistribuciÃ³n geogrÃ¡fica de clusters (k={k})",
            xlabel="Longitud", ylabel="Latitud")
    eje.set_aspect("equal")
    eje.legend(
        handles=[Patch(facecolor=colores[i], label=f"Cluster {i}") for i in range(k)],
        title="Clusters", bbox_to_anchor=(1.02, 1), loc="upper left",
        fontsize=7, ncol=2,
    )
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def guardar_histogramas(analisis, carpeta, k):
    carpeta_medios = carpeta / "por_cluster"
    carpeta_rios = carpeta / "por_rio"
    carpeta_medios.mkdir(parents=True, exist_ok=True)
    carpeta_rios.mkdir(parents=True, exist_ok=True)
    modelo = analisis["modelos"][k]
    datos = analisis["caudales"].copy()
    datos["Cluster"] = modelo.labels_
    medias = datos.groupby("Cluster")[MESES].mean()
    for cluster, fila in medias.iterrows():
        fig, eje = plt.subplots(figsize=(8, 4))
        eje.bar(MESES, fila)
        eje.set(title=f"Perfil medio - Cluster {cluster}",
                xlabel="Mes", ylabel="Caudal normalizado")
        eje.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(carpeta_medios / f"cluster_{cluster:02d}_medio.png", dpi=120)
        plt.close(fig)
        codigos = datos.index[datos["Cluster"] == cluster]
        columnas = 3
        filas = max(1, int(np.ceil(len(codigos) / columnas)))
        fig, ejes = plt.subplots(
            filas, columnas, figsize=(15, max(4, filas * 3.5)), squeeze=False
        )
        for eje, codigo in zip(ejes.flat, codigos):
            eje.bar(MESES, datos.loc[codigo, MESES])
            eje.set_title(f"RÃ­o {codigo}")
            eje.set_xlabel("Mes")
            eje.set_ylabel("Caudal")
            eje.grid(alpha=.3)
        for eje in ejes.flat[len(codigos):]:
            eje.set_visible(False)
        fig.suptitle(f"RÃ­os del cluster {cluster}", y=.995)
        fig.tight_layout()
        fig.savefig(carpeta_rios / f"cluster_{cluster:02d}_rios.png", dpi=120)
        plt.close(fig)
