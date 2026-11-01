from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MESES = [
    "Sep",
    "Oct",
    "Nov",
    "Dic",
    "Ene",
    "Feb",
    "Mar",
    "Abr",
    "May",
    "Jun",
    "Jul",
    "Ago",
]
TAMANO_SOM = (10, 10)
SEMILLA = 42


def coordenada_a_entero(coordenada):
    partes = coordenada.str.extract(
        r"(?P<grados>\d+).*?(?P<minutos>\d+).*?"
        r"(?P<segundos>\d+(?:\.\d+)?).*?(?P<sentido>[SO])",
        expand=True,
    )
    valor_decimal = (
        pd.to_numeric(partes["grados"])
        + pd.to_numeric(partes["minutos"]) / 60
        + pd.to_numeric(partes["segundos"]) / 3600
    )
    return valor_decimal.where(
        ~partes["sentido"].isin(["S", "O"]),
        -valor_decimal,
    ).round().astype(int)


def cargar_rios():
    carpeta = Path(__file__).parent
    archivos = sorted((carpeta / "csv").glob("estaciones_final*.csv"))
    estaciones = pd.concat(
        (pd.read_csv(archivo, encoding="latin1") for archivo in archivos),
        ignore_index=True,
    )
    estaciones["Codigo"] = pd.to_numeric(estaciones["Codigo"], errors="coerce")
    estaciones = estaciones.dropna(subset=["Codigo"]).copy()
    estaciones["Codigo"] = estaciones["Codigo"].astype(int)

    rios = {
        codigo: valores.tolist()
        for codigo, valores in estaciones.groupby("Codigo", sort=True)[MESES]
        .mean()
        .iterrows()
    }

    estaciones["Qmod"] = pd.to_numeric(estaciones["Qmod"], errors="coerce")
    qmod_por_rio = estaciones.groupby("Codigo", sort=True)["Qmod"].mean()
    rios = {
        codigo: [caudal / qmod_por_rio[codigo] for caudal in vector]
        for codigo, vector in rios.items()
    }

    coordenadas = pd.read_csv(
        carpeta / "COORDS2.csv",
        header=None,
        names=[
            "Codigo",
            "Region",
            "Rio",
            "Lugar",
            "Provincia",
            "Latitud",
            "Longitud",
        ],
        encoding="latin1",
    )
    coordenadas["Codigo"] = pd.to_numeric(coordenadas["Codigo"], errors="coerce")
    coordenadas = coordenadas.dropna(subset=["Codigo"]).copy()
    coordenadas["Codigo"] = coordenadas["Codigo"].astype(int)
    coordenadas["Latitud"] = coordenada_a_entero(coordenadas["Latitud"])
    coordenadas["Longitud"] = coordenada_a_entero(coordenadas["Longitud"])
    coordenadas_por_rio = coordenadas.groupby("Codigo")[["Latitud", "Longitud"]].mean()

    for codigo, vector in rios.items():
        if codigo not in coordenadas_por_rio.index:
            raise ValueError(f"No hay coordenadas para el código {codigo}.")
        vector.extend(
            coordenadas_por_rio.loc[codigo].round().astype(int).tolist()
        )
    return rios


def entrenar_som(datos, filas, columnas, epocas=200, semilla=SEMILLA):
    generador = np.random.default_rng(semilla)
    prototipos = generador.normal(
        datos.mean(axis=0),
        datos.std(axis=0) + 1e-12,
        size=(filas, columnas, datos.shape[1]),
    )
    posiciones = np.indices((filas, columnas)).transpose(1, 2, 0)
    radio_inicial = max(filas, columnas) / 2
    radio_final = 1.0
    tasa_inicial = 0.5
    tasa_final = 0.05

    for epoca in range(epocas):
        orden = generador.permutation(len(datos))
        progreso = epoca / max(epocas - 1, 1)
        tasa = tasa_inicial * (tasa_final / tasa_inicial) ** progreso
        radio = radio_inicial * (radio_final / radio_inicial) ** progreso
        for indice in orden:
            vector = datos[indice]
            distancias = np.linalg.norm(prototipos - vector, axis=2)
            neurona_ganadora = np.unravel_index(np.argmin(distancias), distancias.shape)
            distancia_mapa = np.sum(
                (posiciones - neurona_ganadora) ** 2,
                axis=2,
            )
            vecindad = np.exp(-distancia_mapa / (2 * radio**2))
            prototipos += tasa * vecindad[:, :, None] * (vector - prototipos)
    return prototipos


rios = cargar_rios()
codigos_rios = list(rios)
matriz_rios = np.array(list(rios.values()), dtype=float)
prototipos_som = entrenar_som(
    matriz_rios,
    filas=TAMANO_SOM[0],
    columnas=TAMANO_SOM[1],
)

distancias = np.linalg.norm(
    matriz_rios[:, None, None, :] - prototipos_som[None, :, :, :],
    axis=3,
)
neuronas_ganadoras = np.unravel_index(
    np.argmin(distancias.reshape(len(matriz_rios), -1), axis=1),
    TAMANO_SOM,
)
asignaciones_som = pd.DataFrame(
    {
        "Codigo": codigos_rios,
        "Fila": neuronas_ganadoras[0],
        "Columna": neuronas_ganadoras[1],
    }
)

distancia_prototipos = np.zeros(TAMANO_SOM)
for fila in range(TAMANO_SOM[0]):
    for columna in range(TAMANO_SOM[1]):
        vecinos = []
        if fila + 1 < TAMANO_SOM[0]:
            vecinos.append(prototipos_som[fila + 1, columna])
        if columna + 1 < TAMANO_SOM[1]:
            vecinos.append(prototipos_som[fila, columna + 1])
        if vecinos:
            distancia_prototipos[fila, columna] = np.mean(
                [np.linalg.norm(prototipos_som[fila, columna] - vecino) for vecino in vecinos]
            )

plt.figure(figsize=(8, 7))
plt.imshow(distancia_prototipos, cmap="viridis")
plt.colorbar(label="Distancia entre prototipos")
plt.title("SOM 10x10 - mapa de distancias")
plt.xlabel("Columna")
plt.ylabel("Fila")
plt.tight_layout()
plt.show()

print(f"SOM entrenado con {len(rios)} ríos.")
print(f"Prototipos disponibles en prototipos_som: {prototipos_som.shape}")
print(asignaciones_som.sort_values(["Fila", "Columna", "Codigo"]).to_string(index=False))
