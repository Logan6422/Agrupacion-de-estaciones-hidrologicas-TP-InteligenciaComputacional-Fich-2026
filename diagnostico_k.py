from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


KS = range(9, 41)
METRICAS = [
    "k", "ari_boot", "pares_centroides_redundantes",
    "distancia_min_centroides", "exceso_coherencia",
    "cociente_coherencia", "silueta", "calinski_harabasz",
    "davies_bouldin",
]


def cargar_resultados():
    resultados = pd.read_csv(Path(__file__).with_name("resultados_k_etapa2.csv"))
    faltantes = sorted(set(METRICAS) - set(resultados.columns))
    if faltantes:
        raise ValueError(f"Faltan columnas en resultados_k.csv: {faltantes}")
    return resultados[resultados["k"].isin(KS)].copy().sort_values("k")


def diagnosticar_filtros(tabla, ari_piso=0.55):
    diagnostico = tabla[["k", "ari_boot", "pares_centroides_redundantes"]].copy()

    def motivo(fila):
        ari = fila["ari_boot"] < ari_piso
        redundancia = fila["pares_centroides_redundantes"] != 0
        if ari and redundancia:
            return "ambos"
        if ari:
            return "ari"
        if redundancia:
            return "redundancia"
        return "ok"

    diagnostico["motivo"] = diagnostico.apply(motivo, axis=1)
    print("\nDiagnóstico de filtros:")
    print(diagnostico.to_string(index=False))
    print("\nConteo por motivo:")
    print(diagnostico["motivo"].value_counts().reindex(
        ["ok", "ari", "redundancia", "ambos"], fill_value=0
    ))
    return diagnostico


def sensibilidad(tabla):
    filas = []
    rango = tabla[tabla["k"].between(15, 40)]
    for ari_piso in [0.45, 0.5, 0.55, 0.6]:
        for maximo in [0, 1, 2, 3]:
            sobreviven = rango[
                (rango["ari_boot"] >= ari_piso)
                & (rango["pares_centroides_redundantes"] <= maximo)
            ]
            filas.append({
                "ARI_PISO": ari_piso,
                "max_pares_redundantes": maximo,
                "cantidad_k": len(sobreviven),
            })
    matriz = pd.DataFrame(filas).pivot(
        index="ARI_PISO",
        columns="max_pares_redundantes",
        values="cantidad_k",
    )
    print("\nSensibilidad:")
    print(matriz.to_string())
    return matriz


def rankear(tabla):
    candidatos = tabla[
        tabla["k"].between(15, 40)
        & (tabla["ari_boot"] >= 0.5)
        & (tabla["pares_centroides_redundantes"] <= 1)
    ].copy()
    columnas = [
        "ari_boot", "exceso_coherencia", "cociente_coherencia", "silueta"
    ]
    rangos = candidatos[columnas].rank(pct=True)
    candidatos["score"] = rangos.mean(axis=1)
    resultado = candidatos.sort_values("score", ascending=False).head(5)
    print("\nTop 5 por score:")
    print(resultado[["k", "score", *METRICAS[1:]]].to_string(index=False))
    return resultado


def graficar(tabla):
    fig, ejes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    ejes[0].plot(tabla["k"], tabla["ari_boot"], marker="o")
    ejes[0].axhline(0.55, color="red", linestyle="--")
    ejes[0].set_ylabel("ARI bootstrap")
    ejes[1].bar(tabla["k"], tabla["pares_centroides_redundantes"])
    ejes[1].set_ylabel("Pares redundantes")
    ejes[2].plot(tabla["k"], tabla["exceso_coherencia"], marker="o")
    ejes[2].set_ylabel("Exceso coherencia")
    ejes[2].set_xlabel("k")
    for eje in ejes:
        eje.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(Path(__file__).with_name("diagnostico_k_etapa2.png"), dpi=150)
    plt.close(fig)


def ejecutar():
    tabla = cargar_resultados()
    print(tabla[METRICAS].to_string(index=False))
    diagnosticar_filtros(tabla)
    sensibilidad(tabla)
    rankear(tabla)
    graficar(tabla)


if __name__ == "__main__":
    ejecutar()
