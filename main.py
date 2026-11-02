#los resultados; la lógica principal está separada en analisis_rios.py.

from analisis_rios import analizar, graficar_resultados

# Ejecuta todas las etapas: carga, normalización, SOM, K-Means y métricas.
resultados = analizar()

# Abre los gráficos del método del codo y de la silueta.
graficar_resultados(resultados)

# Guarda, para cada código de río, la asignación del clustering óptimo
# y la asignación del clustering teórico de 23 grupos.
resultados["asignaciones"].to_csv("clusters_rios.csv", index=False)

# Muestra todas las métricas calculadas para los valores de k evaluados.
print(resultados["metricas"].to_string(index=False))

# Muestra únicamente la comparación entre los dos valores de k relevantes.
print("\nComparación entre k óptimo y k teórico:")
print(resultados["comparacion"].to_string(index=False))

# Informa las tres referencias utilizadas para interpretar el análisis:
# silueta máxima, cantidad teórica de grupos y punto del método del codo.
print(f"\nk óptimo por silueta: {resultados['k_optimo']}")
print(f"k teórico evaluado: {resultados['k_teorico']}")
print(f"k del codo: {resultados['k_codo']}")
