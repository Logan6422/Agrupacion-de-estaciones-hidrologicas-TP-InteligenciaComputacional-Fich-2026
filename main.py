from analisis_rios import analizar, graficar_resultados


resultados = analizar()
graficar_resultados(resultados)
resultados["asignaciones"].to_csv("clusters_rios.csv", index=False)

print(resultados["metricas"].to_string(index=False))
print("\nComparación entre k óptimo y k teórico:")
print(resultados["comparacion"].to_string(index=False))
print(f"\nk óptimo por silueta: {resultados['k_optimo']}")
print(f"k teórico evaluado: {resultados['k_teorico']}")
print(f"k del codo: {resultados['k_codo']}")
