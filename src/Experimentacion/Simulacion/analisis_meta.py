import random
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
from utils import evaluar_solucion_mejorada , evaluar_solucion_vieja , hay_solape
from utils import generar_escenarios , generar_eventos

class AnalizadorAlgoritmos:
    def __init__(self, n_eventos=150, n_escenarios=30):
        self.n_eventos = n_eventos
        self.n_escenarios = n_escenarios
        self.eventos = generar_eventos()
        self.escenarios = generar_escenarios(self.eventos)

    def analizar_estadisticamente(self, resultados):
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from scipy.stats import spearmanr, ttest_ind, mannwhitneyu
        import seaborn as sns # Importar seaborn para la paleta de colores

        labels = list(resultados.keys())
        # Definir una paleta de colores distintiva para los algoritmos
        colors = sns.color_palette("tab10", len(labels)) # Usar una paleta de 10 colores categóricos
        color_map = {label: colors[i] for i, label in enumerate(labels)}

        print("\n🔍 INICIANDO ANÁLISIS ESTADÍSTICO 🔍\n")

        # 1️⃣ Scatter: Calidad vs Tiempo
        plt.figure(figsize=(10, 6))
        for algo in labels:
            tiempos = resultados[algo]["tiempos"]
            scores = resultados[algo]["scores"]
            plt.scatter(tiempos, scores, alpha=0.7, label=algo, color=color_map[algo]) # Aplicar color por algoritmo
        plt.xlabel("Tiempo de ejecución (s)")
        plt.ylabel("Mejor Score obtenido")
        plt.title("Scatter: Calidad vs Tiempo de Ejecución")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("scatter_calidad_vs_tiempo.png")
        plt.close()
        print("✅ Guardado: scatter_calidad_vs_tiempo.png")

        # 2️⃣ Correlación Iteraciones vs Score
        print("\n🔗 Correlación Iteraciones vs Score (Spearman)")
        for algo in labels:
            iteraciones = resultados[algo]["num_iters"]
            scores = resultados[algo]["scores"]
            corr, pval = spearmanr(iteraciones, scores)
            print(f"{algo}: Spearman rho = {corr:.3f} | p-valor = {pval:.4f}")

        # 3️⃣ Métricas de Robustez
        print("\n📊 Métricas de Robustez")
        robustez_data = []
        for algo in labels:
            scores = resultados[algo]["scores"]
            mean_score = np.mean(scores)
            best_score = np.max(scores)
            perc90 = np.percentile(scores, 90)
            std_dev = np.std(scores)
            robustez_data.append([algo, mean_score, best_score, perc90, std_dev])
            print(f"{algo}:")
            print(f"  ▶️ Media: {mean_score:.2f}")
            print(f"  🥇 Mejor Score: {best_score:.2f}")
            print(f"  🔝 Percentil 90: {perc90:.2f}")
            print(f"  📉 Desviación estándar: {std_dev:.2f}")

        # Guarda tabla de robustez
        df_robustez = pd.DataFrame(robustez_data, columns=["Algoritmo", "Media", "Mejor", "Percentil90", "DesvStd"])
        df_robustez.to_csv("tabla_robustez.csv", index=False)
        print("✅ Guardado: tabla_robustez.csv")

        # 4️⃣ Test de Diferencia de Medias (pares)
        print("\n🔬 Test de Diferencia de Medias (T-test y Mann-Whitney U)")
        test_data = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                algo1 = labels[i]
                algo2 = labels[j]

                scores1 = resultados[algo1]["scores"]
                scores2 = resultados[algo2]["scores"]

                t_stat, p_t = ttest_ind(scores1, scores2, equal_var=False)
                u_stat, p_u = mannwhitneyu(scores1, scores2)

                test_data.append([f"{algo1} vs {algo2}", p_t, p_u])

                print(f"{algo1} vs {algo2}:")
                print(f"  🧪 T-test p-valor: {p_t:.4f}")
                print(f"  🧪 Mann-Whitney U p-valor: {p_u:.4f}")

        # Guarda tabla de tests
        df_tests = pd.DataFrame(test_data, columns=["Comparación", "T-test_p", "MannWhitneyU_p"])
        df_tests.to_csv("tabla_tests_diferencia_medias.csv", index=False)
        print("✅ Guardado: tabla_tests_diferencia_medias.csv")

        print("\n🎉 Análisis estadístico completado 🎉\n")

    def comparar_algoritmos(self, n_ejecuciones=5):
        resultados = {
            "Random": {"scores": [], "tiempos": [], "convergencia": [], "num_iters": [], "num_sols": [], "ciudades": []},
            "HillClimbing": {"scores": [], "tiempos": [], "convergencia": [], "num_iters": [], "num_sols": [], "ciudades": []},
            "SimulatedAnnealing": {"scores": [], "tiempos": [], "convergencia": [], "num_iters": [], "num_sols": [], "ciudades": []}
        }

        for escenario in self.escenarios:
            print(f"\nEvaluando escenario en {escenario['ciudad']}...")

            for _ in range(n_ejecuciones):
                # Búsqueda Aleatoria
                res_random = self.busqueda_aleatoria(escenario)
                resultados["Random"]["scores"].append(res_random["mejor_score"])
                resultados["Random"]["tiempos"].append(res_random["tiempos"][-1])
                resultados["Random"]["convergencia"].append(res_random["scores"])
                resultados["Random"]["num_iters"].append(res_random["num_iter"])
                resultados["Random"]["num_sols"].append(res_random["num_sols"])
                resultados["Random"]["ciudades"].append(escenario['ciudad'])

                # Hill Climbing
                res_hc = self.hill_climbing(escenario)
                resultados["HillClimbing"]["scores"].append(res_hc["mejor_score"])
                resultados["HillClimbing"]["tiempos"].append(res_hc["tiempos"][-1])
                resultados["HillClimbing"]["convergencia"].append(res_hc["scores"])
                resultados["HillClimbing"]["num_iters"].append(res_hc["num_iter"])
                resultados["HillClimbing"]["num_sols"].append(res_hc["num_sols"])
                resultados["HillClimbing"]["ciudades"].append(escenario['ciudad'])

                # Simulated Annealing
                res_sa = self.simulated_annealing(escenario)
                resultados["SimulatedAnnealing"]["scores"].append(res_sa["mejor_score"])
                resultados["SimulatedAnnealing"]["tiempos"].append(res_sa["tiempos"][-1])
                resultados["SimulatedAnnealing"]["convergencia"].append(res_sa["scores"])
                resultados["SimulatedAnnealing"]["num_iters"].append(res_sa["num_iter"])
                resultados["SimulatedAnnealing"]["num_sols"].append(res_sa["num_sols"])
                resultados["SimulatedAnnealing"]["ciudades"].append(escenario['ciudad'])

        # Se llama a generar_graficos y generar_reporte aquí, después de todas las simulaciones.
        self.generar_graficos(resultados)
        self.generar_reporte(resultados)

        return resultados

    def busqueda_aleatoria(self, preferencias, n=10, max_iter=100):
        scores = []
        mejor_score = -1
        mejor_solucion = []
        tiempos = []

        soluciones_generadas = 0
        iter_realizadas = 0

        start_time = time.time()

        for _ in range(max_iter):
            iter_realizadas += 1
            solucion = random.sample(self.eventos, n)

            # Verificar solapes
            valido = True
            for i in range(len(solucion)):
                for j in range(i+1, len(solucion)):
                    if hay_solape(solucion[i], solucion[j]):
                        valido = False
                        break
                if not valido:
                    break

            if not valido:
                continue

            soluciones_generadas += 1

            score = evaluar_solucion_mejorada(solucion, preferencias)
            scores.append(score)

            if score > mejor_score:
                mejor_score = score
                mejor_solucion = solucion.copy()

            tiempos.append(time.time() - start_time)

        return {
            "scores": scores,
            "mejor_score": mejor_score,
            "mejor_solucion": mejor_solucion,
            "tiempos": tiempos,
            "num_iter": iter_realizadas,
            "num_sols": soluciones_generadas
        }

    def hill_climbing(self, preferencias, n=10, max_iter=100):
        current = random.sample(self.eventos, n)
        best = current.copy()
        best_score = evaluar_solucion_mejorada(current, preferencias)
        scores = [best_score]
        tiempos = [0]

        soluciones_generadas = 1  # La inicial cuenta
        iter_realizadas = 0

        start_time = time.time()

        for _ in range(max_iter):
            iter_realizadas += 1
            vecino = current.copy()
            idx = random.randint(0, n-1)

            intentos = 0
            while intentos < 50:
                candidato = random.choice(self.eventos)
                if candidato not in vecino:
                    vecino[idx] = candidato
                    valido = True
                    for i in range(len(vecino)):
                        for j in range(i+1, len(vecino)):
                            if hay_solape(vecino[i], vecino[j]):
                                valido = False
                                break
                        if not valido:
                            break
                    if valido:
                        soluciones_generadas += 1
                        break
                intentos += 1

            if intentos >= 50:
                continue

            score_vecino = evaluar_solucion_vieja(vecino, preferencias)
            if score_vecino > best_score:
                best = vecino.copy()
                best_score = score_vecino
                current = vecino.copy()

            scores.append(best_score)
            tiempos.append(time.time() - start_time)

        return {
            "scores": scores,
            "mejor_score": best_score,
            "mejor_solucion": best,
            "tiempos": tiempos,
            "num_iter": iter_realizadas,
            "num_sols": soluciones_generadas
        }

    def simulated_annealing(self, preferencias, n=10, max_iter=100, T_ini=200, T_min=1):
        current = random.sample(self.eventos, n)
        best = current.copy()
        best_score = evaluar_solucion_mejorada(current, preferencias)
        current_score = best_score
        scores = [best_score]
        tiempos = [0]
        T = T_ini

        soluciones_generadas = 1  # solución inicial
        iter_realizadas = 0

        start_time = time.time()

        for _ in range(max_iter):
            iter_realizadas += 1

            vecino = current.copy()
            idx = random.randint(0, n-1)

            intentos = 0
            while intentos < 50:
                candidato = random.choice(self.eventos)
                if candidato not in vecino:
                    vecino[idx] = candidato

                    valido = True
                    for i in range(len(vecino)):
                        for j in range(i+1, len(vecino)):
                            if hay_solape(vecino[i], vecino[j]):
                                valido = False
                                break
                        if not valido:
                            break

                    if valido:
                        soluciones_generadas += 1
                        break

                intentos += 1

            if intentos >= 50:
                T *= 0.95
                if T < T_min:
                    break
                continue

            score_vecino = evaluar_solucion_vieja(vecino, preferencias)
            delta = score_vecino - current_score

            if delta > 0 or random.random() < math.exp(delta / T):
                current = vecino.copy()
                current_score = score_vecino

                if score_vecino > best_score:
                    best = vecino.copy()
                    best_score = score_vecino

            scores.append(best_score)
            tiempos.append(time.time() - start_time)

            T *= 0.95
            if T < T_min:
                break

        return {
            "scores": scores,
            "mejor_score": best_score,
            "mejor_solucion": best,
            "tiempos": tiempos,
            "num_iter": iter_realizadas,
            "num_sols": soluciones_generadas
        }

    def generar_graficos(self, resultados):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import pandas as pd

        labels = list(resultados.keys())
        # Definir una paleta de colores distintiva para los algoritmos
        colors = sns.color_palette("tab10", len(labels)) 
        color_map = {label: colors[i] for i, label in enumerate(labels)}

        print("\n✅ Generando gráficos ultra completos...\n")

        # Configurar un estilo de Seaborn para gráficos más bonitos
        sns.set_theme(style="whitegrid") # Mantener whitegrid, eliminar palette de aquí

        # 1️⃣ PILA de convergencias por algoritmo (MEJORADO con colores por algoritmo)
        fig, axes = plt.subplots(len(labels), 1, figsize=(12, 4 * len(labels))) 
        if len(labels) == 1:
            axes = [axes]

        for idx, algo in enumerate(labels):
            ax = axes[idx]
            convergencias = resultados[algo]["convergencia"]

            # Filtrar listas de convergencia vacías y calcular longitud mínima válida
            valid_convergencias = [c for c in convergencias if len(c) > 0]
            if not valid_convergencias:
                print(f"Advertencia: No hay datos de convergencia válidos para {algo}. Saltando este gráfico.")
                continue

            min_len = min(len(c) for c in valid_convergencias)

            # Trazar cada curva de convergencia individual con el color del algoritmo y baja opacidad
            for run in valid_convergencias:
                ax.plot(run[:min_len], alpha=0.15, linewidth=0.7, color=color_map[algo]) # Usar color_map

            # Calcular y trazar la media de convergencia con el color del algoritmo y línea destacada
            truncated = [c[:min_len] for c in valid_convergencias]
            conv_promedio = np.mean(truncated, axis=0)
            ax.plot(conv_promedio, color=color_map[algo], linewidth=3, label="Media") # Usar color_map

            ax.set_title(f"Convergencia del algoritmo: {algo}") 
            ax.set_xlabel("Iteración")
            ax.set_ylabel("Score")
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.savefig("grafico_convergencia_mejorado.png") 
        plt.close()
        print("✅ Guardado: grafico_convergencia_mejorado.png")

        # Preparar los datos para boxplot/violinplot de forma que seaborn los coloree por algoritmo
        # Esto es para mejorar la aplicación de colores en estos plots
        df_scores = pd.DataFrame()
        df_tiempos = pd.DataFrame()
        df_iters = pd.DataFrame()
        df_sols = pd.DataFrame()

        for algo in labels:
            df_scores = pd.concat([df_scores, pd.DataFrame({'Score': resultados[algo]['scores'], 'Algoritmo': algo})], ignore_index=True)
            df_tiempos = pd.concat([df_tiempos, pd.DataFrame({'Tiempo': resultados[algo]['tiempos'], 'Algoritmo': algo})], ignore_index=True)
            df_iters = pd.concat([df_iters, pd.DataFrame({'Iteraciones': resultados[algo]['num_iters'], 'Algoritmo': algo})], ignore_index=True)
            df_sols = pd.concat([df_sols, pd.DataFrame({'Soluciones': resultados[algo]['num_sols'], 'Algoritmo': algo})], ignore_index=True)

        # 2️⃣ Boxplot de scores finales
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Algoritmo', y='Score', data=df_scores, palette=color_map) # Usar seaborn y paleta
        plt.title("Distribución de Scores Finales (Boxplot)")
        plt.ylabel("Score")
        plt.grid(True)
        plt.savefig("grafico_scores_boxplot.png")
        plt.close()
        print("✅ Guardado: grafico_scores_boxplot.png")

        # Violinplot de scores finales
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='Algoritmo', y='Score', data=df_scores, palette=color_map) # Usar seaborn y paleta
        plt.title("Distribución de Scores Finales (Violinplot)")
        plt.ylabel("Score")
        plt.grid(True)
        plt.savefig("grafico_scores_violinplot.png")
        plt.close()
        print("✅ Guardado: grafico_scores_violinplot.png")

        # 3️⃣ Boxplot de tiempos de ejecución
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Algoritmo', y='Tiempo', data=df_tiempos, palette=color_map) # Usar seaborn y paleta
        plt.title("Distribución de Tiempos de Ejecución")
        plt.ylabel("Tiempo (s)")
        plt.grid(True)
        plt.savefig("grafico_tiempos_boxplot.png")
        plt.close()
        print("✅ Guardado: grafico_tiempos_boxplot.png")

        # 4️⃣ Boxplot de iteraciones realizadas
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Algoritmo', y='Iteraciones', data=df_iters, palette=color_map) # Usar seaborn y paleta
        plt.title("Distribución de Iteraciones Realizadas")
        plt.ylabel("Iteraciones")
        plt.grid(True)
        plt.savefig("grafico_iteraciones_boxplot.png")
        plt.close()
        print("✅ Guardado: grafico_iteraciones_boxplot.png")

        # 5️⃣ Boxplot de soluciones generadas
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Algoritmo', y='Soluciones', data=df_sols, palette=color_map) # Usar seaborn y paleta
        plt.title("Distribución de Soluciones Generadas")
        plt.ylabel("Soluciones")
        plt.grid(True)
        plt.savefig("grafico_sols_generadas_boxplot.png")
        plt.close()
        print("✅ Guardado: grafico_sols_generadas_boxplot.png")

        # 6️⃣ Histograma de scores finales
        plt.figure(figsize=(10, 6))
        scores_data_flat = [s for sublist in resultados[algo]["scores"] for algo in labels for s in sublist] if any(isinstance(val, list) for val in resultados[labels[0]]["scores"]) else [s for sublist in [resultados[algo]["scores"] for algo in labels] for s in sublist]
        bins = np.linspace(
            min(scores_data_flat), 
            max(scores_data_flat), 
            15
        )
        for i, algo in enumerate(labels):
            plt.hist(resultados[algo]["scores"], bins=bins, alpha=0.5, label=algo, color=color_map[algo]) # Aplicar color por algoritmo
        plt.title("Histograma Comparativo de Scores Finales")
        plt.xlabel("Score")
        plt.ylabel("Frecuencia")
        plt.legend()
        plt.grid(True)
        plt.savefig("grafico_scores_histograma.png")
        plt.close()
        print("✅ Guardado: grafico_scores_histograma.png")

        # 7️⃣ Heatmap de scores promedio por escenario
        heat_data = []
        n_ejecuciones = 5  # Ajusta si es variable o pásalo como argumento, asumiendo 5 por el momento

        for idx, escenario in enumerate(self.escenarios):
            ciudad = escenario["ciudad"]
            for algo in labels:
                start = idx * n_ejecuciones
                end = start + n_ejecuciones
                # Asegurarse de que el slicing no excede los límites
                current_scores = resultados[algo]["scores"][start:end]
                avg_score = np.mean(current_scores) if current_scores else 0 # Manejar caso de lista vacía
                
                heat_data.append({
                    "Algoritmo": algo,
                    "Ciudad": ciudad,
                    "Score": avg_score
                })

        df_heat = pd.DataFrame(heat_data)

        heat = df_heat.pivot_table(index="Ciudad", columns="Algoritmo", values="Score", aggfunc="mean")

        plt.figure(figsize=(12, 8))
        sns.heatmap(heat, annot=True, fmt=".1f", cmap="coolwarm")
        plt.title("Heatmap: Score Promedio por Ciudad y Algoritmo")
        plt.tight_layout()
        plt.savefig("grafico_heatmap_scores.png")
        plt.close()
        print("✅ Guardado: grafico_heatmap_scores.png")

    def generar_reporte(self, resultados):
        reporte = []
        
        for algo in resultados:
            reporte.append({
                "Algoritmo": algo,
                "Score Promedio": np.mean(resultados[algo]["scores"]),
                "Mejor Score": np.max(resultados[algo]["scores"]),
                "Tiempo Promedio (s)": np.mean(resultados[algo]["tiempos"]),
                "Desviación Estándar": np.std(resultados[algo]["scores"]),
                "Iteraciones para Converger": np.mean([len(conv) for conv in resultados[algo]["convergencia"]])
            })
        
        df = pd.DataFrame(reporte)
        df.to_csv("reporte_comparativo.csv", index=False)
        print("\nReporte comparativo:")
        print(df.to_string(index=False))
        print("\nReporte guardado en 'reporte_comparativo.csv'")

if __name__ == "__main__":
    analizador = AnalizadorAlgoritmos(n_eventos=150, n_escenarios=30)
    resultados = analizador.comparar_algoritmos(n_ejecuciones=5)
    analizador.analizar_estadisticamente(resultados)