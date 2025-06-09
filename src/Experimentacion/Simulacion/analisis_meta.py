
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
        
    def busqueda_aleatoria(self, preferencias, n=10, max_iter=100):
        scores = []
        mejor_score = -1
        mejor_solucion = []
        tiempos = []
        
        start_time = time.time()
        
        for _ in range(max_iter):
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
                
            score = evaluar_solucion_vieja(solucion, preferencias)
            scores.append(score)
            
            if score > mejor_score:
                mejor_score = score
                mejor_solucion = solucion.copy()
                
            tiempos.append(time.time() - start_time)
            
        return {
            "scores": scores,
            "mejor_score": mejor_score,
            "mejor_solucion": mejor_solucion,
            "tiempos": tiempos
        }

    def hill_climbing(self, preferencias, n=10, max_iter=100):
        current = random.sample(self.eventos, n)
        best = current.copy()
        best_score = evaluar_solucion_vieja(current, preferencias)
        scores = [best_score]
        tiempos = [0]
        
        start_time = time.time()
        
        for _ in range(max_iter):
            # Generar vecino
            vecino = current.copy()
            idx = random.randint(0, n-1)
            
            intentos = 0
            while intentos < 50:
                candidato = random.choice(self.eventos)
                if candidato not in vecino:
                    vecino[idx] = candidato
                    
                    # Verificar solape
                    valido = True
                    for i in range(len(vecino)):
                        for j in range(i+1, len(vecino)):
                            if hay_solape(vecino[i], vecino[j]):
                                valido = False
                                break
                        if not valido:
                            break
                            
                    if valido:
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
            "tiempos": tiempos
        }
    
    def simulated_annealing(self, preferencias, n=10, max_iter=100, T_ini=100, T_min=1):
        current = random.sample(self.eventos, n)
        best = current.copy()
        best_score = evaluar_solucion_vieja(current, preferencias)
        current_score = best_score
        scores = [best_score]
        tiempos = [0]
        T = T_ini
        
        start_time = time.time()
        
        for _ in range(max_iter):
            # Generar vecino
            vecino = current.copy()
            idx = random.randint(0, n-1)
            
            intentos = 0
            while intentos < 50:
                candidato = random.choice(self.eventos)
                if candidato not in vecino:
                    vecino[idx] = candidato
                    
                    # Verificar solape
                    valido = True
                    for i in range(len(vecino)):
                        for j in range(i+1, len(vecino)):
                            if hay_solape(vecino[i], vecino[j]):
                                valido = False
                                break
                        if not valido:
                            break
                            
                    if valido:
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
            
            # Enfriar
            T *= 0.95
            if T < T_min:
                break
                
        return {
            "scores": scores,
            "mejor_score": best_score,
            "mejor_solucion": best,
            "tiempos": tiempos
        }
    
    def comparar_algoritmos(self, n_ejecuciones=5):
        resultados = {
            "Random": {"scores": [], "tiempos": [], "convergencia": []},
            "HillClimbing": {"scores": [], "tiempos": [], "convergencia": []},
            "SimulatedAnnealing": {"scores": [], "tiempos": [], "convergencia": []}
        }
        
        for escenario in self.escenarios:
            print(f"\nEvaluando escenario en {escenario['ciudad']}...")
            
            for _ in range(n_ejecuciones):
                # Búsqueda Aleatoria
                res_random = self.busqueda_aleatoria(escenario)
                resultados["Random"]["scores"].append(res_random["mejor_score"])
                resultados["Random"]["tiempos"].append(res_random["tiempos"][-1])
                resultados["Random"]["convergencia"].append(res_random["scores"])
                
                # Hill Climbing
                res_hc = self.hill_climbing(escenario)
                resultados["HillClimbing"]["scores"].append(res_hc["mejor_score"])
                resultados["HillClimbing"]["tiempos"].append(res_hc["tiempos"][-1])
                resultados["HillClimbing"]["convergencia"].append(res_hc["scores"])
                
                # Simulated Annealing
                res_sa = self.simulated_annealing(escenario)
                resultados["SimulatedAnnealing"]["scores"].append(res_sa["mejor_score"])
                resultados["SimulatedAnnealing"]["tiempos"].append(res_sa["tiempos"][-1])
                resultados["SimulatedAnnealing"]["convergencia"].append(res_sa["scores"])
        
        self.generar_graficos(resultados)
        self.generar_reporte(resultados)
        
        return resultados
    
    def generar_graficos(self, resultados):
        import matplotlib.pyplot as plt
        import numpy as np

        # 1. Gráfico de convergencia promedio
        plt.figure(figsize=(8, 6))
        for algo in resultados:
            convergencias = [conv for conv in resultados[algo]["convergencia"] if conv]
            if not convergencias:
                print(f"⚠️  No hay datos de convergencia para {algo}")
                continue
            min_len = min(len(conv) for conv in convergencias)
            if min_len == 0:
                print(f"⚠️  Todas las curvas de {algo} tienen longitud 0")
                continue
            truncated = [conv[:min_len] for conv in convergencias]
            conv_promedio = np.mean(truncated, axis=0)
            plt.plot(conv_promedio, label=algo)

        plt.title("Promedio de Convergencia por Iteración")
        plt.xlabel("Iteración")
        plt.ylabel("Score acumulado")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("grafico_convergencia.png")
        plt.close()

        # 2. Gráfico de distribución de scores
        plt.figure(figsize=(8, 6))
        scores_data = [resultados[algo]["scores"] for algo in resultados]
        labels = list(resultados.keys())
        plt.boxplot(scores_data, tick_labels=labels)
        plt.title("Distribución de Puntuaciones Finales")
        plt.ylabel("Score")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("grafico_scores.png")
        plt.close()

        # 3. Gráfico de tiempos de ejecución
        plt.figure(figsize=(8, 6))
        tiempos_data = [resultados[algo]["tiempos"] for algo in resultados]
        plt.boxplot(tiempos_data, tick_labels=labels)
        plt.title("Distribución de Tiempos de Ejecución")
        plt.ylabel("Segundos")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("grafico_tiempos.png")
        plt.close()

        # 4. Gráfico de estabilidad (desviación estándar)
        plt.figure(figsize=(8, 6))
        estabilidad = [np.std(resultados[algo]["scores"]) for algo in resultados]
        plt.bar(labels, estabilidad)
        plt.title("Estabilidad del Algoritmo (Desviación Estándar de Scores)")
        plt.ylabel("Desviación Estándar")
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig("grafico_estabilidad.png")
        plt.close()

        print("Gráficos guardados: 'grafico_convergencia.png', 'grafico_scores.png', 'grafico_tiempos.png', 'grafico_estabilidad.png'")

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