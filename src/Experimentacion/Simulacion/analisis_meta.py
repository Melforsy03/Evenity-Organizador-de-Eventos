import random
import math
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import sys 
import os
import csv
import pandas as pd
import numpy as np

# Añadir carpeta 'src' al path para importar modules.optimizador
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.optimizador import evaluar_solucion_mejorada, generar_solucion_inicial
def simulated_annealing_wrapper(pool, preferencias, n, max_iter=500, T_ini=100.0, T_min=1.0):
    current = generar_solucion_inicial(pool, n)
    best = current.copy()
    best_score = evaluar_solucion_mejorada(current, preferencias)
    T = T_ini
    scores_trace = []

    def hay_solape(e1, e2):
        from datetime import datetime, timedelta
        try:
            fecha1 = e1["temporal_info"]["start"]
            duracion1 = e1["temporal_info"].get("duration_minutes", 120)
            fecha2 = e2["temporal_info"]["start"]
            duracion2 = e2["temporal_info"].get("duration_minutes", 120)
            dt1 = datetime.fromisoformat(fecha1.replace("Z", "+00:00"))
            dt2 = datetime.fromisoformat(fecha2.replace("Z", "+00:00"))
            return not (dt1 + timedelta(minutes=duracion1) <= dt2 or dt2 + timedelta(minutes=duracion2) <= dt1)
        except Exception:
            return False

    for _ in range(max_iter):
        vecino = current.copy()
        idx = random.randint(0, len(vecino) - 1)

        intentos = 0
        while intentos < 100:
            candidato = random.choice(pool)
            if candidato not in vecino:
                vecino[idx] = candidato
                if not any(hay_solape(vecino[i], vecino[j]) for i in range(len(vecino)) for j in range(i+1, len(vecino))):
                    break
            intentos += 1
        else:
            scores_trace.append(best_score)
            T *= 0.95
            if T < T_min:
                break
            continue

        score_vecino = evaluar_solucion_mejorada(vecino, preferencias)
        score_actual = evaluar_solucion_mejorada(current, preferencias)
        delta = score_vecino - score_actual

        if delta > 0 or random.random() < math.exp(delta / T):
            current = vecino
            if score_vecino > best_score:
                best = vecino.copy()
                best_score = score_vecino

        scores_trace.append(best_score)
        T *= 0.95
        if T < T_min:
            break

    return best, best_score, scores_trace

def busqueda_aleatoria(pool, preferencias, n, max_iter=500):
    best_sol = None
    best_score = float('-inf')
    scores = []
    for _ in range(max_iter):
        solucion = generar_solucion_inicial(pool, n)
        score = evaluar_solucion_mejorada(solucion, preferencias)
        if score > best_score:
            best_sol = solucion
            best_score = score
        scores.append(best_score)
    return best_sol, best_score, scores

def hill_climbing(pool, preferencias, n, max_iter=500):
    current = generar_solucion_inicial(pool, n)
    best = current.copy()
    best_score = evaluar_solucion_mejorada(current, preferencias)
    scores_trace = [best_score]

    def hay_solape(e1, e2):
        from datetime import datetime, timedelta
        try:
            fecha1 = e1["temporal_info"]["start"]
            duracion1 = e1["temporal_info"].get("duration_minutes", 120)
            fecha2 = e2["temporal_info"]["start"]
            duracion2 = e2["temporal_info"].get("duration_minutes", 120)
            dt1 = datetime.fromisoformat(fecha1.replace("Z", "+00:00"))
            dt2 = datetime.fromisoformat(fecha2.replace("Z", "+00:00"))
            return not (dt1 + timedelta(minutes=duracion1) <= dt2 or dt2 + timedelta(minutes=duracion2) <= dt1)
        except Exception:
            return False

    for _ in range(max_iter):
        vecino = best.copy()
        idx = random.randint(0, len(vecino) - 1)

        intentos = 0
        while intentos < 100:
            candidato = random.choice(pool)
            if candidato not in vecino:
                vecino[idx] = candidato
                if not any(hay_solape(vecino[i], vecino[j]) for i in range(len(vecino)) for j in range(i+1, len(vecino))):
                    break
            intentos += 1
        else:
            scores_trace.append(best_score)
            continue

        score_vecino = evaluar_solucion_mejorada(vecino, preferencias)
        if score_vecino > best_score:
            best = vecino.copy()
            best_score = score_vecino

        scores_trace.append(best_score)

    return best, best_score, scores_trace

# --- Paso 1: cargar eventos originales ---
def cargar_eventos_originales(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)

# --- Paso 2: generar eventos nuevos ---
def generar_eventos_nuevos(n):
    ciudades = ["Bilbao", "Zaragoza", "Málaga", "Granada", "Palma de Mallorca","Barcelona", "Madrid", "Seville", "Valencia", "Ibiza"]
    categorias = ["technology", "food", "theater", "literature", "wellness","concerts", "sports", "arts", "conferences", "festivals"]
    venues = {
        "Bilbao": ["Azkuna Zentroa", "Bilbao Exhibition Centre", "Teatro Arriaga"],
        "Zaragoza": ["Expo Zaragoza", "Teatro Principal", "Parque Grande"],
        "Málaga": ["Teatro Cervantes", "Muelle Uno", "La Malagueta"],
        "Granada": ["Palacio de Congresos", "Teatro Isabel la Católica", "Alhambra"],
        "Palma de Mallorca": ["Auditorio de Palma", "Plaza Mayor", "Es Baluard"],
        "Barcelona": ["Palau Sant Jordi", "Razzmatazz", "Apolo"],
        "Madrid": ["WiZink Center", "Teatro Kapital", "La Riviera"],
        "Seville": ["FIBES", "Cartuja Center", "Teatro Lope de Vega"],
        "Valencia": ["Palau de la Música", "Ciudad de las Artes", "L'Umbracle"],
        "Ibiza": ["Amnesia", "Ushuaïa", "Chinois"]
    }

    base_date = datetime(2025, 6, 1)
    eventos = []

    for i in range(n):
        ciudad = random.choice(ciudades)
        categoria = random.choice(categorias)
        venue = random.choice(venues[ciudad])

        fecha_evento = base_date + timedelta(days=random.randint(0, 365),
                                             hours=random.randint(0, 23),
                                             minutes=random.randint(0, 59))

        centros = {
            "Bilbao": (43.2630, -2.9350),
            "Zaragoza": (41.6488, -0.8891),
            "Málaga": (36.7213, -4.4214),
            "Granada": (37.1773, -3.5986),
            "Palma de Mallorca": (39.5696, 2.6502),
            "Barcelona": (41.3851, 2.1734),
            "Madrid": (40.4168, -3.7038),
            "Seville": (37.3891, -5.9845),
            "Valencia": (39.4699, -0.3763),
            "Ibiza": (38.9067, 1.4206)
        }
        lat_c, lon_c = centros[ciudad]
        lat = lat_c + random.uniform(-0.03, 0.03)
        lon = lon_c + random.uniform(-0.03, 0.03)

        evento = {
            "metadata": {
                "source": "synthetic_new",
                "processing_timestamp": datetime.utcnow().isoformat() + "Z",
                "original_id": f"sintetico_new_{i+1}",
                "event_version": "2.1",
                "raw_hash": ""
            },
            "basic_info": {
                "title": f"Evento Nuevo Sintético {i+1} en {ciudad}",
                "description": f"Evento sintético {i+1} en {ciudad} - categoría {categoria}",
                "status": "confirmed",
                "visibility": "public"
            },
            "classification": {
                "primary_category": categoria,
                "categories": [],
                "tags": []
            },
            "temporal_info": {
                "start": fecha_evento.isoformat(),
                "end": fecha_evento.isoformat(),
                "timezone": {
                    "name": "Europe/Madrid",
                    "utc_offset": "+0200"
                },
                "is_recurring": False,
                "is_tba": False,
                "duration_minutes": random.choice([60, 90, 120, 180])
            },
            "spatial_info": {
                "venue": {
                    "name": venue,
                    "full_address": f"{venue}, {ciudad}, Spain",
                    "location": {
                        "latitude": lat,
                        "longitude": lon,
                        "geo_accuracy": "high"
                    }
                },
                "area": {
                    "city": ciudad,
                    "country": "Spain"
                }
            },
            "participants": {},
            "external_references": {
                "urls": [
                    {
                        "type": "source",
                        "url": f"https://sintetico.com/eventos/sintetico_new_{i+1}"
                    }
                ],
                "performers": [],
                "images": []
            },
            "raw_data_metadata": {
                "significant_fields_preserved": ["city", "category"],
                "source_specific": {"synthetic_tag": True}
            }
        }
        eventos.append(evento)

    return eventos

# --- Paso 3: crear lista de 30 escenarios ---
def generar_30_escenarios():
    ciudades = ["Bilbao", "Zaragoza", "Málaga", "Granada", "Palma de Mallorca","Barcelona", "Madrid", "Seville", "Valencia", "Ibiza"]
    categorias = ["technology", "food", "theater", "literature", "wellness","concerts", "sports", "arts", "conferences", "festivals"]
    velocidades = [15, 25, 35, 45]
    duraciones = [60, 120, 180, 240]

    escenarios = []
    for _ in range(30):
        escenario = {
            "location": random.choice(ciudades),
            "categories": random.sample(categorias, k=random.randint(1, 3)),
            "velocidad_kmh": random.choice(velocidades),
            "duracion_maxima": random.choice(duraciones)
        }
        escenarios.append(escenario)
    return escenarios

# --- Paso 4: ejecutar simulación para todos los escenarios ---
def comparar_algoritmos_varios_escenarios(eventos, lista_preferencias, n_eventos=10, max_iter=500, repeticiones=3, output_base="resultados_comparacion"):
    from Experimentacion.Simulacion.analisis_meta import simulated_annealing_wrapper, busqueda_aleatoria, hill_climbing

    os.makedirs(output_base, exist_ok=True)

    algoritmos = {
        "SimulatedAnnealing": simulated_annealing_wrapper,
        "BusquedaAleatoria": busqueda_aleatoria,
        "HillClimbing": hill_climbing
    }

    resumen_scores = {}

    for idx, prefs in enumerate(lista_preferencias):
        carpeta = os.path.join(output_base, f"escenario_{idx+1}")
        os.makedirs(carpeta, exist_ok=True)
        print(f"\nEjecutando escenario {idx+1} con preferencias: {prefs}")

        plt.figure(figsize=(12, 7))
        resumen_scores[f"escenario_{idx+1}"] = {}

        for nombre, algoritmo in algoritmos.items():
            print(f" Ejecutando {nombre}...")
            scores_reps = []

            for _ in range(repeticiones):
                _, _, scores_trace = algoritmo(eventos, prefs, n_eventos, max_iter)
                scores_reps.append(scores_trace)

            min_len = min(len(s) for s in scores_reps)
            scores_reps_trunc = [s[:min_len] for s in scores_reps]

            scores_promedio = [sum(col) / len(col) for col in zip(*scores_reps_trunc)]
            resumen_scores[f"escenario_{idx+1}"][nombre] = scores_promedio

            plt.plot(scores_promedio, label=nombre)

        plt.xlabel("Iteración")
        plt.ylabel("Score promedio")
        plt.title(f"Comparación de algoritmos - Escenario {idx+1}")
        plt.legend()
        plt.grid(True)

        ruta_grafico = os.path.join(carpeta, f"comparacion_algoritmos_escenario_{idx+1}.png")
        plt.savefig(ruta_grafico)
        plt.close()
        print(f"Gráfico guardado en {ruta_grafico}")

    # --- Al final de la función agregar resumen cuantitativo ---
    resumen_global_path = os.path.join(output_base, "resumen_global_comparativo.csv")
    with open(resumen_global_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Escenario", "Algoritmo", "Score Final Promedio", "Mejor Score", "Iteraciones Promedio"])

        for escenario, datos_algos in resumen_scores.items():
            for algoritmo, scores in datos_algos.items():
                score_final_promedio = scores[-1] if scores else 0
                mejor_score = max(scores) if scores else 0
                iter_promedio = len(scores)
                writer.writerow([escenario, algoritmo, round(score_final_promedio, 2), round(mejor_score, 2), iter_promedio])

    print(f"Resumen global guardado en {resumen_global_path}")

    # --- Ranking general de algoritmos en los 30 escenarios ---
    algoritmos = list(next(iter(resumen_scores.values())).keys())
    scores_por_algoritmo = {alg: [] for alg in algoritmos}

    for datos_algos in resumen_scores.values():
        for alg in algoritmos:
            score_final = datos_algos[alg][-1]
            scores_por_algoritmo[alg].append(score_final)

    print("\nRanking global promedio de algoritmos (score final promedio en todos los escenarios):")
    ranking = sorted(scores_por_algoritmo.items(), key=lambda x: np.mean(x[1]), reverse=True)
    for i, (alg, scores) in enumerate(ranking, 1):
        print(f"{i}. {alg}: promedio = {np.mean(scores):.2f}")

    return resumen_scores

def grafico_global_comparativo(ruta_csv, output_folder="graficas_resumen_global"):
    os.makedirs(output_folder, exist_ok=True)
    df = pd.read_csv(ruta_csv)

    # Agrupar por algoritmo y calcular promedios sobre todos los escenarios
    df_grouped = df.groupby("Algoritmo").agg({
        "Score Final Promedio": "mean",
        "Mejor Score": "mean",
        "Iteraciones Promedio": "mean"
    }).reset_index()

    algoritmos = df_grouped["Algoritmo"]
    scores_final = df_grouped["Score Final Promedio"]
    mejor_scores = df_grouped["Mejor Score"]
    iter_promedio = df_grouped["Iteraciones Promedio"]

    x = np.arange(len(algoritmos))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))

    bars1 = ax.bar(x - width, scores_final, width, label="Promedio Score Final")
    bars2 = ax.bar(x, mejor_scores, width, label="Promedio Mejor Score")
    bars3 = ax.bar(x + width, iter_promedio, width, label="Promedio Iteraciones")

    ax.set_xlabel("Algoritmos")
    ax.set_ylabel("Valor promedio")
    ax.set_title("Comparación global de algoritmos sobre 30 escenarios")
    ax.set_xticks(x)
    ax.set_xticklabels(algoritmos)
    ax.legend()
    ax.grid(axis='y')

    # Añadir valores encima de las barras
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom")

    plt.tight_layout()
    ruta_guardado = os.path.join(output_folder, "comparacion_global_algoritmos_30_escenarios.png")
    plt.savefig(ruta_guardado)
    plt.close()
    print(f"Gráfico global comparativo guardado en: {ruta_guardado}")
# --- MAIN ---

def main():
    ruta_eventos_original = "eventos_gold.json"  # Cambia si es necesario

    print("Cargando eventos originales...")
    eventos_originales = cargar_eventos_originales(ruta_eventos_original)

    print("Generando eventos nuevos...")
    eventos_nuevos = generar_eventos_nuevos(150)

    print("Generando escenarios...")
    escenarios = generar_30_escenarios()

    print("Ejecutando simulaciones en todos los escenarios...")
    resultados = comparar_algoritmos_varios_escenarios(eventos_nuevos, escenarios, n_eventos=10, max_iter=500, repeticiones=1)
    grafico_global_comparativo("resultados_comparacion/resumen_global_comparativo.csv")
    print("Proceso finalizado.")
    # Aquí podrías guardar resultados, o procesarlos más si quieres

if __name__ == "__main__":
    main()