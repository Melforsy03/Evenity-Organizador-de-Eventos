import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from Experimentacion.Simulacion.analisis_meta import generar_30_escenarios , generar_eventos_nuevos
import os
import sys
import numpy as np
# --- Funciones auxiliares ---
# Añadir carpeta 'src' al path para importar modules.optimizador
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.optimizador import  generar_solucion_inicial
# --- Funciones auxiliares ---

def tiempo_traslado(evento1, evento2, velocidad_kmh=30):
    loc1 = evento1.get("spatial_info", {}).get("venue", {}).get("location", {})
    loc2 = evento2.get("spatial_info", {}).get("venue", {}).get("location", {})
    if not loc1 or not loc2:
        return 0
    coord1 = (loc1.get("latitude"), loc1.get("longitude"))
    coord2 = (loc2.get("latitude"), loc2.get("longitude"))
    if None in coord1 or None in coord2:
        return 0
    distancia_km = geodesic(coord1, coord2).km
    tiempo_horas = distancia_km / velocidad_kmh
    return tiempo_horas * 60

def generar_solucion_inicial(pool: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    def hay_solape(e1, e2):
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

    eventos_validos = [e for e in pool if "temporal_info" in e and e["temporal_info"].get("start")]
    eventos_ordenados = sorted(eventos_validos, key=lambda x: x["temporal_info"].get("start", ""))
    solucion = []

    for evento in eventos_ordenados:
        if len(solucion) >= n:
            break
        if not any(hay_solape(evento, e) for e in solucion):
            solucion.append(evento)

    while len(solucion) < n and eventos_validos:
        candidato = random.choice(eventos_validos)
        if not any(hay_solape(candidato, e) for e in solucion):
            solucion.append(candidato)

    return solucion[:n]

def calcular_huecos_totales(eventos: List[Dict[str, Any]]) -> float:
    if len(eventos) < 2:
        return 0
    tiempos = []
    for e in eventos:
        try:
            start = datetime.fromisoformat(e["temporal_info"]["start"].replace("Z", "+00:00"))
            dur = e["temporal_info"].get("duration_minutes", 120)
            end = start + timedelta(minutes=dur)
            tiempos.append((start, end))
        except:
            pass
    tiempos = sorted(tiempos, key=lambda x: x[0])
    huecos = 0
    for i in range(len(tiempos) - 1):
        diff = (tiempos[i+1][0] - tiempos[i][1]).total_seconds() / 60
        if diff > 0:
            huecos += diff
    return -huecos  # negativo para penalizar

# --- Heurísticas ---

PESOS = {
    "categoria": 1.0,
    "ciudad": 0.8,
    "fecha": 0.5,
    "duracion": 0.3,
    "traslado": -0.7,
    "diversidad_categorias": 0.4,
    "huecos": -0.5,
    "grafo": 1.2
}

def evaluar_solucion_mejorada(eventos: List[Dict[str, Any]], preferencias: Dict[str, Any], scores_grafo: Optional[Dict[str, float]] = None) -> float:
    score_total = 0
    categorias_unicas = set()
    velocidad_kmh = preferencias.get("velocidad_kmh", 30)
    duracion_maxima = preferencias.get("duracion_maxima", None)

    penalizacion_traslados = 0
    for i in range(len(eventos) - 1):
        t_traslado = tiempo_traslado(eventos[i], eventos[i+1], velocidad_kmh)
        if t_traslado > 60:
            penalizacion_traslados += (t_traslado - 60)

    for e in eventos:
        cat = e.get("classification", {}).get("primary_category", "")
        if preferencias.get("categories") and cat in preferencias.get("categories"):
            score_total += PESOS["categoria"]
        categorias_unicas.add(cat)

        ciudad = e.get("spatial_info", {}).get("area", {}).get("city", "")
        if preferencias.get("location") and ciudad == preferencias.get("location"):
            score_total += PESOS["ciudad"]

        fecha_evento = e.get("temporal_info", {}).get("start")
        fechas_disp = preferencias.get("available_dates")
        if fecha_evento and fechas_disp:
            try:
                fecha_evento_dt = datetime.fromisoformat(fecha_evento.replace("Z", "+00:00"))
                inicio_dt = datetime.fromisoformat(fechas_disp[0])
                fin_dt = datetime.fromisoformat(fechas_disp[1])
                if inicio_dt <= fecha_evento_dt <= fin_dt:
                    score_total += PESOS["fecha"]
            except:
                pass

        duracion_evento = e.get("temporal_info", {}).get("duration_minutes")
        if duracion_maxima and duracion_evento:
            if duracion_evento <= duracion_maxima:
                score_total += PESOS["duracion"]
            else:
                score_total -= PESOS["duracion"]

        if scores_grafo:
            nombre = e.get("basic_info", {}).get("title", "")
            score_total += scores_grafo.get(nombre, 0) * PESOS["grafo"]

    score_total += len(categorias_unicas) * PESOS["diversidad_categorias"]
    score_total -= penalizacion_traslados * PESOS["traslado"]

    huecos_totales = calcular_huecos_totales(eventos)
    score_total += huecos_totales * PESOS["huecos"]

    return score_total

def evaluar_solucion_vieja(eventos: List[Dict[str, Any]], preferencias: Dict[str, Any], scores_grafo: Optional[Dict[str, float]] = None) -> float:
    score_total = 0
    categorias_unicas = set()
    velocidad_kmh = preferencias.get("velocidad_kmh", 30)
    duracion_maxima = preferencias.get("duracion_maxima", None)

    penalizacion_traslados = 0
    for i in range(len(eventos)-1):
        t_traslado = tiempo_traslado(eventos[i], eventos[i+1], velocidad_kmh)
        if t_traslado > 60:
            penalizacion_traslados += (t_traslado - 60) * 0.5

    for e in eventos:
        cat = e.get("classification", {}).get("primary_category", "")
        if preferencias.get("categories") and cat in preferencias.get("categories"):
            score_total += 5
        categorias_unicas.add(cat)

        ciudad = e.get("spatial_info", {}).get("area", {}).get("city", "")
        if preferencias.get("location") and ciudad == preferencias.get("location"):
            score_total += 3

        fecha_evento = e.get("temporal_info", {}).get("start")
        fechas_disp = preferencias.get("available_dates")
        if fecha_evento and fechas_disp:
            try:
                fecha_evento_dt = datetime.fromisoformat(fecha_evento.replace("Z", "+00:00"))
                inicio_dt = datetime.fromisoformat(fechas_disp[0])
                fin_dt = datetime.fromisoformat(fechas_disp[1])
                if inicio_dt <= fecha_evento_dt <= fin_dt:
                    score_total += 2
            except:
                pass

        duracion_evento = e.get("temporal_info", {}).get("duration_minutes")
        if duracion_maxima and duracion_evento:
            if duracion_evento <= duracion_maxima:
                score_total += 1
            else:
                score_total -= 1

        if scores_grafo:
            nombre = e.get("basic_info", {}).get("title", "")
            score_total += scores_grafo.get(nombre, 0) * 10

    score_total += len(categorias_unicas) * 2
    score_total -= penalizacion_traslados

    return score_total

def hill_climbing_viejo(pool: List[Dict[str, Any]], preferencias: Dict[str, Any], n: int = 5, max_iter: int = 1000, scores_grafo: Optional[Dict[str, float]] = None) -> Tuple[List[Dict[str, Any]], float]:
    current = generar_solucion_inicial(pool, n)
    best = current.copy()
    best_score = evaluar_solucion_vieja(current, preferencias, scores_grafo)

    def hay_solape(e1, e2):
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
            continue
        score_vecino = evaluar_solucion_vieja(vecino, preferencias, scores_grafo)
        if score_vecino > best_score:
            best = vecino.copy()
            best_score = score_vecino
    return best, best_score

def hill_climbing_mejorado(pool: List[Dict[str, Any]], preferencias: Dict[str, Any], n: int = 5, max_iter: int = 1000, scores_grafo: Optional[Dict[str, float]] = None) -> Tuple[List[Dict[str, Any]], float, List[float]]:
    current = generar_solucion_inicial(pool, n)
    best = current.copy()
    best_score = evaluar_solucion_mejorada(current, preferencias, scores_grafo)
    scores_trace = [best_score]

    def hay_solape(e1, e2):
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
        score_vecino = evaluar_solucion_mejorada(vecino, preferencias, scores_grafo)
        if score_vecino > best_score:
            best = vecino.copy()
            best_score = score_vecino
        scores_trace.append(best_score)
    return best, best_score, scores_trace

def generar_30_escenarios():
    ciudades = ["Bilbao", "Zaragoza", "Málaga", "Granada", "Palma de Mallorca"]
    categorias = ["technology", "food", "theater", "literature", "wellness"]
    velocidades = [15, 25, 35, 45]
    duraciones = [60, 120, 180, 240]
    escenarios = []
    for _ in range(30):
        escenario = {
            "location": random.choice(ciudades),
            "categories": random.sample(categorias, k=random.randint(1,3)),
            "velocidad_kmh": random.choice(velocidades),
            "duracion_maxima": random.choice(duraciones)
        }
        escenarios.append(escenario)
    return escenarios

def graficar_info_detallada(resultados_viejo, resultados_mejorado, traces_mejorado, repeticiones, output_folder="graficas_descriptivas"):
    os.makedirs(output_folder, exist_ok=True)

    viejo = np.array(resultados_viejo)  # shape (num_escenarios, repeticiones)
    mejorado = np.array(resultados_mejorado)

    medias_viejo = np.mean(viejo, axis=1)
    std_viejo = np.std(viejo, axis=1)

    medias_mejorado = np.mean(mejorado, axis=1)
    std_mejorado = np.std(mejorado, axis=1)

    promedio_total_viejo = np.mean(medias_viejo)
    std_total_viejo = np.mean(std_viejo)
    promedio_total_mejorado = np.mean(medias_mejorado)
    std_total_mejorado = np.mean(std_mejorado)

    plt.figure(figsize=(8,6))
    barras = plt.bar(["Viejo", "Mejorado"], [promedio_total_viejo, promedio_total_mejorado], yerr=[std_total_viejo, std_total_mejorado], capsize=10, color=["orange", "green"])
    plt.ylabel("Score promedio final")
    plt.title("Score promedio final ± desviación estándar (30 escenarios)")
    for bar, val in zip(barras, [promedio_total_viejo, promedio_total_mejorado]):
        plt.text(bar.get_x() + bar.get_width()/2, val, f"{val:.2f}", ha='center', va='bottom')
    plt.savefig(os.path.join(output_folder, "promedio_std_scores.png"))
    plt.close()

    plt.figure(figsize=(10,6))
    plt.boxplot([viejo.flatten(), mejorado.flatten()], labels=["Viejo", "Mejorado"])
    plt.ylabel("Score final")
    plt.title("Distribución de scores finales (todas repeticiones y escenarios)")
    plt.savefig(os.path.join(output_folder, "boxplot_scores.png"))
    plt.close()

    traces_array = np.array([np.pad(t, (0, max(map(len,traces_mejorado)) - len(t)), 'edge') for t in traces_mejorado])
    media_trace = np.mean(traces_array, axis=0)
    std_trace = np.std(traces_array, axis=0)

    plt.figure(figsize=(12,6))
    x = np.arange(len(media_trace))
    plt.plot(x, media_trace, label="Score promedio")
    plt.fill_between(x, media_trace - std_trace, media_trace + std_trace, color='green', alpha=0.3, label="±1 std dev")
    plt.xlabel("Iteración")
    plt.ylabel("Score")
    plt.title("Evolución del score promedio con desviación (último escenario)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, "evolucion_score_con_std.png"))
    plt.close()

    dif_porcentual = (medias_mejorado - medias_viejo) / np.abs(medias_viejo) * 100

    plt.figure(figsize=(14,6))
    colores = ["green" if x>0 else "red" for x in dif_porcentual]
    plt.bar(np.arange(1,len(dif_porcentual)+1), dif_porcentual, color=colores)
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("Escenario")
    plt.ylabel("Diferencia porcentual del score (%)")
    plt.title("Diferencia porcentual del score final (Mejorado vs Viejo)")
    plt.savefig(os.path.join(output_folder, "diferencia_porcentual_score.png"))
    plt.close()

    ganador = np.where(medias_mejorado > medias_viejo, "Mejorado", np.where(medias_mejorado < medias_viejo, "Viejo", "Empate"))
    conteo_ganadores = {"Mejorado": np.sum(ganador=="Mejorado"), "Viejo": np.sum(ganador=="Viejo"), "Empate": np.sum(ganador=="Empate")}

    plt.figure(figsize=(8,6))
    plt.bar(conteo_ganadores.keys(), conteo_ganadores.values(), color=["green", "orange", "gray"])
    plt.ylabel("Número de escenarios ganados")
    plt.title("Número de escenarios ganados por algoritmo")
    plt.savefig(os.path.join(output_folder, "numero_escenarios_ganados.png"))
    plt.close()

    print(f"Gráficas descriptivas guardadas en: {output_folder}")

def comparar_en_30_escenarios(eventos, n_eventos=10, max_iter=500, repeticiones=3, output_folder="graficas_descriptivas"):
    os.makedirs(output_folder, exist_ok=True)
    escenarios = generar_30_escenarios()

    # Guardamos resultados por escenario y repetición
    resultados_viejo = []
    resultados_mejorado = []
    traces_mejorado = []

    for i, prefs in enumerate(escenarios, 1):
        print(f"\nEscenario {i}: {prefs}")
        rep_viejo = []
        rep_mejorado = []
        rep_traces = []

        for r in range(repeticiones):
            _, score_viejo = hill_climbing_viejo(eventos, prefs, n_eventos, max_iter)
            rep_viejo.append(score_viejo)

            _, score_mejorado, scores_trace = hill_climbing_mejorado(eventos, prefs, n_eventos, max_iter)
            rep_mejorado.append(score_mejorado)

            if r == repeticiones - 1:
                rep_traces.append(scores_trace)

        resultados_viejo.append(rep_viejo)
        resultados_mejorado.append(rep_mejorado)
        if rep_traces:
            traces_mejorado.extend(rep_traces)

        print(f" Promedio Viejo: {np.mean(rep_viejo):.2f}, Promedio Mejorado: {np.mean(rep_mejorado):.2f}")

    # Graficar info descriptiva
    graficar_info_detallada(resultados_viejo, resultados_mejorado, traces_mejorado, repeticiones, output_folder)

    return resultados_viejo, resultados_mejorado, escenarios


if __name__ == "__main__":
    
    eventos = generar_eventos_nuevos(100)
    resultados_viejo, resultados_mejorado, escenarios = comparar_en_30_escenarios(
        eventos, n_eventos=10, max_iter=500, repeticiones=3, output_folder="Comparacion_HillClim"
    )
    print("Comparación terminada. Revisa la carpeta 'graficas_descriptivas' para los gráficos.")
    
