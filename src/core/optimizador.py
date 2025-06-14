"""
Módulo de optimización de agendas (optimizador.py)

Define funciones para generar una agenda óptima de eventos mediante la técnica de búsqueda
local tipo hill climbing, evaluando cada solución según múltiples criterios ponderados.

Funciones principales:
- `evaluar_solucion_mejorada()`
- `hill_climbing_mejorado()`
- `obtener_eventos_optimales()`

Se consideran aspectos como categoría, ciudad, fechas, duración, traslado entre eventos,
diversidad temática y conectividad en el grafo.
"""
from datetime import datetime, timedelta, timezone
import random
import math
from typing import List, Dict, Any, Tuple, Optional
from geopy.distance import geodesic

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

def ensure_aware(dt):
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt

def evaluar_solucion_mejorada(eventos: List[Dict[str, Any]], 
                              preferencias: Dict[str, Any], 
                              scores_grafo: Optional[Dict[str, float]] = None) -> float:
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
                inicio_dt = ensure_aware(fechas_disp[0])
                fin_dt = ensure_aware(fechas_disp[1])
                fecha_evento_dt = ensure_aware(fecha_evento)
                if inicio_dt <= fecha_evento_dt <= fin_dt:
                    score_total += PESOS["fecha"]
            except Exception as e:
                print(f"⚠️ Error comparando fechas: {e}")

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

def calcular_huecos_totales(eventos: List[Dict[str, Any]]) -> float:
    if len(eventos) < 2:
        return 0
    tiempos = []
    for e in eventos:
        try:
            start = ensure_aware(e["temporal_info"]["start"])
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
    return -huecos

def hay_solape(e1, e2):
    try:
        dt1 = ensure_aware(e1["temporal_info"]["start"])
        dt2 = ensure_aware(e2["temporal_info"]["start"])
        dur1 = e1["temporal_info"].get("duration_minutes", 120)
        dur2 = e2["temporal_info"].get("duration_minutes", 120)
        return not (dt1 + timedelta(minutes=dur1) <= dt2 or dt2 + timedelta(minutes=dur2) <= dt1)
    except Exception:
        return False

def generar_solucion_inicial(pool: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
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
    tiempo_minutos = tiempo_horas * 60
    return tiempo_minutos

def hill_climbing_mejorado(pool: List[Dict[str, Any]],
                           preferencias: Dict[str, Any],
                           n: int = 5,
                           max_iter: int = 1000,
                           scores_grafo: Optional[Dict[str, float]] = None) -> Tuple[List[Dict[str, Any]], float, list]:
    current = generar_solucion_inicial(pool, n)
    best = current.copy()
    best_score = evaluar_solucion_mejorada(current, preferencias, scores_grafo)
    scores_trace = [best_score]

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

def obtener_eventos_optimales(eventos: List[Dict[str, Any]], 
                            preferencias: Dict[str, Any], 
                            cantidad: int = 5, 
                            max_iter: int = 1000,
                            scores_grafo: Optional[Dict[str, float]] = None) -> Tuple[List[Dict[str, Any]], float]:
    if len(eventos) < cantidad:
        raise ValueError(f"No hay suficientes eventos para optimizar (necesarios: {cantidad}, disponibles: {len(eventos)})")
    return hill_climbing_mejorado(eventos, preferencias, n=cantidad, max_iter=max_iter, scores_grafo=scores_grafo)
