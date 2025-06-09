    
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
# --- Funciones auxiliares ---
# Añadir carpeta 'src' al path para importar modules.optimizador
# --- Funciones auxiliares ---

from datetime import datetime, timedelta
import random

def generar_eventos():
    ciudades = ["Bilbao", "Zaragoza", "Málaga", "Granada", "Palma de Mallorca", 
                "Barcelona", "Madrid", "Seville", "Valencia", "Ibiza"]
    categorias = ["technology", "food", "theater", "literature", "wellness", 
                  "concerts", "sports", "arts", "conferences", "festivals"]
    
    eventos = []
    base_date = datetime(2025, 6, 1)
    
    for i in range(100):
        ciudad = random.choice(ciudades)
        categoria = random.choice(categorias)
        
        evento = {
            "metadata": {
                "source": "predicthq",
                "processing_timestamp": (base_date + timedelta(days=random.randint(0, 365))).isoformat() + "Z",
                "original_id": f"event_{i}",
                "event_version": "2.1",
                "raw_hash": f"{random.getrandbits(128):x}"
            },
            "basic_info": {
                "title": f"Evento {i} en {ciudad}",
                "description": f"Evento de categoría {categoria}",
                "status": "confirmed",
                "visibility": "public"
            },
            "classification": {
                "primary_category": categoria,
                "categories": [],
                "tags": []
            },
            "temporal_info": {
                "start": (base_date + timedelta(days=random.randint(0, 365))).isoformat() + "T21:45:00+00:00",
                "end": (base_date + timedelta(days=random.randint(0, 365))).isoformat() + "T23:45:00+00:00",
                "timezone": {
                    "name": "Europe/Madrid",
                    "utc_offset": "+0200"
                },
                "is_recurring": False,
                "is_tba": False
            },
            "spatial_info": {
                "venue": {
                    "name": "Chinois",
                    "full_address": f"{random.choice(ciudades)}",
                    "location": {
                        "latitude": random.uniform(-90, 90),
                        "longitude": random.uniform(-180, 180),
                        "geo_accuracy": "high"
                    }
                },
                "area": {}
            },
            "participants": {},
            "external_references": {
                "urls": [
                    {
                        "type": "source",
                        "url": f"https://predicthq.com/events/event_{i}"
                    }
                ],
                "performers": [],
                "images": []
            },
            "raw_data_metadata": {
                "significant_fields_preserved": [
                    "scope",
                    "entities",
                    "labels"
                ],
                "source_specific": {
                    "scope": "locality",
                    "entity_count": 3
                }
            }
        }
        eventos.append(evento)
        
    return eventos

def generar_escenarios(eventos):
    ciudades = list(set([e["metadata"]["source"] for e in eventos]))
    categorias = list(set([e["classification"]["primary_category"] for e in eventos]))
    
    escenarios = []
    for _ in range(30):
        escenario = {
            "ciudad": random.choice(ciudades),
            "categorias": random.sample(categorias, k=random.randint(1, 3)),
            "duracion_max": random.choice([120, 180, 240]),
            "metadata": {
                "source": "predicthq",
                "processing_timestamp": datetime.now().isoformat() + "Z",
                "original_id": f"escenario_{random.randint(1, 1000)}",
                "event_version": "1.0",
                "raw_hash": f"{random.getrandbits(128):x}"
            }
        }
        escenarios.append(escenario)
        
    return escenarios

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
        def hay_solape(evento1, evento2):
            """Versión optimizada que acepta tanto eventos como índices"""
            if isinstance(evento1, int):
                evento1 = pool[evento1]
            if isinstance(evento2, int):
                evento2 = pool[evento2]
                
            start1 = datetime.fromisoformat(evento1["temporal_info"]["start"])
            end1 = start1 + timedelta(minutes=evento1["temporal_info"]["duration_minutes"])
            
            start2 = datetime.fromisoformat(evento2["temporal_info"]["start"])
            end2 = start2 + timedelta(minutes=evento2["temporal_info"]["duration_minutes"])
            
            return not (end1 <= start2 or end2 <= start1)
        
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
def hay_solape(e1: Dict[str, Any], e2: Dict[str, Any]) -> bool:
    try:
        fecha1 = e1["temporal_info"]["start"]
        duracion1 = e1["temporal_info"].get("duration_minutes", 120)
        fecha2 = e2["temporal_info"]["start"]
        duracion2 = e2["temporal_info"].get("duration_minutes", 120)
        
        dt1 = datetime.fromisoformat(fecha1.replace("Z", "+00:00"))
        dt2 = datetime.fromisoformat(fecha2.replace("Z", "+00:00"))
        
        return not (dt1 + timedelta(minutes=duracion1) <= dt2 or 
                   dt2 + timedelta(minutes=duracion2) <= dt1)
    except Exception:
        return False