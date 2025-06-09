"""
Módulo de construcción y enriquecimiento del grafo de conocimiento (grafo.py)

Este módulo utiliza `networkx` para construir un grafo dirigido multidimensional de eventos,
relacionándolos con ciudades, categorías, artistas, fechas, festividades y similitudes semánticas
basadas en embeddings. También incluye funciones para enriquecer los resultados de búsqueda,
calcular centralidades y exportar el grafo.

Funciones principales:
- `construir_grafo_conocimiento()`
- `enriquecer_resultados_con_razonamiento_avanzado()`
- `generar_y_exportar_grafo_avanzado()`
- `eventos_ordenados_por_pagerank()`
- `enriquecer_resultados_por_similitud()`
"""

import os
import json
import networkx as nx
from typing import List, Dict, Any
import networkx as nx
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.embedding import EventEmbedder

def construir_grafo_conocimiento():
    embedder = EventEmbedder._instance or EventEmbedder()  
    eventos = embedder.events
    G = nx.MultiDiGraph()
    # Diccionario ampliado con más estaciones y festividades
    estaciones_y_fiestas = {
        "invierno": ["12-21", "01-21", "02-21"],  
        "primavera": ["03-21", "04-21", "05-21"],  
        "verano": ["06-21", "07-21", "08-21"],
        "otoño": ["09-21", "10-21", "11-21"],  
        "navidad": ["12-25"], 
        "ano_nuevo": ["01-01"], 
        "halloween": ["10-31"],  
        "san_valentin": ["02-14"], 
        "dia_de_muertos": ["11-01", "11-02"],  
        "ramadan": ["03-23", "04-21"],  
        "viernes_santo": ["04-15"], 
        
    }

    for ev in eventos:
        ev_id = ev.get("id") or ev.get("basic_info", {}).get("title", "unknown_event")
        G.add_node(ev_id, tipo="evento", data=json.dumps(ev)) 

        info = ev.get("basic_info", {})
        cat = ev.get("classification", {})
        spa = ev.get("spatial_info", {})
        tmp = ev.get("temporal_info", {})

        ciudad = spa.get("area", {}).get("city")
        if ciudad:
            G.add_node(ciudad, tipo="ciudad")
            G.add_edge(ev_id, ciudad, tipo="ocurre_en")

        pais = spa.get("area", {}).get("country")
        if pais:
            G.add_node(pais, tipo="pais")
            G.add_edge(ev_id, pais, tipo="ocurre_en")

        categoria = cat.get("primary_category")
        if categoria:
            G.add_node(categoria, tipo="categoría")
            G.add_edge(ev_id, categoria, tipo="es_de_categoria")

        artista = info.get("artist")
        if artista:
            G.add_node(artista, tipo="artista")
            G.add_edge(ev_id, artista, tipo="tiene_artista")

        organizador = info.get("organizer")
        if organizador:
            G.add_node(organizador, tipo="organizador")
            G.add_edge(ev_id, organizador, tipo="organizado_por")

        venue = spa.get("venue", {}).get("name")
        if venue:
            G.add_node(venue, tipo="venue")
            G.add_edge(ev_id, venue, tipo="ocurre_en_venue")

        fecha = tmp.get("start")
        if fecha:
            G.add_node(fecha, tipo="fecha")
            G.add_edge(ev_id, fecha, tipo="ocurre_en_fecha")

            # Relacionar con estaciones y festividades
            for estacion, fechas in estaciones_y_fiestas.items():
                if any(fecha.startswith(d) for d in fechas):
                    G.add_node(estacion, tipo="estacion_festividad")
                    G.add_edge(ev_id, estacion, tipo="ocurre_en_estacion")

    # Relaciones de similitud entre eventos
    textos = [embedder.build_event_text(ev) for ev in eventos]
    embeddings = embedder.model.encode(textos, convert_to_numpy=True, normalize_embeddings=True)
    umbral_sim = 0.88

    for i in range(len(eventos)):
        for j in range(i + 1, len(eventos)):
            sim = np.dot(embeddings[i], embeddings[j])
            if sim > umbral_sim:
                id1 = eventos[i].get("id") or eventos[i].get("basic_info", {}).get("title", f"ev_{i}")
                id2 = eventos[j].get("id") or eventos[j].get("basic_info", {}).get("title", f"ev_{j}")
                G.add_edge(id1, id2, tipo="similar_a", peso=sim)

    return G

def enriquecer_resultados_con_razonamiento_avanzado(eventos, grafo, consulta, k=10):
    consulta = consulta.lower()
    resultados_con_score = []

    pagerank = nx.pagerank(grafo) if len(grafo.nodes) > 0 else {}

    for ev in eventos:
        ev_id = ev.get("id") or ev.get("basic_info", {}).get("title", "unknown_event")
        score = 1.0

        if grafo.has_node(ev_id):
            for vecino in grafo.neighbors(ev_id):
                edge_data = grafo.get_edge_data(ev_id, vecino)
                if not edge_data:
                    continue

                for _, attrs in edge_data.items():
                    tipo = attrs.get("tipo")
                    if tipo == "similar_a":
                        score += attrs.get("peso", 0.1)
                    elif tipo in ["ocurre_en", "ocurre_en_venue", "ocurre_en_fecha"] and consulta in vecino.lower():
                        score += 0.25
                    elif tipo in ["es_de_categoria", "tiene_artista", "organizado_por"] and consulta in vecino.lower():
                        score += 0.2

        # Bonus por centralidad
        score += pagerank.get(ev_id, 0)

        resultados_con_score.append((ev, score))

    resultados_con_score.sort(key=lambda x: x[1], reverse=True)
    return resultados_con_score[:k]

def exportar_grafo(G, output_path="grafo_eventos.graphml"):
    nx.write_graphml(G, output_path)
    print(f"[📦] Grafo exportado a {output_path}")

def get_knowledge_graph() -> nx.MultiDiGraph:
    return generar_y_exportar_grafo_avanzado()

def calcular_pagerank(G: nx.Graph, alpha=0.85) -> dict:
    """
    Calcula PageRank para todos los nodos del grafo.
    """
    pr = nx.pagerank(G, alpha=alpha)
    return pr

def eventos_ordenados_por_pagerank(G: nx.Graph):
    """
    Devuelve la lista de eventos ordenados por su score de PageRank descendente.
    """
    pr = calcular_pagerank(G)
    eventos = {n: score for n, score in pr.items() if G.nodes[n].get("tipo") == "evento"}
    ordenados = sorted(eventos.items(), key=lambda x: x[1], reverse=True)
    return ordenados  # Lista de (evento, score)

def generar_y_exportar_grafo_avanzado(output_path="grafo_eventos.graphml") -> str:
    """
    Construye el grafo de conocimiento y lo exporta en formato GraphML a la carpeta /grafo en la raíz del proyecto.

    Args:
        output_path (str): Nombre del archivo (sin ruta completa).

    Returns:
        str: Resumen textual del grafo generado.
    """
    # Subir desde src/core/ hasta la raíz y apuntar a carpeta /grafo
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    output_file = os.path.join(project_root, "grafo", output_path)

    # Asegurar que la carpeta exista
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Generar y exportar el grafo
    G = construir_grafo_conocimiento()
    exportar_grafo(G, output_file)
    eventos_ordenados = eventos_ordenados_por_pagerank(G)

    resumen = f"Grafo exportado a {output_file} con {len(G.nodes)} nodos y {len(G.edges)} relaciones.\n"
    resumen += f"Top 5 eventos por PageRank:\n"
    for ev, score in eventos_ordenados[:5]:
        resumen += f" - {ev}: {score:.5f}\n"
    return resumen

