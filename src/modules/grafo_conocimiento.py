import os
import json
import networkx as nx
from embedding import load_events_from_folder
from typing import List, Dict, Any

def construir_grafo_conocimiento(folder_path="eventos_mejorados"):
    eventos = load_events_from_folder(folder_path)
    G = nx.MultiDiGraph()

    for evento in eventos:
        eid = evento.get("metadata", {}).get("original_id", evento.get("basic_info", {}).get("title", "evento_desconocido"))
        enombre = evento.get("basic_info", {}).get("title", "Evento sin título")
        ciudad = evento.get("spatial_info", {}).get("area", {}).get("city", "Ciudad desconocida")
        categoria = evento.get("classification", {}).get("primary_category", "Sin categoría")
        performers = evento.get("participants", {}).get("performers", [])

        # Nodo del evento
        G.add_node(enombre, tipo="evento")

        # Nodo de ciudad
        G.add_node(ciudad, tipo="ciudad")
        G.add_edge(enombre, ciudad, tipo="ocurre_en")

        # Nodo de categoría
        G.add_node(categoria, tipo="categoría")
        G.add_edge(enombre, categoria, tipo="es_de_tipo")

        # Nodo de país
        pais = evento.get("spatial_info", {}).get("area", {}).get("country")
        if pais:
            G.add_node(pais, tipo="pais")
            G.add_edge(ciudad, pais, tipo="ubicado_en")

        # Nodo de organizadores
        organizadores = evento.get("participants", {}).get("organizers", [])
        for org in organizadores:
            nombre_org = org.get("name", "Organizador desconocido")
            G.add_node(nombre_org, tipo="organizador")
            G.add_edge(nombre_org, enombre, tipo="organiza")

        # Nodo por cada performer
        for p in performers:
            nombre_p = p.get("name", "Artista desconocido")
            G.add_node(nombre_p, tipo="artista")
            G.add_edge(nombre_p, enombre, tipo="actúa_en")

    print(f"[✔] Grafo generado con {len(G.nodes)} nodos y {len(G.edges)} relaciones.")
    return G

def exportar_grafo(G, output_path="grafo_eventos.graphml"):
    nx.write_graphml(G, output_path)
    print(f"[📦] Grafo exportado a {output_path}")

def get_knowledge_graph(folder_path="eventos_mejorados") -> nx.MultiDiGraph:
    return construir_grafo_conocimiento(folder_path)

# ======================== NUEVAS FUNCIONES DE RAZONAMIENTO AVANZADO ========================

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

def distancia_entre_nodos(G: nx.Graph, nodo_origen: str, nodo_destino: str) -> int:
    """
    Calcula la distancia (número de aristas) más corta entre dos nodos.
    Retorna un entero o None si no hay camino.
    """
    try:
        dist = nx.shortest_path_length(G, source=nodo_origen, target=nodo_destino)
        return dist
    except nx.NetworkXNoPath:
        return None

def enriquecer_resultados_con_razonamiento_avanzado(eventos: list, G: nx.Graph, query: str, k=10):
    pr = calcular_pagerank(G)
    palabras_clave = query.lower().split()
    nodos_relacionados = [n for n in G.nodes if any(pk in n.lower() for pk in palabras_clave)]

    eventos_con_score = []
    for ev in eventos:
        nombre_evento = ev.get("basic_info", {}).get("title", "")
        if nombre_evento not in G:
            score_pr = 0
            score_prox = 100
        else:
            score_pr = pr.get(nombre_evento, 0)
            distancias = []
            for nodo_rel in nodos_relacionados:
                d = distancia_entre_nodos(G, nombre_evento, nodo_rel)
                if d is not None:
                    distancias.append(d)
            score_prox = min(distancias) if distancias else 100

        score_combinado = score_pr - 0.1 * score_prox
        if any(pk in nombre_evento.lower() for pk in palabras_clave):
            score_combinado += 0.2

        eventos_con_score.append((ev, score_combinado))

    eventos_con_score.sort(key=lambda x: x[1], reverse=True)
    return eventos_con_score[:k]  # Aquí retornamos lista de (evento, score)


# Función para generar y exportar grafo con mejoras integradas
def generar_y_exportar_grafo_avanzado(input_folder="eventos_mejorados", output_path="grafo_eventos.graphml") -> str:
    G = construir_grafo_conocimiento(input_folder)
    exportar_grafo(G, output_path)
    pr = calcular_pagerank(G)
    eventos_ordenados = eventos_ordenados_por_pagerank(G)

    resumen = f"Grafo exportado a {output_path} con {len(G.nodes)} nodos y {len(G.edges)} relaciones.\n"
    resumen += f"Top 5 eventos por PageRank:\n"
    for ev, score in eventos_ordenados[:5]:
        resumen += f" - {ev}: {score:.5f}\n"
    return resumen



def enriquecer_resultados_por_similitud(eventos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mejora la relevancia si los eventos comparten artistas o categorías"""
    eventos_enriquecidos = []
    for ev in eventos:
        score = 0
        artistas = {p.get("name") for p in ev.get("participants", {}).get("performers", []) if p.get("name")}
        categoria = ev.get("classification", {}).get("primary_category")

        for otro in eventos:
            if otro == ev:
                continue
            otros_artistas = {p.get("name") for p in otro.get("participants", {}).get("performers", []) if p.get("name")}
            otra_cat = otro.get("classification", {}).get("primary_category")

            if artistas & otros_artistas:
                score += 2
            if categoria and categoria == otra_cat:
                score += 1

        eventos_enriquecidos.append((ev, score))

    eventos_enriquecidos.sort(key=lambda x: x[1], reverse=True)
    return [ev for ev, _ in eventos_enriquecidos]
