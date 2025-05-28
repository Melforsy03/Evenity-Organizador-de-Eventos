from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
from datetime import datetime
import os
from typing import List, Dict, Any, Tuple, Optional
import logging
import json
from dateutil import tz
import pytz
from geopy.distance import geodesic
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import os
def make_timezone_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=tz.UTC)
    return dt

def load_events_from_folder(folder_path):
    events = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    event = json.load(f)
                    events.append(event)
                except Exception as e:
                    print(f"Error leyendo {filename}: {e}")
    print(f"Cargados {len(events)} eventos desde {folder_path}")
    return events

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EventEmbedder:
    def __init__(self, model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.events = []
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "model": model_name,
            "version": "2.0"  # Versión actualizada
        }
        self.model.to("cpu")
        self.shards = {}
    def build_shards(self, eventos, shard_key_func):
        from numpy.linalg import norm

        grupos = defaultdict(list)
        for ev in eventos:
            key = shard_key_func(ev)
            grupos[key].append(ev)

        for key, grupo_eventos in grupos.items():
            textos = [self.build_event_text(ev) for ev in grupo_eventos]
            vecs = self.model.encode(textos, convert_to_numpy=True)
            vecs = vecs / norm(vecs, axis=1, keepdims=True)

            index = faiss.IndexFlatIP(vecs.shape[1])
            index.add(vecs)
            self.shards[key] = (index, grupo_eventos)
    def build_event_text(self, event: Dict[str, Any]) -> str:
        parts = []
        basic = event.get("basic_info", {})
        if basic.get("title"):
            parts.append(f"Título: {basic['title']}")
        if basic.get("description"):
            parts.append(f"Descripción: {basic['description']}")

        classification = event.get("classification", {})
        if classification.get("primary_category"):
            parts.append(f"Categoría: {classification['primary_category']}")
        if classification.get("categories"):
            parts.append(f"Subcategorías: {', '.join(classification['categories'])}")

        spatial = event.get("spatial_info", {})
        venue = spatial.get("venue", {})
        if venue.get("name"):
            parts.append(f"Lugar: {venue['name']}")
        if venue.get("full_address"):
            parts.append(f"Dirección: {venue['full_address']}")
        
        if spatial.get("area", {}).get("city"):
            parts.append(f"Ciudad: {spatial['area']['city']}")

        temporal = event.get("temporal_info", {})
        if temporal.get("start"):
            parts.append(f"Fecha: {temporal['start']}")

        performers = event.get("participants", {}).get("performers", [])
        nombres = [p.get("name", "").strip() for p in performers if p.get("name")]
        if nombres:
            artistas = ", ".join(nombres)
            parts.append(f"Artistas: {artistas}")
        
        organizadores = event.get("participants", {}).get("organizers", [])
        nombres_org = [o.get("name") for o in organizadores if o.get("name")]
        if nombres_org:
            partes_org = ", ".join(nombres_org)
            parts.append(f"Organizadores: {partes_org}")

        # Añadir popularidad si existe
        popularidad = event.get("raw_data_metadata", {}).get("source_specific", {}).get("popularity")
        if popularidad:
            parts.append(f"Popularidad: {popularidad}")

        return ". ".join(parts) + "."

    def filtered_search(self, query: str, city: str = None, user_coords: tuple = None, max_km: int = None, k: int = 5):
            candidatos = self.events

            # Filtrar por ciudad (eventos y sus índices)
            if city:
                candidatos = [e for e in candidatos if e.get("spatial_info", {}).get("area", {}).get("city") == city]

            # Filtrar por distancia
            if user_coords and max_km:
                filtrados = []
                for ev in candidatos:
                    loc = ev.get("spatial_info", {}).get("venue", {}).get("location", {})
                    if loc.get("latitude") and loc.get("longitude"):
                        dist = geodesic(user_coords, (loc["latitude"], loc["longitude"])).km
                        if dist <= max_km:
                            filtrados.append(ev)
                candidatos = filtrados

            if not candidatos:
                return [], []

            # Obtener índices de candidatos para filtrar embeddings y crear subíndice
            idx_candidatos = [self.events.index(ev) for ev in candidatos]

            # Extraer embeddings para candidatos
            embeddings_candidatos = self.embeddings[idx_candidatos]

            # Crear índice temporal FAISS para candidatos
            index_temp = faiss.IndexFlatIP(embeddings_candidatos.shape[1])
            index_temp.add(embeddings_candidatos)

            # Vectorizar consulta
            query_vec = self.model.encode([query], convert_to_numpy=True)
            query_vec = query_vec / np.linalg.norm(query_vec)

            distances, indices = index_temp.search(query_vec, k)

            resultados = [candidatos[i] for i in indices[0] if i < len(candidatos)]
            scores = distances[0].tolist()

            return resultados, scores

    def generate_embeddings(self, events):
            self.events = events
            texts = [self.build_event_text(e) for e in events]
            self.embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            return self.embeddings
    
    def build_index(self, embeddings: np.ndarray, index_type: str = "IVFFlat"):
            dimension = embeddings.shape[1]
            if index_type == "FlatL2":
                self.index = faiss.IndexFlatL2(dimension)
            elif index_type == "IVFFlat":
                nlist = 100
                quantizer = faiss.IndexFlatL2(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                if not self.index.is_trained:
                    self.index.train(embeddings)
            else:
                raise ValueError(f"Tipo de índice no soportado: {index_type}")

            self.index.add(embeddings)
            print(f"Índice FAISS creado con {self.index.ntotal} embeddings")
    def save(self, output_dir: str = "embedding_data"):
            os.makedirs(output_dir, exist_ok=True)
            faiss.write_index(self.index, os.path.join(output_dir, "eventos.index"))
            with open(os.path.join(output_dir, "eventos_metadata.pkl"), "wb") as f:
                pickle.dump({
                    "events": self.events,
                }, f)
            np.save(os.path.join(output_dir, "eventos_embeddings.npy"), self.embeddings)
            print(f"Datos guardados en {output_dir}")

    @classmethod
    def load(cls, input_dir: str = "embedding_data"):
        embedder = cls()
        embedder.index = faiss.read_index(os.path.join(input_dir, "eventos.index"))
        with open(os.path.join(input_dir, "eventos_metadata.pkl"), "rb") as f:
            data = pickle.load(f)
            embedder.events = data["events"]
        embedder.embeddings = np.load(os.path.join(input_dir, "eventos_embeddings.npy"))
        
        # ✅ Construir shards por ciudad
        embedder.build_shards(embedder.events, shard_key_func=lambda ev: ev.get("spatial_info", {}).get("area", {}).get("city", "desconocido"))
        
        print(f"Cargados {len(embedder.events)} eventos y embeddings")
        return embedder

    def search(self, query, shard_key, k=5):
            if shard_key not in self.shards:
                return [], []

            index, eventos = self.shards[shard_key]
            q_vec = self.model.encode([query], convert_to_numpy=True)
            q_vec /= np.linalg.norm(q_vec)

            D, I = index.search(q_vec, k)
            resultados = [eventos[i] for i in I[0]]
            scores = D[0].tolist()
            return resultados, scores

    
    def hybrid_search(self, query: str, keywords: List[str] = None, k: int = 5):
        semantic_results, scores = self.search(query, k*2)
        
        if keywords:
            filtered = []
            for event, score in zip(semantic_results, scores):
                event_text = self.build_event_text(event).lower()
                if any(keyword.lower() in event_text for keyword in keywords):
                    filtered.append((event, score))
            
            filtered.sort(key=lambda x: x[1], reverse=True)
            return [x[0] for x in filtered[:k]], [x[1] for x in filtered[:k]]
        
        return semantic_results[:k], scores[:k]
    
    def recommend_events(self, user_preferences: Dict[str, Any], k: int = 5):
        candidatos = self.events

        city = user_preferences.get("location")
        if city:
            candidatos = [e for e in candidatos if e.get("spatial_info", {}).get("area", {}).get("city") == city]


        time_filter = user_preferences.get("available_dates")
        if time_filter:
            start_date, end_date = time_filter
            candidatos = [e for e in candidatos if self._is_event_in_date_range(e, start_date, end_date)]

        preferred_categories = user_preferences.get("categories", [])
        candidatos.sort(key=lambda e: self._calculate_relevance_score(e, preferred_categories), reverse=True)

        return candidatos[:k]

    def _evento_cercano(self, ev, user_coords, max_km):
        loc = ev.get("spatial_info", {}).get("venue", {}).get("location", {})
        if loc.get("latitude") and loc.get("longitude"):
            dist = geodesic(user_coords, (loc["latitude"], loc["longitude"])).km
            return dist <= max_km
        return False

    
    def _calculate_relevance_score(self, event: Dict[str, Any], preferred_categories: List[str]) -> float:
        score = 0
        event_category = event.get("classification", {}).get("primary_category", "")
        if event_category in preferred_categories:
            score += 2.0
        
        popularity = event.get("raw_data_metadata", {}).get("source_specific", {}).get("popularity", 0)
        score += min(popularity * 0.01, 1.0)
        
        start_time = event.get("temporal_info", {}).get("start")
        if start_time:
            try:
                days_until = (datetime.fromisoformat(start_time) - datetime.now()).days
                score += max(0, (30 - days_until) / 30)
            except:
                pass
        
        return score
    
    def _is_event_in_date_range(self, event: Dict[str, Any], start_date: str, end_date: str) -> bool:
        event_date = event.get("temporal_info", {}).get("start")
        if not event_date:
            return False
        
        try:
            event_dt = datetime.fromisoformat(event_date.replace("Z", "+00:00"))
            start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            return start_dt <= event_dt <= end_dt
        except:
            return False
    def enriquecer_resultados_con_grafo(self, eventos: List[Dict[str, Any]], G, query: str) -> List[Dict[str, Any]]:
            """Ordena eventos según su conectividad en el grafo con la categoría o artista consultado."""
            palabras_clave = query.lower().split()
            eventos_enriquecidos = []

            for ev in eventos:
                score = 0
                nodo_evento = ev.get("basic_info", {}).get("title", "")
                if not G.has_node(nodo_evento):
                    eventos_enriquecidos.append((ev, score))
                    continue

                vecinos = G[nodo_evento]
                for vecino in vecinos:
                    for palabra in palabras_clave:
                        if palabra in vecino.lower():
                            score += 1
                eventos_enriquecidos.append((ev, score))

            eventos_enriquecidos.sort(key=lambda x: x[1], reverse=True)
            return [ev for ev, _ in eventos_enriquecidos]
    def enriquecer_resultados_con_grafo(self, eventos: List[Dict[str, Any]], G, query: str) -> List[Dict[str, Any]]:
        """Ordena eventos según su conectividad en el grafo con la categoría o artista consultado."""
        palabras_clave = query.lower().split()
        eventos_enriquecidos = []

        for ev in eventos:
            score = 0
            nodo_evento = ev.get("basic_info", {}).get("title", "")
            if not G.has_node(nodo_evento):
                eventos_enriquecidos.append((ev, score))
                continue

            vecinos = G[nodo_evento]
            for vecino in vecinos:
                for palabra in palabras_clave:
                    if palabra in vecino.lower():
                        score += 1
            eventos_enriquecidos.append((ev, score))

        eventos_enriquecidos.sort(key=lambda x: x[1], reverse=True)
        return [ev for ev, _ in eventos_enriquecidos]

    def explicar_recomendacion(self, event: Dict[str, Any], preferencias: Dict[str, Any]) -> str:
        explicacion = []

        cat = event.get("classification", {}).get("primary_category")
        if cat in preferencias.get("categories", []):
            explicacion.append(f"✔ Categoría preferida: {cat}")

        ciudad = event.get("spatial_info", {}).get("area", {}).get("city")
        if ciudad == preferencias.get("location"):
            explicacion.append(f"✔ Ubicación coincide: {ciudad}")

        fecha = event.get("temporal_info", {}).get("start")
        fechas_disp = preferencias.get("available_dates")
        if fecha and fechas_disp:
            explicacion.append(f"✔ Está dentro de tus fechas: {fecha}")

        if not explicacion:
            return "⚠️ Recomendado por proximidad semántica general."
        return " · ".join(explicacion)


from crawler import EventScraper
from procesamiento import EventProcessor
def run_embedding():
        # Obtener la carpeta donde está este script que contiene run_embedding
        current_dir = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(current_dir, "../../eventos_mejorados")  # ruta absoluta a la carpeta

        eventos_normalizados = load_events_from_folder(folder_path)

        if not eventos_normalizados:
            print("No se encontraron eventos para procesar.")
            return

        embedder = EventEmbedder()
        embeddings = embedder.generate_embeddings(eventos_normalizados)
        embedder.build_index(embeddings, index_type="IVFFlat")
        embedder.save(output_dir="embedding_data")
        print("Proceso completado: índice creado y guardado.")

def fallback_api_call(query: str, start_date: str = None, end_date: str = None, source="seatgeek"):
    scraper = EventScraper()
    processor = EventProcessor()
    nuevos_eventos = []

    if source == "seatgeek":
        url = "https://api.seatgeek.com/2/events"
        params = {
            "client_id": scraper.sources_config["seatgeek"]["client_id"],
            "q": query,
            "per_page": 10
        }
        if start_date:
            params["datetime_local.gte"] = start_date
        if end_date:
            params["datetime_local.lte"] = end_date

        response = scraper._make_request(url, params=params)
        if response:
            data = response.json()
            for e in data.get("events", []):
                ev = processor.process_event(e, "seatgeek")
                if "error" not in ev:
                    nuevos_eventos.append(ev)

    return nuevos_eventos
