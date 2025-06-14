
"""
Módulo de embedding semántico y búsqueda vectorial (embedding.py)

Este módulo define la clase singleton `EventEmbedder`, que permite:
- Generar representaciones vectoriales (embeddings) de eventos
- Construir y consultar un índice FAISS para búsquedas semánticas
- Filtrar resultados por ciudad, coordenadas o fechas
- Enriquecer y recomendar eventos según preferencias del usuario
- Guardar/cargar datos vectoriales

También incluye funciones auxiliares como `run_embedding()` y `fallback_api_call()` 
para situaciones de regeneración y recuperación en caso de baja cobertura.
"""

from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
from datetime import datetime
import os
import json
import threading
from typing import List, Dict, Any, Optional
import logging
from dateutil import tz
from geopy.distance import geodesic
from collections import defaultdict
import sys
from math import radians, sin, cos, sqrt, atan2


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scraping.crawler import EventScraper
from core.procesamiento import EventProcessor
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

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
def haversine(coord1, coord2):
    # Coordenadas: (lat, lon)
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    R = 6371  # Radio de la Tierra en km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c  # distancia en km
class EventEmbedder:
    # Variables de clase (compartidas por todas las instancias)
    _instance = None
    _lock = threading.Lock()
    _model = None
    _index = None
    _events = None
    _embeddings = None
    _shards = None
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EventEmbedder, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    
    def __init__(self, model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
        with self._lock:
            if not self._initialized:
                logger.info("🔄 Inicializando EventEmbedder...")
                
                # Inicializar variables de instancia
                self.model_name = model_name
                self.model = self._load_model(model_name)
                self.index = None
                self.events = []
                self.shards = {}
                self.metadata = {
                    "created_at": datetime.now().isoformat(),
                    "model": model_name,
                    "version": "2.1"
                }
                
                # Inicializar variables de clase si no existen
                if EventEmbedder._model is None:
                    EventEmbedder._model = self.model
                if EventEmbedder._index is None:
                    EventEmbedder._index = self.index
                if EventEmbedder._events is None:
                    EventEmbedder._events = self.events
                if EventEmbedder._embeddings is None:
                    EventEmbedder._embeddings = None
                if EventEmbedder._shards is None:
                    EventEmbedder._shards = {}
                
                self._initialized = True
                logger.info("✅ EventEmbedder completamente inicializado")

    def _load_model(self, model_name: str):
        """Carga el modelo de SentenceTransformer"""
        logger.info(f"⏳ Cargando modelo {model_name}...")
        model = SentenceTransformer(model_name)
        model.to("cpu")  # Usar CPU por defecto
        logger.info(f"✅ Modelo {model_name} cargado")
        return model

    def generate_embeddings(self, events: List[Dict[str, Any]]):
        """Genera embeddings para los eventos y los almacena"""
        if not events:
            raise ValueError("La lista de eventos está vacía")
        
        logger.info("🧠 Generando embeddings...")
        self.events = events
        EventEmbedder._events = events
        
        texts = [self.build_event_text(e) for e in events]
        EventEmbedder._embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # Normalizar embeddings
        norms = np.linalg.norm(EventEmbedder._embeddings, axis=1, keepdims=True)
        EventEmbedder._embeddings = EventEmbedder._embeddings / norms
        
        logger.info(f"✅ Embeddings generados. Dimensión: {EventEmbedder._embeddings.shape}")
        return EventEmbedder._embeddings
    def normalize_embeddings(self, embeddings):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return embeddings / norms
    
    def cargar_embeddings(self, archivo: str):
        with open(archivo, "rb") as f:
            data = pickle.load(f)
            self.events= data["eventos"]
            self.embeddings = np.array(data["embeddings"]).astype("float32")
            self.embeddings = self.normalize_embeddings(self.embeddings)

            dim = self.embeddings.shape[1]
            self.index = faiss.IndexHNSWFlat(dim, 32)
            self.index.hnsw.efSearch = 64
            self.index.add(self.embeddings)

        print(f"✅ {len(self.events)} eventos cargados y embebidos.")
        
    def cargar_shards(self, carpeta: str):
        for archivo in os.listdir(carpeta):
            if archivo.endswith(".pkl"):
                ciudad = archivo[:-4]
                with open(os.path.join(carpeta, archivo), "rb") as f:
                    data = pickle.load(f)
                    emb = np.array(data["embeddings"]).astype("float32")
                    emb = self.normalize_embeddings(emb)

                    idx = faiss.IndexFlatIP(emb.shape[1])
                    idx.add(emb)
                    self.shards[ciudad] = (idx, data["eventos"])

        print(f"✅ {len(self.shards)} shards de ciudades cargados.")
        
    def build_index(self, embeddings: np.ndarray, index_type: str = "IVFFlat"):
        """Construye el índice FAISS para búsqueda eficiente"""
        if embeddings is None:
            raise ValueError("Primero debe generar embeddings")
            
        logger.info(f"🔨 Construyendo índice {index_type}...")
        dimension = embeddings.shape[1]
        
        if index_type == "FlatL2":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "IVFFlat":
            nlist = min(100, len(embeddings))
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            if not self.index.is_trained:
                self.index.train(embeddings)
        else:
            raise ValueError(f"Tipo de índice no soportado: {index_type}")
        
        self.index.add(embeddings)
        EventEmbedder._index = self.index
        logger.info(f"✅ Índice construido. Total embeddings: {self.index.ntotal}")

    def filtered_search(
        self,
        query: str,
        city: str = None,
        user_coords: tuple = None,
        max_km: int = None,
        category: str = None,
        start_date: str = None,
        end_date: str = None,
        k: int = 100,
        page: int = 1,
        limit: int = 10
    ) -> list:
        all_eventos = []

        for shard, (index, eventos) in self.shards.items():
            q_vec = self.model.encode([query], convert_to_numpy=True)
            q_vec = self.normalize_embeddings(q_vec.astype("float32"))
            D, I = index.search(q_vec, k)
            indices = I[0]
            scores = D[0]

            resultados = [
                {**eventos[i], "score": float(scores[j])}
                for j, i in enumerate(indices)
                if i >= 0 and i < len(eventos)
            ]

            # Filtro por ciudad (desde el shard)
            if city and shard != city:
                continue

            # Filtro por coordenadas si aplica
            if user_coords and max_km:
                resultados = [
                    ev for ev in resultados
                    if self._is_event_near(ev, user_coords, max_km)
                ]

            # Filtro por categoría
            if category:
                resultados = [
                    ev for ev in resultados
                    if category.lower() in (
                        ev.get("classification", {}).get("primary_category", "") or ""
                    ).lower()
                ]

            # Filtro por fecha
            if start_date or end_date:
                def parse_date(d):
                    try:
                        dt = datetime.fromisoformat(d.replace("Z", "+00:00"))
                        return dt if dt.tzinfo else dt.replace(tzinfo=tz.UTC)
                    except:
                        return None

                start_dt = parse_date(start_date) if start_date else None
                end_dt = parse_date(end_date) if end_date else None

                resultados = [
                    ev for ev in resultados
                    if self._event_in_range(ev, start_dt, end_dt)
                ]

            all_eventos.extend(resultados)

        # Ordenar por score descendente
        all_eventos.sort(key=lambda x: x["score"], reverse=True)

        # Paginación
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        return all_eventos[start_idx:end_idx]

    def _event_in_range(self, event, start_dt, end_dt):
        fecha_raw = event.get("temporal_info", {}).get("start")
        if not fecha_raw:
            return False

        try:
            fecha = datetime.fromisoformat(fecha_raw.replace("Z", "+00:00"))
            if fecha.tzinfo is None:
                fecha = fecha.replace(tzinfo=tz.UTC)

            if start_dt and fecha < start_dt:
                return False
            if end_dt and fecha > end_dt:
                return False
            return True
        except Exception:
            return False
    
    def _is_event_near(self, event: Dict[str, Any], user_coords: tuple, max_km: float) -> bool:
        """Verifica si un evento está cerca de las coordenadas del usuario"""
        loc = event.get("spatial_info", {}).get("venue", {}).get("location", {})
        if not (loc.get("latitude") and loc.get("longitude")):
            return False
            
        event_coords = (loc["latitude"], loc["longitude"])
        distance = geodesic(user_coords, event_coords).km
        return distance <= max_km

    @classmethod
    def get_embeddings(cls):
        if cls._embeddings is None:
            raise ValueError("Embeddings no han sido generados. Llama a generate_embeddings() primero.")
        return cls._embeddings

    @classmethod
    def clear_cache(cls):
        cls._model = None
        cls._index = None
        cls._events = None
        cls._embeddings = None
        cls._instance = None
        print("✅ Caché limpiada")

    def build_shards(self, eventos, shard_key_func):
        if not self.shards:
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
            print(f"✅ Shards construidos y almacenados en caché.")
        else:
            print(f"⚠️ Usando shards previamente construidos.")

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

        popularidad = event.get("raw_data_metadata", {}).get("source_specific", {}).get("popularity")
        if popularidad:
            parts.append(f"Popularidad: {popularidad}")

        return ". ".join(parts) + "."
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise RuntimeError("EventEmbedder no ha sido inicializado")
        return cls._instance
    
    def save(self, output_dir: str = "embedding_data"):
        os.makedirs(output_dir, exist_ok=True)
        
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(output_dir, "eventos.index"))
        
        if self.events is not None:
            with open(os.path.join(output_dir, "eventos_metadata.pkl"), "wb") as f:
                pickle.dump({"events": self.events}, f)
        
        if EventEmbedder._embeddings is not None:
            np.save(os.path.join(output_dir, "eventos_embeddings.npy"), EventEmbedder._embeddings)

        print(f"✅ Datos guardados en {output_dir}")

    @classmethod
    def load(cls, input_dir: str = "embedding_data", model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
        if cls._instance is None:
            cls._instance = cls(model_name)
        
        embedder = cls._instance
        
        path_index = os.path.join(input_dir, "eventos.index")
        path_metadata = os.path.join(input_dir, "eventos_metadata.pkl")
        path_embeddings = os.path.join(input_dir, "eventos_embeddings.npy")

        if not all(os.path.exists(p) for p in [path_index, path_metadata, path_embeddings]):
            raise FileNotFoundError("No se encontraron archivos de embeddings necesarios")

        embedder.index = faiss.read_index(path_index)
        EventEmbedder._index = embedder.index

        with open(path_metadata, "rb") as f:
            data = pickle.load(f)
            embedder.events = data["events"]
            EventEmbedder._events = embedder.events

        EventEmbedder._embeddings = np.load(path_embeddings, allow_pickle=True)

        embedder.build_shards(
            embedder.events,
            shard_key_func=lambda ev: ev.get("spatial_info", {}).get("area", {}).get("city", "desconocido")
        )

        print(f"✅ Cargados {len(embedder.events)} eventos desde {input_dir}")
        return embedder

    def search(self, query: str, shard_key: str = None, k: int = 30):
        q_vec = self.model.encode([query], convert_to_numpy=True)
        q_vec = self.normalize_embeddings(q_vec.astype("float32"))

        resultados = []

        if shard_key and shard_key in self.shards:
            index, eventos = self.shards[shard_key]
            D, I = index.search(q_vec, k)
            indices = I[0]
            scores = D[0]

            resultados = [
                {**eventos[i], "score": float(scores[j])}
                for j, i in enumerate(indices) if 0 <= i < len(eventos)
            ]
            return resultados, scores.tolist()

        # Si no se especifica shard_key o no es válido, buscar en todos los shards
        all_resultados = []
        for ciudad, (index, eventos) in self.shards.items():
            D, I = index.search(q_vec, k)
            indices = I[0]
            scores = D[0]

            for j, i in enumerate(indices):
                if 0 <= i < len(eventos):
                    all_resultados.append({**eventos[i], "score": float(scores[j])})

        return all_resultados[:k], []

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
            def make_aware(dt):
                dt_obj = datetime.fromisoformat(dt.replace("Z", "+00:00"))
                return dt_obj if dt_obj.tzinfo else dt_obj.replace(tzinfo=tz.UTC)

            event_dt = make_aware(event_date)
            start_dt = make_aware(start_date)
            end_dt = make_aware(end_date)

            return start_dt <= event_dt <= end_dt
        except Exception as e:
            print(f"[Error fecha] {e}")
            return False
    def enriquecer_resultados_con_grafo(self, eventos: List[Dict[str, Any]], G, query: str) -> List[Dict[str, Any]]:
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

    def load_index(self, path: str):
        if os.path.exists(path):
            self.index = faiss.read_index(path)
            EventEmbedder._index = self.index
            print(f"✅ Índice FAISS cargado desde {path}")
        else:
            raise FileNotFoundError(f"No se encontró el índice FAISS en {path}")
    def add_new_events(self, nuevos_eventos: List[Dict[str, Any]]):
        textos_nuevos = [self.build_event_text(ev) for ev in nuevos_eventos]
        nuevos_embeddings = self.model.encode(textos_nuevos, convert_to_numpy=True)

        # Añadir al índice existente
        if self.index is not None:
            self.index.add(nuevos_embeddings)
        else:
            raise ValueError("El índice FAISS no ha sido inicializado")

        # Añadir a eventos y embeddings existentes
        self.events.extend(nuevos_eventos)
        if EventEmbedder._embeddings is not None:
            EventEmbedder._embeddings = np.vstack([EventEmbedder._embeddings, nuevos_embeddings])
        else:
            EventEmbedder._embeddings = nuevos_embeddings

        print(f"✅ {len(nuevos_eventos)} nuevos eventos añadidos al índice.")
    def save_event_data(self, output_dir: str = "embedding_data"):
        os.makedirs(output_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(output_dir, "eventos.index"))
        with open(os.path.join(output_dir, "eventos_metadata.pkl"), "wb") as f:
            pickle.dump({"events": self.events}, f)
        np.save(os.path.join(output_dir, "eventos_embeddings.npy"), EventEmbedder._embeddings)
        print(f"✅ Eventos y embeddings guardados incrementalmente.")

def run_embedding():
    if EventEmbedder._embeddings is not None:
        print("⚠️ Usando embeddings en caché. Omitiendo generación.")
        return EventEmbedder._instance

    current_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_dir, "../../eventos_mejorados")
    
    eventos_normalizados = load_events_from_folder(folder_path)
    if not eventos_normalizados:
        print("No se encontraron eventos para procesar.")
        return None
    
    embedder = EventEmbedder()
    embedder.generate_embeddings(eventos_normalizados)
    embedder.build_index(EventEmbedder._embeddings, index_type="IVFFlat")
    embedder.save(output_dir="embedding_data")
    
    return embedder
def ordenar_y_paginar(eventos: list, page: int = 1, limit: int = 10) -> list:
    """
    Ordena una lista de eventos primero por score (descendente),
    luego por fecha de inicio (ascendente), y aplica paginación.
    """

    def extraer_fecha(ev):
        fecha_str = ev.get("temporal_info", {}).get("start")
        try:
            return datetime.fromisoformat(fecha_str)
        except:
            return datetime.max  # Coloca eventos sin fecha al final

    # Ordenar primero por mayor score, luego por fecha más próxima
    eventos_ordenados = sorted(
        eventos,
        key=lambda ev: (-ev.get("score", 0), extraer_fecha(ev))
    )

    # Aplicar paginación
    inicio = (page - 1) * limit
    fin = inicio + limit
    return eventos_ordenados[inicio:fin]

def fallback_api_call(query: str, start_date: str = None, end_date: str = None,
                      ciudad: str = None, categoria: str = None, source="seatgeek"):
    scraper = EventScraper()
    processor = EventProcessor()
    nuevos_eventos = []
    
   
    if source == "seatgeek":
        url = "https://api.seatgeek.com/2/events"
        params = {
            "client_id": scraper.sources_config["seatgeek"]["client_id"],
            #"q": query,
            "per_page": 10
        }
        if start_date:
            params["datetime_local.gte"] = start_date
        if end_date:
            params["datetime_local.lte"] = end_date
        if ciudad:
            params["venue.city"] = ciudad
        if categoria:
            params["taxonomies.name"] = categoria

        response = scraper._make_request(url, params=params)
        if response:
            data = response.json()
            for e in data.get("events", []):
                ev = processor.process_event(e, "seatgeek")
                if "error" not in ev:
                    nuevos_eventos.append(ev)

    elif source == "predicthq":
        url = f"{scraper.sources_config['predicthq']['base_url']}/events/"
        headers = {
            "Authorization": f"Bearer {scraper.sources_config['predicthq']['api_key']}",
            "Accept": "application/json"
        }
        params = {
            #"q": query,
            "limit": 10,
            #"country": "US"
        }
        if start_date:
            params["active.gte"] = start_date
        if end_date:
            params["active.lte"] = end_date
        if categoria:
            params["category"] = categoria
        if ciudad:
            params["place.scope"] = ciudad

        response = scraper._make_request(url, params=params, headers=headers)
        if response:
            data = response.json()
            for e in data.get("results", []):
                ev = processor.process_event(e, "predicthq")
                if "error" not in ev:
                    nuevos_eventos.append(ev)

    elif source == "ticketmaster":
        url = f"{scraper.sources_config['ticketmaster']['base_url']}/events.json"
        params = {
            "apikey": scraper.sources_config["ticketmaster"]["api_key"],
            #"keyword": query,
            #"countryCode": "US",
            "size": 10
        }
        if start_date:
            params["startDateTime"] = start_date + "T00:00:00Z"
        if end_date:
            params["endDateTime"] = end_date + "T23:59:59Z"
        if ciudad:
            params["city"] = ciudad
        if categoria:
            params["classificationName"] = categoria

        response = scraper._make_request(url, params=params)
        if response:
            data = response.json()
            for e in data.get("_embedded", {}).get("events", []):
                ev = processor.process_event(e, "ticketmaster")
                if "error" not in ev:
                    nuevos_eventos.append(ev)

    # 🔐 Garantizar máximo de 10
    return nuevos_eventos[:10]
