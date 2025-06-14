# servidor_api.py
"""
Módulo de endpoints principales de la API (servidor_api.py)

Versión corregida con:
- Mejor manejo de concurrencia
- Sincronización adecuada de recursos compartidos
- Mejor serialización de eventos
- Manejo robusto de errores
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from flask import Flask, request, jsonify
from core.embedding import EventEmbedder
from agentes.sistema_multiagente import Mensaje, obtener_bandeja
import threading
from functools import wraps
import time
import queue
from api.servidor_base import app, logger , response_queues , response_queues_lock
from api.contexto_global import obtener_bandeja
import uuid
from datetime import datetime, date , timezone
from copy import deepcopy

_embedder = None
_embedder_lock = threading.Lock()
_initialized = False
resultados_api = {}
resultados_api_lock = threading.Lock()

def get_response_queues():
    return response_queues, response_queues_lock

def get_embedder(allow_empty=True, max_wait=30):
    global _embedder, _initialized
    if _initialized:
        return _embedder

    start_time = time.time()
    while time.time() - start_time < max_wait:
        with _embedder_lock:
            if not _initialized:
                try:
                    logger.info("Initializing embedder...")
                    _embedder = EventEmbedder._instance
                    try:
                        _embedder.load("embedding_data")
                        logger.info("Embedder data loaded successfully")
                    except FileNotFoundError:
                        if allow_empty:
                            logger.warning("No embedding data found, initializing empty embedder")
                        else:
                            continue
                    _initialized = True
                    return _embedder
                except Exception as e:
                    logger.error(f"Error initializing embedder: {str(e)}")
                    time.sleep(1)
    raise TimeoutError("Timeout waiting for embedder initialization")

def ensure_initialized(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            get_embedder(allow_empty=True)
        except Exception as e:
            return jsonify({
                "status": "initializing",
                "message": "Embedder being initialized",
                "details": str(e)
            }), 503
        return f(*args, **kwargs)
    return decorated_function

def limpiar_eventos(lista):
    """Limpia los eventos para serialización segura"""
    nuevos = []
    for ev in lista:
        try:
            nuevo = deepcopy(ev)
            # Eliminar campos problemáticos
            for key in list(nuevo.keys()):
                if key.startswith('__'):  # Eliminar todos los campos internos
                    del nuevo[key]
                elif isinstance(nuevo[key], (datetime, date)):
                    nuevo[key] = str(nuevo[key])  # Convertir fechas a string
            nuevos.append(nuevo)
        except Exception as e:
            logger.error(f"Error limpiando evento: {e}")
    return nuevos

def enviar_y_esperar_respuesta(receptor, contenido, respuesta_key, timeout=60):
    bandeja_destino = obtener_bandeja(receptor)
    if bandeja_destino is None:
        raise RuntimeError(f"❌ Bandeja para '{receptor}' no encontrada")

    bandeja_api = obtener_bandeja("api")
    if bandeja_api is None:
        raise RuntimeError("❌ Bandeja de la API no está registrada")

    logger.info(f"📨 [API] Enviando a {receptor}: {contenido}")
    bandeja_destino.put(Mensaje("api", receptor, contenido))

    inicio = time.time()
    while time.time() - inicio < timeout:
        try:
            respuesta = bandeja_api.get(timeout=1)
            if isinstance(respuesta, Mensaje):
                if (respuesta.receptor == "api" and
                    isinstance(respuesta.contenido, dict) and
                    respuesta.contenido.get("key") == respuesta_key):
                    return respuesta.contenido.get("data")
                else:
                    # No era nuestra respuesta, lo volvemos a poner
                    bandeja_api.put(respuesta)
        except queue.Empty:
            continue
    raise TimeoutError(f"No se recibió respuesta del agente '{receptor}' para clave '{respuesta_key}'")

@app.route("/status")
@ensure_initialized
def status():
    embedder = get_embedder(allow_empty=True)
    return jsonify({
        "status": "operational",
        "events_loaded": len(embedder.events),
        "model_ready": embedder.model is not None
    })

@app.route("/init_status")
def init_status():
    return jsonify({
        "initialized": _initialized,
        "has_embeddings": _embedder.events is not None if _initialized else False
    })

@app.route("/ciudades", methods=["GET"])
@ensure_initialized
def get_ciudades():
    embedder = get_embedder()
    ciudades = set()
    for ev in embedder.events:
        if ev.get("spatial_info", {}).get("area", {}).get("city"):
            ciudades.add(ev["spatial_info"]["area"]["city"])
    return jsonify(sorted(ciudades))

@app.route("/categorias", methods=["GET"])
@ensure_initialized
def get_categorias():
    embedder = get_embedder()
    categorias = set()
    for ev in embedder.events:
        if ev.get("classification", {}).get("primary_category"):
            categorias.add(ev["classification"]["primary_category"])
    return jsonify(sorted(categorias))

@app.route("/buscar", methods=["POST"])
@ensure_initialized
def buscar():
    try:
        data = request.json or {}
        clave_respuesta = data.get("respuesta") or f"res_busqueda_{uuid.uuid4().hex}"
        
        # Crear una cola específica para esta solicitud
        with response_queues_lock:
            response_queues[clave_respuesta] = queue.Queue()
            response_queue = response_queues[clave_respuesta]

        contenido = {
            "query": data.get("query", ""),
            "ciudad": data.get("ciudad"),
            "categoria": data.get("categoria"),
            "fecha_inicio": data.get("fecha_inicio"),
            "fecha_fin": data.get("fecha_fin"),
            "respuesta": clave_respuesta
        }

        logger.info(f"📨 [API] Enviando a busqueda_interactiva: {contenido}")
        
        # Enviar mensaje
        bandeja_busqueda = obtener_bandeja("busqueda_interactiva")
        if bandeja_busqueda:
            bandeja_busqueda.put(Mensaje(
                emisor="api",
                receptor="busqueda_interactiva",
                contenido=contenido
            ))
        else:
            logger.error("❌ Bandeja de búsqueda no disponible")
            return jsonify({"status": "error", "mensaje": "Servicio no disponible"}), 503

        # Esperar respuesta en la cola específica
        try:
            respuesta = response_queue.get(timeout=60)  # 60 segundos timeout
            eventos_limpios = limpiar_eventos(respuesta.get("data", []))
            
            logger.info(f"✅ [API] Enviando {len(eventos_limpios)} eventos a frontend")
            return jsonify({
                "status": "ok",
                "eventos": eventos_limpios,
                "mensaje": respuesta.get("mensaje", ""),
                "total": len(eventos_limpios)
            })
        except queue.Empty:
            logger.error(f"❌ [API] Timeout esperando resultados para {clave_respuesta}")
            return jsonify({
                "status": "error",
                "mensaje": "El agente de búsqueda no respondió a tiempo"
            }), 504
        finally:
            with response_queues_lock:
                response_queues.pop(clave_respuesta, None)

    except Exception as e:
        logger.error(f"❌ Error crítico en /buscar: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "mensaje": "Error interno del servidor"
        }), 500
def to_aware(iso_str):
    dt = datetime.fromisoformat(iso_str)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

@app.route("/agenda", methods=["POST"])
@ensure_initialized
def generar_agenda_optima():
    try:
        data = request.json or {}
        eventos_entrada = data.get("eventos", [])
        fecha_inicio = data.get("fecha_inicio")
        fecha_fin = data.get("fecha_fin")

        preferencias = {
            "location": data.get("ciudad"),
            "categories": data.get("categorias", []),
        }

        if fecha_inicio and fecha_fin:
            preferencias["available_dates"] = (
                to_aware(fecha_inicio),
                to_aware(fecha_fin)
            )

        clave_respuesta = "res_agenda_" + uuid.uuid4().hex
        with response_queues_lock:
            response_queues[clave_respuesta] = queue.Queue()
        contenido = {
            "preferencias": preferencias,
            "eventos_filtrados": eventos_entrada,
            "respuesta": clave_respuesta
        }

        bandeja = obtener_bandeja("optimizador")
        if not bandeja:
            return jsonify({"error": "Optimizador no disponible"}), 500
        bandeja.put(Mensaje("api", "optimizador", contenido))

        try:
            respuesta = response_queues[clave_respuesta].get(timeout=60)
        except queue.Empty:
            return jsonify({"error": "Timeout generando agenda"}), 504

        agenda = respuesta.get("data", {}).get("agenda", [])
        score = respuesta.get("data", {}).get("score", 0)

        agenda_simple = limpiar_eventos(agenda)

        return jsonify({
            "agenda": agenda_simple,
            "count": len(agenda_simple),
            "score": score
        })

    except Exception as e:
        logger.error(f"Error en /agenda: {e}", exc_info=True)
        return jsonify({"error": "Error interno"}), 500