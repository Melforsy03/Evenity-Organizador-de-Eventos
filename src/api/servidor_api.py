# servidor_api.py
"""
Módulo de endpoints principales de la API (servidor_api.py)

Define todos los endpoints REST del sistema, incluyendo:
- Verificación de estado de inicialización
- Consulta de ciudades y categorías disponibles
- Búsqueda semántica de eventos
- Generación de agenda óptima

Este módulo se integra con el sistema multiagente a través de `get_bandeja_global()` y `Mensaje`.
También inicializa el `EventEmbedder` si no está cargado.
"""

from flask import Flask, request, jsonify
from core.embedding import EventEmbedder
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agentes.sistema_multiagente import  Mensaje 
import threading
from functools import wraps
import time
import queue
from servidor_base import app, logger
from contexto_global import get_bandeja_global

_embedder = None
_embedder_lock = threading.Lock()
_initialized = False

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
                    _embedder = EventEmbedder()
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

# Función de comunicación con los agentes
def enviar_y_esperar_respuesta(receptor, contenido, respuesta_key, timeout=10):
    bandeja = get_bandeja_global()
    bandeja.put(Mensaje("api", receptor, contenido))
    inicio = time.time()
    while time.time() - inicio < timeout:
        try:
            respuesta = bandeja.get_nowait()
            if respuesta.receptor == "api" and isinstance(respuesta.contenido, dict) and respuesta.contenido.get("key") == respuesta_key:
                return respuesta.contenido.get("data")
        except queue.Empty:
            time.sleep(0.2)
    raise TimeoutError(f"No se recibió respuesta del agente '{receptor}' para '{respuesta_key}'")

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
        contenido = {
            "query": data.get("query", ""),
            "ciudad": data.get("ciudad"),
            "categoria": data.get("categoria"),
            "fecha_inicio": data.get("fecha_inicio"),
            "fecha_fin": data.get("fecha_fin"),
            "respuesta": "res_busqueda"
        }
        resultado = enviar_y_esperar_respuesta("busqueda_interactiva", contenido, "res_busqueda")
        return jsonify(resultado)
    except TimeoutError as e:
        return jsonify({"error": "Agente de búsqueda no respondió a tiempo"}), 504
    except Exception as e:
        logger.error(f"Error en /buscar: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/agenda", methods=["POST"])
@ensure_initialized
def generar_agenda_optima():
    try:
        data = request.json or {}
        preferencias = {
            "location": data.get("ciudad"),
            "categories": data.get("categorias", []),
            "available_dates": (
                data.get("fecha_inicio"),
                data.get("fecha_fin")
            ) if data.get("fecha_inicio") and data.get("fecha_fin") else None
        }
        contenido = {
            "preferencias": preferencias,
            "respuesta": "res_agenda"
        }
        resultado = enviar_y_esperar_respuesta("optimizador", contenido, "res_agenda")
        return jsonify({
            "agenda": resultado.get("agenda", []),
            "count": len(resultado.get("agenda", [])),
            "score": resultado.get("score", 0)
        })
    except TimeoutError as e:
        return jsonify({"error": "Agente optimizador no respondió a tiempo"}), 504
    except Exception as e:
        logger.error(f"Error en /agenda: {str(e)}")
        return jsonify({"error": str(e)}), 500

