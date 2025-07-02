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
import requests  
_embedder = None
_embedder_lock = threading.Lock()
_initialized = False
resultados_api = {}
resultados_api_lock = threading.Lock()
import google.generativeai as genai



# Tu clave API de Gemini
GEMINI_API_KEY = "AIzaSyAkwcMwEZgmqqUAIiln89CJAD3mhaeXufQ"

# Configura la API de Gemini una sola vez al inicio de tu aplicación
genai.configure(api_key=GEMINI_API_KEY)

# Define el modelo que quieres usar (por ejemplo, "gemini-pro" para texto general)
# Puedes cambiarlo a "gemini-1.5-pro-latest" o "gemini-1.0-pro" si lo prefieres
GEMINI_MODEL = "gemini-1.5-flash"
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

import unicodedata

def normalizar(texto):
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )

def es_pregunta_valida_llm(pregunta: str) -> tuple[bool, list[str]]:
    """
    Valida si una pregunta es pertinente y procesable por el sistema multiagente
    utilizando un LLM externo (Gemini).
    Si no es válida, el LLM intenta sugerir preguntas similares.
    Retorna: (es_valida: bool, recomendaciones: list[str])
    """
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)

        # Prepara el prompt para el modelo.
        # Le pedimos que responda con SI/NO y, si es NO, que dé 3 ejemplos.
        prompt_para_gemini = (
            "Dada la siguiente pregunta, ¿es relevante para un sistema de gestión de eventos, "
            "planificación de ocio o búsqueda de actividades? "
            "Responde 'SI' si es relevante o 'NO' si no lo es.\n"
            "Si tu respuesta es 'NO', añade en una nueva línea 3 ejemplos de preguntas relevantes "
            "para este sistema, separadas por ';'.\n\n"
            f"Pregunta: \"{pregunta}\"\n"
            "Respuesta:"
        )

        response = model.generate_content(
            prompt_para_gemini,
            safety_settings={
                "HARASSMENT": "BLOCK_NONE",
                "HATE": "BLOCK_NONE",
                "SEXUAL": "BLOCK_NONE",
                "DANGEROUS": "BLOCK_NONE",
            },
            request_options={"timeout": 60}
        )

        respuesta_llm_raw = response.text.strip().lower()
        logger.info(f"LLM Gemini evaluó '{pregunta}' con raw: '{respuesta_llm_raw}'")

        es_valida = "si" in respuesta_llm_raw or "sí" in respuesta_llm_raw
        recomendaciones = []

        if not es_valida:
            # Si la respuesta no contiene 'si', buscamos recomendaciones
            partes = respuesta_llm_raw.split('\n', 1) # Divide en max 2 partes por el primer salto de línea
            if len(partes) > 1:
                # La segunda parte debería contener las recomendaciones
                # Eliminamos el prefijo 'ejemplos:' o similar si lo hay
                reco_str = partes[1].strip()
                # Elimina cualquier prefijo común que el LLM pueda añadir
                reco_str = reco_str.replace("ejemplos:", "").replace("ejemplos de preguntas:", "").strip()
                
                # Divide por ';' y limpia cada recomendación
                recomendaciones = [
                    r.strip().capitalize() # Capitalizamos la primera letra de cada recomendación
                    for r in reco_str.split(';') if r.strip()
                ]
                # Limitar a un máximo de 3 recomendaciones para mantener la concisión
                recomendaciones = recomendaciones[:3]


        return es_valida, recomendaciones

    except Exception as e:
        logger.error(f"Error al validar o generar recomendaciones con Gemini: {e}")
        # En caso de error, asumimos que no es válida y no damos recomendaciones
        return False, []


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

@ensure_initialized
@app.route("/buscar", methods=["POST"])
def buscar():
    try:
        data = request.json or {}
        query_usuario = data.get("query", "").strip()

        if not query_usuario:
            return jsonify({"status": "error", "mensaje": "Consulta vacía"}), 400

        # Validación con Gemini y obtención de recomendaciones
        es_valida, recomendaciones = es_pregunta_valida_llm(query_usuario)

        if not es_valida:
            logger.info(f"❌ Consulta fuera de dominio: {query_usuario}")
            mensaje_respuesta = " Esta consulta no está relacionada con eventos o actividades. Reformúlala por favor."
            
            return jsonify({
                "status": "invalid",
                "mensaje": mensaje_respuesta,
                "eventos": [],
                "sugerencias": recomendaciones # Añadimos las sugerencias al JSON de respuesta
            }), 200

        clave_respuesta = data.get("respuesta") or f"res_busqueda_{uuid.uuid4().hex}"

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

        try:
            respuesta = response_queue.get(timeout=60)
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
            "respuesta": clave_respuesta,
            "eventos_filtrados": eventos_entrada
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