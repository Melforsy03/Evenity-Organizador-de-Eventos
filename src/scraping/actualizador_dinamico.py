"""
Módulo de actualización dinámica de eventos (actualizador_dinamico.py)

Este módulo revisa periódicamente los eventos existentes, detecta aquellos que están obsoletos,
incompletos o modificados, y los actualiza desde las fuentes originales (PredictHQ, SeatGeek, Ticketmaster).
Luego, regenera embeddings solo para los eventos actualizados (actualización incremental).

Funciones principales:
- `revisar_y_actualizar_eventos()`
- `incremental_reembed()`
- `actualizar_eventos_y_embeddings()`
"""

import os
import json
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.procesamiento import EventProcessor, detect_source_from_filename, calcular_hash_json
from scraping.crawler import EventScraper
from core.embedding import EventEmbedder
from scraping.scraping_incremental import archivos_a_procesar

def is_event_obsolete_or_incomplete(event):
    required_fields = [
        event.get("basic_info", {}).get("title"),
        event.get("temporal_info", {}).get("start"),
        event.get("spatial_info", {}).get("venue", {}).get("name")
    ]
    if any(f is None or f == "" for f in required_fields):
        return True

    try:
        fecha = event["temporal_info"]["start"]
        fecha_evento = datetime.fromisoformat(fecha.replace("Z", "+00:00"))
        return fecha_evento < datetime.now(fecha_evento.tzinfo)
    except:
        return True

def buscar_evento_por_id(fuente, event_id):
    scraper = EventScraper()
    if fuente == "seatgeek":
        url = f"https://api.seatgeek.com/2/events/{event_id}"
        params = {"client_id": scraper.sources_config["seatgeek"]["client_id"]}
        resp = scraper._make_request(url, params)
        if resp:
            return resp.json()
    elif fuente == "predicthq":
        url = f"https://api.predicthq.com/v1/events/{event_id}"
        headers = {
            "Authorization": f"Bearer {scraper.sources_config['predicthq']['api_key']}",
            "Accept": "application/json"
        }
        resp = scraper._make_request(url, headers=headers)
        if resp:
            return resp.json()
    elif fuente == "ticketmaster":
        url = f"https://app.ticketmaster.com/discovery/v2/events/{event_id}.json"
        params = {"apikey": scraper.sources_config["ticketmaster"]["api_key"]}
        resp = scraper._make_request(url, params)
        if resp:
            return resp.json()
    return None

def revisar_y_actualizar_eventos(input_folder="./eventos_mejorados"):
    import requests

    processor = EventProcessor()
    archivos_para_procesar = archivos_a_procesar(input_folder)
    archivos_actualizados = []

    for archivo in archivos_para_procesar:
        if not archivo.endswith(".json"):
            continue

        path = os.path.join(input_folder, archivo)
        with open(path, "r", encoding="utf-8") as f:
            evento = json.load(f)

        if evento.get("obsoleto", False):
            print(f"[SKIP] Evento obsoleto: {archivo}")
            continue

        fuente = detect_source_from_filename(archivo)
        if not fuente:
            print(f"[SKIP] Fuente no detectada para {archivo}")
            continue

        original_id = evento.get("metadata", {}).get("original_id")
        if not original_id:
            print(f"[SKIP] Sin original_id: {archivo}")
            continue

        actualizar = False
        nuevo_evento_raw = None

        if is_event_obsolete_or_incomplete(evento):
            print(f"[ACTUALIZAR] Evento incompleto u obsoleto: {archivo}")
            actualizar = True
        else:
            try:
                nuevo_evento_raw = buscar_evento_por_id(fuente, original_id)
            except requests.HTTPError as e:
                if e.response.status_code == 404:
                    print(f"[OBSOLETO] Evento no encontrado: {archivo}")
                    evento["obsoleto"] = True
                    with open(path, "w", encoding="utf-8") as f_out:
                        json.dump(evento, f_out, ensure_ascii=False, indent=2)
                    continue
                else:
                    raise

            if nuevo_evento_raw:
                nuevo_hash = calcular_hash_json(nuevo_evento_raw)
                hash_actual = evento.get("metadata", {}).get("raw_hash")
                if nuevo_hash != hash_actual:
                    print(f"[ACTUALIZAR] Evento modificado: {archivo}")
                    actualizar = True
            else:
                print(f"[SKIP] No se pudo obtener evento actualizado para {archivo}")

        if actualizar:
            if not nuevo_evento_raw:
                nuevo_evento_raw = buscar_evento_por_id(fuente, original_id)
                if not nuevo_evento_raw:
                    print(f"[SKIP] No se pudo obtener evento actualizado para {archivo}")
                    continue
            evento_procesado = processor.process_event(nuevo_evento_raw, fuente)
            with open(path, "w", encoding="utf-8") as f_out:
                json.dump(evento_procesado, f_out, ensure_ascii=False, indent=2)
            archivos_actualizados.append(archivo)

    print(f"[✔] {len(archivos_actualizados)} eventos actualizados.")
    return archivos_actualizados

def incremental_reembed(archivos_actualizados, eventos_folder="./eventos_mejorados", output_folder="./embedding_data"):
    if not archivos_actualizados:
        print("⚠️ No hay eventos actualizados para re-embeddear.")
        return

    embedder = EventEmbedder._instance
    embedder.load_index(os.path.join(output_folder, "eventos.index"))

    nuevos_eventos = []
    for archivo in archivos_actualizados:
        with open(os.path.join(eventos_folder, archivo), "r", encoding="utf-8") as f:
            ev = json.load(f)
            nuevos_eventos.append(ev)

    embedder.add_new_events(nuevos_eventos)
    embedder.save_event_data(output_folder)

def actualizar_eventos_y_embeddings(input_folder="./eventos_mejorados", output_folder="./embedding_data"):
    print("🔁 Revisando y actualizando eventos...")
    archivos_actualizados = revisar_y_actualizar_eventos(input_folder)

    print("🧠 Actualizando embeddings incrementalmente...")
    incremental_reembed(archivos_actualizados, input_folder, output_folder)

    return f"{len(archivos_actualizados)} eventos actualizados y embeddings incrementales aplicados."

if __name__ == "__main__":
    resultado = actualizar_eventos_y_embeddings()
    print(resultado)
