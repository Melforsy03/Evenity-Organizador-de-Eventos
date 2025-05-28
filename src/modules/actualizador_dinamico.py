import os
import json
from datetime import datetime
from procesamiento import EventProcessor, detect_source_from_filename , calcular_hash_json
from crawler import EventScraper
from embedding import EventEmbedder
from embedding import load_events_from_folder
from scraping_incremental import archivos_a_procesar


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

    eventos_actualizados = 0
    processor = EventProcessor()

    archivos_para_procesar = archivos_a_procesar(input_folder)

    for archivo in archivos_para_procesar:
        if not archivo.endswith(".json"):
            continue

        path = os.path.join(input_folder, archivo)
        with open(path, "r", encoding="utf-8") as f:
            evento = json.load(f)

        # Saltar eventos ya marcados como obsoletos
        if evento.get("obsoleto", False):
            print(f"[SKIP] Evento marcado como obsoleto: {archivo}")
            continue

        fuente = detect_source_from_filename(archivo)
        if not fuente:
            print(f"[SKIP] Fuente no detectada para {archivo}")
            continue

        original_id = evento.get("metadata", {}).get("original_id")
        if not original_id:
            print(f"[SKIP] Sin original_id en {archivo}")
            continue

        actualizar = False
        if is_event_obsolete_or_incomplete(evento):
            print(f"[ACTUALIZAR] Evento obsoleto o incompleto: {archivo}")
            actualizar = True
        else:
            nuevo_evento_raw = None
            try:
                nuevo_evento_raw = buscar_evento_por_id(fuente, original_id)
            except requests.HTTPError as e:
                if e.response.status_code == 404:
                    print(f"[OBSOLETO] Evento no encontrado (404): {archivo}")
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
                    print(f"[ACTUALIZAR] Evento modificado detectado: {archivo}")
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
            eventos_actualizados += 1

    print(f"[✔] {eventos_actualizados} eventos actualizados.")
    return eventos_actualizados


def reembed_all_events(input_folder="./eventos_mejorados", output_folder="./embedding_data"):
    eventos = load_events_from_folder(input_folder)
    if not eventos:
        print("No hay eventos válidos para generar embeddings.")
        return
    embedder = EventEmbedder()
    embeddings = embedder.generate_embeddings(eventos)
    embedder.build_index(embeddings, index_type="IVFFlat")
    embedder.save(output_folder)
    print(f"Embeddings actualizados y guardados en {output_folder}")

def actualizar_y_reembed():
    revisar_y_actualizar_eventos()
    reembed_all_events()

def actualizar_eventos_y_embeddings(input_folder="./eventos_mejorados", output_folder="./embedding_data"):
    """
    Revisa eventos obsoletos/incompletos, los actualiza y regenera los embeddings.
    Útil para ejecutar como parte del backend automáticamente.
    """
    print("🔁 Revisando y actualizando eventos...")
    total_actualizados = revisar_y_actualizar_eventos(input_folder)

    print("🧠 Recalculando embeddings...")
    reembed_all_events(input_folder, output_folder)

    return f"{total_actualizados} eventos actualizados y embeddings regenerados."

if __name__ == "__main__":
    actualizar_eventos_y_embeddings()
