
"""
Módulo de procesamiento de eventos (procesamiento.py)

Este módulo contiene la clase `EventProcessor` encargada de transformar datos crudos de eventos 
provenientes de distintas fuentes (PredictHQ, SeatGeek, Ticketmaster) en una estructura enriquecida
y estandarizada. También incluye validaciones, normalización de fechas, y detección de fuente.

Funciones auxiliares permiten el procesamiento masivo desde carpetas y la detección automática
de la fuente a partir del nombre del archivo.
"""
import json
import os
from datetime import datetime
import pytz
import uuid
import hashlib
from typing import Dict, Any, Optional, List

def calcular_hash_json(data):
    json_str = json.dumps(data, sort_keys=True)
    return hashlib.md5(json_str.encode('utf-8')).hexdigest()

class EventProcessor:
    def __init__(self):
        self.supported_sources = {
            'predicthq': self._process_predicthq,
            'seatgeek': self._process_seatgeek,
            'ticketmaster': self._process_ticketmaster
        }
    def process_event(self, raw_data: Dict[str, Any], source: str) -> Dict[str, Any]:
        """Procesa un evento según su fuente y devuelve un formato estandarizado enriquecido"""
        processor = self.supported_sources.get(source.lower())
        if not processor:
            raise ValueError(f"Fuente no soportada: {source}. Fuentes válidas: {list(self.supported_sources.keys())}")
        
        try:
            # Estructura base del evento
            event = {
                "metadata": {
                    "source": source,
                    "processing_timestamp": datetime.utcnow().isoformat() + "Z",
                    "original_id": raw_data.get("id"),
                    "event_version": "2.1"
                },
                "basic_info": {
                    "title": None,
                    "description": None,
                    "status": "confirmed",
                    "visibility": "public"
                },
                "classification": {
                    "primary_category": None,
                    "categories": [],
                    "tags": []
                },
                "temporal_info": {
                    "start": None,
                    "end": None,
                    "timezone": None,
                    "duration_minutes": None,
                    "is_recurring": False
                },
                "spatial_info": {
                    "venue": {
                        "name": None,
                        "full_address": None,
                        "address_components": {},
                        "location": {
                            "latitude": None,
                            "longitude": None,
                            "geo_accuracy": None
                        },
                        "additional_info": {}
                    },
                    "area": {
                        "city": None,
                        "region": None,
                        "country": None
                    }
                },
                "participants": {
                    "performers": [],
                    "organizers": [],
                    "attendees_expected": None
                },
                "external_references": {
                    "urls": [],
                    "images": [],
                    "social_media": []
                },
                "raw_data_metadata": {
                    "significant_fields_preserved": [],
                    "source_specific": {}
                }
            }

            # Procesamiento específico por fuente
            processor(raw_data, event)
            
            # Post-procesamiento común
            self._post_process_event(event)
            
            # Validación final de campos obligatorios (más flexible para venue)
            missing_fields = []

            if not event["basic_info"].get("title"):
                missing_fields.append("title")
            if not event["temporal_info"].get("start"):
                missing_fields.append("start date")
            
            # Validación más flexible para venue
            venue = event["spatial_info"]["venue"]
            has_venue_info = (
                venue.get("name") or
                venue.get("full_address") or
                (venue.get("location", {}).get("latitude") and venue.get("location", {}).get("longitude"))
            )

            if not has_venue_info:
                missing_fields.append("venue information (name, address or location)")

            if missing_fields:
                raise ValueError(f"Faltan campos obligatorios: {', '.join(missing_fields)}")

            event["metadata"]["raw_hash"] = calcular_hash_json(raw_data)
            return event
            
        except Exception as e:
            error_id = str(uuid.uuid4())
            return {
                "error": {
                    "id": error_id,
                    "message": f"Error procesando evento: {str(e)}",
                    "source": source,
                    "original_data": raw_data
                },
                "metadata": {
                    "processing_timestamp": datetime.utcnow().isoformat() + "Z",
                    "status": "processing_error"
                }
            }
    
    def _process_predicthq(self, raw_data: Dict[str, Any], event: Dict[str, Any]):
        """Procesamiento específico para eventos de PredictHQ"""
        # Basic info
        event["basic_info"].update({
            "title": raw_data.get("title"),
            "description": raw_data.get("description"),
            "status": "confirmed"  # PredictHQ generalmente no tiene estado cancelado
        })
        
        # Classification
        event["classification"].update({
            "primary_category": raw_data.get("category"),
            "tags": raw_data.get("labels", [])
        })
        
        # Temporal info
        self._process_temporal_info(
            event,
            start=raw_data.get("start"),
            end=raw_data.get("end"),
            timezone=raw_data.get("timezone"),
            duration=raw_data.get("duration")
        )
        
        # Spatial info and venue
        location = raw_data.get("location", {})
        entities = raw_data.get("entities", [])
        
        # Procesar venue desde entities
        venue_entity = next((e for e in entities if e.get("type") == "venue"), None)
        if venue_entity:
            event["spatial_info"]["venue"].update({
                "name": venue_entity.get("name"),
                "full_address": venue_entity.get("formatted_address"),
                "location": {
                    "latitude": location.get("lat"),
                    "longitude": location.get("lon"),
                    "geo_accuracy": "high"
                }
            })
        elif location and location.get("lat") is not None and location.get("lon") is not None:
            event["spatial_info"]["venue"]["location"] = {
                "latitude": location.get("lat"),
                "longitude": location.get("lon"),
                "geo_accuracy": "medium"
            }

        
        # Procesar performers desde entities
        performers = [e for e in entities if e.get("type") in ["person", "performer"]]
        event["participants"]["performers"] = [
            {
                "name": p.get("name"),
                "type": p.get("type"),
                "external_ids": [p.get("entity_id")],
                "role": "main" if p.get("type") == "person" else "supporting"
            } for p in performers
        ]
        
        # External references
        if raw_data.get("id"):
            event["external_references"]["urls"].append({
                "type": "source",
                "url": f"https://predicthq.com/events/{raw_data.get('id')}"
            })
        
        # Preservar metadatos importantes del raw data
        event["raw_data_metadata"]["significant_fields_preserved"] = [
            "scope", "labels", "entities"
        ]
        event["raw_data_metadata"]["source_specific"] = {
            "scope": raw_data.get("scope"),
            "entity_count": len(entities)
        }
    
    def _process_seatgeek(self, raw_data: Dict[str, Any], event: Dict[str, Any]):
        """Procesamiento específico para eventos de SeatGeek"""
        # Basic info
        event["basic_info"].update({
            "title": raw_data.get("title"),
            "status": "confirmed"  # SeatGeek generalmente no tiene estado cancelado
        })
        
        # Classification
        taxonomies = raw_data.get("taxonomies", [])
        if taxonomies:
            event["classification"].update({
                "primary_category": taxonomies[0].get("name"),
                "categories": [t.get("name") for t in taxonomies[1:] if t.get("name")]
            })
        
        # Temporal info
        self._process_temporal_info(
            event,
            start=raw_data.get("datetime_local"),
            timezone=raw_data.get("venue", {}).get("timezone")
        )
        
        # Spatial info and venue
        venue = raw_data.get("venue", {})
        location = venue.get("location", {})
        
        # Generar nombre alternativo si es necesario
        venue_name = venue.get("name")
        if not venue_name and venue.get("city"):
            venue_name = f"Venue in {venue.get('city')}"
        
        address_components = {
            "street": venue.get("address"),
            "city": venue.get("city"),
            "state": venue.get("state"),
            "country": venue.get("country"),
            "postal_code": venue.get("postal_code")
        }
        
        event["spatial_info"].update({
            "venue": {
                "name": venue_name,
                "full_address": ", ".join(
                    filter(None, [
                        venue.get("address"),
                        venue.get("city"),
                        venue.get("state"),
                        venue.get("country")
                    ])
                ),
                "address_components": address_components,
                "location": {
                    "latitude": location.get("lat"),
                    "longitude": location.get("lon"),
                    "geo_accuracy": "high"
                },
                "additional_info": {
                    "venue_id": venue.get("id"),
                    "timezone": venue.get("timezone"),
                    "capacity": venue.get("capacity")
                }
            },
            "area": {
                "city": venue.get("city"),
                "region": venue.get("state"),
                "country": venue.get("country")
            }
        })
        
        # Participants
        performers = raw_data.get("performers", [])
        event["participants"]["performers"] = [
            {
                "name": p.get("name"),
                "type": p.get("type"),
                "external_ids": [str(p.get("id"))],
                "images": [img for img in [
                    p.get("images", {}).get("huge"),
                    p.get("image")
                ] if img],
                "role": "main" if p.get("primary") else "supporting"
            } for p in performers
        ]
        
        # External references
        if raw_data.get("url"):
            event["external_references"]["urls"].append({
                "type": "ticketing",
                "url": raw_data.get("url")
            })
        
        # Images from performers
        for performer in performers:
            if performer.get("image"):
                event["external_references"]["images"].append({
                    "type": "performer",
                    "url": performer.get("image"),
                    "performer_name": performer.get("name")
                })
        
        # Preservar metadatos importantes del raw data
        event["raw_data_metadata"]["significant_fields_preserved"] = [
            "taxonomies", "performers", "stats"
        ]
        event["raw_data_metadata"]["source_specific"] = {
            "score": raw_data.get("score"),
            "popularity": raw_data.get("stats", {}).get("event_count")
        }
    
    def _process_ticketmaster(self, raw_data: Dict[str, Any], event: Dict[str, Any]):
        """Procesamiento específico para eventos de Ticketmaster"""
        # Basic info
        event["basic_info"].update({
            "title": raw_data.get("name"),
            "description": raw_data.get("info"),
            "status": "cancelled" if (
                raw_data.get("dates", {}).get("status", {}).get("code") == "cancelled"
            ) else "confirmed"
        })
        
        # Classification
        classifications = raw_data.get("classifications", [])
        if classifications:
            classification = classifications[0]
            event["classification"].update({
                "primary_category": classification.get("segment", {}).get("name"),
                "categories": [
                    classification.get("genre", {}).get("name"),
                    classification.get("subGenre", {}).get("name")
                ]
            })
        
        # Temporal info
        dates = raw_data.get("dates", {})
        start = dates.get("start", {})
        self._process_temporal_info(
            event,
            start=start.get("localDate"),
            timezone=dates.get("timezone"),
            is_tba=start.get("timeTBA") or start.get("dateTBA")
        )
        
        # Spatial info and venue
        venue = raw_data.get("venue", {})
        
        # Generar nombre alternativo si es necesario
        venue_name = venue.get("name")
        if not venue_name:
            if venue.get("address") and venue.get("city"):
                venue_name = f"{venue.get('address')} ({venue.get('city')})"
            elif venue.get("city"):
                venue_name = f"Venue in {venue.get('city')}"
            else:
                venue_name = "Unknown venue"
        
        address_components = {
            "street": venue.get("address"),
            "city": venue.get("city"),
            "country": venue.get("country")
        }
        
        event["spatial_info"].update({
            "venue": {
                "name": venue_name,
                "full_address": ", ".join(
                    filter(None, [
                        venue.get("address"),
                        venue.get("city"),
                        venue.get("country")
                    ])
                ),
                "address_components": address_components,
                "additional_info": {
                    "venue_type": "specific"
                }
            },
            "area": {
                "city": venue.get("city"),
                "country": venue.get("country")
            }
        })
        
        # Participants
        attractions = raw_data.get("attractions", [])
        event["participants"]["performers"] = [
            {
                "name": a.get("name"),
                "type": a.get("type"),
                "external_ids": [a.get("id")],
                "url": a.get("url"),
                "role": "main"
            } for a in attractions
        ]
        
        # External references
        if raw_data.get("url"):
            event["external_references"]["urls"].append({
                "type": "ticketing",
                "url": raw_data.get("url")
            })
        
        # Preservar metadatos importantes del raw data
        event["raw_data_metadata"]["significant_fields_preserved"] = [
            "classifications", "attractions", "dates"
        ]
        event["raw_data_metadata"]["source_specific"] = {
            "event_type": raw_data.get("type"),
            "span_multiple_days": dates.get("spanMultipleDays")
        }
    
    def _process_temporal_info(self, event: Dict[str, Any], 
                             start: Optional[str] = None, 
                             end: Optional[str] = None,
                             timezone: Optional[str] = None,
                             duration: Optional[int] = None,
                             is_tba: bool = False):
        """Procesamiento común para información temporal"""
        temporal = event["temporal_info"]
        
        # Procesar timezone
        if timezone:
            try:
                tz = pytz.timezone(timezone)
                temporal["timezone"] = {
                    "name": timezone,
                    "utc_offset": datetime.now(tz).strftime("%z")
                }
            except:
                temporal["timezone"] = {"name": timezone}
        
        # Procesar fechas
        temporal["start"] = self._normalize_datetime(start, timezone)
        temporal["end"] = self._normalize_datetime(end, timezone)
        
        # Calcular duración si no existe
        if not temporal["end"] and duration:
            temporal["duration_minutes"] = round(duration / 60) if duration else None
        
        # Marcar si es TBA (To Be Announced)
        temporal["is_tba"] = is_tba
    
    def _normalize_datetime(self, dt_str: Optional[str], timezone: Optional[str]) -> Optional[str]:
        """Normaliza formatos de fecha/hora a ISO 8601"""
        if not dt_str:
            return None
            
        try:
            # Intentar parsear con timezone si está disponible
            if timezone and "T" in dt_str:
                tz = pytz.timezone(timezone)
                dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = tz.localize(dt)
                return dt.isoformat()
            
            # Parsear fechas simples (sin hora)
            if "T" not in dt_str:
                dt = datetime.strptime(dt_str, "%Y-%m-%d")
                return dt.date().isoformat()
                
            # Parsear formato ISO básico
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00")).isoformat()
            
        except ValueError:
            # Si falla el parsing, devolver el valor original
            return dt_str
    
    def _post_process_event(self, event: Dict[str, Any]):
        """Limpieza y normalización post-procesamiento"""
        def remove_empty_fields(d):
            if not isinstance(d, dict):
                return d
            return {k: remove_empty_fields(v) for k, v in d.items() if v not in [None, "", [], {}]}
        
        cleaned_event = remove_empty_fields(event)
        
        # Asegurar arrays vacíos en lugar de nulos para consistencia
        for field in ["categories", "tags", "performers", "urls", "images"]:
            path = ["classification", field] if field in ["categories", "tags"] else ["external_references", field]
            current = cleaned_event
            for part in path[:-1]:
                current = current.setdefault(part, {})
            current.setdefault(path[-1], [])
        
        # Actualizar metadatos de preservación
        preserved_fields = cleaned_event.get("raw_data_metadata", {}).get("significant_fields_preserved", [])
        cleaned_event["raw_data_metadata"]["significant_fields_preserved"] = list(set(preserved_fields))
        
        # Actualizar el evento original
        event.clear()
        event.update(cleaned_event)

    def procesar_evento_desde_archivo(filepath: str) -> Dict[str, Any]:
        """
        Procesa un archivo JSON de evento crudo y devuelve el evento enriquecido.
        """
        processor = EventProcessor()

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            source = detect_source_from_filename(os.path.basename(filepath))
            if not source:
                return {"error": "No se pudo detectar la fuente automáticamente"}

            resultado = processor.process_event(raw_data, source)
            return resultado

        except Exception as e:
            return {
                "error": f"Excepción durante procesamiento: {str(e)}",
                "file": filepath
            }
def detect_source_from_filename(filename: str) -> Optional[str]:
        """Detecta la fuente basado en el nombre del archivo"""
        fname = filename.lower()
        if "predicthq" in fname:
            return "predicthq"
        elif "seatgeek" in fname:
            return "seatgeek"
        elif "ticketmaster" in fname or "tm" in fname:
            return "ticketmaster"
        return None

def procesar_json():
        """Función principal para ejecutar el procesamiento"""
        input_folder = "./eventos_completos"
        output_folder = "./eventos_mejorados"
        clean_folder(input_folder, output_folder)
def clean_folder(input_folder: str, output_folder: str, verbose: bool = True):
        """Procesa todos los archivos JSON en la carpeta de entrada y guarda los resultados"""
        processor = EventProcessor()
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        processed_files = 0
        errors = 0
        skipped_files = []
        
        for filename in [f for f in os.listdir(input_folder) if f.endswith(".json")]:
            input_path = os.path.join(input_folder, filename)
            output_filename = f"{os.path.splitext(filename)[0]}_enhanced.json"
            output_path = os.path.join(output_folder, output_filename)
            
            try:
                with open(input_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                
                source = detect_source_from_filename(filename)
                if not source:
                    if verbose:
                        print(f"[SKIP] No se pudo detectar fuente para {filename}")
                    skipped_files.append(filename)
                    continue
                
                # Procesar el evento
                processed_event = processor.process_event(raw_data, source)
                
                # Manejar errores de procesamiento
                if "error" in processed_event:
                    errors += 1
                    if verbose:
                        print(f"[ERROR] Procesando {filename}: {processed_event['error']['message']}")
                    continue
                
                # Guardar el resultado
                with open(output_path, "w", encoding="utf-8") as f_out:
                    json.dump(processed_event, f_out, ensure_ascii=False, indent=2)
                
                processed_files += 1
                if verbose:
                    print(f"[OK] {filename} procesado y guardado como {output_filename}")
                    
            except Exception as e:
                errors += 1
                if verbose:
                    print(f"[ERROR] Error procesando {filename}: {str(e)}")
        
        # Resumen final
        if verbose:
            print("\nResumen de procesamiento:")
            print(f"- Archivos procesados correctamente: {processed_files}")
            print(f"- Archivos con errores: {errors}")
            if skipped_files:
                print(f"- Archivos omitidos (fuente no reconocida): {len(skipped_files)}")
                print("  " + "\n  ".join(skipped_files))
