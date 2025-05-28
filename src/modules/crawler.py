import requests
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Dict, List, Optional
from urllib.parse import urljoin
import sys
import io

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except Exception:
    pass  # Ignorar si falla (por ejemplo en Streamlit)
# Configuración básica de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawler.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

class EventScraper:
    def __init__(self):
        self.output_dir = "eventos_completos"
        self.max_workers = 5  # Número de hilos para scraping paralelo
        self.request_delay = 0.5  # Segundos entre requests al mismo dominio
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configuración de todas las APIs
        self.sources_config = {
            "predicthq": {
                "api_key": "MipULnByYK9BwqPHec0hLOXJDLF8ZY6MqVo0i0EI",
                "base_url": "https://api.predicthq.com/v1",
                "max_events": 2000  # Aumentado para alcanzar 2000+
            },
            "seatgeek": {
                "client_id": "NTAzNTA5OTh8MTc0ODAyNzgyMi42NzgxNDY0",
                "base_url": "https://api.seatgeek.com/2",
                "max_events": 500
            },
            "ticketmaster": {
                "api_key": "WQGelSMVlChnAOa4AJWp3JQ9TFNlWc95",
                "base_url": "https://app.ticketmaster.com/discovery/v2",
                "max_events": 1500  # Aumentado para alcanzar 2000+
            }
        }

    def _make_request(self, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None) -> Optional[requests.Response]:
        """Realiza una petición HTTP con manejo de errores"""
        default_headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json"
        }
        
        if headers:
            default_headers.update(headers)
            
        try:
            response = requests.get(url, params=params, headers=default_headers, timeout=15)
            
            # Manejar específicamente errores 403
            if response.status_code == 403:
                logging.warning(f"Acceso denegado (403) para {url}. Verifica tu API key o permisos.")
                return None
                
            response.raise_for_status()
            return response
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error al hacer request a {url}: {str(e)}")
            return None

    def save_event(self, data: Dict, source: str, event_id: int) -> None:
        """Guarda un evento en un archivo JSON"""
        filename = os.path.join(self.output_dir, f"{source}_evento_{event_id}.json")
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logging.info(f"[SUCCESS] Evento guardado: {filename}")
        except Exception as e:
            logging.error(f"Error guardando evento {event_id} de {source}: {e}")
    def scrape_seatgeek_events(self) -> int:
            config = self.sources_config["seatgeek"]
            base_url = f"{config['base_url']}/events"
            client_id = config["client_id"].split('|')[0]
            max_events = config["max_events"]

            per_page = 100
            saved_events = 0
            page = 1

            while saved_events < max_events:
                params = {
                    "client_id": client_id,
                    "per_page": per_page,
                    "page": page,
                    # Puedes agregar filtros aquí, ejemplo:
                    # "venue.city": "Brussels",
                    # "datetime_utc.gte": "2025-06-01T00:00:00",
                    # "datetime_utc.lte": "2025-12-31T23:59:59",
                }

                response = self._make_request(base_url, params)
                if not response:
                    break

                data = response.json()
                events = data.get("events", [])
                if not events:
                    break

                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = []
                    for event in events:
                        event_id = saved_events + 1
                        futures.append(executor.submit(self.process_seatgeek_event, event, event_id))
                        saved_events += 1
                        if saved_events >= max_events:
                            break
                    # Esperar a que terminen todos
                    for future in futures:
                        future.result()

                logging.info(f"Guardados {saved_events} eventos SeatGeek, página {page}")
                page += 1
                time.sleep(self.request_delay)

            logging.info(f"[SUCCESS] Total de eventos SeatGeek guardados: {saved_events}")
            return saved_events

    def process_seatgeek_event(self, event_data: Dict, event_id: int) -> None:
        try:
            # Extraer campos relevantes para tu organizador
            event = {
                "id": event_data.get("id"),
                "title": event_data.get("title"),
                "type": event_data.get("type"),
                "url": event_data.get("url"),
                "datetime_local": event_data.get("datetime_local"),
                "venue": event_data.get("venue", {}),
                "performers": event_data.get("performers", []),
                "taxonomies": event_data.get("taxonomies", []),
                "stats": event_data.get("stats", {}),
                "score": event_data.get("score"),
                "source": "seatgeek"
            }
            self.save_event(event, "seatgeek", event_id)
        except Exception as e:
            logging.error(f"Error procesando evento SeatGeek {event_id}: {e}")

    # ====================== PREDICTHQ API ======================
    def scrape_predicthq_events(self) -> int:
        """Extrae eventos de PredictHQ API"""
        config = self.sources_config["predicthq"]
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Accept": "application/json"
        }
        
        params = {
            "country": "ES",  # España
            "active.gte": "2025-06-01",
            "active.lte": "2025-12-31",
            "limit": 100  # Máximo permitido por petición
        }
        
        saved_events = 0
        offset = 0
        
        while saved_events < config["max_events"]:
            params["offset"] = offset
            response = self._make_request(
                f"{config['base_url']}/events/",
                headers=headers,
                params=params
            )
            
            if not response:
                break
                
            data = response.json()
            events = data.get("results", [])
            
            if not events:
                break
                
            # Procesamiento en paralelo
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for event in events:
                    executor.submit(self.process_predicthq_event, event, saved_events + 1)
                    saved_events += 1
                    if saved_events >= config["max_events"]:
                        break
            
            offset += len(events)
            time.sleep(self.request_delay)
            
        logging.info(f"[SUCCESS] Total de eventos PredictHQ guardados: {saved_events}")
        return saved_events

    def process_predicthq_event(self, event_data: Dict, event_id: int) -> None:
        """Procesa y guarda un evento de PredictHQ"""
        try:
            event = {
                "id": event_data.get("id"),
                "title": event_data.get("title"),
                "description": event_data.get("description"),
                "category": event_data.get("category"),
                "labels": event_data.get("labels", []),
                "start": event_data.get("start"),
                "end": event_data.get("end"),
                "duration": event_data.get("duration"),
                "timezone": event_data.get("timezone"),
                "location": {
                    "lat": event_data.get("location", [])[0] if event_data.get("location") else None,
                    "lon": event_data.get("location", [])[1] if event_data.get("location") else None,
                    "address": event_data.get("place_hierarchies", [])[0] if event_data.get("place_hierarchies") else None
                },
                "scope": event_data.get("scope"),
                "entities": event_data.get("entities", []),
                "source": "predicthq"
            }
            
            self.save_event(event, "predicthq", event_id)
        except Exception as e:
            logging.error(f"Error procesando evento PredictHQ {event_id}: {e}")

    # ====================== TICKETMASTER API ======================
    def scrape_ticketmaster_events(self) -> int:
        """Extrae eventos de Ticketmaster API"""
        config = self.sources_config["ticketmaster"]
        params = {
            "apikey": config["api_key"],
            "countryCode": "ES",  # Añade filtro por país
            "size": 50,  # Aumentado a 100 por petición
            "startDateTime": "2025-06-01T00:00:00Z",
            "endDateTime": "2025-12-31T23:59:59Z"
        }
        
        saved_events = 0
        page = 0
        
        while saved_events < config["max_events"]:
            params["page"] = page
            response = self._make_request(
                f"{config['base_url']}/events.json",
                params=params
            )
            
            if not response:
                break
                
            data = response.json()
            events = data.get("_embedded", {}).get("events", [])
            
            if not events:
                break
                
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for event in events:
                    executor.submit(self.process_ticketmaster_event, event, saved_events + 1)
                    saved_events += 1
                    if saved_events >= config["max_events"]:
                        break
            
            page += 1
            time.sleep(self.request_delay)
            
        logging.info(f"[SUCCESS] Total de eventos Ticketmaster guardados: {saved_events}")
        return saved_events

    def process_ticketmaster_event(self, event_data: Dict, event_id: int) -> None:
        """Procesa y guarda un evento de Ticketmaster"""
        try:
            attractions = event_data.get("_embedded", {}).get("attractions", [])
            attractions_list = [{
                "name": a.get("name"),
                "type": a.get("type"),
                "url": a.get("url")
            } for a in attractions]

            venue_info = {}
            venues = event_data.get("_embedded", {}).get("venues", [])
            if venues:
                v = venues[0]
                venue_info = {
                    "name": v.get("name"),
                    "address": v.get("address", {}).get("line1", ""),
                    "city": v.get("city", {}).get("name", ""),
                    "country": v.get("country", {}).get("name", "")
                }

            event = {
                "id": event_data.get("id"),
                "name": event_data.get("name"),
                "type": event_data.get("type"),
                "url": event_data.get("url"),
                "info": event_data.get("info", ""),
                "classifications": event_data.get("classifications", []),
                "dates": event_data.get("dates", {}),
                "attractions": attractions_list,
                "venue": venue_info,
                "source": "ticketmaster"
            }
            
            self.save_event(event, "ticketmaster", event_id)
        except Exception as e:
            logging.error(f"Error procesando evento Ticketmaster {event_id}: {e}")

    # ====================== EJECUCIÓN PRINCIPAL ======================
    def run_all_scrapers(self) -> None:
        """Ejecuta todos los scrapers disponibles y verifica el mínimo de 2000 eventos"""
        logging.info("Iniciando scraping de eventos...")
        total_events = 0
        
        try:
             total_events += self.scrape_seatgeek_events()
             #total_events += self.scrape_predicthq_events()
             total_events += self.scrape_ticketmaster_events()

        except Exception as e:
            logging.error(f"Error en el proceso principal de scraping: {e}")
        finally:
            logging.info(f"Proceso de scraping completado. Total de eventos recolectados: {total_events}")
            if total_events >= 2000:
                logging.info("¡Objetivo de 2000+ eventos alcanzado!")
            else:
                logging.warning(f"Se recolectaron {total_events} eventos (menos de 2000).")


def ejecutar_scraping_fuentes(source="all", country="ES", start="2025-06-01", end="2025-12-31"):
    """
    Ejecuta el scraping de eventos desde las fuentes especificadas.

    Args:
        source (str): Fuente específica o "all". Opciones: "seatgeek", "predicthq", "ticketmaster", "all".
        country (str): Código de país (para PredictHQ y Ticketmaster).
        start (str): Fecha de inicio (YYYY-MM-DD).
        end (str): Fecha de fin (YYYY-MM-DD).

    Returns:
        int: Total de eventos recolectados.
    """
    scraper = EventScraper()
    total = 0

    try:
        if source in ["all", "seatgeek"]:
            total += scraper.scrape_seatgeek_events()

        if source in ["all", "predicthq"]:
            scraper.sources_config["predicthq"]["country"] = country
            scraper.sources_config["predicthq"]["start"] = start
            scraper.sources_config["predicthq"]["end"] = end
            total += scraper.scrape_predicthq_events()

        if source in ["all", "ticketmaster"]:
            scraper.sources_config["ticketmaster"]["country"] = country
            scraper.sources_config["ticketmaster"]["start"] = start
            scraper.sources_config["ticketmaster"]["end"] = end
            total += scraper.scrape_ticketmaster_events()

    except Exception as e:
        logging.error(f"❌ Error en ejecución del scraper: {str(e)}")

    logging.info(f"[🧮] Total eventos recolectados: {total}")
    return total
