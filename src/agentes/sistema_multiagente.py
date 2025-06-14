"""
Módulo principal del sistema multiagente (sistema_multiagente.py)

Coordina agentes especializados (scraping, procesamiento, actualización, grafo, optimización, etc.)
para mantener un flujo continuo de adquisición, mejora y recomendación de eventos.

Define:
- La clase base `AgenteBase` y el formato de `Mensaje`
- Agentes específicos: Scraper, Procesador, Actualizador, Grafo, Optimizador, Embedding, Fallback, Búsqueda
- Ciclos de ejecución concurrentes para cada agente
- Funciones de arranque y actualización periódica
- Lógica de fallback si hay escasez de resultados

Este módulo implementa el patrón de diseño productor-consumidor entre hilos.
"""
from queue import Queue
from threading import Thread
from abc import ABC, abstractmethod
import time
import subprocess
import json
import os
import sys
from datetime import date
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import networkx as nx
from core.embedding import fallback_api_call, run_embedding, load_events_from_folder, ordenar_y_paginar, EventEmbedder
from scraping.crawler import EventScraper
from core.procesamiento import procesar_json,EventProcessor
from scraping.actualizador_dinamico import actualizar_eventos_y_embeddings
from core.grafo_conocimiento import get_knowledge_graph 
from core.optimizador import obtener_eventos_optimales
from api.lanzar_api import iniciar_api  
from api.servidor_base import response_queues_lock , response_queues
from api.contexto_global import registrar_bandeja , obtener_bandeja
from datetime import datetime
from copy import deepcopy
import uuid

# === Mensaje para comunicación entre agentes ===
class Mensaje:
    def __init__(self, emisor, receptor, contenido):
        self.emisor = emisor
        self.receptor = receptor
        self.contenido = contenido

# === Clase base para agentes ===
class AgenteBase(ABC):
    def __init__(self, nombre, bandeja_entrada):
        self.nombre = nombre
        self.bandeja_entrada = bandeja_entrada

    def enviar(self, receptor, contenido):
        self.bandeja_entrada.put(Mensaje(self.nombre, receptor, contenido))

    @abstractmethod
    def ejecutar(self):
        pass
class AgenteCoordinador(AgenteBase):
    def __init__(self, nombre, bandeja_entrada):
        super().__init__(nombre, bandeja_entrada)
        self.estado = {
            "scraping": False,
            "procesamiento": False,
            "actualizacion": False,
            "grafo": False,
            "agenda_generada": False
        }

    def ejecutar(self):
        print("🧠 [Coordinador] Escuchando mensajes de coordinación...")
        while True:
            try:
                msg = self.bandeja_entrada.get()
                if msg.receptor != self.nombre:
                    continue

                contenido = msg.contenido

                if contenido == "scraping_terminado":
                    self.estado["scraping"] = True
                    self.enviar("procesador", "scraping_terminado")

                elif contenido == "eventos_procesados":
                    self.estado["procesamiento"] = True
                    self.enviar("actualizador", "eventos_procesados")

                elif contenido == "embeddings_actualizados":
                    self.estado["actualizacion"] = True
                    self.enviar("grafo", "embeddings_actualizados")

                elif contenido == "grafo_listo":
                    self.estado["grafo"] = True
                    self.enviar("optimizador", "grafo_listo")

                elif contenido == "resetear":
                    self.resetear_estado()

            except Exception as e:
                print(f"❌ [Coordinador] Error en ciclo principal: {e}")

    def resetear_estado(self):
        for key in self.estado:
            self.estado[key] = False

# === Agente 1: Scraper ===
class AgenteScraper(AgenteBase):
    def run_once(self, intento=1):
        eventos_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../eventos_completos"))
        
        if os.path.exists(eventos_dir) and os.listdir(eventos_dir):
            print("📁 [Scraper] Datos crudos ya presentes. Omitiendo scraping.")
            self.enviar("procesador", "scraping_terminado")
            return
        
        print(f"🔄 [Scraper] Iniciando scraping (intento {intento})...")
        try:
            scraper = EventScraper()
            scraper.run_all_scrapers()
            print("✅ [Scraper] Scraping completado exitosamente.")
            self.enviar("procesador", "scraping_terminado")
            self.enviar("coordinador", "scraping_terminado")

        except Exception as e:
            print(f"❌ [Scraper] Error durante scraping: {e}")
            if intento < 3:
                time.sleep(5 * intento)
                self.run_once(intento + 1)
            else:
                print("🛑 [Scraper] Fallo permanente tras 3 intentos.")
                self.enviar("procesador", "scraping_fallido")
    def ejecutar(self):
        while True:
            try:
                msg = self.bandeja_entrada.get()
                if msg.receptor == self.nombre and msg.contenido in ["scraping_terminado", "iniciar_scraping"]:
                    self.run_once()
            except Exception as e:
                print(f"[Scraper] ❌ Error: {e}")

                  
# === Agente 2: Procesador ===
class AgenteProcesador(AgenteBase):
    def run_once(self):
        # Verificar si la carpeta "eventos_mejorados" no tiene archivos o no existe
        eventos_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../eventos_mejorados"))
        if os.path.exists(eventos_dir) and os.listdir(eventos_dir):
            self.enviar("actualizador", "eventos_procesados")
            print("⏸️ [Procesador] La carpeta de eventos ya tiene archivos. Omitiendo procesamiento.")
        else:
            print("🧪 [Procesador] Procesando eventos...")
            # Llamada a la función de procesamiento de eventos
            procesar_json()
            print("✅ [Procesador] Procesamiento completo.")
            self.enviar("actualizador", "eventos_procesados")
            self.enviar("coordinador", "eventos_procesados")

    def ejecutar(self):
        while True:
            try:
                msg = self.bandeja_entrada.get()
                if msg.receptor == self.nombre and msg.contenido == "scraping_terminado":
                    self.run_once()
            except Exception as e:
                print(f"[Procesador] ❌ Error: {e}")
# === Agente 3: Actualizador ===
class AgenteActualizador(AgenteBase):
    def ejecutar(self):
        while True:
            try:
                msg = self.bandeja_entrada.get()
                if msg.receptor == self.nombre and msg.contenido in ["eventos_procesados", "forzar_actualizacion"]:
                    self.run_once()
            except Exception as e:
                print(f"[Actualizador] ❌ Error: {e}")


    def run_once(self, intento=1):
        if not self.necesita_actualizacion():
            print("⏸️ [Actualizador] Datos recientes. No se actualiza.")
            return

        print(f"🔁 [Actualizador] Actualizando eventos y embeddings (intento {intento})...")
        try:
            actualizar_eventos_y_embeddings()
            print("✅ [Actualizador] Actualización terminada.")
            self.enviar("grafo", "embeddings_actualizados")
            self.enviar("coordinador", "embeddings_actualizados")
        except Exception as e:
            print(f"❌ [Actualizador] Error al actualizar: {e}")
            if intento < 3:
                time.sleep(5 * intento)
                self.run_once(intento + 1)
            else:
                print("🛑 [Actualizador] Fallo permanente tras 3 intentos.")


    def necesita_actualizacion(self):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            path_index = os.path.join(script_dir, "embedding_data/eventos.index")
            last_modified = os.path.getmtime(path_index)
            tiempo_actual = time.time()
            return (tiempo_actual - last_modified) > (60 * 60 * 24 * 60)
        except:
            return True  # actualizar si no existe
# === Agente 4: Grafo de Conocimiento ===
class AgenteGrafo(AgenteBase):
    def run_once(self, intento=1):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        file_path = os.path.join(project_root, "grafo", "grafo_eventos.graphml")

        if os.path.exists(file_path):
            print("⚠️ Grafo ya existente y actualizado, omitiendo la reconstrucción.")
            try:
                self.grafo = nx.read_graphml(file_path)
                print("✅ [Grafo] Grafo cargado desde archivo existente.")
            except Exception as e:
                print(f"❌ Error al cargar el grafo existente: {e}")
                self.enviar("optimizador", "error_en_grafo")
                return
        else:
            print(f"🧠 [Grafo] Construyendo grafo de conocimiento (intento {intento})...")
            try:
                self.grafo = get_knowledge_graph()
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                nx.write_graphml(self.grafo, file_path)
                print("✅ [Grafo] Grafo construido y guardado.")
            except Exception as e:
                print(f"❌ Error al construir el grafo: {e}")
                if intento < 3:
                    time.sleep(5 * intento)
                    self.run_once(intento + 1)
                else:
                    self.enviar("optimizador", "error_en_grafo")
                    return

        self.enviar("optimizador", "grafo_listo")
        self.enviar("coordinador", "grafo_listo")

        self.enviar("busqueda_interactiva", {"grafo": self.grafo})

    def ejecutar(self):
        while True:
            try:
                msg = self.bandeja_entrada.get()
                if msg.receptor == self.nombre and msg.contenido == "embeddings_actualizados":
                    self.run_once()
            except Exception as e:
                print(f"[Grafo] ❌ Error: {e}")
# === Agente 5: Embedding y Búsqueda ===
class AgenteEmbedding(AgenteBase):
    def run_once(self):
        print("💾 [Embedding] Generando embeddings...")
        run_embedding()
        print("✅ [Embedding] Embeddings listos.")
        self.enviar("coordinador", "embeddings_actualizados")

    def ejecutar(self):
        while True:
            try:
                msg = self.bandeja_entrada.get()
                if msg.receptor == self.nombre:
                    if msg.contenido == "refrescar_embeddings":
                        self.run_once()
            except Exception as e:
                print(f"[Embedding] ❌ Error: {e}")
# === Agente 6: Optimizador ===
class AgenteOptimizador(AgenteBase):
    def __init__(self, nombre, bandeja_entrada):
        super().__init__(nombre, bandeja_entrada)
        self.grafo = None
        
    def ejecutar(self):
        print("🎯 [Optimizador] Agente activo y escuchando...")
        while True:
            try:
                msg = self.bandeja_entrada.get()

                # Si el grafo está listo
                if msg.receptor == self.nombre and msg.contenido == "grafo_listo":
                    self.run_once()

                # Si viene una petición con preferencias y posiblemente eventos filtrados
                if (
                    msg.receptor == self.nombre
                    and isinstance(msg.contenido, dict)
                    and "preferencias" in msg.contenido
                ):
                    preferencias = msg.contenido["preferencias"]
                    respuesta_key = msg.contenido["respuesta"]

                    try:
                        eventos = msg.contenido.get("eventos_filtrados")
                        if not eventos:
                            embedder = EventEmbedder._instance
                            eventos = embedder.events

                        scores = {
                            ev.get("basic_info", {}).get("title", ""): 1.0
                            for ev in eventos
                        }

                        agenda, score, _ = obtener_eventos_optimales(
                            eventos,
                            preferencias,
                            cantidad=5,
                            scores_grafo=scores,
                        )

                        self.enviar_respuesta_api(respuesta_key, agenda, score)

                    except Exception as e:
                        print(f"❌ [Optimizador] Error: {e}")
                        self.enviar_error_api(respuesta_key, str(e))

            except Exception as e:
                print(f"🔥 [Optimizador] Error crítico: {e}")
                time.sleep(1)


    def enviar_respuesta_api(self, clave_respuesta, agenda, score):
        """Envía respuesta por todos los canales disponibles"""
        respuesta = {
            "key": clave_respuesta,
            "data": {
                "agenda": agenda,
                "score": score
            }
        }
        
        # 1. Sistema de colas dedicadas
        try:
            with response_queues_lock:
                if clave_respuesta in response_queues:
                    response_queues[clave_respuesta].put(respuesta)
                    print("✅ [Optimizador] Respuesta enviada a cola dedicada")
        except Exception as e:
            print(f"⚠️ [Optimizador] Error en cola dedicada: {e}")
        
        # 2. Sistema tradicional de mensajes
        try:
            bandeja_api = obtener_bandeja("api")
            if bandeja_api:
                bandeja_api.put(Mensaje(
                    emisor=self.nombre,
                    receptor="api",
                    contenido=respuesta
                ))
                print("✅ [Optimizador] Respuesta enviada a bandeja API")
        except Exception as e:
            print(f"⚠️ [Optimizador] Error enviando a bandeja API: {e}")

    def enviar_error_api(self, clave_respuesta, error):
        """Envía mensaje de error"""
        self.enviar_respuesta_api(clave_respuesta, [], error)
class AgenteGapFallback(AgenteBase):
    def __init__(self, nombre, bandeja_entrada):
        super().__init__(nombre, bandeja_entrada)
        

    def ejecutar(self):
        print("📡 [GapFallback] Agente activo y escuchando...")
        while True:
            try:
                msg = self.bandeja_entrada.get()
                if msg.receptor != self.nombre or not isinstance(msg.contenido, dict):
                    continue

                print(f"⚠️ [GapFallback] Disparo solicitado: {msg.contenido}")

                query = msg.contenido.get("query", "")
                ciudad = msg.contenido.get("ciudad", None)
                categoria = msg.contenido.get("categoria", None)
                respuesta_clave = msg.contenido.get("respuesta", f"fallback_resultado_{uuid.uuid4().hex}")

                # Procesamiento de fechas
                fecha_inicio = self.normalizar_fecha(msg.contenido.get("fecha_inicio"))
                fecha_fin = self.normalizar_fecha(msg.contenido.get("fecha_fin"))

                # Validación de filtros mínimos
                if not any([ciudad, categoria, fecha_inicio, fecha_fin]):
                    print("🚫 [GapFallback] Fallback ignorado: no hay filtros suficientes.")
                    self.enviar_respuesta_error(
                        respuesta_clave,
                        "No se ejecutó fallback: se requiere al menos ciudad, categoría o fecha"
                    )
                    continue

                # Llamada al API de fallback
                eventos = fallback_api_call(
                    query=query,
                    start_date=fecha_inicio.isoformat() if fecha_inicio else None,
                    end_date=fecha_fin.isoformat() if fecha_fin else None,
                    source="ticketmaster"
                )

                if eventos:
                    # Procesar y ordenar eventos
                    eventos_ordenados = ordenar_y_paginar(eventos, page=1, limit=10)
                    print(f"🎯 [GapFallback] Recuperados {len(eventos)} eventos")

                    # Enviar respuesta
                    self.enviar_respuesta_exitosa(
                        respuesta_clave,
                        eventos_ordenados,
                        f"Eventos recuperados vía fallback: {len(eventos)}"
                    )

                    # Agregar eventos al sistema
                    embedder = EventEmbedder._instance
                    embedder.add_new_events(eventos)
                    print(f"📥 [GapFallback] {len(eventos)} eventos agregados dinámicamente")
                else:
                    self.enviar_respuesta_error(respuesta_clave, "No se encontraron eventos en el fallback")
                    
            except Exception as e:
                print(f"❌ [GapFallback] Error: {e}")
                self.enviar_respuesta_error(
                    f"fallback_error_{uuid.uuid4().hex}",
                    f"Error en fallback: {str(e)}"
                )
       

    def normalizar_fecha(self, fecha):
        """Normaliza fechas de diferentes formatos a datetime.date"""
        if not fecha:
            return None
        try:
            if isinstance(fecha, str):
                return datetime.fromisoformat(fecha.split('T')[0]).date()
            elif isinstance(fecha, (datetime, date)):
                return fecha.date() if isinstance(fecha, datetime) else fecha
            return None
        except Exception as e:
            print(f"⚠️ [GapFallback] Error normalizando fecha: {e}")
            return None

    def enviar_respuesta_exitosa(self, clave_respuesta, datos, mensaje=""):
        """Envía respuesta exitosa por ambos canales"""
        respuesta = {
            "key": clave_respuesta,
            "data": datos,
            "mensaje": mensaje,
            "status": "success"
        }

        # 1. Sistema de colas dedicadas
        try:
            with self.response_queues_lock:
                if clave_respuesta in self.response_queues:
                    self.response_queues[clave_respuesta].put(respuesta)
                    print(f"✅ [GapFallback] Respuesta enviada a cola dedicada ({clave_respuesta})")
        except Exception as e:
            print(f"⚠️ [GapFallback] Error en cola dedicada: {e}")

        # 2. Sistema tradicional de mensajes
        try:
            bandeja_api = obtener_bandeja("api")
            if bandeja_api:
                bandeja_api.put(Mensaje(
                    emisor=self.nombre,
                    receptor="api",
                    contenido=respuesta
                ))
                print(f"✅ [GapFallback] Respuesta enviada a bandeja API tradicional")
        except Exception as e:
            print(f"⚠️ [GapFallback] Error enviando a bandeja API: {e}")
   

    def enviar_respuesta_error(self, clave_respuesta, mensaje_error):
        """Envía mensaje de error"""
        print(f"⚠️ [GapFallback] Enviando error: {mensaje_error}")
        self.enviar_respuesta_exitosa(
            clave_respuesta,
            [],
            mensaje_error
        )             
class AgenteBusquedaInteractiva(AgenteBase):
    def __init__(self, nombre, bandeja_entrada):
        super().__init__(nombre, bandeja_entrada)
        self.embedder = None
        self.last_embedder_check = 0
        

    def get_embedder(self):
        """Obtiene el embedder con cache de 5 segundos"""
        current_time = time.time()
        if self.embedder is None or (current_time - self.last_embedder_check) > 5:
            self.embedder = EventEmbedder.get_instance()
            self.last_embedder_check = current_time
        return self.embedder

    def ejecutar(self):
        print("📡 [BusquedaInteractiva] Agente activo y escuchando...")
        while True:
            try:
                msg = self.bandeja_entrada.get()
                if msg.receptor != self.nombre:
                    self.bandeja_entrada.put(msg)
                    continue

                if not isinstance(msg.contenido, dict):
                    print(f"⚠️ [BusquedaInteractiva] Mensaje inválido ignorado: {msg}")
                    continue

                query = msg.contenido.get("query", "").strip()
                ciudad = msg.contenido.get("ciudad")
                categoria = msg.contenido.get("categoria")
                fecha_inicio = msg.contenido.get("fecha_inicio")
                fecha_fin = msg.contenido.get("fecha_fin")
                clave_respuesta = msg.contenido.get("respuesta", f"res_busqueda_{uuid.uuid4().hex}")

                print(f"🔍 [BusquedaInteractiva] Nueva búsqueda - Query: '{query}' | Ciudad: {ciudad} | Categoría: {categoria}")

                if not query:
                    self.enviar_respuesta_error(clave_respuesta, "La consulta no puede estar vacía")
                    continue

                try:
                    embedder = self.get_embedder()
                    if embedder is None:
                        raise RuntimeError("Embedder no disponible")
                except Exception as e:
                    self.enviar_respuesta_error(clave_respuesta, f"Error en el sistema de búsqueda: {str(e)}")
                    continue

                try:
                    usar_filtros = any([ciudad, categoria])

                    if usar_filtros:
                        if ciudad and ciudad in embedder.shards and not categoria :
                            resultados, _ = embedder.search(query=query, shard_key=ciudad, k=60)
                        elif ciudad and categoria:
                            resultados, _ = embedder.filtered_search(
                                query=query,
                                city=ciudad,
                                category=categoria,
                                fecha_inicio=fecha_inicio,
                                fecha_fin=fecha_fin,
                                k=60
                            )
                        else :
                            resultados = embedder.filtered_search(
                                query=query,
                                city= None,
                                category=categoria,
                                fecha_inicio=fecha_inicio,
                                fecha_fin=fecha_fin,
                                k=60)
                    else:
                        resultados, _ = embedder.search(query=query, k=60 , shard_key=None)

                    eventos_finales = limpiar_eventos(resultados)
                    total_resultados = len(eventos_finales)

                    print(f"📊 [BusquedaInteractiva] Resultados encontrados: {total_resultados}")

                    self.enviar_respuesta_exitosa(
                        clave_respuesta,
                        eventos_finales,
                        f"Se encontraron {total_resultados} evento(s)"
                    )

                    if total_resultados < 3:
                        self.disparar_fallback(
                            query=query,
                            ciudad=ciudad,
                            categoria=categoria,
                            fecha_inicio=fecha_inicio,
                            fecha_fin=fecha_fin,
                            clave_respuesta=clave_respuesta
                        )

                except Exception as e:
                    print(f"❌ [BusquedaInteractiva] Error en búsqueda: {str(e)}")
                    self.enviar_respuesta_error(clave_respuesta, f"Error al procesar la búsqueda: {str(e)}")

            except Exception as e:
                print(f"🔥 [BusquedaInteractiva] Error crítico en ciclo principal: {str(e)}")
                time.sleep(1)
            self.enviar("coordinador", "busqueda_exitosa")

    def enviar_respuesta_exitosa(self, clave_respuesta, datos, mensaje=""):
        """Envía una respuesta exitosa por todos los canales disponibles"""
        respuesta = {
            "key": clave_respuesta,
            "data": datos,
            "mensaje": mensaje,
            "status": "success"
        }

        # 1. Enviar a través del sistema de colas dedicado (nuevo)
        try:
            with response_queues_lock:
                if clave_respuesta in response_queues:
                    response_queues[clave_respuesta].put(respuesta)
                    print(f"✅ [BusquedaInteractiva] Respuesta enviada a cola dedicada")
        except Exception as e:
            print(f"⚠️ [BusquedaInteractiva] Error enviando a cola dedicada: {str(e)}")

        # 2. Enviar a través del sistema tradicional de mensajes (backup)
        try:
            bandeja_api = obtener_bandeja("api")
            if bandeja_api:
                bandeja_api.put(Mensaje(
                    emisor=self.nombre,
                    receptor="api",
                    contenido=respuesta
                ))
                print(f"✅ [BusquedaInteractiva] Respuesta enviada a bandeja API tradicional")
        except Exception as e:
            print(f"⚠️ [BusquedaInteractiva] Error enviando a bandeja API: {str(e)}")

    def enviar_respuesta_error(self, clave_respuesta, mensaje_error):
        """Envía un mensaje de error"""
        print(f"⚠️ [BusquedaInteractiva] Enviando error: {mensaje_error}")
        self.enviar_respuesta_exitosa(
            clave_respuesta,
            [],
            mensaje_error
        )

    def disparar_fallback(self, query, ciudad, categoria, fecha_inicio, fecha_fin, clave_respuesta):
        """Dispara el mecanismo de fallback para obtener más resultados"""
        print("⚠️ [BusquedaInteractiva] Activando fallback por baja cobertura")
        
        fallback_msg = {
            "query": query,
            "ciudad": ciudad,
            "categoria": categoria,
            "fecha_inicio": fecha_inicio,
            "fecha_fin": fecha_fin,
            "respuesta": clave_respuesta
        }

        try:
            bandeja_fallback = obtener_bandeja("gapfallback")
            if bandeja_fallback:
                bandeja_fallback.put(Mensaje(
                    emisor=self.nombre,
                    receptor="gapfallback",
                    contenido=fallback_msg
                ))
                print(f"✅ [BusquedaInteractiva] Mensaje de fallback enviado")
            else:
                print("❌ [BusquedaInteractiva] No se pudo obtener bandeja para fallback")
        except Exception as e:
            print(f"❌ [BusquedaInteractiva] Error al disparar fallback: {str(e)}")
            
def arranque_secuencial():
    from core.embedding import load_events_from_folder

    eventos_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../eventos_mejorados"))
    eventos = load_events_from_folder(eventos_dir)

    try:
        # ✅ Cargar embeddings existentes o generarlos
        embedder = EventEmbedder.load("embedding_data")
        print("✅ Embeddings cargados desde disco.")
    except Exception as e:
        print(f"⚠️ No se pudo cargar embeddings: {e}")
        print("🔁 Generando embeddings desde cero...")
        embedder = EventEmbedder()
        embedder.generate_embeddings(eventos)
        embedder.build_index(EventEmbedder._embeddings, index_type="IVFFlat")
        embedder.save("embedding_data")

    # ✅ Crear agentes y registrar sus bandejas
    scraper = AgenteScraper("scraper", Queue())
    procesador = AgenteProcesador("procesador", Queue())
    embedding = AgenteEmbedding("embedding", Queue())
    grafo = AgenteGrafo("grafo", Queue())

    registrar_bandeja("scraper", scraper.bandeja_entrada)
    registrar_bandeja("procesador", procesador.bandeja_entrada)
    registrar_bandeja("embedding", embedding.bandeja_entrada)
    registrar_bandeja("grafo", grafo.bandeja_entrada)

    # ✅ Ejecutar agentes principales una vez
    scraper.run_once()
    procesador.run_once()
    embedding.run_once()
    grafo.run_once()
    
# === Función para iniciar agentes en threads (escuchando mensajes) ===
def iniciar_threads():
    agentes = [
        AgenteCoordinador("coordinador", Queue()),
        AgenteScraper("scraper", Queue()),
        AgenteProcesador("procesador", Queue()),
        AgenteActualizador("actualizador", Queue()),
        AgenteGrafo("grafo", Queue()),
        AgenteEmbedding("embedding", Queue()),
        AgenteOptimizador("optimizador", Queue()),
        AgenteGapFallback("gapfallback", Queue()),
        AgenteBusquedaInteractiva("busqueda_interactiva", Queue()),
    ]

    for agente in agentes:
        registrar_bandeja(agente.nombre, agente.bandeja_entrada)

    hilos = [Thread(target=a.ejecutar, daemon=True) for a in agentes]
    for h in hilos:
        h.start()
    print("✅ Todos los agentes con bandeja propia están activos.")

# === Función para enviar mensaje que inicia la cadena de actualización periódica ===
def disparar_actualizacion_periodica(bandeja):
    # Inicia el proceso enviando el mensaje que dispara el flujo
    bandeja.put(Mensaje("periodico", "scraper", "scraping_terminado"))

# === Hilo que corre actualización cada 7 días ===
def hilo_actualizacion_periodica(bandeja):
    while True:
        print("⏳ Esperando 7 días para próxima actualización...")
        time.sleep(7 * 24 * 3600)  # 7 días
        print("🔄 Actualización periódica iniciada")
        disparar_actualizacion_periodica(bandeja)

def cargar_ciudades_categorias():
    """Carga dinámica de ciudades y categorías disponibles"""
    try:
        embedder = EventEmbedder()
        if not embedder.events:
            eventos = load_events_from_folder("eventos_mejorados")
            embedder.events = eventos

        ciudades = set()
        categorias = set()
        
        for ev in embedder.events:
            if ev.get("spatial_info", {}).get("area", {}).get("city"):
                ciudades.add(ev["spatial_info"]["area"]["city"])
            if ev.get("classification", {}).get("primary_category"):
                categorias.add(ev["classification"]["primary_category"])
        
        return sorted(ciudades), sorted(categorias)
    except Exception as e:
        print(f"⚠️ Error cargando ciudades/categorías: {e}")
        return [], []

def actualizar_visual(bandeja):
    """Envía las ciudades/categorías disponibles al frontend"""
    while True:
        time.sleep(30)  # Actualizar cada 30 segundos
        try:
            ciudades, categorias = cargar_ciudades_categorias()
            bandeja.put(Mensaje(
                "sistema",
                "visual",
                {"ciudades": ciudades, "categorias": categorias}
            ))
        except Exception as e:
            print(f"⚠️ Error actualizando frontend: {e}")
def limpiar_eventos(lista):
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
            print(f"⚠️ Error limpiando evento: {e}")
    return nuevos
# Reemplazar la parte del lanzamiento de Streamlit con:
def verificar_api_lista():
    import requests
    for _ in range(30):  # Intentar por 30 segundos máximo
        try:
            res = requests.get("http://localhost:8502/status", timeout=2)
            if res.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

if __name__ == "__main__":
    print("🚀 Iniciando sistema multiagente...")

   # 1. Inicialización del sistema
    bandeja = Queue()
    registrar_bandeja("coordinador", bandeja)

    registrar_bandeja("api", bandeja)
    # 2. Ejecutar arranque secuencial PRIMERO
    print("⚙️ Ejecutando arranque secuencial...")
    arranque_secuencial()

    # 3. Iniciar API después de tener embeddings
    print("🌐 Iniciando servidor API...")
    api_thread = Thread(target=iniciar_api)
    api_thread.start()

    # 4. Iniciar agentes en segundo plano
    print("🤖 Iniciando agentes en segundo plano...")
    iniciar_threads()

    # 5. Lanzar frontend cuando todo esté listoP
    if verificar_api_lista():
        print("🖥️ Lanzando interfaz visual...")
        subprocess.Popen([
            "streamlit", 
            "run", 
            "./src/app/visual.py",
            "--server.address=localhost",
            "--server.port=8501"
        ])
    else:
        print("❌ No se pudo verificar el estado de la API")

    
    api_thread.join()
