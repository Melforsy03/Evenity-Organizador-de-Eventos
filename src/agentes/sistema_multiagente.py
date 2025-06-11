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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import networkx as nx
from core.embedding import fallback_api_call, EventEmbedder
from scraping.crawler import EventScraper
from core.procesamiento import procesar_json,EventProcessor
from scraping.actualizador_dinamico import actualizar_eventos_y_embeddings
from core.grafo_conocimiento import get_knowledge_graph 
from core.embedding import run_embedding , EventEmbedder
from core.optimizador import obtener_eventos_optimales
from core.embedding import load_events_from_folder, EventEmbedder
from api.lanzar_api import iniciar_api  
from api.contexto_global import registrar_bandeja , obtener_bandeja
from datetime import datetime
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
    def run_once(self, intento=1):
        print(f"🎯 [Optimizador] Generando agenda óptima (intento {intento})...")
        try:
            embedder = EventEmbedder._instance
            eventos = embedder.events[:50]
            preferencias = {"location": None, "categories": [], "available_dates": None}
            scores = {ev.get("basic_info", {}).get("title", ""): 1.0 for ev in eventos}
            print(f"[Optimizador] Recibido mensaje de API con preferencias: {preferencias}")

            agenda, score, _ = obtener_eventos_optimales(eventos, preferencias, cantidad=5, scores_grafo=scores)
            print(f"✅ [Optimizador] Agenda generada (score: {score:.2f})")
            for i, ev in enumerate(agenda, 1):
                print(f"{i}. {ev.get('basic_info', {}).get('title', 'Sin título')}")
        except Exception as e:
            print(f"❌ [Optimizador] Error al generar agenda: {e}")
            if intento < 3:
                time.sleep(5 * intento)
                self.run_once(intento + 1)
            else:
                print("🛑 [Optimizador] Fallo permanente tras 3 intentos.")

    def ejecutar(self):
        while True:
            try:
                msg = self.bandeja_entrada.get()

                # ✅ 1. Si el grafo está listo, generar agenda por cuenta propia (modo automático)
                if msg.receptor == self.nombre and msg.contenido == "grafo_listo":
                    self.run_once()

                # ✅ 2. Si viene una petición desde la API con preferencias
                if msg.receptor == self.nombre and isinstance(msg.contenido, dict) and "preferencias" in msg.contenido:
                    preferencias = msg.contenido["preferencias"]
                    respuesta_key = msg.contenido["respuesta"]

                    embedder = EventEmbedder._instance
                    eventos = embedder.events
                    scores = {ev.get("basic_info", {}).get("title", ""): 1.0 for ev in eventos}

                    try:
                        from core.optimizador import obtener_eventos_optimales
                        agenda, score, _ = obtener_eventos_optimales(
                            eventos, preferencias, cantidad=5, scores_grafo=scores
                        )

                        self.enviar("api", {
                            "key": respuesta_key,
                            "data": {
                                "agenda": agenda,
                                "score": score
                            }
                        })
                    except Exception as e:
                        self.enviar("api", {
                            "key": respuesta_key,
                            "error": str(e)
                        })

            except Exception as e:
                print(f"[Optimizador] ❌ Error: {e}")

class AgenteGapFallback(AgenteBase):
    def ejecutar(self):
        print("📡 [GapFallback] Agente activo y escuchando...")
        while True:
            try:
                msg = self.bandeja_entrada.get()
                if msg.receptor != self.nombre or not isinstance(msg.contenido, dict):
                    continue

                print(f"⚠️ [GapFallback] Disparo solicitado: {msg.contenido}")

                query = msg.contenido.get("query", "")
                ciudad = msg.contenido.get("ciudad", None)  # aún no usado
                categoria = msg.contenido.get("categoria", None)  # aún no usado
                respuesta_clave = msg.contenido.get("respuesta", "fallback_resultado")

                fecha_inicio = msg.contenido.get("fecha_inicio")
                fecha_fin = msg.contenido.get("fecha_fin")

                # Conversión segura si llegan fechas tipo string
                if isinstance(fecha_inicio, str):
                    fecha_inicio = fecha_inicio.split("T")[0]
                if isinstance(fecha_fin, str):
                    fecha_fin = fecha_fin.split("T")[0]

                # Si no hay ni ciudad, ni categoría, ni fechas => NO DISPARAR
                if not any([ciudad, categoria, fecha_inicio, fecha_fin]):
                    print("🚫 [GapFallback] Fallback ignorado: no hay filtros suficientes.")
                    self.enviar("api", {
                        "key": respuesta_clave,
                        "data": [],
                        "mensaje": "No se ejecutó fallback: se requiere al menos ciudad, categoría o fecha"
                    })
                    continue
                eventos = fallback_api_call(
                    query=query,
                    start_date=fecha_inicio,
                    end_date=fecha_fin,
                    source="ticketmaster"  # puedes cambiar esto dinámicamente
                )

                print(f"🎯 [GapFallback] Recuperados {len(eventos)} eventos")

                self.enviar("api", {
                    "key": respuesta_clave,
                    "data": eventos,
                    "mensaje": f"Eventos recuperados vía fallback: {len(eventos)}"
                })
                if eventos:
                    embedder = EventEmbedder.get_instance()
                    embedder.agregar_eventos(eventos)
                    print(f"📥 [GapFallback] {len(eventos)} eventos agregados dinámicamente al sistema") 
                    
            except Exception as e:
                print(f"❌ [GapFallback] Error: {e}")
                self.enviar("api", {
                    "key": "fallback_resultado",
                    "error": str(e)
                })
                
class AgenteBusquedaInteractiva(AgenteBase):
    def ejecutar(self):
        print("📡 [BusquedaInteractiva] Agente activo y escuchando...")
        while True:
            try:
                msg = self.bandeja_entrada.get()
                if msg.receptor != self.nombre:
                    self.bandeja_entrada.put(msg)  # volver a poner si no es nuestro
                    continue
                if not isinstance(msg.contenido, dict):
                    print(f"⚠️ [BusquedaInteractiva] Mensaje inválido ignorado: {msg}")
                    continue


                query = msg.contenido.get("query", "").strip()
                ciudad = msg.contenido.get("ciudad")
                categoria = msg.contenido.get("categoria")
                fecha_inicio = msg.contenido.get("fecha_inicio")
                fecha_fin = msg.contenido.get("fecha_fin")
                clave_respuesta = msg.contenido.get("respuesta", "res_busqueda")

                print(f"🔍 [BusquedaInteractiva] Recibida query: '{query}'")

                if not query:
                    self.enviar("api", {
                        "key": clave_respuesta,
                        "data": [],
                        "mensaje": "Consulta vacía"
                    })
                    continue

                embedder = EventEmbedder.get_instance()

                # === Mejora 1: usar shards si hay ciudad
                if ciudad and ciudad in embedder.shards:
                    resultados, _ = embedder.search(query, shard_key=ciudad, k=20)
                else:
                    resultados = embedder.buscar_eventos(query, top_k=20)

                # === Mejora 2: convertir fechas solo una vez
                fecha_inicio_dt = datetime.fromisoformat(fecha_inicio).date() if fecha_inicio else None
                fecha_fin_dt = datetime.fromisoformat(fecha_fin).date() if fecha_fin else None

                # === Mejora 3: aplicar filtros
                eventos_filtrados = []
                for ev in resultados:
                    ciudad_ev = ev.get("spatial_info", {}).get("area", {}).get("city")
                    categoria_ev = ev.get("classification", {}).get("primary_category")
                    fecha_ev_str = ev.get("temporal_info", {}).get("start")

                    if ciudad and ciudad != ciudad_ev:
                        continue
                    if categoria and categoria != categoria_ev:
                        continue
                    if fecha_ev_str:
                        try:
                            fecha_ev = datetime.fromisoformat(fecha_ev_str).date()
                            if fecha_inicio_dt and fecha_ev < fecha_inicio_dt:
                                continue
                            if fecha_fin_dt and fecha_ev > fecha_fin_dt:
                                continue
                        except:
                            continue

                    eventos_filtrados.append(ev)

                print(f"📊 [BusquedaInteractiva] Eventos tras filtro: {len(eventos_filtrados)}")

                # === Mejora 4: responder siempre, incluso con pocos eventos
                self.enviar("api", {
                    "key": clave_respuesta,
                    "data": eventos_filtrados,
                    "mensaje": f"Se encontraron {len(eventos_filtrados)} evento(s) tras búsqueda principal"
                })

                # Si son menos de 2, disparar fallback
                if len(eventos_filtrados) < 2:
                    print("⚠️ [BusquedaInteractiva] Disparando fallback por baja cobertura")

                    fallback_msg = {
                        "query": query,
                        "ciudad": ciudad,
                        "categoria": categoria,
                        "fecha_inicio": fecha_inicio,
                        "fecha_fin": fecha_fin,
                        "respuesta": clave_respuesta
                    }
                    bandeja_fallback = obtener_bandeja("gapfallback")
                    if bandeja_fallback:
                        bandeja_fallback.put(Mensaje(
                            emisor=self.nombre,
                            receptor="gapfallback",
                            contenido=fallback_msg
                        ))
                    else:
                        print("❌ No se pudo obtener la bandeja para gapfallback")

            except Exception as e:
                print(f"❌ [BusquedaInteractiva] Error: {e}")

def arranque_secuencial(bandeja):
    eventos_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../eventos_mejorados"))
    eventos = load_events_from_folder(eventos_dir)

    try:
        # ✅ Intentar cargar embeddings existentes
        embedder = EventEmbedder.load("embedding_data")
        print("✅ Embeddings cargados desde disco.")
    except Exception as e:
        print(f"⚠️ No se pudo cargar embeddings: {e}")
        print("🔁 Generando embeddings desde cero...")
        embedder = EventEmbedder()
        embedder.generate_embeddings(eventos)
        embedder.build_index(EventEmbedder._embeddings, index_type="IVFFlat")
        embedder.save("embedding_data")

        scraper = AgenteScraper("scraper", bandeja)
        procesador = AgenteProcesador("procesador", bandeja)
        embedding = AgenteEmbedding("embedding", bandeja)
        grafo = AgenteGrafo("grafo", bandeja)

    scraper.run_once()
    procesador.run_once()
    embedding.run_once()
    grafo.run_once()

# === Función para iniciar agentes en threads (escuchando mensajes) ===
def iniciar_threads():
    agentes = [
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
    registrar_bandeja("api", bandeja)
    # 2. Ejecutar arranque secuencial PRIMERO
    print("⚙️ Ejecutando arranque secuencial...")
    arranque_secuencial(bandeja)

    # 3. Iniciar API después de tener embeddings
    print("🌐 Iniciando servidor API...")
    api_thread = Thread(target=iniciar_api)
    api_thread.start()

    # 4. Iniciar agentes en segundo plano
    print("🤖 Iniciando agentes en segundo plano...")
    iniciar_threads(bandeja)

    # 5. Lanzar frontend cuando todo esté listo
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
