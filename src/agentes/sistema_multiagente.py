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
from core.grafo_conocimiento import get_knowledge_graph , enriquecer_resultados_con_razonamiento_avanzado
from core.embedding import run_embedding , EventEmbedder
from core.optimizador import obtener_eventos_optimales
from core.embedding import load_events_from_folder, EventEmbedder
from api.lanzar_api import iniciar_api  
from api.contexto_global import set_bandeja_global

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
                if msg.receptor == self.nombre and msg.contenido == "scraping_terminado":
                    # Acción si se desea al recibir el mensaje
                    pass
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
    def run_once(self):
        print("🕳️ [GapFallback] Analizando cobertura de eventos...")
        embedder = EventEmbedder._instance
        eventos_actuales = embedder.events

        # --- ANALIZAR GAPS ---
        ciudad_eventos = {}
        categoria_eventos = {}
        fecha_eventos = {}

        for ev in eventos_actuales:
            ciudad = ev.get("spatial_info", {}).get("area", {}).get("city")
            cat = ev.get("classification", {}).get("primary_category")
            fecha = ev.get("temporal_info", {}).get("start", "")

            if ciudad:
                ciudad_eventos.setdefault(ciudad, []).append(ev)
            if cat:
                categoria_eventos.setdefault(cat, []).append(ev)
            if fecha:
                fecha_clave = fecha[:7]  # agrupar por mes (YYYY-MM)
                fecha_eventos.setdefault(fecha_clave, []).append(ev)

        ciudades_con_pocos = [c for c, evs in ciudad_eventos.items() if len(evs) < 5]
        categorias_con_pocos = [c for c, evs in categoria_eventos.items() if len(evs) < 5]
        meses_con_pocos = [m for m, evs in fecha_eventos.items() if len(evs) < 5]

        print(f"🏙️ Ciudades con pocos eventos: {ciudades_con_pocos}")
        print(f"📂 Categorías con pocos eventos: {categorias_con_pocos}")
        print(f"📅 Meses con pocos eventos: {meses_con_pocos}")

        # --- SCRAPING DE FALLBACK ---
        queries = []

        for ciudad in ciudades_con_pocos:
            queries.append(f"eventos en {ciudad}")
        for cat in categorias_con_pocos:
            queries.append(f"eventos de {cat}")
        for mes in meses_con_pocos:
            queries.append(f"eventos en {mes}")

        nuevos_eventos = []
        processor = EventProcessor()
        titulos_existentes = {e.get("basic_info", {}).get("title", "").lower() for e in eventos_actuales}

        for query in set(queries):
            for fuente in ["seatgeek", "predicthq", "ticketmaster"]:
                eventos_api = fallback_api_call(query, source=fuente)
                for ev in eventos_api:
                    titulo = ev.get("basic_info", {}).get("title", "").lower()
                    if titulo and titulo not in titulos_existentes:
                        nuevos_eventos.append(ev)

        if not nuevos_eventos:
            print("❌ [GapFallback] No se encontraron eventos nuevos en fallback.")
            return

        print(f"🆕 [GapFallback] Se recuperaron {len(nuevos_eventos)} eventos nuevos.")

        # --- GUARDAR NUEVOS EVENTOS COMO ARCHIVOS INDIVIDUALES ---
        output_folder = "eventos_mejorados"
        os.makedirs(output_folder, exist_ok=True)

        for ev in nuevos_eventos:
            titulo = ev.get("basic_info", {}).get("title", "").replace(" ", "_")[:50]
            safe_name = "".join(c for c in titulo if c.isalnum() or c in "_").rstrip("_")
            nombre_archivo = f"gap_event_{safe_name}.json"
            path = os.path.join(output_folder, nombre_archivo)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(ev, f, ensure_ascii=False, indent=2)

        # --- CARGAR ÍNDICE Y AÑADIR NUEVOS EMBEDDINGS ---
        try:
            embedder.load_index("embedding_data/eventos.index")
        except FileNotFoundError:
            print("⚠️ Índice inexistente, se generará uno nuevo.")
            # Generar desde cero
            embedder.generate_embeddings(eventos_actuales + nuevos_eventos)
            embedder.build_index(EventEmbedder._embeddings, index_type="IVFFlat")
            embedder.save("embedding_data")
            print("✅ Índice creado desde cero con nuevos eventos.")
            return

        embedder.add_new_events(nuevos_eventos)
        embedder.save_event_data("embedding_data")

        print("✅ [GapFallback] Nuevos embeddings generados e integrados.")
        self.enviar("visual", "actualizar_embedding")

    def ejecutar(self):
        while True:
            try:
                msg = self.bandeja_entrada.get()
                if msg.receptor == self.nombre and msg.contenido in ["verificar_gaps", "forzar_fallback"]:
                    self.run_once()
            except Exception as e:
                print(f"[GapFallback] ❌ Error: {e}")

class AgenteBusquedaInteractiva(AgenteBase):
    def __init__(self, nombre, bandeja_entrada):
        super().__init__(nombre, bandeja_entrada)
        from core.embedding import EventEmbedder
        self.embedder = EventEmbedder._instance
        self.grafo = None

    def ejecutar(self):
        while True:
            try:
                msg = self.bandeja_entrada.get()

                if msg.receptor == self.nombre and isinstance(msg.contenido, dict) and "query" in msg.contenido:
                    query = msg.contenido.get("query", "").strip()
                    ciudad = msg.contenido.get("ciudad", "").strip()
                    fecha_i = msg.contenido.get("fecha_inicio")
                    fecha_f = msg.contenido.get("fecha_fin")
                    categoria = msg.contenido.get("categoria", "").strip()
                    respuesta_key = msg.contenido.get("respuesta", "res_busqueda")

                    # ✅ Validar que la query esté presente
                    if not query:
                        print("⚠️ [BusquedaInteractiva] Query vacía. Ignorando búsqueda.")
                        self.enviar("api", {
                            "key": respuesta_key,
                            "data": [],
                            "error": "Debe ingresar una consulta de búsqueda (query)."
                        })
                        continue

                    print(f"🔍 [BusquedaInteractiva] Procesando búsqueda: {query}")

                    # Búsqueda semántica inicial
                    resultados, _ = self.embedder.filtered_search(query, k=100)

                    # Filtro por ciudad (si se proporciona)
                    if ciudad:
                        resultados = [ev for ev in resultados if ev.get("spatial_info", {}).get("area", {}).get("city", "").lower() == ciudad.lower()]

                    # Guardar tamaño antes de aplicar categoría
                    count_pre_categoria = len(resultados)

                    # Filtro por categoría (si se proporciona)
                    if categoria:
                        resultados = [ev for ev in resultados if ev.get("classification", {}).get("primary_category") == categoria]

                    # Filtro por fechas
                    if fecha_i and fecha_f:
                        resultados = [
                            ev for ev in resultados
                            if ev.get("temporal_info", {}).get("start")
                            and fecha_i <= ev["temporal_info"]["start"][:10] <= fecha_f
                        ]

                    # Enriquecer con grafo si está disponible
                    if resultados and self.grafo:
                        resultados = [ev for ev, _ in enriquecer_resultados_con_razonamiento_avanzado(resultados, self.grafo, query, k=50)]

                    # 🔥 Disparar fallback si hay pocas coincidencias
                    condiciones = [
                        len(resultados) == 0,
                        (categoria and count_pre_categoria <= 1),
                        (fecha_i and fecha_f and all(
                            not ev.get("temporal_info", {}).get("start") or not (fecha_i <= ev["temporal_info"]["start"][:10] <= fecha_f)
                            for ev in self.embedder.events
                        ))
                    ]
                    if any(condiciones):
                        print("🛑 [BusquedaInteractiva] Disparando fallback por baja cobertura...")
                        self.enviar("gapfallback", "forzar_fallback")

                    # Guardar resultados
                    with open("resultados_interactivos.json", "w", encoding="utf-8") as f:
                        json.dump(resultados, f, ensure_ascii=False, indent=2)

                    print(f"✅ [BusquedaInteractiva] Resultados guardados: {len(resultados)} eventos")

                    # Responder a la API si aplica
                    if respuesta_key:
                        self.enviar("api", {
                            "key": respuesta_key,
                            "data": resultados
                        })

                elif msg.receptor == self.nombre and isinstance(msg.contenido, dict) and "grafo" in msg.contenido:
                    self.grafo = msg.contenido["grafo"]
                    print("📥 [BusquedaInteractiva] Grafo recibido desde AgenteGrafo")

            except Exception as e:
                print(f"[BusquedaInteractiva] ❌ Error: {e}")
                self.enviar("visual", {"error": str(e)})

# === Función para arranque secuencial ===
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
def iniciar_threads(bandeja):
    agentes = [
        AgenteScraper("scraper", bandeja),
        AgenteProcesador("procesador", bandeja),
        AgenteActualizador("actualizador", bandeja),
        AgenteGrafo("grafo", bandeja),
        AgenteEmbedding("embedding", bandeja),
        AgenteOptimizador("optimizador", bandeja),
        AgenteGapFallback("gapfallback", bandeja),
        AgenteBusquedaInteractiva("busqueda_interactiva", bandeja),

    ]

    hilos = [Thread(target=a.ejecutar, daemon=True) for a in agentes]
    for h in hilos:
        h.start()
    print("✅ Todos los agentes en modo escucha iniciados.")

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

# En sistema_multiagente.py, modificar el main:
if __name__ == "__main__":
    print("🚀 Iniciando sistema multiagente...")

   # 1. Inicialización del sistema
    bandeja = Queue()
    set_bandeja_global(bandeja)      

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
            "./src/modules/visual.py",
            "--server.address=localhost",
            "--server.port=8501"
        ])
    else:
        print("❌ No se pudo verificar el estado de la API")

    
    api_thread.join()
