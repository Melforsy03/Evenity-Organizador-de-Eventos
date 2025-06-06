from queue import Queue
from threading import Thread
from abc import ABC, abstractmethod
import time
import subprocess
import json
import os
from embedding import fallback_api_call, EventEmbedder
from crawler import EventScraper
from procesamiento import procesar_json,EventProcessor
from actualizador_dinamico import actualizar_eventos_y_embeddings
from grafo_conocimiento import get_knowledge_graph , enriquecer_resultados_con_razonamiento_avanzado
from embedding import run_embedding , EventEmbedder
from optimizador import obtener_eventos_optimales

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
    def run_once(self):
        if os.path.exists("../../embedding_data/index.faiss"):
            print("📁 [Scraper] Datos ya presentes. Omitiendo scraping.")
            self.enviar("procesador", "scraping_terminado")
            return

        print("📡 [Scraper] Scrapeando eventos...")
        scraper = EventScraper()
        scraper.run_all_scrapers()
        print("✅ [Scraper] Scraping completo.")
        self.enviar("procesador", "scraping_terminado")


    def ejecutar(self):
        while True:
            msg = self.bandeja_entrada.get()
            if msg.receptor == self.nombre and msg.contenido == "scraping_terminado":
                # Aquí puedes reaccionar a mensajes si quieres
                pass


# === Agente 2: Procesador ===
class AgenteProcesador(AgenteBase):
    def run_once(self):
        print("🧪 [Procesador] Procesando eventos...")
        procesar_json()
        print("✅ [Procesador] Procesamiento completo.")
        self.enviar("actualizador", "eventos_procesados")

    def ejecutar(self):
        while True:
            msg = self.bandeja_entrada.get()
            if msg.receptor == self.nombre and msg.contenido == "scraping_terminado":
                self.run_once()


# === Agente 3: Actualizador ===
class AgenteActualizador(AgenteBase):
    def run_once(self):
        if not self.necesita_actualizacion():
            print("⏸️ [Actualizador] Datos recientes. No se actualiza.")
            return

        print("🔁 [Actualizador] Actualizando eventos y embeddings...")
        actualizar_eventos_y_embeddings()
        print("✅ [Actualizador] Actualización terminada.")
        self.enviar("grafo", "embeddings_actualizados")

    def necesita_actualizacion(self):
        try:
            last_modified = os.path.getmtime("embedding_data/index.faiss")  # o el archivo que quieras
            tiempo_actual = time.time()
            # 60 segundos × 60 minutos × 24 horas × 60 días ≈ dos meses
            return (tiempo_actual - last_modified) > (60 * 60 * 24 * 60)
        except:
            return True  # actualizar si no existe

# === Agente 4: Grafo de Conocimiento ===
class AgenteGrafo(AgenteBase):
    def run_once(self):
        print("🧠 [Grafo] Construyendo grafo de conocimiento...")
        self.grafo = get_knowledge_graph()
        print("✅ [Grafo] Grafo construido.")

        # Enviar grafo a otros agentes
        self.enviar("optimizador", "grafo_listo")
        self.enviar("busqueda_interactiva", {"grafo": self.grafo})  # 🔁 Nuevo

    def ejecutar(self):
        while True:
            msg = self.bandeja_entrada.get()
            if msg.receptor == self.nombre and msg.contenido == "embeddings_actualizados":
                self.run_once()


# === Agente 5: Embedding y Búsqueda ===
class AgenteEmbedding(AgenteBase):
    def run_once(self):
        print("💾 [Embedding] Generando embeddings...")
        run_embedding()
        print("✅ [Embedding] Embeddings listos.")

    def ejecutar(self):
        while True:
            msg = self.bandeja_entrada.get()
            if msg.receptor == self.nombre:
                if msg.contenido == "refrescar_embeddings":
                    self.run_once()


# === Agente 6: Optimizador ===
class AgenteOptimizador(AgenteBase):
    def run_once(self):
        print("🎯 [Optimizador] Generando agenda óptima...")
        embedder = EventEmbedder.load("embedding_data")
        eventos = embedder.events[:50]
        preferencias = {"location": None, "categories": [], "available_dates": None}
        scores = {ev.get("basic_info", {}).get("title", ""): 1.0 for ev in eventos}
        agenda, score = obtener_eventos_optimales(eventos, preferencias, cantidad=5, scores_grafo=scores)
        print(f"✅ [Optimizador] Agenda generada (score: {score:.2f})")
        for i, ev in enumerate(agenda, 1):
            print(f"{i}. {ev.get('basic_info', {}).get('title', 'Sin título')}")

    def ejecutar(self):
        while True:
            msg = self.bandeja_entrada.get()
            if msg.receptor == self.nombre and msg.contenido == "grafo_listo":
                self.run_once()

class AgenteVisual(AgenteBase):
    def __init__(self, nombre, bandeja_entrada):
        super().__init__(nombre, bandeja_entrada)
        self.proceso_streamlit = None

    def run_once(self):
        print("🖼️ [Visual] Iniciando interfaz Streamlit...")
        # Ejecutar Streamlit como subproceso
        self.proceso_streamlit = subprocess.Popen(
            ["streamlit", "run", "visual.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("✅ [Visual] Streamlit lanzado.")

    def ejecutar(self):
        self.run_once()
        while True:
            msg = self.bandeja_entrada.get()
            if msg.receptor == self.nombre:
                if msg.contenido == "actualizar_embedding":
                    print("🔁 [Visual] Señal recibida para actualizar embeddings")
                    with open("embedding_data_nuevo/signal.txt", "w") as f:
                        f.write("reload")

class AgenteGapFallback(AgenteBase):
    def run_once(self):
        print("🕳️ [GapFallback] Analizando cobertura de eventos...")
        embedder = EventEmbedder.load("embedding_data")
        eventos = embedder.events

        # --- ANALIZAR GAPS ---
        ciudad_eventos = {}
        categoria_eventos = {}
        fecha_eventos = {}

        for ev in eventos:
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

        for query in set(queries):
            for fuente in ["seatgeek", "predicthq", "ticketmaster"]:
                eventos_api = fallback_api_call(query, source=fuente)
                for ev in eventos_api:
                    titulo = ev.get("basic_info", {}).get("title")
                    if titulo and titulo.lower() not in {e.get("basic_info", {}).get("title", "").lower() for e in eventos + nuevos_eventos}:
                        nuevos_eventos.append(ev)


        if not nuevos_eventos:
            print("❌ [GapFallback] No se encontraron eventos nuevos en fallback.")
            return

        print(f"🆕 [GapFallback] Se recuperaron {len(nuevos_eventos)} eventos nuevos.")

        # --- GUARDAR NUEVOS EVENTOS ---
        output_folder = "eventos_mejorados"
        os.makedirs(output_folder, exist_ok=True)

        for ev in nuevos_eventos:
            titulo = ev.get("basic_info", {}).get("title", "").replace(" ", "_")[:50]
            safe_name = "".join(c for c in titulo if c.isalnum() or c in "_").rstrip("_")
            nombre_archivo = f"gap_event_{safe_name}.json"
            path = os.path.join(output_folder, nombre_archivo)

            with open(path, "w", encoding="utf-8") as f:
                json.dump(ev, f, ensure_ascii=False, indent=2)

        # --- REGENERAR EMBEDDINGS CON NUEVOS ---
        embedder.events = eventos + nuevos_eventos
        texts = [embedder.build_event_text(ev) for ev in embedder.events]
        embedder.embeddings = embedder.model.encode(texts, convert_to_numpy=True)

        embedder.build_index(embedder.embeddings, index_type="IVFFlat")
        embedder.save("embedding_data_nuevo")

        print("✅ [GapFallback] Nuevos embeddings generados y guardados.")
        self.enviar("visual", "actualizar_embedding")

    def ejecutar(self):
        while True:
            msg = self.bandeja_entrada.get()
            if msg.receptor == self.nombre and msg.contenido == "verificar_gaps":
                self.run_once()
# === Agente 9: Conversacional ===
class AgenteBusquedaInteractiva(AgenteBase):
    def __init__(self, nombre, bandeja_entrada):
        super().__init__(nombre, bandeja_entrada)
        self.embedder = EventEmbedder.load("embedding_data")
        self.grafo = None  # se cargará por mensaje

    def ejecutar(self):
        while True:
            msg = self.bandeja_entrada.get()
            
            # 🔁 Nuevo: recibir el grafo desde AgenteGrafo
            if msg.receptor == self.nombre and isinstance(msg.contenido, dict) and "grafo" in msg.contenido:
                self.grafo = msg.contenido["grafo"]
                print("📥 [BusquedaInteractiva] Grafo recibido desde AgenteGrafo")
                continue  # nada más que hacer con este mensaje

            # 🔍 Procesar búsqueda
            if msg.receptor == self.nombre and isinstance(msg.contenido, dict):
                if self.grafo is None:
                    print("⚠️ [BusquedaInteractiva] No se puede procesar: el grafo aún no fue recibido.")
                    continue

                query = msg.contenido.get("query")
                ciudad = msg.contenido.get("ciudad")
                fecha_i = msg.contenido.get("fecha_inicio")
                fecha_f = msg.contenido.get("fecha_fin")
                categoria = msg.contenido.get("categoria")

                print(f"🔍 [BusquedaInteractiva] Procesando búsqueda: {query}")
                resultados, _ = self.embedder.filtered_search(query, k=200)

                # Filtros
                if ciudad:
                    resultados = [ev for ev in resultados if ev.get("spatial_info", {}).get("area", {}).get("city") == ciudad]
                if categoria:
                    resultados = [ev for ev in resultados if ev.get("classification", {}).get("primary_category") == categoria]
                if fecha_i and fecha_f:
                    resultados = [
                        ev for ev in resultados
                        if ev.get("temporal_info", {}).get("start") and
                        fecha_i <= ev["temporal_info"]["start"][:10] <= fecha_f
                    ]

                # Enriquecer con grafo
                if resultados:
                    resultados = [ev for ev, _ in enriquecer_resultados_con_razonamiento_avanzado(resultados, self.grafo, query, k=50)]

                with open("resultados_interactivos.json", "w", encoding="utf-8") as f:
                    json.dump(resultados, f, ensure_ascii=False, indent=2)

                print(f"✅ [BusquedaInteractiva] Resultados guardados: {len(resultados)}")

# === Función para arranque secuencial ===
def arranque_secuencial(bandeja):
    scraper = AgenteScraper("scraper", bandeja)
    procesador = AgenteProcesador("procesador", bandeja)
    embedding = AgenteEmbedding("embedding", bandeja)
    grafo = AgenteGrafo("grafo", bandeja)
    optimizador = AgenteOptimizador("optimizador", bandeja)

    #scraper.run_once()
    #procesador.run_once()
    embedding.run_once()
    grafo.run_once()
    optimizador.run_once()

    # ✅ Solo cuando todo esté listo, lanza visual
    visual = AgenteVisual("visual", bandeja)
    Thread(target=visual.ejecutar, daemon=True).start()

# === Función para iniciar agentes en threads (escuchando mensajes) ===
def iniciar_threads(bandeja):
    agentes = [
        AgenteScraper("scraper", bandeja),
        AgenteProcesador("procesador", bandeja),
        AgenteActualizador("actualizador", bandeja),
        AgenteGrafo("grafo", bandeja),
        AgenteEmbedding("embedding", bandeja),
        AgenteOptimizador("optimizador", bandeja),
        AgenteVisual("visual", bandeja),
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


# === Función principal para arrancar el sistema ===
def iniciar_sistema_multiagente():
    bandeja = Queue()

    print("🚀 Ejecutando arranque secuencial inicial...")
    arranque_secuencial(bandeja)

    print("🚀 Iniciando agentes en modo escucha...")
    iniciar_threads(bandeja)

    print("🚀 Lanzando hilo para actualizaciones periódicas (cada 7 días)...")
    t_actualizacion = Thread(target=hilo_actualizacion_periodica, args=(bandeja,), daemon=True)
    t_actualizacion.start()

    print("✅ Sistema multiagente iniciado correctamente. (AgenteVisual no arrancado aquí)")

    # Mantener el hilo principal vivo
    while True:
        time.sleep(3600)

def lanzar_visual(bandeja):
    visual = AgenteVisual("visual", bandeja)
    visual.ejecutar()

if __name__ == "__main__":
    print("✅ Iniciando script...")
    bandeja = Queue()

    arranque_secuencial(bandeja)
    iniciar_threads(bandeja)

    # Lanzar flujo inicial
    bandeja.put(Mensaje("inicial", "scraper", "scraping_terminado"))

    # Lanzar actualizaciones periódicas
    t_actualizacion = Thread(target=hilo_actualizacion_periodica, args=(bandeja,), daemon=True)
    t_actualizacion.start()

    while True:
        time.sleep(3600)
