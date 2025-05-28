
from queue import Queue
from threading import Thread
from abc import ABC, abstractmethod

# === Sistema de mensajería entre agentes ===
class Mensaje:
    def __init__(self, emisor, receptor, contenido):
        self.emisor = emisor
        self.receptor = receptor
        self.contenido = contenido

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
    def ejecutar(self):
        from crawler import EventScraper
        print("📡 [Scraper] Scrapeando eventos...")
        scraper = EventScraper()
        scraper.run_all_scrapers()
        print("✅ [Scraper] Scraping completo.")
        self.enviar("procesador", "scraping_terminado")

# === Agente 2: Procesador ===
class AgenteProcesador(AgenteBase):
    def ejecutar(self):
        from procesamiento import procesar_json
        while True:
            msg = self.bandeja_entrada.get()
            if msg.receptor == self.nombre and msg.contenido == "scraping_terminado":
                print("🧪 [Procesador] Procesando eventos...")
                procesar_json()
                print("✅ [Procesador] Procesamiento completo.")
                self.enviar("actualizador", "eventos_procesados")

# === Agente 3: Actualizador ===
class AgenteActualizador(AgenteBase):
    def ejecutar(self):
        from actualizador_dinamico import actualizar_eventos_y_embeddings
        while True:
            msg = self.bandeja_entrada.get()
            if msg.receptor == self.nombre and msg.contenido == "eventos_procesados":
                print("🔁 [Actualizador] Actualizando eventos y embeddings...")
                actualizar_eventos_y_embeddings()
                print("✅ [Actualizador] Actualización terminada.")
                self.enviar("grafo", "embeddings_actualizados")

# === Agente 4: Grafo de Conocimiento ===
class AgenteGrafo(AgenteBase):
    def ejecutar(self):
        from grafo_conocimiento import get_knowledge_graph
        while True:
            msg = self.bandeja_entrada.get()
            if msg.receptor == self.nombre and msg.contenido == "embeddings_actualizados":
                print("🧠 [Grafo] Construyendo grafo de conocimiento...")
                self.grafo = get_knowledge_graph()
                print("✅ [Grafo] Grafo construido.")
                self.enviar("optimizador", "grafo_listo")

# === Agente 5: Embedding y Búsqueda ===
class AgenteEmbedding(AgenteBase):
    def ejecutar(self):
        from embedding import EventEmbedder
        while True:
            msg = self.bandeja_entrada.get()
            if msg.receptor == self.nombre:
                if msg.contenido == "refrescar_embeddings":
                    print("💾 [Embedding] Regenerando embeddings...")
                    embedder = EventEmbedder()
                    embedder.run_embedding()
                    print("✅ [Embedding] Embeddings listos.")

# === Agente 6: Optimizador ===
class AgenteOptimizador(AgenteBase):
    def ejecutar(self):
        from embedding import EventEmbedder
        from optimizador import obtener_eventos_optimales
        while True:
            msg = self.bandeja_entrada.get()
            if msg.receptor == self.nombre and msg.contenido == "grafo_listo":
                print("🎯 [Optimizador] Generando agenda óptima...")
                embedder = EventEmbedder.load("embedding_data")
                eventos = embedder.events[:50]
                preferencias = {"location": None, "categories": [], "available_dates": None}
                scores = {ev.get("basic_info", {}).get("title", ""): 1.0 for ev in eventos}
                agenda, score = obtener_eventos_optimales(eventos, preferencias, cantidad=5, scores_grafo=scores)
                print(f"✅ [Optimizador] Agenda generada (score: {score:.2f})")
                for i, ev in enumerate(agenda, 1):
                    print(f"{i}. {ev.get('basic_info', {}).get('title', 'Sin título')}")

# === Agente 7: Visual (UI separado) ===
class AgenteVisual(AgenteBase):
    def ejecutar(self):
        import visual
        print("🖥️ [Visual] Interfaz en ejecución (independiente).")
        visual  # Esto simplemente asegura que el módulo esté disponible

# === Coordinador ===
def iniciar_sistema_multiagente():
    bandeja = Queue()

    agentes = [
        AgenteScraper("scraper", bandeja),
        AgenteProcesador("procesador", bandeja),
        AgenteActualizador("actualizador", bandeja),
        AgenteGrafo("grafo", bandeja),
        AgenteEmbedding("embedding", bandeja),
        AgenteOptimizador("optimizador", bandeja),
        AgenteVisual("visual", bandeja)
    ]

    hilos = [Thread(target=a.ejecutar) for a in agentes]
    for h in hilos:
        h.start()

if __name__ == "__main__":
    iniciar_sistema_multiagente()
