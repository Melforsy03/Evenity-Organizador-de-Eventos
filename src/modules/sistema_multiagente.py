from queue import Queue
from threading import Thread
from abc import ABC, abstractmethod
import time

from crawler import EventScraper
from procesamiento import procesar_json
from actualizador_dinamico import actualizar_eventos_y_embeddings
from grafo_conocimiento import get_knowledge_graph
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
        print("🔁 [Actualizador] Actualizando eventos y embeddings...")
        actualizar_eventos_y_embeddings()
        print("✅ [Actualizador] Actualización terminada.")
        self.enviar("grafo", "embeddings_actualizados")

    def ejecutar(self):
        while True:
            msg = self.bandeja_entrada.get()
            if msg.receptor == self.nombre and msg.contenido == "eventos_procesados":
                self.run_once()


# === Agente 4: Grafo de Conocimiento ===
class AgenteGrafo(AgenteBase):
    def run_once(self):
        print("🧠 [Grafo] Construyendo grafo de conocimiento...")
        self.grafo = get_knowledge_graph()
        print("✅ [Grafo] Grafo construido.")
        self.enviar("optimizador", "grafo_listo")

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


# === Función para iniciar agentes en threads (escuchando mensajes) ===
def iniciar_threads(bandeja):
    agentes = [
        AgenteScraper("scraper", bandeja),
        AgenteProcesador("procesador", bandeja),
        AgenteActualizador("actualizador", bandeja),
        AgenteGrafo("grafo", bandeja),
        AgenteEmbedding("embedding", bandeja),
        AgenteOptimizador("optimizador", bandeja),
        # AgenteVisual NO va aquí
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


if __name__ == "__main__":
    iniciar_sistema_multiagente()

