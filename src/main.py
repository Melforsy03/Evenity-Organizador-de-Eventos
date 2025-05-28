import os
from modules.crawler import ejecutar_scraping_fuentes
from modules.procesamiento import clean_folder
from modules.embedding import EventEmbedder, load_events_from_folder
from modules.grafo_conocimiento import generar_y_exportar_grafo

def pipeline_completo():
    # --- Paso 1: Scraping ---
    print("📥 Iniciando scraping de eventos...")
    total_eventos = ejecutar_scraping_fuentes(source="all", country="ES", start="2025-06-01", end="2025-12-31")
    print(f"✅ Eventos descargados: {total_eventos}")

    # --- Paso 2: Procesamiento ---
    print("\n🔄 Procesando eventos raw a formato enriquecido...")
    input_folder = "./eventos_completos"
    output_folder = "./eventos_mejorados"
    clean_folder(input_folder, output_folder, verbose=True)

    # --- Paso 3: Generación embeddings ---
    print("\n🧠 Generando embeddings y creando índice vectorial...")
    eventos = load_events_from_folder(output_folder)
    if not eventos:
        print("⚠️ No se encontraron eventos enriquecidos. Terminado.")
        return

    embedder = EventEmbedder()
    embeddings = embedder.generate_embeddings(eventos)
    embedder.build_index(embeddings, index_type="IVFFlat")
    embedder.save(output_dir="embedding_data")
    print("✅ Embeddings generados y guardados.")

    # --- Paso 4: Construcción grafo de conocimiento ---
    print("\n🌐 Construyendo y exportando grafo de conocimiento...")
    resumen = generar_y_exportar_grafo(input_folder=output_folder, output_path="grafo_eventos.graphml")
    print(f"✅ {resumen}")

    print("\n🎉 Pipeline completo finalizado con éxito.")

if __name__ == "__main__":
    pipeline_completo()
