startup_script = """#!/bin/bash

echo "🚀 Iniciando el sistema de organizador inteligente de eventos..."

# Paso 1: Actualizar eventos y regenerar embeddings
echo "🔁 Actualizando eventos y generando embeddings..."
python3 -c "from actualizador_dinamico_mejorado import actualizar_eventos_y_embeddings; print(actualizar_eventos_y_embeddings())"

# Paso 2: Generar grafo de conocimiento internamente (sin exportar)
echo "🧠 Construyendo grafo de conocimiento..."
python3 -c "from grafo_conocimiento_mejorado import get_knowledge_graph; G = get_knowledge_graph(); print(f'Grafo con {len(G.nodes)} nodos y {len(G.edges)} relaciones.')"

# Paso 3: Confirmar carga de embeddings
echo "📦 Cargando índice de búsqueda..."
python3 -c "from embedding_final import EventEmbedder; embedder = EventEmbedder.load(); print(f'{len(embedder.events)} eventos cargados en memoria.')"

echo "✅ Sistema preparado. Puedes lanzar consultas desde la interfaz."
"""

# Guardar como archivo
from pathlib import Path

startup_path = Path("/mnt/data/startup.sh")
with startup_path.open("w", encoding="utf-8") as f:
    f.write(startup_script)

# Hacer ejecutable
startup_path.chmod(0o755)

startup_path.name
