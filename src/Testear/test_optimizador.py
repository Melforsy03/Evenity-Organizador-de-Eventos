
import sys
import os

# Añadir la ruta absoluta de la carpeta externa al sys.path
ruta = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules'))
sys.path.insert(0, ruta)

from optimizador import obtener_eventos_optimales
from embedding import load_events_from_folder

def test_optimizador():
    eventos = load_events_from_folder("eventos_mejorados")
    preferencias = {
        "categories": ["Music"],
        "location": "Madrid",
        "available_dates": ("2025-06-01", "2025-12-31")
    }
    eventos_filtrados = [e for e in eventos if "temporal_info" in e and e["temporal_info"].get("start")]
    top_eventos, score = obtener_eventos_optimales(eventos, preferencias, cantidad=5)
    print(f"[Optimizador] Score: {score}")
    for i, ev in enumerate(top_eventos, 1):
        print(f"{i}. {ev['basic_info']['title']} - {ev['temporal_info']['start']}")

if __name__ == "__main__":

    test_optimizador()
