import os
import sys
ruta = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules'))
sys.path.insert(0, ruta)

from procesamiento import procesar_json

def test_procesamiento():
    print("[Procesamiento] Iniciando...")
    procesar_json()
    # Contar archivos procesados
    path = "./eventos_mejorados"
    procesados = [f for f in os.listdir(path) if f.endswith(".json")]
    print(f"[Procesamiento] Archivos procesados en {path}: {len(procesados)}")

if __name__ == "__main__":
    test_procesamiento()
