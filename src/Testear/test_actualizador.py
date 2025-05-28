import sys
import os
# Añadir la ruta absoluta de la carpeta externa al sys.path
ruta = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules'))
sys.path.insert(0, ruta)

from actualizador_dinamico import actualizar_eventos_y_embeddings

def test_actualizador():
    print("[Actualizador] Iniciando actualización y re-embedding...")
    resultado = actualizar_eventos_y_embeddings()
    print("[Actualizador]", resultado)

if __name__ == "__main__":
    test_actualizador()
