import sys
import os

# Añadir la ruta absoluta de la carpeta externa al sys.path
ruta = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules'))
sys.path.insert(0, ruta)

from grafo_conocimiento import generar_y_exportar_grafo

def test_grafo():
    resumen = generar_y_exportar_grafo()
    print("[Grafo] Resumen:")
    print(resumen)

if __name__ == "__main__":
    test_grafo()
