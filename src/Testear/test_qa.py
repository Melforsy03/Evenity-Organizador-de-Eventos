import sys
import os

# Añadir la ruta absoluta de la carpeta externa al sys.path
ruta = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules'))
sys.path.insert(0, ruta)

from qa_rag import EventQA

def test_qa():
    qa = EventQA()
    pregunta = "¿Qué conciertos hay en Madrid en junio?"
    respuesta = qa.responder(pregunta, ciudad="Madrid", fechas=("2025-06-01", "2025-06-30"))
    print("[QA] Pregunta:", pregunta)
    print("[QA] Respuesta:", respuesta)

if __name__ == "__main__":
    test_qa()
