"""
Módulo de arranque de la API (lanzar_api.py)

Este módulo importa la instancia Flask y el logger desde `servidor_base`, y lanza el servidor
Flask utilizando los endpoints definidos en `servidor_api`.

Es utilizado como punto de entrada directo desde otros hilos o scripts.
"""

from api.servidor_base import app, logger

def iniciar_api():
    import api.servidor_api
    logger.info("✅ API iniciada en http://localhost:8502")
    app.run(host="0.0.0.0", port=8502)