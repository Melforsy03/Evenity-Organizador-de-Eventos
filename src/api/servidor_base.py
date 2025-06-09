# servidor_base.py
"""
Módulo base de la API (servidor_base.py)

Inicializa y configura una instancia del servidor Flask con un endpoint básico de prueba de vida (ping).
Este módulo no define lógica de negocio, sino que actúa como punto de entrada común para los demás módulos de la API.

"""

from flask import Flask
import logging

# Inicialización del servidor Flask
app = Flask(__name__)

# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ruta mínima para prueba de vida
@app.route("/ping")
def ping():
    return "pong"
