# contexto_global.py

"""
Módulo de contexto compartido

Permite registrar y acceder a las bandejas (colas) por agente.
"""

from queue import Queue

_bandejas_por_agente = {}

def registrar_bandeja(nombre_agente: str, bandeja: Queue):
    _bandejas_por_agente[nombre_agente] = bandeja

def obtener_bandeja(nombre_agente: str) -> Queue:
    return _bandejas_por_agente.get(nombre_agente)

def get_all_bandejas():
    return _bandejas_por_agente
