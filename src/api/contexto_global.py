# contexto_global.py
"""
Módulo de contexto compartido (contexto.py)

Define una bandeja global para facilitar la comunicación entre módulos sin necesidad de
pasar explícitamente la referencia entre funciones. Este patrón es útil en el sistema multiagente
donde múltiples hilos comparten una cola de mensajes común.

Funciones:
- `set_bandeja_global(bandeja)`
- `get_bandeja_global()`
"""

_bandeja_global = None

def set_bandeja_global(bandeja):
    global _bandeja_global
    _bandeja_global = bandeja

def get_bandeja_global():
    return _bandeja_global
