import streamlit as st
import os
import json
from datetime import datetime
from embedding import EventEmbedder
from sistema_multiagente import Mensaje
from optimizador import obtener_eventos_optimales
import logging
from tqdm import tqdm
from functools import partialmethod
from grafo_conocimiento import get_knowledge_graph

# --- Logging y tqdm ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

# --- Recarga automática inicial si no hay modelo cargado ---
def check_reload_signal():
    path_signal = "embedding_data_nuevo/signal.txt"
    if os.path.exists(path_signal):
        os.remove(path_signal)
        try:
            nuevo = EventEmbedder.load("embedding_data_nuevo")
            st.session_state.embedder = nuevo
            st.session_state.grafo = get_knowledge_graph()
            st.toast("🔁 Nuevos eventos cargados con éxito", icon="✨")
            return True
        except Exception as e:
            st.error(f"Error recargando embeddings: {e}")
    return False

if "embedder" not in st.session_state or "grafo" not in st.session_state:
    check_reload_signal()

# --- Recarga silenciosa si hay signal.txt (sin molestar al usuario) ---
def intentar_actualizar_embeddings_silenciosamente():
    path_signal = "embedding_data_nuevo/signal.txt"
    if os.path.exists(path_signal):
        os.remove(path_signal)
        try:
            nuevo = EventEmbedder.load("embedding_data_nuevo")
            st.session_state.embedder = nuevo
            st.session_state.grafo = get_knowledge_graph()
            st.toast("🔁 Nuevos eventos cargados en segundo plano", icon="✨")
            return True
        except Exception as e:
            st.warning(f"⚠️ Falló la recarga de nuevos embeddings: {e}")
    return False

# === Inicialización de estado ===
if "categorias_embedder" not in st.session_state:
    embedder = EventEmbedder.load("embedding_data")
    st.session_state.categorias_embedder = sorted(set(
        ev.get("classification", {}).get("primary_category")
        for ev in embedder.events
        if ev.get("classification", {}).get("primary_category")
    ))

if "resultados" not in st.session_state:
    st.session_state.resultados = []
if "historial_consultas" not in st.session_state:
    st.session_state.historial_consultas = []
if "pagina_actual" not in st.session_state:
    st.session_state.pagina_actual = 0
if "resultados_por_pagina" not in st.session_state:
    st.session_state.resultados_por_pagina = 10

# === FORMULARIO ===
st.title("🔍 Búsqueda Inteligente de Eventos")

with st.form("form_busqueda_agente"):
    pregunta_libre = st.text_input("¿Qué evento estás buscando?", placeholder="Ej: conciertos en Valencia")
    ciudad_p = st.text_input("Ciudad (opcional)", placeholder="Ej: Valencia")
    fecha_i = st.date_input("Desde", value=None, key="fecha_i_busqueda")
    fecha_f = st.date_input("Hasta", value=None, key="fecha_f_busqueda")
    categoria_p = st.selectbox("Categoría (opcional)", [""] + st.session_state.categorias_embedder, key="cat_busqueda")
    enviar = st.form_submit_button("Buscar")

if enviar and pregunta_libre.strip():
    intentar_actualizar_embeddings_silenciosamente()  # 🔄 Recarga si es necesario
    msg = {
        "query": pregunta_libre.strip(),
        "ciudad": ciudad_p.strip() or None,
        "fecha_inicio": fecha_i.isoformat() if fecha_i else None,
        "fecha_fin": fecha_f.isoformat() if fecha_f else None,
        "categoria": categoria_p or None
    }
    st.session_state.historial_consultas.append(msg)
    st.success("🔍 Tu búsqueda fue enviada al agente. Esperando resultados...")

# === Mostrar resultados ===
if os.path.exists("resultados_interactivos.json"):
    with open("resultados_interactivos.json", "r", encoding="utf-8") as f:
        resultados = json.load(f)
        st.session_state.resultados = resultados
        st.session_state.pagina_actual = 0

# === Historial de consultas ===
if st.session_state.historial_consultas:
    with st.expander("🕓 Historial de consultas"):
        for idx, consulta in enumerate(reversed(st.session_state.historial_consultas[-5:]), 1):
            st.markdown(f"**{idx}.** {consulta['query']} ({consulta.get('ciudad', '-')}, {consulta.get('categoria', '-')})")

# === Paginación ===
resultados = st.session_state.resultados
total = len(resultados)
por_pagina = st.session_state.resultados_por_pagina
pagina = st.session_state.pagina_actual
inicio = pagina * por_pagina
fin = inicio + por_pagina

if total > 0:
    st.markdown("## 📋 Resultados encontrados:")
    iconos = {
        "Music": "🎤", "Sports": "⚽", "Arts": "🎨",
        "Theatre": "🎭", "Business": "💼", "Education": "📚"
    }

    for ev in resultados[inicio:fin]:
        title = ev.get("basic_info", {}).get("title", "Sin título")
        fecha = ev.get("temporal_info", {}).get("start", "-")
        ciudad = ev.get("spatial_info", {}).get("area", {}).get("city", "Desconocida")
        categoria = ev.get("classification", {}).get("primary_category", "General")
        icono = iconos.get(categoria, "📌")
        st.markdown(f"### {icono} {title}")
        st.markdown(f"📅 Fecha: {fecha}")
        st.markdown(f"📍 Ciudad: {ciudad}")
        st.markdown(f"🏷️ Categoría: {categoria}")
        st.markdown("---")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if pagina > 0:
            if st.button("⬅️ Anterior"):
                st.session_state.pagina_actual -= 1
    with col2:
        st.markdown(f"<div style='text-align:center;'>Página {pagina+1} de {((total-1)//por_pagina)+1}</div>", unsafe_allow_html=True)
    with col3:
        if fin < total:
            if st.button("➡️ Siguiente"):
                st.session_state.pagina_actual += 1

# === Generar agenda óptima ===
st.header("🧠 Generar Agenda Óptima")

if resultados and st.button("Generar agenda óptima"):
    with st.spinner("Optimizando agenda con tus preferencias..."):
        preferencias = {
            "location": None,
            "categories": [],
            "available_dates": None,
        }
        eventos_validos = resultados[:50]
        top_eventos, score = obtener_eventos_optimales(
            eventos_validos, preferencias, cantidad=min(5, len(eventos_validos)), scores_grafo=None
        )

        st.success(f"✅ Agenda generada (score: {score:.2f})")
        for i, ev in enumerate(top_eventos, 1):
            titulo = ev.get("basic_info", {}).get("title", "Evento sin título")
            fecha = ev.get("temporal_info", {}).get("start", "-")
            ciudad = ev.get("spatial_info", {}).get("area", {}).get("city", "Ciudad desconocida")
            st.markdown(f"### {i}. {titulo}")
            st.markdown(f"📅 {fecha} | 📍 {ciudad}")
            st.markdown("---")
