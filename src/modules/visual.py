import streamlit as st
import os
from embedding import EventEmbedder
from grafo_conocimiento import get_knowledge_graph
from optimizador import obtener_eventos_optimales
from datetime import datetime
from grafo_conocimiento import enriquecer_resultados_con_razonamiento_avanzado
import numpy as np
from tqdm import tqdm
import logging
from functools import partialmethod

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Deshabilitar tqdm para evitar conflictos con Streamlit
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
# --- Carga y cacheo de modelos ---
@st.cache_resource
def cargar_modelos():
    embedder = EventEmbedder.load("embedding_data")
    grafo = get_knowledge_graph()
    return embedder, grafo

# --- Inicializar ---
if "embedder" not in st.session_state or "grafo" not in st.session_state:
    st.session_state.embedder, st.session_state.grafo = cargar_modelos()
# --- Estado de la aplicación ---
if "resultados" not in st.session_state:
    st.session_state.resultados = []
if "page" not in st.session_state:
    st.session_state.page = 0
if "total_pages" not in st.session_state:
    st.session_state.total_pages = 1
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# --- Función para realizar búsqueda ---
def realizar_busqueda():
    with st.spinner("Buscando los eventos más relevantes..."):
        try:
            st.session_state.last_query = st.session_state.input_query
            # Guardar filtros actuales
            ciudad = st.session_state.get("filter_ciudad")
            categorias = st.session_state.get("filter_categorias", [])
            fecha_inicio = st.session_state.get("filter_fecha_inicio")
            fecha_fin = st.session_state.get("filter_fecha_fin")

            filtros = {
                "ciudad": ciudad if ciudad != "Todas las ciudades" else None,
                "categorias": categorias,
                "fecha_inicio": fecha_inicio.isoformat() if fecha_inicio else None,
                "fecha_fin": fecha_fin.isoformat() if fecha_fin else None,
            }
            st.session_state.filters = filtros

            # Búsqueda semántica
            if filtros["ciudad"]:
                resultados, scores = st.session_state.embedder.search(
                    st.session_state.last_query,
                    shard_key=filtros["ciudad"],
                    k=200
                )
            else:
                resultados, scores = st.session_state.embedder.filtered_search(
                    st.session_state.last_query,
                    k=200
                )

            # Filtrar por categorías y fechas
            if resultados:
                if filtros["categorias"]:
                    resultados = [
                        ev for ev in resultados
                        if ev.get("classification", {}).get("primary_category") in filtros["categorias"]
                    ]

                if filtros["fecha_inicio"] and filtros["fecha_fin"]:
                    start = filtros["fecha_inicio"]
                    end = filtros["fecha_fin"]
                    resultados = [
                        ev for ev in resultados
                        if ev.get("temporal_info", {}).get("start") and
                        start <= ev["temporal_info"]["start"][:10] <= end
                    ]

            if resultados and st.session_state.grafo:
                eventos_scores = enriquecer_resultados_con_razonamiento_avanzado(
                    resultados,
                    st.session_state.grafo,
                    st.session_state.last_query,
                    k=200
                )
                # Separamos la lista de eventos y la lista de scores para guardar
                st.session_state.resultados = [ev for ev, score in eventos_scores]
                st.session_state.resultados_scores = {ev.get("basic_info", {}).get("title", ""): score for ev, score in eventos_scores}
            else:
                st.session_state.resultados_scores = {}

            st.session_state.page = 0
            st.session_state.total_pages = max(1, (len(st.session_state.resultados) - 1) // st.session_state.select_k + 1)

            if not resultados:
                st.warning("No encontramos eventos locales con esos criterios. Intentando buscar en línea...")

                from embedding import fallback_api_call  # asegúrate de que esté disponible
                query = st.session_state.last_query
                start = filtros.get("fecha_inicio")
                end = filtros.get("fecha_fin")

                nuevos_eventos = fallback_api_call(query, start, end)
                
                if nuevos_eventos:
                    st.success(f"🎉 Hemos encontrado {len(nuevos_eventos)} eventos relevantes en línea.")
                    
                    # Agregar temporalmente los nuevos eventos al embedder
                    st.session_state.embedder.events.extend(nuevos_eventos)
                    nuevos_embeddings = st.session_state.embedder.model.encode(
                        [st.session_state.embedder.build_event_text(ev) for ev in nuevos_eventos],
                        convert_to_numpy=True
                    )
                    st.session_state.embedder.embeddings = np.vstack([st.session_state.embedder.embeddings, nuevos_embeddings])
                    st.session_state.embedder.build_index(st.session_state.embedder.embeddings)

                    resultados = nuevos_eventos
                    st.session_state.resultados = nuevos_eventos
                    st.session_state.total_pages = max(1, (len(nuevos_eventos) - 1) // st.session_state.select_k + 1)
                    st.session_state.page = 0
                else:
                    st.error("No encontramos eventos, ni siquiera con búsqueda externa. ¿Quieres ampliar las fechas o cambiar de ciudad?")

        except Exception as e:
            st.error(f"Error en la búsqueda: {str(e)}")
            st.session_state.resultados = []

# --- UI PRINCIPAL ---
st.title("🎟️ Buscador Avanzado de Eventos Mejorado")
st.markdown("Encuentra y organiza los mejores eventos con filtros potentes y navegación fácil.")

# --- Buscador visible ---
with st.container():
    st.text_input(
        "Buscar eventos:",
        placeholder="Ej. 'concierto jazz', 'taller pintura'",
        value=st.session_state.last_query,
        key="input_query"
    )
    st.selectbox("Resultados por página", [5, 10, 15], index=0, key="select_k")

    if st.button("🔍 Buscar eventos"):
        realizar_busqueda()

# --- Filtros avanzados ---
with st.expander("🔧 Filtros avanzados (opcional)"):
    ciudades_disponibles = sorted(set(
        ev.get("spatial_info", {}).get("area", {}).get("city")
        for ev in st.session_state.embedder.events
        if ev.get("spatial_info", {}).get("area", {}).get("city")
    ))
    st.selectbox(
        "Filtrar por ciudad:",
        ["Todas las ciudades"] + ciudades_disponibles,
        index=0,
        key="filter_ciudad"
    )
    categorias_disponibles = sorted(set(
        ev.get("classification", {}).get("primary_category")
        for ev in st.session_state.embedder.events
        if ev.get("classification", {}).get("primary_category")
    ))
    st.multiselect(
        "Filtrar por categorías:",
        options=categorias_disponibles,
        default=[],
        key="filter_categorias"
    )
    st.date_input("Fecha inicio (desde):", value=None, key="filter_fecha_inicio")
    st.date_input("Fecha fin (hasta):", value=None, key="filter_fecha_fin")

# --- Mostrar resultados con diseño limpio ---
if st.session_state.resultados:
    k = st.session_state.select_k
    start_idx = st.session_state.page * k
    end_idx = start_idx + k
    current_results = st.session_state.resultados[start_idx:end_idx]

    for ev in current_results:
        categoria = ev.get('classification', {}).get('primary_category', 'General')
        color = {
            "Music": "#FF6B6B",
            "Sports": "#4ECDC4",
            "Arts": "#FFBE0B",
            "Business": "#8338EC",
            "Education": "#06D6A0"
        }.get(categoria, "#7F7F7F")

        venue_name = ev.get('spatial_info', {}).get('venue', {}).get('name', 'Lugar no especificado')
        city = ev.get('spatial_info', {}).get('area', {}).get('city', None)
        duration = ev.get('temporal_info', {}).get('duration_minutes', None)
        url_info = ev.get('external_references', {}).get('urls', [{}])[0].get('url', '#')
        title = ev.get('basic_info', {}).get('title', 'Evento sin título')
        start_date = ev.get('temporal_info', {}).get('start', 'Fecha no definida')

        with st.container():
            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(f"### {title}")
                loc_text = f"📍 {venue_name}" + (f" | {city}" if city else "")
                st.markdown(loc_text)
                st.markdown(f"📅 {start_date}")
                if duration:
                    st.markdown(f"⏳ Duración: {duration} minutos")
                st.markdown(f"🏷️ Categoría: **{categoria}**")
                st.markdown(f"[🔗 Más info]({url_info})")

                performers = ev.get("participants", {}).get("performers", [])
                if performers:
                    with st.expander("👥 Ver participantes"):
                        for p in performers:
                            st.markdown(f"- 🎤 **{p.get('name', 'Nombre no disponible')}**")
                            if p.get("wiki_bio"):
                                st.caption(p["wiki_bio"])

            with cols[1]:
                images = ev.get("external_references", {}).get("images", [])
                if images:
                    st.image(images[0].get("url"), use_container_width=True)

        st.markdown("---")

    # --- Paginación ---
    col1, col2, col3, col4, col5 = st.columns([1,1,2,1,1])
    if col1.button("⏮️ Primera", key="first_page"):
        st.session_state.page = 0
    if col2.button("⬅️ Anterior", key="prev_page"):
        if st.session_state.page > 0:
            st.session_state.page -= 1
    col3.markdown(f"Página {st.session_state.page + 1} de {st.session_state.total_pages} | Total resultados: {len(st.session_state.resultados)}")
    if col4.button("Siguiente ➡️", key="next_page"):
        if st.session_state.page < st.session_state.total_pages - 1:
            st.session_state.page += 1
    if col5.button("Última ⏭️", key="last_page"):
        st.session_state.page = st.session_state.total_pages - 1
# --- Generar Agenda Óptima ---
st.header("✨ Generar Agenda Óptima")

if st.session_state.resultados and st.button("Generar agenda óptima"):
    with st.spinner("Optimizando tu agenda de eventos..."):
        try:
            preferencias = {
                "location": st.session_state.filters.get("ciudad"),
                "categories": st.session_state.filters.get("categorias"),
                "available_dates": None,
            }
            if st.session_state.filters.get("fecha_inicio") and st.session_state.filters.get("fecha_fin"):
                preferencias["available_dates"] = (
                    st.session_state.filters["fecha_inicio"],
                    st.session_state.filters["fecha_fin"],
                )
            eventos_validos = st.session_state.resultados[:50]
            scores_grafo = {ev.get("basic_info", {}).get("title", ""): st.session_state.resultados_scores.get(ev.get("basic_info", {}).get("title", ""), 0)
                for ev in eventos_validos}

            top_eventos, score = obtener_eventos_optimales(
                eventos_validos, preferencias, cantidad=min(5, len(eventos_validos)), scores_grafo=scores_grafo
            )

            st.success(f"🎯 Agenda óptima generada (puntuación: {score:.1f})")

            for i, ev in enumerate(top_eventos, 1):
                with st.container():
                    st.markdown(f"### {i}. {ev.get('basic_info', {}).get('title', 'Evento sin título')}")
                    venue_name = ev.get('spatial_info', {}).get('venue', {}).get('name', 'Lugar no especificado')
                    city = ev.get('spatial_info', {}).get('area', {}).get('city', None)
                    location_display = f"📍 {venue_name}"
                    if city:
                        location_display += f" | {city}"
                    st.markdown(location_display)
                    st.markdown(f"📅 **Fecha:** {ev.get('temporal_info', {}).get('start', '-')}")
                    motivo = st.session_state.embedder.explicar_recomendacion(ev, preferencias)
                    st.markdown(f"🧠 **Por qué está en tu agenda:** {motivo}")
                    st.markdown("---")
        except Exception as e:
            st.error(f"Error al generar agenda: {str(e)}")

# --- Mensaje inicial ---
if not st.session_state.last_query:
    st.info("💡 Escribe qué eventos te interesan y haz clic en Buscar")