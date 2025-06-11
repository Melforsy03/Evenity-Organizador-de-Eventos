"""
Módulo visual.py

Interfaz de usuario principal desarrollada con Streamlit. Permite realizar búsquedas semánticas
de eventos, filtrar por ciudad, categoría y rango de fechas, y generar una agenda personalizada
basada en preferencias del usuario.

Este módulo se conecta a una API local (http://localhost:8502) expuesta por Flask para obtener
información sobre eventos.
"""

import streamlit as st
import requests
from datetime import date
import time

st.set_page_config(page_title="Buscador de Eventos", layout="wide")
st.title("🔍 Explorador de Eventos")

# Variables globales de estado
ciudades_disponibles = []
categorias_disponibles = []

@st.cache_data
def cargar_datos():
    try:
        ciudades = requests.get("http://localhost:8502/ciudades", timeout=5).json()
        categorias = requests.get("http://localhost:8502/categorias", timeout=5).json()
        return ciudades, categorias
    except Exception as e:
        st.error(f"❌ No se pudo conectar a la API: {e}")
        return [], []

# Lógica principal
ciudades_disponibles, categorias_disponibles = cargar_datos()

query = st.text_input("Consulta de búsqueda", "")
col1, col2 = st.columns(2)
with col1:
    ciudad = st.selectbox("📍 Ciudad (opcional)", [""] + ciudades_disponibles)
with col2:
    categoria = st.selectbox("📂 Categoría (opcional)", [""] + categorias_disponibles)

col3, col4 = st.columns(2)
with col3:
    fecha_inicio = st.date_input("Fecha inicio", date.today())
with col4:
    fecha_fin = st.date_input("Fecha fin", date.today())

# === BÚSQUEDA ===
if st.button("🔍 Buscar eventos"):
    if not query.strip():
        st.warning("⚠️ Debes escribir una consulta.")
    else:
        with st.spinner("Buscando eventos..."):
            try:
                respuesta = requests.post(
                    "http://localhost:8502/buscar",
                    json={
                        "query": query,
                        "ciudad": ciudad if ciudad else None,
                        "categoria": categoria if categoria else None,
                        "fecha_inicio": str(fecha_inicio),
                        "fecha_fin": str(fecha_fin)
                    },
                    timeout=30
                )
                if respuesta.status_code == 200:
                    eventos = respuesta.json()
                    st.success(f"🎯 Se encontraron {len(eventos)} eventos.")
                    for ev in eventos:
                        st.markdown(f"### {ev.get('basic_info', {}).get('title', 'Sin título')}")
                        st.write(ev.get('basic_info', {}).get('description', ''))
                        st.write(ev.get('temporal_info', {}).get('start', ''))
                        st.write("---")
                else:
                    st.error(f"Error: {respuesta.status_code} - {respuesta.text}")
            except Exception as e:
                st.error(f"No se pudo conectar con el backend: {e}")

# === AGENDA PERSONALIZADA ===
st.subheader("🧠 Generar Agenda Recomendada")
if st.button("📅 Generar agenda personalizada"):
    with st.spinner("Generando agenda..."):
        try:
            res = requests.post(
                "http://localhost:8502/agenda",
                json={
                    "ciudad": ciudad if ciudad else None,
                    "categorias": [categoria] if categoria else [],
                    "fecha_inicio": str(fecha_inicio),
                    "fecha_fin": str(fecha_fin)
                },
                timeout=30
            )
            if res.status_code == 200:
                data = res.json()
                agenda = data.get("agenda", [])
                score = data.get("score", 0)
                st.success(f"✅ Agenda generada con score: {score:.2f}")
                for i, ev in enumerate(agenda, 1):
                    st.markdown(f"**{i}. {ev.get('basic_info', {}).get('title', 'Sin título')}**")
                    st.write(ev.get('basic_info', {}).get('description', ''))
                    st.write(ev.get('temporal_info', {}).get('start', ''))
                    st.write("---")
            else:
                st.error(f"❌ Error al generar agenda: {res.status_code} - {res.text}")
        except Exception as e:
            st.error(f"❌ Error en agenda: {e}")
