import streamlit as st
import requests
from datetime import date
import uuid

# Configuración inicial
st.set_page_config(page_title="Evenity", layout="wide")
st.title("🔍 Explorador de Eventos")

# Estado inicial
if 'clave_respuesta' not in st.session_state:
    st.session_state.clave_respuesta = f"res_busqueda_{uuid.uuid4().hex}"

if 'pagina_actual' not in st.session_state:
    st.session_state.pagina_actual = 0

@st.cache_data(ttl=300)
def cargar_datos():
    try:
        ciudades = requests.get("http://localhost:8502/ciudades", timeout=3).json()
        categorias = requests.get("http://localhost:8502/categorias", timeout=3).json()
        return ciudades, categorias
    except:
        st.error("❌ No se pudo cargar ciudades o categorías")
        return [], []

def mostrar_evento(ev):
    with st.container():
        titulo = ev.get("basic_info", {}).get("title", "Sin título")
        descripcion = ev.get("basic_info", {}).get("description", "Sin descripción")

        st.markdown(f"## 🎫 {titulo}")

        imagenes = ev.get("external_references", {}).get("images", [])
        if imagenes:
            st.image(imagenes[0].get("url"), use_column_width=True)

        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("### 📍 Información del evento")
            lugar = ev.get("spatial_info", {}).get("venue", {}).get("name", "")
            direccion = ev.get("spatial_info", {}).get("venue", {}).get("full_address", "")
            ciudad = ev.get("spatial_info", {}).get("area", {}).get("city", "")
            if lugar:
                st.markdown(f"- **Lugar:** {lugar}")
            if direccion:
                st.markdown(f"- **Dirección:** {direccion}")
            elif ciudad:
                st.markdown(f"- **Ciudad:** {ciudad}")

            fecha = ev.get("temporal_info", {}).get("start", "Fecha no especificada")
            duracion = ev.get("temporal_info", {}).get("duration_minutes")
            st.markdown(f"- **Fecha:** {fecha}")
            if duracion:
                st.markdown(f"- **Duración:** {duracion} minutos")

            popularidad = ev.get("raw_data_metadata", {}).get("source_specific", {}).get("popularity")
            if popularidad:
                st.markdown(f"- **Popularidad estimada:** 🌟 {popularidad}")

        with col2:
            st.markdown("### 👥 Participantes")
            artistas = ev.get("participants", {}).get("performers", [])
            if artistas:
                nombres = ", ".join([a.get("name", "") for a in artistas if a.get("name")])
                st.markdown(f"**Artistas:** {nombres}")
            organizadores = ev.get("participants", {}).get("organizers", [])
            if organizadores:
                nombres_org = ", ".join([o.get("name", "") for o in organizadores if o.get("name")])
                st.markdown(f"**Organizadores:** {nombres_org}")

            st.markdown("### 🗂️ Clasificación")
            categoria = ev.get("classification", {}).get("primary_category")
            subcategorias = ev.get("classification", {}).get("categories", [])
            if categoria:
                st.markdown(f"- **Categoría principal:** {categoria}")
            if subcategorias:
                st.markdown(f"- **Subcategorías:** {', '.join(subcategorias)}")

        st.markdown("### 📝 Descripción")
        st.write(descripcion[:500] + "..." if len(descripcion) > 500 else descripcion)

        urls = ev.get("external_references", {}).get("urls", [])
        if urls:
            st.markdown("### 🔗 Más información")
            for url in urls:
                st.markdown(f"- [{url.get('type', 'Enlace')}]({url.get('url')})")

        st.markdown("---")

# === CARGA INICIAL ===
ciudades_disponibles, categorias_disponibles = cargar_datos()

colA, colB = st.columns([2, 1], gap="large")

with colA:
    st.subheader("🔎 Buscar eventos")
    query = st.text_input("Consulta de búsqueda", "", key="query_input")

    c1, c2 = st.columns(2)
    with c1:
        ciudad = st.selectbox("📍 Ciudad (opcional)", [""] + ciudades_disponibles, key="ciudad_select")
    with c2:
        categoria = st.selectbox("📂 Categoría (opcional)", [""] + categorias_disponibles, key="categoria_select")

    c3, c4 = st.columns(2)
    with c3:
        fecha_inicio = st.date_input("Fecha inicio", date.today(), key="fecha_inicio")
    with c4:
        fecha_fin = st.date_input("Fecha fin", date.today(), key="fecha_fin")

    if st.button("🔍 Buscar eventos", key="buscar_btn"):
        if not query.strip():
            st.warning("⚠️ Por favor ingresa un término de búsqueda")
        else:
            with st.spinner("Buscando eventos..."):
                try:
                    respuesta = requests.post(
                        "http://localhost:8502/buscar",
                        json={
                            "query": query,
                            "ciudad": ciudad or None,
                            "categoria": categoria or None,
                            "fecha_inicio": fecha_inicio.strftime("%Y-%m-%d"),
                            "fecha_fin": fecha_fin.strftime("%Y-%m-%d"),
                            "respuesta": st.session_state.clave_respuesta
                        },
                        timeout=15
                    )
                    if respuesta.status_code != 200:
                        st.error(f"❌ Error del servidor (código {respuesta.status_code})")
                        st.stop()

                    data = respuesta.json()
                    eventos = data.get("eventos", [])
                    mensaje = data.get("mensaje", "")

                    if eventos:
                        st.success(f"🎯 {mensaje}")
                        total_paginas = (len(eventos) - 1) // 10 + 1

                        st.write(f"Página {st.session_state.pagina_actual + 1} de {total_paginas}")
                        inicio = st.session_state.pagina_actual * 10
                        fin = inicio + 10
                        for ev in eventos[inicio:fin]:
                            mostrar_evento(ev)

                        col_prev, col_next = st.columns([1, 1])
                        with col_prev:
                            if st.button("⬅️ Anterior") and st.session_state.pagina_actual > 0:
                                st.session_state.pagina_actual -= 1
                                st.experimental_rerun()
                        with col_next:
                            if st.button("➡️ Siguiente") and st.session_state.pagina_actual < total_paginas - 1:
                                st.session_state.pagina_actual += 1
                                st.experimental_rerun()
                    else:
                        st.info("ℹ️ No se encontraron eventos que coincidan con tu búsqueda")

                except requests.exceptions.Timeout:
                    st.error("⏳ El servidor tardó demasiado en responder")
                except Exception as e:
                    st.error(f"❌ Error inesperado: {str(e)}")

with colB:
    st.subheader("🧠 Agenda personalizada")
    st.markdown("Generada según ciudad, categoría y fechas.")

    if st.button("📅 Generar agenda personalizada", key="agenda_btn"):
        with st.spinner("Generando recomendaciones..."):
            try:
                res = requests.post(
                    "http://localhost:8502/agenda",
                    json={
                        "ciudad": ciudad or None,
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

                    if agenda:
                        st.success(f"✅ Agenda generada (puntaje: {score:.2f}/5)")
                        for i, ev in enumerate(agenda, 1):
                            with st.expander(f"🗓️ Evento {i}"):
                                mostrar_evento(ev)
                    else:
                        st.info("ℹ️ No se pudo generar una agenda con tus preferencias")
                else:
                    st.error(f"❌ Error al generar agenda: {res.text}")

            except requests.exceptions.Timeout:
                st.error("⏳ El servidor tardó demasiado en generar la agenda")
            except Exception as e:
                st.error(f"❌ Error inesperado: {str(e)}")
