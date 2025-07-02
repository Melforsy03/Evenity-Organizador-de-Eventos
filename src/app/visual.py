import streamlit as st
import requests
from datetime import date
import uuid

st.set_page_config(page_title="Evenity", layout="wide")
st.title("🔍 Explorador de Eventos")

if 'clave_respuesta' not in st.session_state:
    st.session_state.clave_respuesta = f"res_busqueda_{uuid.uuid4().hex}"

if 'pagina_actual' not in st.session_state:
    st.session_state.pagina_actual = 0

if 'fecha_inicio' not in st.session_state:
    st.session_state.fecha_inicio = date.today()

if 'fecha_fin' not in st.session_state:
    st.session_state.fecha_fin = date.today()

if 'eventos_encontrados' not in st.session_state:
    st.session_state.eventos_encontrados = []
    # Añadimos un estado para las sugerencias de la última consulta inválida
    st.session_state.sugerencias_llm = [] 

if 'agenda_generada' not in st.session_state:
    st.session_state.agenda_generada = []
    st.session_state.agenda_score = 0

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
        st.session_state.fecha_inicio = st.date_input("Fecha inicio", st.session_state.fecha_inicio, key="fecha_inicio_input")
    with c4:
        st.session_state.fecha_fin = st.date_input("Fecha fin", st.session_state.fecha_fin, key="fecha_fin_input")

    if st.button("🔍 Buscar eventos", key="buscar_btn"):
        if not query.strip():
            st.warning("⚠️ Por favor ingresa un término de búsqueda")
            st.session_state.sugerencias_llm = [] # Limpiar sugerencias si la consulta está vacía
        else:
            with st.spinner("Buscando eventos..."):
                try:
                    respuesta = requests.post(
                        "http://localhost:8502/buscar",
                        json={
                            "query": query,
                            "ciudad": ciudad or None,
                            "categoria": categoria or None,
                            "fecha_inicio": st.session_state.fecha_inicio.strftime("%Y-%m-%d"),
                            "fecha_fin": st.session_state.fecha_fin.strftime("%Y-%m-%d"),
                            "respuesta": st.session_state.clave_respuesta
                        },
                        timeout=15
                    )
                    
                    data = respuesta.json() # Procesar siempre la respuesta JSON

                    # Aquí está el cambio principal: manejar la respuesta de la API
                    if data.get("status") == "invalid":
                        st.warning(data.get("mensaje", "Lo sentimos, la consulta realizada no es válida en este contexto. Reformúlala."))
                        st.session_state.sugerencias_llm = data.get("sugerencias", []) # Guardar las sugerencias
                        st.session_state.eventos_encontrados = [] # Limpiar eventos encontrados si la consulta es inválida
                        st.session_state.agenda_generada = [] # Limpiar agenda

                    elif data.get("status") == "ok":
                        st.session_state.sugerencias_llm = [] # Limpiar sugerencias si la consulta es válida
                        eventos = data.get("eventos", [])
                        st.session_state.eventos_encontrados = eventos
                        st.session_state.pagina_actual = 0

                        if len(eventos) >= 5:
                            st.info("🧠 Generando agenda automáticamente desde resultados...")
                            r2 = requests.post("http://localhost:8502/agenda", json={
                                "ciudad": ciudad or None,
                                "categorias": [categoria] if categoria else [],
                                "fecha_inicio": str(st.session_state.fecha_inicio),
                                "fecha_fin": str(st.session_state.fecha_fin),
                                "eventos": eventos
                            }, timeout=30)
                            if r2.ok:
                                d = r2.json()
                                st.session_state.agenda_generada = d.get("agenda", [])
                                st.session_state.agenda_score = d.get("score", 0)
                            else:
                                st.warning("⚠️ No se pudo generar agenda")
                        else:
                            st.session_state.agenda_generada = [] # Limpiar agenda si no hay suficientes eventos
                            st.session_state.agenda_score = 0
                    else: # Otros errores de la API (ej. status_code no 200, pero JSON válido con status error)
                        st.error(f"❌ Error del servidor: {data.get('mensaje', 'Error desconocido')}")
                        st.session_state.sugerencias_llm = []
                        st.session_state.eventos_encontrados = []
                        st.session_state.agenda_generada = []

                except requests.exceptions.Timeout:
                    st.error("❌ La conexión con el servidor API se agotó. Inténtalo de nuevo más tarde.")
                    st.session_state.sugerencias_llm = []
                    st.session_state.eventos_encontrados = []
                    st.session_state.agenda_generada = []
                except requests.exceptions.ConnectionError:
                    st.error("❌ No se pudo conectar con el servidor API. Asegúrate de que esté en ejecución.")
                    st.session_state.sugerencias_llm = []
                    st.session_state.eventos_encontrados = []
                    st.session_state.agenda_generada = []
                except Exception as e:
                    st.error(f"❌ Error inesperado: {e}")
                    st.session_state.sugerencias_llm = []
                    st.session_state.eventos_encontrados = []
                    st.session_state.agenda_generada = []

    # Mostrar sugerencias si existen, fuera del bloque del botón para que persistan
    if st.session_state.sugerencias_llm:
        st.markdown("---")
        st.markdown("### Sugerencias de búsqueda:")
        for sug in st.session_state.sugerencias_llm:
            st.info(f"- **{sug}**")
        st.markdown("---")

    eventos = st.session_state.eventos_encontrados
    if eventos:
        total_paginas = (len(eventos) - 1) // 10 + 1
        st.write(f"Página {st.session_state.pagina_actual + 1} de {total_paginas}")
        inicio = st.session_state.pagina_actual * 10
        fin = inicio + 10
        for ev in eventos[inicio:fin]:
            mostrar_evento(ev)

        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("⬅️ Anterior") and st.session_state.pagina_actual > 0:
                st.session_state.pagina_actual -= 1
                st.rerun()
        with col_next:
            if st.button("➡️ Siguiente") and st.session_state.pagina_actual < total_paginas - 1:
                st.session_state.pagina_actual += 1
                st.rerun()
    elif not st.session_state.sugerencias_llm and query.strip() and st.session_state.query_input: # Solo si la query no está vacía y no hay sugerencias, ni eventos
        st.info("No se encontraron eventos para tu búsqueda.")


with colB:
    st.subheader("🧠 Agenda generada")
    if st.session_state.agenda_generada:
        st.markdown(f"🎯 Puntaje total: **{st.session_state.agenda_score:.2f}**")
        for i, ev in enumerate(st.session_state.agenda_generada, 1):
            with st.expander(f"🗓️ Evento {i}"):
                mostrar_evento(ev)
    else:
        st.info("La agenda se mostrará aquí si se encuentran al menos 5 eventos.")