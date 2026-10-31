"""RAGuitoBot: interfaz Streamlit del agente conversacional (capa de presentación).

Pantallas (II.3, Práctica 7): P-01 Inicio de sesión · P-02 Chat (estudiante) ·
P-03 Panel, P-04 Actualizar normatividad y P-05 Reporte (investigador/evaluador).

Uso (desde RAGASTESIS):  streamlit run APP/app.py
"""
import time

import streamlit as st

import auth
import servicios

st.set_page_config(page_title="RAGuitoBot", page_icon="🎓", layout="centered")
MAX_INTENTOS, ESPERA_S = 5, 30


@st.cache_resource
def _con():
    return auth.conectar(servicios.DB)


def inicio_sesion():
    """P-01: valida el formulario y redirige según el rol (RF-19, RF-20)."""
    st.title("RAGuitoBot")
    st.caption("Asistente académico · Educación con especialidad de Electrónica e Informática (UNE)")
    bloqueo = st.session_state.get("bloqueado_hasta", 0)
    with st.form("login"):
        codigo = st.text_input("Código", placeholder="EST0001").strip()
        clave = st.text_input("Contraseña", type="password")
        enviar = st.form_submit_button("Ingresar", type="primary", width="stretch")
    if not enviar:
        return
    if time.time() < bloqueo:  # freno básico ante intentos repetidos (OWASP A07)
        st.error(f"Demasiados intentos. Espera {int(bloqueo - time.time())} s.")
        return
    error = auth.validar_formulario(codigo, clave)
    usuario = None if error else auth.autenticar(_con(), codigo, clave)
    if usuario is None:
        n = st.session_state.get("intentos", 0) + 1
        st.session_state["intentos"] = n
        if n >= MAX_INTENTOS:
            st.session_state["bloqueado_hasta"], st.session_state["intentos"] = time.time() + ESPERA_S, 0
        st.error(error or "Código o contraseña incorrectos.")  # mensaje genérico: no revela qué campo falló
        return
    st.session_state.clear()
    st.session_state["usuario"] = usuario
    st.rerun()


def cerrar_sesion():
    st.session_state.clear()
    st.rerun()


usuario = st.session_state.get("usuario")
if usuario is None:
    pagina = st.navigation([st.Page(inicio_sesion, title="Inicio de sesión", icon="🔐")], position="hidden")
else:
    if usuario["rol"] == "estudiante":
        paginas = [st.Page("paginas/chat.py", title="Chat", icon="💬", default=True)]
    else:
        paginas = [st.Page("paginas/panel.py", title="Panel del evaluador", icon="🧪", default=True),
                   st.Page("paginas/normatividad.py", title="Actualizar normatividad", icon="📄"),
                   st.Page("paginas/reporte.py", title="Reporte RAGAS", icon="📊")]
    pagina = st.navigation(paginas)
    with st.sidebar:
        st.markdown(f"**{usuario['codigo']}** · {usuario['rol']}")
        st.button("Cerrar sesión", on_click=cerrar_sesion, width="stretch")
pagina.run()
