"""P-02 Chat conversacional (estudiante). RF-02 a RF-10, RF-15, RF-16, RF-17."""
import streamlit as st

import servicios

usuario = st.session_state["usuario"]
BIENVENIDA = ("¡Hola! Soy RAGuitoBot. Puedo responder preguntas sobre tus notas, tu promedio, tu avance, "
              "tu horario y tu matrícula, y sobre el Reglamento General de la UNE. Escribe tu pregunta.")
# sugerencias genéricas: no coinciden con las tareas del test de usuario, para no sesgarlo
SUGERENCIAS = ["¿Qué cursos estoy llevando este ciclo?", "¿Qué clases tengo el lunes?",
               "¿Puedo reservar mi matrícula?"]

st.title("RAGuitoBot")
historial = st.session_state.setdefault("historial", [])

with st.chat_message("assistant"):
    st.write(BIENVENIDA)
if not historial:
    cols = st.columns(len(SUGERENCIAS))
    for col, texto in zip(cols, SUGERENCIAS, strict=True):
        if col.button(texto, width="stretch"):
            st.session_state["pendiente"] = texto

for m in historial:
    with st.chat_message(m["rol"]):
        st.write(m["texto"])

pregunta = st.chat_input("Escribe tu pregunta...", max_chars=500) or st.session_state.pop("pendiente", None)
if pregunta:
    historial.append({"rol": "user", "texto": pregunta})
    with st.chat_message("user"):
        st.write(pregunta)
    with st.chat_message("assistant"):
        with st.spinner("RAGuitoBot está escribiendo..."):
            try:
                # RF-17: el ID sale de la sesión iniciada, nunca del texto de la pregunta
                texto = servicios.agente().responder(usuario["estudiante_id"], pregunta)["respuesta"]
            except Exception:  # noqa: BLE001 - servicio del LLM caído, cuota agotada, red...
                texto = "No pude responder en este momento. Inténtalo de nuevo en unos segundos."
        st.write(texto)
    historial.append({"rol": "assistant", "texto": texto})
    st.rerun()
