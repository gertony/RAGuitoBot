"""P-02 Chat conversacional (estudiante). RF-02 a RF-10, RF-15, RF-16, RF-17."""
import re

import streamlit as st

import servicios

usuario = st.session_state["usuario"]
BIENVENIDA = ("¡Hola! Soy RAGuitoBot. Puedo responder preguntas sobre tus notas, tu promedio, tu avance, "
              "tu horario y tu matrícula, y sobre el Reglamento General de la UNE. Escribe tu pregunta.")
# sugerencias genéricas: no coinciden con las tareas del test de usuario, para no sesgarlo
SUGERENCIAS = ["¿Qué cursos estoy llevando este ciclo?", "¿Qué clases tengo el lunes?",
               "¿Puedo reservar mi matrícula?"]
ABSTENCION = "No tengo esa información en los registros disponibles."
PISTA = ("Puedo ayudarte con tus notas, promedio, créditos, horario, cursos pendientes o disponibles, y con el "
         "Reglamento General de la UNE. Prueba a preguntarlo de otra forma.")


def fuentes(contextos):
    """Encabezados [documento | artículo | título] de los fragmentos normativos; los datos propios, en una línea."""
    normas = [m.group(1) for c in contextos if (m := re.match(r"\[([^\]]+)\]", c))]
    propios = any(not c.startswith("[") for c in contextos)
    return list(dict.fromkeys(normas)) + (["Tus datos del registro académico"] if propios else [])


def mostrar(m):
    st.write(m["texto"])
    if m.get("fuentes"):
        with st.expander("Fuentes consultadas"):
            for f in m["fuentes"]:
                st.markdown(f"- {f}")

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
        mostrar(m)

pregunta = st.chat_input("Escribe tu pregunta...", max_chars=500) or st.session_state.pop("pendiente", None)
if pregunta:
    historial.append({"rol": "user", "texto": pregunta})
    with st.chat_message("user"):
        st.write(pregunta)
    previas = [m["texto"] for m in historial[:-1] if m["rol"] == "user"]  # memoria para «¿y en 2023-I?»
    with st.chat_message("assistant"):
        with st.spinner("RAGuitoBot está escribiendo..."):
            try:
                # RF-17: el ID sale de la sesión iniciada, nunca del texto de la pregunta
                r = servicios.agente().responder(usuario["estudiante_id"], pregunta, historial=previas)
                texto, citas = r["respuesta"], fuentes(r.get("contextos", []))
                if texto.strip() == ABSTENCION:
                    texto, citas = f"{texto} {PISTA}", []
            except Exception:  # noqa: BLE001 - servicio del LLM caído, cuota agotada, red...
                texto, citas = "No pude responder en este momento. Inténtalo de nuevo en unos segundos.", []
        mostrar({"texto": texto, "fuentes": citas})
    historial.append({"rol": "assistant", "texto": texto, "fuentes": citas})
    st.rerun()
