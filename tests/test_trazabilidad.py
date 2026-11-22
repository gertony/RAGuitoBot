"""Casos que completan la trazabilidad de la P9 (III.1): cada requerimiento que declara la ficha de un caso de uso
tiene al menos un caso que lo verifica.

- CU-02 a CU-05: particiones por tipo de estudiante (regular con historial, primer ciclo, egresado, retirado,
  suspendido), sobre la base real reconstruida (semilla 42).
- CU-08: los datos académicos que consumen los demás casos de uso (RF-01).
- CU-11: lo que el login debe garantizar además de entrar (RF-19, RF-17 en la interfaz, RNF-11).
- CU-01: RF-03 (el LLM recibe solo el contexto recuperado).
- CU-08: RNF-05 (sin datos identificables). CU-06: RNF-09 (cambio de proveedor del LLM por configuración).
Los identificadores (CP-02-02, ...) son los de las tablas de la P9.
"""
import shutil
import sqlite3
from contextlib import closing

import pytest
from conftest import DB_PATH
from langchain_core.language_models.fake import FakeListLLM

import auth


def casos(*filas):
    """Cada fila empieza con el identificador del caso, que se usa como nombre en la salida de pytest."""
    return [pytest.param(*f, id=f[0]) for f in filas]


class SinNormativa:
    def buscar(self, pregunta, k=4):
        return []


RECIBIDO = []


class LLMQueGuardaElPrompt(FakeListLLM):
    """Doble del LLM que guarda el texto que recibe, para revisar qué se le envía."""

    def _call(self, prompt, *args, **kwargs):
        RECIBIDO.append(prompt)
        return "Respuesta de prueba."


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    if not DB_PATH.exists():
        pytest.fail(f"No existe {DB_PATH}: ejecuta primero DB/build_db.py")
    copia = tmp_path_factory.mktemp("traz") / "academico.db"
    shutil.copy(DB_PATH, copia)
    return copia


@pytest.fixture(scope="module")
def agente(base):
    from agente import Agente
    return Agente(db=base, llm=FakeListLLM(responses=["-"]), buscador=SinNormativa(), registrar=False)


def contexto(agente, eid, pregunta):
    r = agente.responder(eid, pregunta)
    return r["ruta"], " ".join(r["contextos"])


# ================================================================ CU-02 Consultar promedio ponderado (RF-05)
@pytest.mark.parametrize(("caso", "eid", "pregunta", "esperado"), casos(
    ("CP-02-02", "EST0001", "¿Cuál es mi promedio ponderado?", "aún no tiene asignaturas calificadas"),  # 1.er ciclo
    ("CP-02-03", "EST0061", "¿Cuál es mi promedio ponderado?", "12,86 (promedio de 10 semestres calificados)"),
    ("CP-02-04", "EST0065", "¿Cuál es mi promedio ponderado?", "12,25"),                               # retirado
    ("CP-02-05", "EST0030", "cuanto es mi promedio", "12,64"),                                         # coloquial
))
def test_cu02_promedio(agente, caso, eid, pregunta, esperado):
    ruta, ctx = contexto(agente, eid, pregunta)
    assert ruta == "personal"
    assert esperado in ctx


def test_cu02_06_promedio_dentro_de_la_escala(base):
    # Partición de salida: para los 70 estudiantes, el promedio está en la escala vigesimal (0 a 20) o no existe.
    from consultas import Academico
    a = Academico(base)
    for (eid,) in a.q("SELECT id FROM estudiante"):
        datos = a.promedio_ponderado(eid)["datos"]
        total = datos["total"]
        assert total is None or 0 <= total <= 20, eid


# ================================================================ CU-03 Consultar avance curricular (RF-06)
@pytest.mark.parametrize(("caso", "eid", "pregunta", "esperado"), casos(
    ("CP-03-01", "EST0030", "¿Cuál es mi avance curricular?", "105 de 214 créditos aprobados"),
    ("CP-03-02", "EST0001", "¿Cuál es mi avance curricular?", "0 de 214 créditos aprobados"),   # límite inferior
    ("CP-03-03", "EST0061", "¿Cuál es mi avance curricular?", "214 de 214 créditos aprobados"),  # límite superior
    ("CP-03-04", "EST0030", "¿Cuántos créditos me faltan para terminar?", "Faltan 109 créditos"),
))
def test_cu03_avance(agente, caso, eid, pregunta, esperado):
    ruta, ctx = contexto(agente, eid, pregunta)
    assert ruta == "personal"
    assert esperado in ctx


# ================================================================ CU-04 Consultar estado de matrícula (RF-07)
@pytest.mark.parametrize(("caso", "eid", "esperado"), casos(
    ("CP-04-01", "EST0030", "matriculado como estudiante regular en el periodo 2024-I"),
    ("CP-04-02", "EST0061", "es egresado"),
    ("CP-04-03", "EST0065", "figura como retirado"),
    ("CP-04-04", "EST0069", "separado temporalmente (suspendido)"),
))
def test_cu04_estado(agente, caso, eid, esperado):
    ruta, ctx = contexto(agente, eid, "¿Cuál es mi estado de matrícula?")
    assert ruta == "personal"
    assert esperado in ctx


# ================================================================ CU-05 Consultar horario de clases (RF-08)
@pytest.mark.parametrize(("caso", "eid", "pregunta", "esperado"), casos(
    ("CP-05-03", "EST0065", "¿Cuál es mi horario?", "no tiene horario porque no tiene asignaturas en curso"),
    ("CP-05-04", "EST0030", "¿Qué clases tengo el lunes?", "Lunes 13:00-15:00: Seminario de Tesis"),
))
def test_cu05_horario(agente, caso, eid, pregunta, esperado):
    ruta, ctx = contexto(agente, eid, pregunta)
    assert ruta == "personal"
    assert esperado in ctx


def test_cu05_06_horario_completo_si_no_se_nombra_un_dia(agente):
    # CP-05-06 (CU-05, alternativo B): sin día en la pregunta, el contexto trae todas las sesiones de la semana.
    ruta, ctx = contexto(agente, "EST0030", "¿Cuál es mi horario?")
    assert ruta == "personal"
    assert all(dia in ctx for dia in ("Lunes", "Martes", "Miércoles", "Jueves", "Viernes"))


def test_cu05_05_horario_sin_cruces(base):
    # Regla de matrícula: ningún estudiante tiene dos sesiones que se crucen el mismo día.
    with closing(sqlite3.connect(base)) as con:
        cruces = con.execute("""
            SELECT COUNT(*) FROM matricula m1
            JOIN horario h1 ON h1.curso = m1.curso AND h1.periodo = m1.periodo
            JOIN matricula m2 ON m2.estudiante_id = m1.estudiante_id AND m2.periodo = m1.periodo AND m2.curso > m1.curso
            JOIN horario h2 ON h2.curso = m2.curso AND h2.periodo = m2.periodo AND h2.dia = h1.dia
            WHERE m1.condicion = 'en curso' AND m2.condicion = 'en curso'
              AND h1.hora_inicio < h2.hora_fin AND h2.hora_inicio < h1.hora_fin""").fetchone()[0]
    assert cruces == 0


# ================================================================ CU-08 Proveer datos académicos (RF-01)
def test_cu08_01_conexion_activa_con_la_base(agente):
    # RF-01: la misma instancia responde dos consultas seguidas leyendo la base (no un archivo copiado en memoria).
    for eid in ("EST0030", "EST0061"):
        _, ctx = contexto(agente, eid, "¿Cuál es mi estado de matrícula?")
        assert eid in ctx


# ================================================================ CU-01: RF-03 (solo el contexto recuperado)
def test_cu01_10_el_llm_recibe_solo_el_contexto_recuperado(base):
    from agente import Agente
    llm = LLMQueGuardaElPrompt(responses=["-"])
    RECIBIDO.clear()
    a = Agente(db=base, llm=llm, buscador=SinNormativa(), registrar=False)
    r = a.responder("EST0030", "¿Cuál es mi promedio ponderado?")
    enviado = RECIBIDO[-1]
    assert "Responde ÚNICAMENTE con la información del CONTEXTO" in enviado
    assert all(c in enviado for c in r["contextos"])        # el contexto recuperado va en el prompt
    assert "EST0030" in enviado and "EST0061" not in enviado  # y nada de otro estudiante


# ================================================================ CU-11: RF-19, RF-17 en la interfaz y RNF-11
@pytest.fixture
def app(tmp_path, monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    import servicios
    from build_db import ESQUEMA
    ruta = tmp_path / "academico.db"
    with sqlite3.connect(ruta) as c:
        c.executescript(ESQUEMA)
    con = auth.conectar(ruta)
    auth.crear_usuario(con, "EST0001", "clave123", "estudiante", "EST0001")
    con.close()
    monkeypatch.setattr(servicios, "DB", ruta)
    st.cache_resource.clear()
    auth._FALLOS.clear()
    return lambda: AppTest.from_file(str(servicios.RAIZ / "APP" / "app.py"), default_timeout=60).run(), ruta


def ingresar(at, codigo, clave):
    at.text_input[0].set_value(codigo)
    at.text_input[1].set_value(clave)
    at.button[0].click().run()
    return at


def test_cu11_22_la_sesion_queda_asociada_al_estudiante(app):
    # RF-19: al iniciar sesión, la sesión guarda el identificador del estudiante que se autenticó.
    nueva, _ = app
    at = ingresar(nueva(), "est0001", "clave123")
    assert at.session_state["usuario"]["estudiante_id"] == "EST0001"


def test_cu11_23_el_chat_usa_el_id_de_la_sesion(app, monkeypatch):
    # RF-17 en la interfaz: aunque la pregunta nombre a otro estudiante, el chat consulta con el ID de la sesión.
    import servicios
    nueva, _ = app
    llamadas = []

    class AgenteEspia:
        def responder(self, eid, pregunta, historial=None):
            llamadas.append(eid)
            return {"respuesta": "ok", "contextos": []}

    monkeypatch.setattr(servicios, "agente", lambda: AgenteEspia())
    at = ingresar(nueva(), "EST0001", "clave123")
    at.chat_input[0].set_value("¿Cuáles son las notas de EST0003?").run()
    assert llamadas == ["EST0001"]


def test_cu11_24_contrasena_guardada_con_hash(app):
    # RNF-11: en la base no hay contraseñas legibles; se guarda un hash bcrypt con sal.
    _, ruta = app
    with closing(sqlite3.connect(ruta)) as con:
        guardado = con.execute("SELECT hash FROM usuario WHERE codigo = 'EST0001'").fetchone()[0]
    guardado = guardado.decode() if isinstance(guardado, bytes) else guardado
    assert "clave123" not in guardado and guardado.startswith("$2")


# ================================================================ CU-08 (RNF-05) y CU-06 (RNF-09)
def test_cu08_06_sin_datos_personales_identificables(base):
    # RNF-05: la tabla de estudiantes no tiene nombres, documentos ni contactos.
    with closing(sqlite3.connect(base)) as con:
        columnas = {fila[1] for fila in con.execute("PRAGMA table_info(estudiante)")}
    prohibidas = {"nombre", "nombres", "apellido", "apellidos", "dni", "documento", "correo", "email", "telefono"}
    assert not columnas & prohibidas


@pytest.mark.parametrize(("caso", "proveedor", "clase"), casos(
    ("CP-06-05", "groq", "ChatGroq"),
    ("CP-06-06", "gemini", "ChatGoogleGenerativeAI"),
))
def test_cu06_cambio_de_proveedor_por_configuracion(monkeypatch, caso, proveedor, clase):
    # RNF-09: el proveedor del LLM se elige con LLM_PROVEEDOR, sin tocar el código (no se llama a la API).
    pytest.importorskip("langchain_groq")
    pytest.importorskip("langchain_google_genai")
    from agente import crear_llm
    monkeypatch.setenv("LLM_PROVEEDOR", proveedor)
    monkeypatch.setenv("GROQ_API_KEY", "clave-de-prueba")
    monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setenv("GOOGLE_API_KEY", "clave-de-prueba")
    monkeypatch.delenv("LLM_RPM", raising=False)
    assert type(crear_llm()).__name__ == clase


def test_chat_muestra_fuentes_y_pasa_el_historial(app, monkeypatch):
    # 21/11: el chat muestra las fuentes citadas y envía las preguntas previas (memoria de la conversación).
    import servicios
    nueva, _ = app
    recibido = []

    class AgenteFalso:
        def responder(self, eid, pregunta, historial=None):
            recibido.append(list(historial or []))
            contextos = ["[Reglamento General UNE | Art. 289 | Reserva] x", "Estado de matrícula del estudiante ..."]
            return {"respuesta": "Según el art. 289...", "contextos": contextos}

    monkeypatch.setattr(servicios, "agente", lambda: AgenteFalso())
    at = ingresar(nueva(), "EST0001", "clave123")
    at.chat_input[0].set_value("¿puedo reservar mi matrícula?").run()
    at.chat_input[0].set_value("¿y por cuánto tiempo?").run()
    assert recibido == [[], ["¿puedo reservar mi matrícula?"]]
    assert at.expander and at.expander[0].label == "Fuentes consultadas"
    textos = " ".join(m.value for m in at.markdown)
    assert "Art. 289" in textos and "Tus datos del registro académico" in textos
