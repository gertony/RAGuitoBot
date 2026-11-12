"""Casos de caja negra por escenario de caso de uso (P9 · III.1.3.1), sobre la base real reconstruida (semilla 42).

Cada caso parte de un escenario de la Fase 1 (flujo normal o alternativo) y de una partición de la entrada.
Se verifica lo que el agente le entrega al LLM: la ruta y el contexto recuperado. El LLM se reemplaza por un
doble que no consume cuota, así que el caso comprueba que los datos correctos llegan a la generación; el
texto final que redacta el LLM se evalúa aparte (RAGAS y sesión de aceptación).
Los identificadores (CP-01-01, ...) son los de las tablas de la P9.
"""
import shutil

import pytest
from conftest import DB_PATH
from langchain_core.language_models.fake import FakeListLLM


class SinNormativa:
    def buscar(self, pregunta, k=4):
        return []


@pytest.fixture(scope="module")
def agente(tmp_path_factory):
    if not DB_PATH.exists():
        pytest.fail(f"No existe {DB_PATH}: ejecuta primero DB/build_db.py")
    copia = tmp_path_factory.mktemp("cu") / "academico.db"
    shutil.copy(DB_PATH, copia)
    from agente import Agente
    return Agente(db=copia, llm=FakeListLLM(responses=["-"]), buscador=SinNormativa(), registrar=False)


def casos(*filas):
    """Cada fila empieza con el identificador del caso (CP-xx-yy), que se usa como nombre en la salida de pytest."""
    return [f if isinstance(f, type(pytest.param())) else pytest.param(*f, id=f[0]) for f in filas]


def contexto(agente, eid, pregunta):
    r = agente.responder(eid, pregunta)
    return r["ruta"], " ".join(r["contextos"])


# ---------------------------------------------------------------- CU-01 Consultar nota por asignatura
@pytest.mark.parametrize(("caso", "pregunta", "esperado"), casos(
    ("CP-01-01", "¿Qué nota saqué en Inglés I?", "Inglés I (ACFB0102)"),          # flujo normal
    ("CP-01-02", "que nota saque en ingles 1", "Inglés I (ACFB0102)"),            # sin tildes, minúsculas, cifra
    ("CP-01-03", "nota de ingls I", "Inglés I (ACFB0102)"),                       # error de tipeo
    ("CP-01-04", "¿Qué nota saqué en Metodología del Trabajo Universitario?",     # alt. B: sin periodo
     "2022-II: 7 (desaprobado); 2023-I: 12 (aprobado)"),
    ("CP-01-05", "¿Qué nota saqué en Electrónica de Potencia II?",                # alt. A: no matriculado
     "no registra matrícula en Electrónica de Potencia II"),
))
def test_cu01_nota(agente, caso, pregunta, esperado):
    ruta, ctx = contexto(agente, "EST0016", pregunta)
    assert ruta == "personal"
    assert esperado in ctx


def test_cu01_06_asignatura_inexistente_no_inventa_curso(agente):
    # Alt. A: el curso no existe en la malla. El router no debe asociar la pregunta a ningún curso real.
    ruta, ctx = contexto(agente, "EST0016", "¿Qué nota saqué en Cocina Molecular?")
    assert "Registro del estudiante EST0016 en" not in ctx


def test_cu01_07_id_de_otro_estudiante_en_el_texto(agente):
    # RF-17: el texto nombra a EST0003, pero el contexto es solo del estudiante de la sesión.
    _, ctx = contexto(agente, "EST0016", "¿Cuál es la nota de EST0003 en Inglés I?")
    assert "EST0016" in ctx and "EST0003" not in ctx


# ---------------------------------------------------------------- CU-02 a CU-05 (flujo análogo a CU-01)
@pytest.mark.parametrize(("caso", "eid", "pregunta", "esperado"), casos(
    ("CP-02-01", "EST0030", "¿Cuál es mi promedio ponderado?", "estudiante EST0030: 12,64"),
    ("CP-05-01", "EST0033", "¿Qué clases tengo el lunes?", "Lunes 15:00-18:00: Electrónica de Potencia I"),
    ("CP-05-02", "EST0061", "¿Cuál es mi horario?", "no tiene horario porque no tiene asignaturas en curso"),
))
def test_cu02_cu05_consultas_personales(agente, caso, eid, pregunta, esperado):
    ruta, ctx = contexto(agente, eid, pregunta)
    assert ruta == "personal"
    assert esperado in ctx


# ---------------------------------------------------------------- CU-12 Prerrequisitos y posibilidad de matrícula
@pytest.mark.parametrize(("caso", "eid", "curso", "motivo"), casos(
    ("CP-12-01", "EST0001", "Geografía General", "Sí puede matricularse"),
    ("CP-12-02", "EST0001", "Lenguaje y Comunicación II", "le falta aprobar el prerrequisito"),
    ("CP-12-03", "EST0007", "Lenguaje y Comunicación I", "ya la aprobó"),
    ("CP-12-04", "EST0001", "Lenguaje y Comunicación I", "la está cursando actualmente"),
    ("CP-12-05", "EST0061", "Inglés III", "su estado es 'egresado'"),
))
def test_cu12_puede_llevar(agente, caso, eid, curso, motivo):
    ruta, ctx = contexto(agente, eid, f"¿Puedo llevar {curso} el próximo ciclo?")
    assert ruta == "personal"
    assert motivo in ctx


# ------------------------------------------------ CU-09 Normatividad (ruta; la recuperación se mide con RAGAS)
@pytest.mark.parametrize(("caso", "eid", "pregunta", "ruta_esperada"), casos(
    ("CP-09-01", None, "¿Puedo reservar mi matrícula y por cuánto tiempo?", "normativa"),
    ("CP-09-02", None, "que pasa si jalo un curso 3 veces", "normativa"),
    ("CP-09-03", "EST0069", "Desaprobé Lenguaje y Comunicación I varias veces, ¿qué consecuencia tiene eso para mí?",
     "mixta"),
    pytest.param("CP-09-04", "EST0016", "¿Qué dice el reglamento sobre el horario del comedor?", "normativa",
                 marks=pytest.mark.xfail(strict=True, reason="D-11: «horario» se toma como consulta personal"),
                 id="CP-09-04"),
))
def test_cu09_ruta(agente, caso, eid, pregunta, ruta_esperada):
    ruta, _ = contexto(agente, eid, pregunta)
    assert ruta == ruta_esperada
