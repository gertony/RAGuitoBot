"""Pruebas de caja blanca (P9 · III.1): recorren las ramas de los cálculos y decisiones más sensibles.

Usan una base mínima construida con el mismo esquema de build_db.py, con datos elegidos a mano para que
cada rama tenga un resultado conocido de antemano. No dependen de academico.db ni de la API del LLM: el
LLM se reemplaza por un modelo falso de LangChain (FakeListLLM) y el buscador de normativa por un doble.

Casos: CB-01 promedio ponderado · CB-02 puede_llevar · CB-03 router por reglas · CB-04 búsqueda de cursos ·
CB-05 desempate del LLM y sesión · CB-06 registro de la consulta (RF-12) y aislamiento (RF-17) ·
CB-07 sesiones concurrentes (regresión del defecto D-02).
"""
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import pytest
from langchain_core.language_models.fake import FakeListLLM

from build_db import ESQUEMA

CURSOS = [  # codigo, nombre, ciclo, creditos, electivo, grupo
    ("A01", "Matemática Básica I", 1, 4, 0, None),
    ("A02", "Matemática Básica II", 2, 4, 0, None),
    ("A03", "Circuitos Eléctricos III", 3, 3, 0, None),
    ("B01", "Lenguaje y Comunicación I", 1, 2, 0, None),
    ("C00", "Física I", 2, 3, 0, None),
    ("C01", "Física II", 3, 3, 0, None),
    ("E01", "Taller de Robótica", 3, 2, 1, "G1"),
]
PRERREQ = [("A02", "A01"), ("A03", "A02"), ("C01", "C00")]
ESTUDIANTES = [  # id, ciclo, periodo, estado
    ("EST0001", 3, "2025-I", "matriculado regular"),
    ("EST0002", 10, "2024-II", "egresado"),
    ("EST0003", 1, "2025-I", "matriculado regular"),
]
MATRICULAS = [  # estudiante, curso, periodo, intento, nota, condicion
    ("EST0001", "A01", "2024-I", 1, 14, "aprobado"),
    ("EST0001", "B01", "2024-I", 1, 8, "desaprobado"),
    ("EST0001", "B01", "2024-II", 2, 16, "aprobado"),
    ("EST0001", "A02", "2024-II", 1, 13, "aprobado"),
    ("EST0001", "A03", "2025-I", 1, None, "en curso"),
]


@pytest.fixture
def ruta_db(tmp_path):
    ruta = tmp_path / "mini.db"
    con = sqlite3.connect(ruta)
    con.executescript(ESQUEMA)
    con.execute("INSERT INTO carrera VALUES (1, 'Ing. de Sistemas', 'UNE', 'FATEC', 'prueba', 20)")
    con.executemany("INSERT INTO curso VALUES (?, ?, ?, ?, 2, 2, ?, ?)", CURSOS)
    con.executemany("INSERT INTO prerrequisito VALUES (?, ?)", PRERREQ)
    con.executemany("INSERT INTO estudiante VALUES (?, 'u', 'M', 2024, 'regular', 'Lima', ?, ?, 0, 0, ?, 'prueba')",
                    ESTUDIANTES)
    con.executemany("INSERT INTO matricula (estudiante_id, curso, periodo, intento, nota, condicion) "
                    "VALUES (?, ?, ?, ?, ?, ?)", MATRICULAS)
    con.execute("INSERT INTO horario (curso, periodo, dia, hora_inicio, hora_fin, aula) "
                "VALUES ('A03', '2025-I', 'Martes', '08:00', '10:00', 'L-1')")
    con.commit()
    con.close()
    return ruta


@pytest.fixture
def academico(ruta_db):
    from consultas import Academico
    return Academico(ruta_db)


# ---------------------------------------------------------------- CB-01 promedio ponderado (RF-05)
def test_cb01_promedio_ponderado_por_semestre(academico):
    # 2024-I: (14·4 + 8·2) / 6 = 12.00 · 2024-II: (16·2 + 13·4) / 6 = 14.00 · total = (12 + 14) / 2 = 13.00
    # El curso en curso (nota NULL) no entra en el cálculo.
    d = academico.promedio_ponderado("EST0001")["datos"]
    assert d["parciales"] == {"2024-I": 12.0, "2024-II": 14.0}
    assert d["total"] == 13.0
    assert d["ultimo_semestre"] == "2024-II"


def test_cb01_promedio_sin_notas(academico):
    r = academico.promedio_ponderado("EST0003")
    assert r["datos"]["total"] is None
    assert "aún no tiene asignaturas calificadas" in r["texto"]


# ------------------------------------------------ CB-02 ¿puede matricularse? (RF-18), una prueba por rama
@pytest.mark.parametrize("eid, curso, puede, motivo", [
    ("EST0001", "A01", False, "ya la aprobó"),
    ("EST0001", "A03", False, "la está cursando actualmente"),
    ("EST0002", "C00", False, "su estado es 'egresado'"),
    ("EST0001", "C01", False, "le falta aprobar el prerrequisito: Física I"),
    ("EST0001", "C00", True, "cumple los prerrequisitos y no la ha aprobado"),
], ids=["aprobado", "en-curso", "estado-no-regular", "falta-prerrequisito", "cumple"])
def test_cb02_puede_llevar(academico, eid, curso, puede, motivo):
    d = academico.puede_llevar(eid, curso)["datos"]
    assert (d["puede"], d["motivo"]) == (puede, motivo)


# ---------------------------------------------------------------- CB-03 router por reglas (RF-02, RF-16)
@pytest.mark.parametrize("pregunta, ruta, intencion", [
    ("¿Cuál es mi promedio ponderado?", "personal", "promedio"),
    ("¿Qué dice el reglamento sobre la reserva de matrícula?", "normativa", None),
    ("Desaprobé Lenguaje y Comunicación I varias veces, ¿qué pasa si la vuelvo a desaprobar?", "mixta", "desaprobados"),
    ("Hola", None, None),
], ids=["personal", "normativa", "mixta", "indeciso"])
def test_cb03_router_reglas(academico, pregunta, ruta, intencion):
    from agente import router_reglas
    r = router_reglas(pregunta, academico)
    assert r["ruta"] == ruta
    if intencion:
        assert intencion in r["intenciones"]


def test_cb03_repitencia_agrega_estado(academico):
    # La repitencia puede cambiar el estado de matrícula (art. 282): la regla agrega la intención "estado".
    from agente import router_reglas
    r = router_reglas("Desaprobé Lenguaje y Comunicación I varias veces", academico)
    assert {"desaprobados", "estado"} <= set(r["intenciones"])


# ---------------------------------------------------------------- CB-04 búsqueda del curso en lenguaje natural
@pytest.mark.parametrize("texto, codigo", [
    ("¿puedo llevar circuitos electricos 3?", "A03"),     # número arábigo → romano (estrategia 1)
    ("mi nota de fisica II", "C01"),                      # nombre completo; se prefiere el más largo
    ("matematica basik 2", "A02"),                       # error de tipeo (estrategia 2 o 3)
    ("¿cuándo abre la biblioteca?", None),                # ningún curso
], ids=["romano", "mas-largo", "tipeo", "ninguno"])
def test_cb04_buscar_curso(academico, texto, codigo):
    assert academico.buscar_curso(texto) == codigo


# ---------------------------------------------------------------- CB-05 y CB-06: agente con dobles (sin API)
class BuscadorFalso:
    def buscar(self, pregunta, k=4):
        return ["Art. 282 (fragmento de prueba)."]


def agente(ruta_db, respuestas):
    from agente import Agente
    return Agente(db=ruta_db, llm=FakeListLLM(responses=respuestas), buscador=BuscadorFalso())


def test_cb05_desempate_del_llm(ruta_db):
    a = agente(ruta_db, ['{"ruta": "personal"}', "Respuesta de prueba."])
    r = a.responder("EST0001", "Hola")
    assert (r["ruta"], r["router"], r["intenciones"]) == ("personal", "llm", ["resumen"])


def test_cb05_llm_fallido_va_a_normativa(ruta_db):
    a = agente(ruta_db, ["no es un JSON", "Respuesta de prueba."])
    r = a.responder("EST0001", "Hola")
    assert (r["ruta"], r["router"]) == ("normativa", "llm-fallido")


def test_cb05_sin_sesion_no_hay_datos_personales(ruta_db):
    a = agente(ruta_db, ["Respuesta de prueba."])
    r = a.responder(None, "¿Cuál es mi promedio ponderado?")
    assert r["ruta"] == "normativa"
    assert all("EST" not in c for c in r["contextos"])


def test_cb06_registra_consulta_con_latencia(ruta_db):
    a = agente(ruta_db, ["Respuesta de prueba."])
    a.responder("EST0001", "¿Cuál es mi promedio ponderado?")
    with closing(sqlite3.connect(ruta_db)) as con:
        fila = con.execute("SELECT estudiante_id, ruta, latencia_ms FROM log_consultas").fetchone()
    assert fila[0] == "EST0001" and fila[1] == "personal" and fila[2] > 0


def test_cb06_id_de_la_sesion_y_no_del_texto(ruta_db):
    # RF-17: aunque la pregunta nombre a otro estudiante, el contexto es solo del estudiante de la sesión.
    a = agente(ruta_db, ["Respuesta de prueba."])
    r = a.responder("EST0001", "¿Cuál es el promedio de EST0003?")
    assert r["contextos"] and all("EST0003" not in c for c in r["contextos"])


# ---------------------------------------------------------------- CB-07 regresión del defecto D-02 (concurrencia)
class LLMLento(FakeListLLM):
    """Simula la demora de la API para que las consultas de varias sesiones se crucen."""

    def _call(self, *args, **kwargs):
        time.sleep(0.2)
        return "Respuesta de prueba."


def test_cb07_sesiones_concurrentes_no_se_mezclan(ruta_db):
    # La app comparte una sola instancia del agente entre sesiones. Con 8 consultas en paralelo de dos
    # estudiantes, cada respuesta y cada fila del registro deben corresponder al estudiante que preguntó.
    from agente import Agente
    a = Agente(db=ruta_db, llm=LLMLento(responses=["-"]), buscador=BuscadorFalso())
    ids = ["EST0001", "EST0003"] * 8

    with ThreadPoolExecutor(max_workers=8) as ex:
        salidas = list(ex.map(lambda eid: (eid, a.responder(eid, "¿Cuál es mi promedio ponderado?")), ids))

    assert all(all(eid in c for c in r["contextos"]) for eid, r in salidas)
    with closing(sqlite3.connect(ruta_db)) as con:
        filas = con.execute("SELECT estudiante_id, contextos FROM log_consultas").fetchall()
    assert len(filas) == 16
    assert all(eid in ctx for eid, ctx in filas)
