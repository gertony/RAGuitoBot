"""RF-17 y RNF-05: las consultas de un estudiante solo devuelven sus propios registros."""
import re

import pytest
from conftest import DB_PATH

FUNCIONES = ["estado_matricula", "promedio_ponderado", "avance_curricular", "notas_periodo",
             "cursos_en_curso", "horario", "desaprobados", "resumen"]


@pytest.fixture(scope="module")
def academico():
    from consultas import Academico
    return Academico(DB_PATH)


@pytest.fixture(scope="module")
def ids(academico):
    return [r[0] for r in academico.q("SELECT id FROM estudiante ORDER BY id")]


def test_matriculas_solo_del_estudiante(academico, ids):
    for eid in ids:
        assert {m["estudiante_id"] for m in academico._matriculas(eid)} <= {eid}


@pytest.mark.parametrize("funcion", FUNCIONES)
def test_texto_no_menciona_a_otro_estudiante(academico, ids, funcion):
    for eid in ids:
        texto = getattr(academico, funcion)(eid)["texto"]
        assert set(re.findall(r"EST\d{4}", texto)) <= {eid}, (eid, texto[:120])
