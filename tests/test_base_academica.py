"""Integridad de academico.db construida con build_db.py, semilla 42 (tesis, §4.3.1 y Tabla 10)."""
import sqlite3

import pytest
from conftest import DB_PATH


@pytest.fixture(scope="module")
def con():
    if not DB_PATH.exists():
        pytest.fail(f"No existe {DB_PATH}: ejecuta primero DB/build_db.py")
    c = sqlite3.connect(DB_PATH)
    yield c
    c.close()


@pytest.mark.parametrize(("tabla", "esperado"), [
    ("estudiante", 70), ("curso", 82), ("prerrequisito", 23), ("matricula", 2905),
    ("horario", 164), ("carrera", 1), ("meta", 14)])
def test_conteos_semilla_42(con, tabla, esperado):
    (n,) = con.execute(f"SELECT COUNT(*) FROM {tabla}").fetchone()  # noqa: S608 - nombre de tabla fijo
    assert n == esperado


def test_nota_coherente_con_condicion(con):
    (n,) = con.execute("SELECT COUNT(*) FROM matricula WHERE (condicion='aprobado' AND nota<11) OR "
                       "(condicion='desaprobado' AND nota>=11) OR (condicion='en curso' AND nota IS NOT NULL)"
                       ).fetchone()
    assert n == 0


def test_prerrequisito_aprobado_en_periodo_anterior(con):
    """RF-18: ningún curso se llevó sin haber aprobado antes su prerrequisito."""
    violaciones = con.execute("""
        SELECT m.estudiante_id, m.curso, p.requiere, m.periodo FROM matricula m
        JOIN prerrequisito p ON p.curso = m.curso
        WHERE NOT EXISTS (SELECT 1 FROM matricula a WHERE a.estudiante_id = m.estudiante_id
                          AND a.curso = p.requiere AND a.condicion = 'aprobado' AND a.periodo < m.periodo)
    """).fetchall()
    assert violaciones == []


def test_egresados_completan_214_creditos(con):
    filas = con.execute("""
        SELECT e.id, SUM(c.creditos) FROM estudiante e
        JOIN (SELECT DISTINCT estudiante_id, curso FROM matricula WHERE condicion = 'aprobado') m
             ON m.estudiante_id = e.id
        JOIN curso c ON c.codigo = m.curso
        WHERE e.estado = 'egresado' GROUP BY e.id
    """).fetchall()
    assert len(filas) == 4
    assert all(abs(cr - 214) < 1e-6 for _, cr in filas)
