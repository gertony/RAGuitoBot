"""CU-10 / RF-11: actualizar notas sin reconstruir la base."""
import sqlite3

import pandas as pd
import pytest


@pytest.fixture()
def con(tmp_path):
    from build_db import ESQUEMA
    c = sqlite3.connect(tmp_path / "a.db")
    c.executescript(ESQUEMA)
    c.execute("INSERT INTO curso VALUES ('C1','Curso 1',1,4,2,2,0,NULL)")
    c.execute("INSERT INTO estudiante VALUES ('EST0001','u','M',2023,'P','S',2,'2024-I',1,4,"
              "'matriculado regular','real')")
    c.execute("INSERT INTO matricula (estudiante_id,curso,periodo,intento,nota,condicion) "
              "VALUES ('EST0001','C1','2024-I',1,NULL,'en curso')")
    return c


def test_rechaza_datos_invalidos(con):
    from actualizacion import validar
    df = pd.DataFrame([{"estudiante_id": "EST9999", "curso": "ZZ", "periodo": "2024-3", "nota": 25}])
    assert len(validar(df, con)) == 4


def test_reemplaza_y_agrega(con):
    from actualizacion import aplicar, validar
    df = pd.DataFrame([{"estudiante_id": "EST0001", "curso": "C1", "periodo": "2024-I", "nota": 9},
                       {"estudiante_id": "EST0001", "curso": "C1", "periodo": "2024-II", "nota": 14}])
    assert validar(df, con) == []
    assert aplicar(df, con) == (1, 1)
    filas = con.execute("SELECT periodo, intento, nota, condicion FROM matricula ORDER BY periodo").fetchall()
    assert filas == [("2024-I", 1, 9, "desaprobado"), ("2024-II", 2, 14, "aprobado")]
