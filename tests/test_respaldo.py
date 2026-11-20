"""P10: el plan de reversión depende de que el respaldo se pueda restaurar sin pérdida de datos."""
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "DB"))
import respaldo  # noqa: E402


@pytest.fixture
def entorno(tmp_path):
    db = tmp_path / "academico.db"
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE matricula (id INTEGER PRIMARY KEY, nota INTEGER);
        CREATE TABLE log_consultas (id INTEGER PRIMARY KEY, pregunta TEXT);
        INSERT INTO matricula (nota) VALUES (14), (9), (18);
        INSERT INTO log_consultas (pregunta) VALUES ('¿cuál es mi promedio?');
    """)
    con.commit()
    con.close()
    chroma = tmp_path / "chroma_e5"
    chroma.mkdir()
    (chroma / "indice.bin").write_bytes(b"indice")
    return db, chroma, tmp_path / "RESPALDOS"


def test_crear_respaldo_copia_base_indice_y_log(entorno):
    db, chroma, base = entorno
    destino = respaldo.crear("prueba", db, chroma, base)
    assert (destino / "chroma_e5" / "indice.bin").read_bytes() == b"indice"
    assert "¿cuál es mi promedio?" in (destino / "log_consultas.csv").read_text(encoding="utf-8")
    m = respaldo.conteos(destino / "academico.db", destino / "chroma_e5")
    assert m["matricula"] == 3 and m["integridad"] == "ok"


def test_restaurar_vuelve_al_estado_respaldado_y_guarda_el_actual(entorno):
    db, chroma, base = entorno
    destino = respaldo.crear("antes", db, chroma, base)
    con = sqlite3.connect(db)  # un cambio posterior que se quiere revertir (p. ej., un CSV erróneo del CU-10)
    con.execute("UPDATE matricula SET nota = 0")
    con.commit()
    con.close()
    (chroma / "indice.bin").write_bytes(b"roto")

    respaldo.restaurar(destino, db, chroma, base)

    con = sqlite3.connect(db)
    assert [r[0] for r in con.execute("SELECT nota FROM matricula ORDER BY id")] == [14, 9, 18]
    con.close()
    assert (chroma / "indice.bin").read_bytes() == b"indice"
    previos = [d for d in base.iterdir() if d.name.endswith("pre_restauracion")]
    assert len(previos) == 1  # el estado con el error también quedó guardado
