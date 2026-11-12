"""Particiones de equivalencia y valores límite de CU-10 y CU-11 (P9 · III.1.3.1).

CU-11 se prueba sobre la app real (app.py) con streamlit.testing (AppTest): cada AppTest es una sesión
nueva del navegador, así que también permite probar el bloqueo entre sesiones (defecto D-07).
CU-10 se prueba con la misma validación que usa el panel del evaluador (actualizacion.validar).
"""
import io
import sqlite3

import pandas as pd
import pytest

import auth


def casos(*filas):
    """Cada fila empieza con el identificador del caso, que se usa como nombre en la salida de pytest."""
    return [pytest.param(*f, id=f[0]) for f in filas]


# ================================================================ CU-11 Iniciar sesión
FORMATO = "El código de estudiante tiene el formato EST seguido de 4 dígitos (p. ej., EST0001)."
VACIO = "Ingresa tu código y tu contraseña."
CORTA = "La contraseña tiene al menos 6 caracteres."
GENERICO = "Código o contraseña incorrectos."


@pytest.fixture
def app(tmp_path, monkeypatch):
    """La app real, con una base temporal que tiene una cuenta de estudiante y una de evaluador."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    import servicios
    from build_db import ESQUEMA
    ruta = tmp_path / "academico.db"
    with sqlite3.connect(ruta) as c:
        c.executescript(ESQUEMA)
    con = auth.conectar(ruta)
    auth.crear_usuario(con, "EST0001", "clave123", "estudiante", "EST0001")
    auth.crear_usuario(con, "EVAL01", "evalua99", "evaluador")
    con.close()
    monkeypatch.setattr(servicios, "DB", ruta)
    st.cache_resource.clear()
    auth._FALLOS.clear()
    raiz = servicios.RAIZ

    def nueva_sesion():
        return AppTest.from_file(str(raiz / "APP" / "app.py"), default_timeout=60).run()
    return nueva_sesion


def ingresar(at, codigo, clave):
    at.text_input[0].set_value(codigo)
    at.text_input[1].set_value(clave)
    at.button[0].click().run()
    errores = [e.value for e in at.error]
    usuario = at.session_state["usuario"] if "usuario" in at.session_state else None
    return errores, usuario


@pytest.mark.parametrize(("caso", "codigo", "clave", "rol"), casos(
    ("CP-11-01", "EST0001", "clave123", "estudiante"),       # flujo normal
    ("CP-11-02", "EVAL01", "evalua99", "evaluador"),          # flujo normal, otro rol (RF-20)
    ("CP-11-03", "est0001", "clave123", "estudiante"),        # código en minúsculas
    ("CP-11-04", "  EST0001 ", "clave123", "estudiante"),     # espacios al inicio y al final
))
def test_cu11_credenciales_validas(app, caso, codigo, clave, rol):
    errores, usuario = ingresar(app(), codigo, clave)
    assert errores == [] and usuario["rol"] == rol


@pytest.mark.parametrize(("caso", "codigo", "clave", "mensaje"), casos(
    ("CP-11-05", "", "clave123", VACIO),                     # código vacío
    ("CP-11-06", "EST0001", "", VACIO),                      # contraseña vacía
    ("CP-11-07", "", "", VACIO),                             # ambos vacíos
    ("CP-11-08", "EST", "clave123", FORMATO),                # solo el prefijo
    ("CP-11-09", "EST001", "clave123", FORMATO),             # 3 dígitos (límite inferior − 1)
    ("CP-11-10", "EST00011", "clave123", FORMATO),           # 5 dígitos (límite superior + 1)
    ("CP-11-11", "ESTABCD", "clave123", FORMATO),            # letras en lugar de dígitos
    ("CP-11-12", "0001", "clave123", GENERICO),              # solo los dígitos: no es un código válido
    ("CP-11-13", "EST9999", "clave123", GENERICO),           # formato válido, código inexistente
    ("CP-11-14", "EST0001", "12345", CORTA),                 # contraseña de 5 caracteres (límite − 1)
    ("CP-11-15", "EST0001", "clave1", GENERICO),             # 6 caracteres (límite), pero incorrecta
    ("CP-11-16", "EST0001", "CLAVE123", GENERICO),           # mayúsculas: la contraseña distingue
    ("CP-11-17", "' OR '1'='1", "x' OR '1'='1", GENERICO),   # inyección SQL
))
def test_cu11_entradas_invalidas(app, caso, codigo, clave, mensaje):
    errores, usuario = ingresar(app(), codigo, clave)
    assert errores == [mensaje] and usuario is None


def test_cu11_18_bloqueo_tras_5_intentos(app):
    at = app()
    for _ in range(auth.MAX_INTENTOS):
        assert ingresar(at, "EST0001", "malamala")[0] == [GENERICO]
    errores, usuario = ingresar(at, "EST0001", "clave123")  # la correcta también espera
    assert errores[0].startswith("Demasiados intentos") and usuario is None


def test_cu11_19_el_bloqueo_sigue_en_una_sesion_nueva(app):
    # D-07: antes, recargar la página (sesión nueva) reiniciaba el contador y saltaba el bloqueo.
    at = app()
    for _ in range(auth.MAX_INTENTOS):
        ingresar(at, "EST0001", "malamala")
    errores, usuario = ingresar(app(), "EST0001", "clave123")
    assert errores[0].startswith("Demasiados intentos") and usuario is None


def test_cu11_20_el_bloqueo_vence_a_los_30_s():
    auth._FALLOS.clear()
    for _ in range(auth.MAX_INTENTOS):
        auth.registrar_fallo("EST0002", ahora=1000)
    assert auth.segundos_bloqueo("EST0002", ahora=1000) == auth.ESPERA_S
    assert auth.segundos_bloqueo("est0002", ahora=1000 + auth.ESPERA_S) == 0


# ================================================================ CU-10 Actualizar la base académica
ENCABEZADO = "estudiante_id,curso,periodo,nota\n"


@pytest.fixture
def con():
    from build_db import ESQUEMA
    c = sqlite3.connect(":memory:")
    c.executescript(ESQUEMA)
    c.execute("INSERT INTO curso VALUES ('C1','Curso 1',1,4,2,2,0,NULL)")
    c.execute("INSERT INTO estudiante VALUES ('EST0001','u','M',2023,'P','S',2,'2024-I',1,4,"
              "'matriculado regular','real')")
    yield c
    c.close()


def leer(filas, encabezado=ENCABEZADO):
    # igual que el panel del evaluador (paginas/panel.py)
    return pd.read_csv(io.StringIO(encabezado + filas), dtype={"estudiante_id": str, "curso": str, "periodo": str})


@pytest.mark.parametrize(("caso", "filas", "valido"), casos(
    ("CP-10-01", "EST0001,C1,2024-I,0", True),                # nota en el límite inferior
    ("CP-10-02", "EST0001,C1,2024-I,20", True),               # nota en el límite superior
    ("CP-10-03", "EST0001,C1,2024-I,-1", False),              # límite inferior − 1
    ("CP-10-04", "EST0001,C1,2024-I,21", False),              # límite superior + 1
    ("CP-10-05", "EST0001,C1,2024-I,10.5", False),            # decimal
    ("CP-10-06", "EST0001,C1,2024-I,", False),                # nota vacía
    ("CP-10-07", "EST0001,C1,2024-I,abc", False),             # texto (D-08: antes detenía la carga)
    ("CP-10-08", "EST0001,C1,2024-III,12", False),            # periodo inexistente
    ("CP-10-09", "EST0001,C1,2024-i,12", False),              # periodo en minúsculas
    ("CP-10-10", "EST0001,C1,24-I,12", False),                # año con 2 dígitos
    ("CP-10-11", "EST9999,C1,2024-I,12", False),              # estudiante inexistente
    ("CP-10-12", "EST0001,ZZ,2024-I,12", False),              # curso inexistente
    ("CP-10-13", "EST0001,C1,2024-I,12\nEST0001,C1,2024-I,15", False),  # fila repetida (D-10)
    ("CP-10-14", "", False),                                  # solo el encabezado (D-09)
))
def test_cu10_validacion(con, caso, filas, valido):
    from actualizacion import validar
    assert (validar(leer(filas), con) == []) is valido


def test_cu10_15_falta_una_columna(con):
    from actualizacion import validar
    assert validar(leer("EST0001,C1,2024-I", "estudiante_id,curso,periodo\n"), con) == ["Faltan columnas: nota"]
