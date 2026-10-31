"""Inicio de sesión y roles (RF-19, RF-20, RNF-11).

La tabla usuario guarda solo el hash bcrypt de la contraseña. El estudiante queda vinculado a su
estudiante_id: ese ID es el que se pasa al agente, nunca uno escrito en la pregunta (RF-17).
"""
import re
import sqlite3

import bcrypt

ESQUEMA = """
CREATE TABLE IF NOT EXISTS usuario (
    codigo        TEXT PRIMARY KEY,
    hash          TEXT NOT NULL,
    rol           TEXT NOT NULL CHECK (rol IN ('estudiante', 'evaluador')),
    estudiante_id TEXT REFERENCES estudiante(id),
    activo        INTEGER NOT NULL DEFAULT 1,
    CHECK ((rol = 'estudiante') = (estudiante_id IS NOT NULL))
);
"""
FORMATO_ESTUDIANTE = re.compile(r"^EST\d{4}$")
MIN_CLAVE = 6


def conectar(ruta_db):
    con = sqlite3.connect(ruta_db, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(ESQUEMA)
    return con


def validar_formulario(codigo, clave):
    """Validaciones de P-01 (II.3.3.2). Devuelve un mensaje de error o None."""
    if not codigo or not clave:
        return "Ingresa tu código y tu contraseña."
    if codigo.upper().startswith("EST") and not FORMATO_ESTUDIANTE.match(codigo.upper()):
        return "El código de estudiante tiene el formato EST seguido de 4 dígitos (p. ej., EST0001)."
    if len(clave) < MIN_CLAVE:
        return f"La contraseña tiene al menos {MIN_CLAVE} caracteres."
    return None


def autenticar(con, codigo, clave):
    """Devuelve {'codigo', 'rol', 'estudiante_id'} o None. El error mostrado no dice qué campo falló."""
    fila = con.execute("SELECT codigo, hash, rol, estudiante_id FROM usuario WHERE codigo = ? AND activo = 1",
                       (codigo.upper(),)).fetchone()
    if fila is None or not bcrypt.checkpw(clave.encode(), fila["hash"].encode()):
        return None
    return {"codigo": fila["codigo"], "rol": fila["rol"], "estudiante_id": fila["estudiante_id"]}


def crear_usuario(con, codigo, clave, rol, estudiante_id=None):
    h = bcrypt.hashpw(clave.encode(), bcrypt.gensalt()).decode()
    con.execute("INSERT OR REPLACE INTO usuario (codigo, hash, rol, estudiante_id) VALUES (?, ?, ?, ?)",
                (codigo.upper(), h, rol, estudiante_id))
    con.commit()
