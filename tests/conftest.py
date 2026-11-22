"""Rutas del proyecto para pytest: las pruebas importan el código tal como está en el repositorio."""
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
for carpeta in ("APP", "AGENTE", "TESTBED", "DB", "CHROMADB2"):
    sys.path.insert(0, str(RAIZ / carpeta))
DB_PATH = Path(os.environ.get("DB_PATH", RAIZ / "DB" / "academico.db"))

import pytest  # noqa: E402

# Variables de configuración del .env que cambian el comportamiento del agente. Algunos módulos (APP/servicios.py,
# DB/respaldo.py) cargan el .env al importarse; sin esto, las pruebas dependerían de la configuración local del
# desarrollador (p. ej. ROUTER_MODO=union desde el 20/11). Cada prueba parte de los valores por defecto (línea base)
# y la que necesita otro valor lo fija con monkeypatch.setenv.
CONFIG = ("ROUTER_MODO", "NORMALIZAR", "RECUPERACION", "K_NORMATIVA", "CHROMA_RUTA", "LLM_RPM")


@pytest.fixture(autouse=True)
def _configuracion_por_defecto(monkeypatch):
    for var in CONFIG:
        monkeypatch.delenv(var, raising=False)
