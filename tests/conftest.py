"""Rutas del proyecto para pytest: las pruebas importan el código tal como está en el repositorio."""
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
for carpeta in ("APP", "AGENTE", "TESTBED", "DB", "CHROMADB2"):
    sys.path.insert(0, str(RAIZ / carpeta))
DB_PATH = Path(os.environ.get("DB_PATH", RAIZ / "DB" / "academico.db"))
