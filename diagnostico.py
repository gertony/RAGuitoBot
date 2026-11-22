"""Diagnóstico: qué ruta, intenciones y contexto recibe el LLM para una pregunta (y la respuesta).

Uso (desde la raíz):  python diagnostico.py EST0001 "jalé un curso el semestre pasado?"
"""
import sys
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent
load_dotenv(RAIZ / ".env")
sys.path[:0] = [str(RAIZ / "AGENTE"), str(RAIZ / "TESTBED")]
import agente  # noqa: E402

print("agente.py cargado desde:", agente.__file__)
print("¿agente.py es la versión v4pre?", "jalaste" in Path(agente.__file__).read_text(encoding="utf-8"))
import consultas  # noqa: E402
print("¿consultas.py es la versión v4pre?", hasattr(consultas.Academico, "cursos_pendientes"))
ag = agente.Agente(registrar=False)
r = ag.responder(sys.argv[1], sys.argv[2])
print("\nruta:", r["ruta"], "| router:", r.get("router"), "| intenciones:", r.get("intenciones"))
print("contextos:")
for c in r["contextos"]:
    print("  -", c[:200])
print("\nrespuesta:", r["respuesta"])
