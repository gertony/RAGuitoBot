"""Exactitud del enrutador en el testbed (35) y en el conjunto de retención (15), por modo.

Uso (desde la raíz del proyecto, con el .env configurado):
    python EVALUACION/evaluar_router.py               # compara ROUTER_MODO=reglas y llm
    python EVALUACION/evaluar_router.py --modos reglas  # solo reglas (no llama al LLM: gratis)
    python EVALUACION/evaluar_router.py --modos union --confirmacion  # USO ÚNICO: suma el conjunto de confirmación

Métricas:
  exacta     la ruta coincide con la esperada
  cobertura  la ruta trae todas las fuentes que la respuesta necesita (mixta cubre a personal y a normativa)
Guarda el detalle en PRUEBAS/evidencias/router_<fecha>.csv
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "AGENTE"))
try:
    from dotenv import load_dotenv
    load_dotenv(RAIZ / ".env")
except ImportError:
    pass

import agente  # noqa: E402

CONJUNTOS = {"testbed": RAIZ / "TESTBED" / "testbed.json", "retencion": RAIZ / "TESTBED" / "testbed_retencion.json"}
# Escrito antes de programar el modo union (sha256 b4898d7a…), para confirmar una sola vez el modo elegido
CONFIRMACION = RAIZ / "TESTBED" / "testbed_confirmacion.json"
# En la app siempre hay un estudiante en sesión (CU-11). Las preguntas normativas del testbed no tienen estudiante y
# _enrutar fuerza entonces la ruta normativa, lo que no refleja el uso real: se evalúan con una sesión activa.
SESION_POR_DEFECTO = "EST0030"
CUBRE = {"personal": {"personal"}, "normativa": {"normativa"}, "mixta": {"personal", "normativa", "mixta"}}


class _SinBuscador:  # el enrutador no consulta ChromaDB
    def buscar(self, *a, **k):
        return []


def evaluar(modo, llm=None, conjuntos=None):
    os.environ["ROUTER_MODO"] = modo
    ag = agente.Agente(buscador=_SinBuscador(), registrar=False, llm=llm)
    filas = []
    for conjunto, ruta in (conjuntos or CONJUNTOS).items():
        for par in json.loads(ruta.read_text(encoding="utf-8")):
            eid = par.get("estudiante_id") or SESION_POR_DEFECTO
            x = ag._enrutar({"pregunta": par["pregunta"], "estudiante_id": eid})
            esperada, obtenida = par["ruta_esperada"], x["ruta"]
            filas.append({"modo": modo, "conjunto": conjunto, "id": par["id"], "tipo": par["tipo"],
                          "esperada": esperada, "obtenida": obtenida, "router": x["router"],
                          "exacta": int(obtenida == esperada),
                          "cobertura": int(esperada in CUBRE.get(obtenida, set())),
                          "intenciones": "|".join(x["intenciones"])})
    return filas


def resumen(filas):
    print(f"\n{'modo':8} {'conjunto':10} {'tipo':10} {'n':>3} {'exacta':>8} {'cobertura':>10}")
    orden = {"testbed": 0, "retencion": 1, "confirmacion": 2}
    claves = sorted({(f["modo"], f["conjunto"]) for f in filas}, key=lambda k: (k[0], orden.get(k[1], 3)))
    for modo, conjunto in claves:
        for tipo in ("personal", "normativa", "mixta", "TOTAL"):
            g = [f for f in filas if f["modo"] == modo and f["conjunto"] == conjunto and tipo in (f["tipo"], "TOTAL")]
            if g:
                print(f"{modo:8} {conjunto:10} {tipo:10} {len(g):>3} {sum(f['exacta'] for f in g):>4}/{len(g):<3} "
                      f"{sum(f['cobertura'] for f in g):>6}/{len(g):<3}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modos", nargs="+", default=["reglas", "llm", "union"], choices=["reglas", "llm", "union"])
    ap.add_argument("--confirmacion", action="store_true", help="agrega el conjunto de confirmación (uso único)")
    a = ap.parse_args()
    conjuntos = dict(CONJUNTOS, **({"confirmacion": CONFIRMACION} if a.confirmacion else {}))
    todas = []
    for m in a.modos:
        todas += evaluar(m, conjuntos=conjuntos)
    resumen(todas)
    salida = RAIZ / "PRUEBAS" / "evidencias" / f"router_{datetime.now():%Y%m%d_%H%M}.csv"
    salida.parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(todas[0]))
        w.writeheader()
        w.writerows(todas)
    print(f"\nDetalle: {salida.relative_to(RAIZ)}")
