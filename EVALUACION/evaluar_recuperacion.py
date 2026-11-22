"""Acierto de la recuperación normativa (sin LLM: gratis) en el testbed, la retención y la confirmación.

Uso (desde la raíz del proyecto):
    python EVALUACION/evaluar_recuperacion.py                       # índice actual (CHROMADB2/chroma_e5)
    python EVALUACION/evaluar_recuperacion.py --indices CHROMADB2/chroma_e5 CHROMADB2/chroma_e5_v2

Por cada consulta normativa o mixta se mira si entre los k fragmentos recuperados está alguna de sus fuentes
esperadas (acierto@k). Se prueban: búsqueda densa o híbrida, con o sin normalización, k = 4 o 6.
Guarda el detalle en PRUEBAS/evidencias/recuperacion_<fecha>.csv
"""
import argparse
import csv
import json
import re
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "AGENTE"))
import agente  # noqa: E402

CONJUNTOS = {"testbed": "testbed.json", "retencion": "testbed_retencion.json", "confirmacion": "testbed_confirmacion.json"}
# El conjunto ciego (testbed_ciego.json, 21/11) NO se usa aquí a propósito: estas mediciones sirven para elegir la
# configuración, y el ciego solo se evalúa una vez, en la corrida final, con la configuración ya fijada.
RG, LEY, PLAN = "Reglamento General UNE", "Ley Universitaria N.° 30220", "Plan de estudios"


def esperadas(par):
    """Predicados sobre (texto, metadatos) que cuentan como acierto para esa consulta."""
    preds = []
    for f in par["fuentes_esperadas"]:
        m = re.search(r"art\.\s*(\d+)", f)
        if m and ("RG" in f or "Reglamento" in f):
            preds.append(lambda t, md, a=m.group(1): md.get("documento") == RG and str(md.get("articulo")) == a)
        elif m and ("Ley" in f):
            preds.append(lambda t, md, a=m.group(1): md.get("documento") == LEY and str(md.get("articulo")) == a)
        elif "sumilla" in f:
            cod = f.split()[-1]
            preds.append(lambda t, md, c=cod: str(md.get("documento", "")).startswith(PLAN) and md.get("articulo") == c)
        elif "reglas de matrícula" in f:
            preds.append(lambda t, md: str(md.get("documento", "")).startswith(PLAN) and "Reglas de matrícula" in str(md.get("titulo")))
    if par["id"] == "M03":  # GT corregido el 18/11: la excepción del bachillerato automático también es correcta
        preds.append(lambda t, md: "bachillerato automático" in t.lower())
    return preds


def consultas():
    for conjunto, archivo in CONJUNTOS.items():
        ruta = RAIZ / "TESTBED" / archivo
        if not ruta.exists():
            continue
        for par in json.loads(ruta.read_text(encoding="utf-8")):
            if par["tipo"] in ("normativa", "mixta") and esperadas(par):
                yield conjunto, par


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", nargs="+", default=["CHROMADB2/chroma_e5"])
    a = ap.parse_args()
    pares = list(consultas())
    filas = []
    emb = None
    for indice in a.indices:
        base = agente.BuscadorNormativa(ruta_chroma=RAIZ / indice, embedding=emb)
        emb = base.embedding  # el modelo se carga una sola vez
        for modo, normalizar, k, div in product(["denso", "hibrido"], [False, True], [4, 6], [False, True]):
            b = agente.BuscadorNormativa(ruta_chroma=RAIZ / indice, embedding=emb, modo=modo, normalizar=normalizar, k=k,
                                         diversificar=div)
            b._bm25, b._docs, b._metas = getattr(base, "_bm25", None), getattr(base, "_docs", None), getattr(base, "_metas", None)
            for conjunto, par in pares:
                res = b.buscar_con_meta(par["pregunta"])
                ok = any(p(t, md) for t, md in res for p in esperadas(par))
                filas.append({"indice": Path(indice).name, "modo": modo, "normalizar": int(normalizar), "k": k,
                              "diversificar": int(div), "articulos_distintos": len({(m.get("documento"),
                                                                                    str(m.get("articulo"))) for _, m in res}),
                              "conjunto": conjunto, "id": par["id"], "tipo": par["tipo"], "acierto": int(ok),
                              "recuperados": " || ".join(t.split("]")[0].strip("[") for t, _ in res)})
            if modo == "hibrido" and base._bm25 is None:
                base._bm25, base._docs, base._metas = b._bm25, b._docs, b._metas

    print(f"\n{'índice':14} {'modo':8} {'norm':>4} {'k':>2} {'div':>3} | {'testbed':>8} {'retención':>10} {'confirm.':>9} | "
          f"{'TOTAL':>7} | art. distintos")
    for clave in dict.fromkeys((f["indice"], f["modo"], f["normalizar"], f["k"], f["diversificar"]) for f in filas):
        g = [f for f in filas if (f["indice"], f["modo"], f["normalizar"], f["k"], f["diversificar"]) == clave]
        celda = lambda c: (lambda h: f"{sum(x['acierto'] for x in h)}/{len(h)}" if h else "-")([x for x in g if x["conjunto"] == c])
        media_art = sum(x["articulos_distintos"] for x in g) / len(g)
        print(f"{clave[0]:14} {clave[1]:8} {clave[2]:>4} {clave[3]:>2} {clave[4]:>3} | {celda('testbed'):>8} "
              f"{celda('retencion'):>10} {celda('confirmacion'):>9} | {sum(x['acierto'] for x in g):>3}/{len(g)} | "
              f"{media_art:.2f}")
    salida = RAIZ / "PRUEBAS" / "evidencias" / f"recuperacion_{datetime.now():%Y%m%d_%H%M}.csv"
    salida.parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print(f"\nDetalle: {salida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
