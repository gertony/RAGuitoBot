"""Prueba de carga del proceso crítico de RAGuitoBot (P9 · III.1): responder una consulta.

Simula N estudiantes que consultan a la vez sobre UNA sola instancia del agente, igual que la app de
Streamlit (servicios.agente() comparte la instancia entre sesiones, un hilo por sesión). Cada usuario
virtual envía C consultas del testbed con una pausa aleatoria entre ellas y se mide la latencia de punta
a punta de cada una (recuperación + LLM). Se contrasta con el RNF-01: percentil 95 ≤ 10 000 ms.

Por qué no k6 ni JMeter: Streamlit no expone una API HTTP para la consulta (usa WebSocket), así que esas
herramientas medirían el transporte de Streamlit y no el proceso crítico. La concurrencia se reproduce
en Python con hilos, que es como la app atiende las sesiones.

Uso (desde la raíz del repositorio, con .env configurado; consume cuota del LLM):
    python PRUEBAS/carga.py                      # 8 usuarios × 5 consultas (el grupo del SUS)
    python PRUEBAS/carga.py --usuarios 4 --consultas 3
Salida: PRUEBAS/resultados_carga.csv (una fila por consulta) y el resumen en pantalla.
"""
import argparse
import csv
import json
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
for carpeta in ("AGENTE", "TESTBED"):
    sys.path.insert(0, str(RAIZ / carpeta))

UMBRAL_MS = 10_000  # RNF-01
SALIDA = RAIZ / "PRUEBAS" / "resultados_carga.csv"


def percentil(valores, p):
    """Percentil con interpolación lineal (mismo criterio que numpy.percentile por defecto)."""
    v = sorted(valores)
    k = (len(v) - 1) * p / 100
    i = int(k)
    return v[i] if i + 1 >= len(v) else v[i] + (v[i + 1] - v[i]) * (k - i)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--usuarios", type=int, default=8, help="usuarios concurrentes (por defecto, 8: grupo del SUS)")
    ap.add_argument("--consultas", type=int, default=5, help="consultas por usuario")
    ap.add_argument("--pausa", type=float, nargs=2, default=(2.0, 5.0), help="pausa entre consultas, en segundos")
    ap.add_argument("--semilla", type=int, default=42)
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(RAIZ / ".env")
    from agente import Agente

    testbed = json.loads((RAIZ / "TESTBED" / "testbed.json").read_text(encoding="utf-8"))
    rng = random.Random(args.semilla)  # noqa: S311 (simula pausas de usuarios, no es criptografía)
    plan = [rng.sample(testbed, args.consultas) for _ in range(args.usuarios)]

    print("Cargando el agente (una sola instancia, como en la app)...")
    agente = Agente(registrar=False)  # no se mezclan las consultas de carga con el registro real (RF-12)
    agente.responder(testbed[0]["estudiante_id"], testbed[0]["pregunta"])  # calentamiento, no se mide

    filas, candado = [], threading.Lock()

    def usuario(n):
        for par in plan[n]:
            inicio = datetime.now().isoformat(timespec="seconds")
            t0 = time.perf_counter()
            try:
                r = agente.responder(par["estudiante_id"], par["pregunta"])
                error = ""
                propio = all(par["estudiante_id"] in c for c in r["contextos"] if "estudiante EST" in c)
            except Exception as e:  # se registra y se sigue: un fallo también es un resultado (p. ej., 429)
                error, propio = type(e).__name__ + ": " + str(e)[:120], ""
            ms = (time.perf_counter() - t0) * 1000
            with candado:
                filas.append({"usuario": n + 1, "par": par["id"], "tipo": par["tipo"], "inicio": inicio,
                              "latencia_ms": round(ms, 1), "error": error, "contexto_propio": propio})
            time.sleep(rng.uniform(*args.pausa))

    print(f"Carga: {args.usuarios} usuarios × {args.consultas} consultas...")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.usuarios) as ex:
        list(ex.map(usuario, range(args.usuarios)))
    duracion = time.perf_counter() - t0

    with SALIDA.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(sorted(filas, key=lambda x: x["inicio"]))

    ok = [f["latencia_ms"] for f in filas if not f["error"]]
    errores = [f for f in filas if f["error"]]
    mezclas = [f for f in filas if f["contexto_propio"] is False]
    print("\n================ RESUMEN DE LA PRUEBA DE CARGA ================")
    print(f"Usuarios concurrentes        : {args.usuarios}")
    print(f"Consultas enviadas           : {len(filas)} en {duracion:.0f} s")
    print(f"Consultas con error          : {len(errores)}")
    for e in sorted({x['error'].split(':')[0] for x in errores}):
        print(f"   - {e}: {sum(x['error'].startswith(e) for x in errores)}")
    print(f"Contexto de otro estudiante  : {len(mezclas)}")
    if ok:
        p95 = percentil(ok, 95)
        print(f"Latencia media               : {statistics.mean(ok):.0f} ms")
        print(f"Latencia mediana             : {statistics.median(ok):.0f} ms")
        print(f"Latencia percentil 95        : {p95:.0f} ms")
        print(f"Latencia máxima              : {max(ok):.0f} ms")
        print(f"RNF-01 (p95 ≤ {UMBRAL_MS} ms)    : {'CUMPLE' if p95 <= UMBRAL_MS else 'NO CUMPLE'}")
    print(f"Detalle: {SALIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
