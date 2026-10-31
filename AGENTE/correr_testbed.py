"""Corre el testbed completo a través del agente y guarda cada resultado apenas se obtiene.

- Guarda en resultados_testbed.jsonl (una línea por par): pregunta, ground truth, respuesta, contextos,
  ruta, latencia. Si se corta (cuota de Gemini, internet), al volver a correrlo continúa donde quedó.
- Espera --pausa segundos entre preguntas para no superar el límite de solicitudes por minuto del plan gratuito.
- Al final imprime la latencia media, la mediana y el percentil 95 (RNF-01: p95 ≤ 10 s) y la ruta del router.

Uso (desde RAGASTESIS/AGENTE):  python evaluar_ragas.py
Lee el testbed de ../TESTBED/testbed.json y guarda los resultados en AGENTE/resultados_testbed.jsonl.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from agente import Agente

CARPETA = Path(__file__).resolve().parent            # RAGASTESIS/AGENTE: aquí se guardan los resultados
TESTBED = CARPETA.parent / "TESTBED" / "testbed.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testbed", default=str(TESTBED))
    ap.add_argument("--salida", default=str(CARPETA / "resultados_testbed.jsonl"))
    ap.add_argument("--pausa", type=float, default=6.0, help="segundos entre preguntas (cuota gratuita)")
    ap.add_argument("--solo", nargs="*", help="ids a correr, p. ej. P01 N02 M01")
    args = ap.parse_args()

    testbed = json.load(open(args.testbed, encoding="utf-8"))
    if args.solo:
        testbed = [t for t in testbed if t["id"] in args.solo]
    salida = Path(args.salida)
    hechos = set()
    if salida.exists():
        hechos = {json.loads(l)["id"] for l in salida.read_text(encoding="utf-8").splitlines() if l.strip()}
    pendientes = [t for t in testbed if t["id"] not in hechos]
    print(f"Testbed: {len(testbed)} pares | ya hechos: {len(hechos)} | por correr: {len(pendientes)}")

    agente = Agente()
    for n, t in enumerate(pendientes, start=1):
        try:
            r = agente.responder(t["estudiante_id"], t["pregunta"])
        except Exception as e:  # cuota agotada, red, etc.: se guarda lo avanzado y se puede reanudar
            print(f"\n✗ Error en {t['id']}: {e}\nLo avanzado quedó guardado. Vuelve a correr el script para continuar.")
            break
        fila = {"id": t["id"], "tipo": t["tipo"], "categoria": t["categoria"], "estudiante_id": t["estudiante_id"],
                "pregunta": t["pregunta"], "ground_truth": t["ground_truth"], "respuesta": r["respuesta"],
                "contextos": r["contextos"], "ruta_esperada": t["ruta_esperada"], "ruta": r["ruta"],
                "router": r["router"], "intenciones": r["intenciones"], "latencia_ms": r["latencia_ms"]}
        with salida.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(fila, ensure_ascii=False) + "\n")
        marca = "✓" if r["ruta"] == t["ruta_esperada"] else "≠"
        print(f"[{n}/{len(pendientes)}] {t['id']} {marca} ruta={r['ruta']:<9} {r['latencia_ms']:>7.0f} ms | {r['respuesta'][:90]}")
        if n < len(pendientes):
            time.sleep(args.pausa)

    filas = [json.loads(l) for l in salida.read_text(encoding="utf-8").splitlines() if l.strip()] if salida.exists() else []
    if filas:
        lat = sorted(f["latencia_ms"] for f in filas)
        p95 = lat[min(len(lat) - 1, int(round(0.95 * len(lat))) - 1)]
        print(f"\n== Resumen ({len(filas)} pares)")
        print(f"Latencia: media {statistics.mean(lat):.0f} ms | mediana {statistics.median(lat):.0f} ms | p95 {p95:.0f} ms "
              f"| RNF-01 (p95 ≤ 10 000 ms): {'cumple' if p95 <= 10000 else 'no cumple'}")
        print(f"Ruta del router correcta: {sum(f['ruta'] == f['ruta_esperada'] for f in filas)}/{len(filas)}")


if __name__ == "__main__":
    main()