"""Prepara la evaluación RAGAS pendiente (tesis): reintento de NaN y, si se pide, ground truth corregido de M03.

Uso (desde la carpeta EVALUACION):
    python preparar_ragas.py                  # solo quita de ragas_por_par.csv los pares con alguna métrica NaN
    python preparar_ragas.py --corregir-m03   # además corrige el ground truth de M03

Antes de tocar nada, guarda copias *.bak_AAAAMMDD_HHMM de cada archivo que modifica.
"""
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

CARPETA = Path(__file__).parent
RAIZ = CARPETA.parent
METRICAS = ["faithfulness", "answer_relevancy", "llm_context_precision_with_reference", "context_recall"]
GT_M03 = ("Sí. Aprobó los 214 de 214 créditos del plan y concluyó sus estudios en 2023-I, dentro de los años "
          "académicos 2020 a 2023. Por la excepción del bachillerato automático (Quinta Disposición Complementaria "
          "del Reglamento General de la UNE; Décima Cuarta Disposición Complementaria Transitoria de la Ley 30220, "
          "incorporada por la Ley 31183 y ampliada por la Ley 31359), puede solicitar el grado de Bachiller en forma "
          "automática, exonerado de los demás requisitos del numeral 45.1 del artículo 45 (idioma extranjero y "
          "trabajo de investigación).")
SELLO = datetime.now().strftime("%Y%m%d_%H%M")


def respaldar(ruta):
    copia = ruta.with_name(f"{ruta.name}.bak_{SELLO}")
    shutil.copy2(ruta, copia)
    print(f"Copia: {copia.name}")


def quitar_nan():
    ruta = CARPETA / "ragas_por_par.csv"
    df = pd.read_csv(ruta)
    cols = [c for c in df.columns if c in METRICAS]
    con_nan = df[df[cols].isna().any(axis=1)]
    if con_nan.empty:
        print("ragas_por_par.csv no tiene pares con NaN.")
        return
    respaldar(ruta)
    df[~df.index.isin(con_nan.index)].to_csv(ruta, index=False, encoding="utf-8")
    print(f"Quitados para reevaluar: {', '.join(con_nan['id'])} (se recalculan sus 4 métricas)")


def corregir_m03():
    jsonl = RAIZ / "AGENTE" / "resultados_testbed.jsonl"  # evaluar_ragas.py toma el ground truth de aquí
    respaldar(jsonl)
    lineas = []
    for linea in jsonl.read_text(encoding="utf-8").splitlines():
        if linea.strip():
            fila = json.loads(linea)
            if fila["id"] == "M03":
                fila["ground_truth"] = GT_M03
            linea = json.dumps(fila, ensure_ascii=False)
        lineas.append(linea)
    jsonl.write_text("\n".join(lineas) + "\n", encoding="utf-8")

    testbed = RAIZ / "TESTBED" / "testbed.json"  # se corrige también aquí para que ambos coincidan
    respaldar(testbed)
    pares = json.loads(testbed.read_text(encoding="utf-8"))
    for par in pares:
        if par["id"] == "M03":
            par["ground_truth"] = GT_M03
    testbed.write_text(json.dumps(pares, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Ground truth de M03 corregido en resultados_testbed.jsonl y testbed.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corregir-m03", action="store_true")
    args = ap.parse_args()
    quitar_nan()
    if args.corregir_m03:
        corregir_m03()
