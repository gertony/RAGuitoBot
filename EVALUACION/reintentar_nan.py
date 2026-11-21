"""Quita de un ragas_por_par_<etiqueta>.csv las filas con alguna métrica NaN, para que evaluar_ragas.py las reintente.

Uso:  python EVALUACION/reintentar_nan.py final
"""
import sys
from pathlib import Path

import pandas as pd

ruta = Path(__file__).resolve().parent / f"ragas_por_par_{sys.argv[1]}.csv"
df = pd.read_csv(ruta)
nan = df[df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].isna().any(axis=1)]
print("Se reintentarán:", ", ".join(nan["id"]) or "ninguno")
df.drop(nan.index).to_csv(ruta, index=False, encoding="utf-8")
