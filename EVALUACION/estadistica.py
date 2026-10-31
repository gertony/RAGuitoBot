"""Análisis estadístico de los resultados RAGAS (4.4.3 y 4.4.4).

Para cada métrica (Faithfulness, Answer Relevancy, Context Precision, Context Recall):
- Descriptivo: n, media, desviación estándar, mediana, mínimo y máximo.
- IC 95 % de la media por bootstrap percentil (10 000 remuestreos, semilla 42).
- Prueba de Wilcoxon de una muestra, unilateral: H0: mediana < 0.85 ; H1: mediana >= 0.85 (alfa = 0.05).
  Se usa Wilcoxon porque los puntajes RAGAS están acotados en [0, 1] y no se asume normalidad.
- Se reporta además la prueba de normalidad de Shapiro-Wilk como justificación.

Entrada:  ragas_por_par.csv (de evaluar_ragas.py)
Salida:   estadistica_ragas.csv  + impresión en pantalla

Uso (desde RAGASTESIS/EVALUACION):  python estadistica.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CARPETA = Path(__file__).resolve().parent
METRICAS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
UMBRAL, ALFA, B, SEMILLA = 0.85, 0.05, 10_000, 42


def analizar(x, rng):
    x = np.asarray([v for v in x if v == v], dtype=float)  # sin NaN
    n = len(x)
    fila = {"n": n}
    if n == 0:
        return fila
    boot = rng.choice(x, size=(B, n), replace=True).mean(axis=1)
    fila.update({
        "media": x.mean(), "de": x.std(ddof=1) if n > 1 else np.nan, "mediana": np.median(x),
        "min": x.min(), "max": x.max(),
        "ic95_inf": np.percentile(boot, 2.5), "ic95_sup": np.percentile(boot, 97.5),
    })
    if n >= 3 and np.ptp(x) > 0:
        fila["shapiro_p"] = stats.shapiro(x).pvalue
    d = x - UMBRAL
    if np.any(d != 0):
        res = stats.wilcoxon(d, alternative="greater", zero_method="wilcox")
        fila["wilcoxon_W"] = res.statistic
        fila["wilcoxon_p"] = res.pvalue
        fila["rechaza_H0"] = "sí" if res.pvalue < ALFA else "no"
    return fila


def main():
    ruta = CARPETA / "ragas_por_par.csv"
    if not ruta.exists():
        sys.exit(f"No existe {ruta}")
    df = pd.read_csv(ruta).drop_duplicates("id", keep="last")
    rng = np.random.default_rng(SEMILLA)
    filas = []
    for grupo, sub in [("global", df)] + [(t, df[df.tipo == t]) for t in ("personal", "normativa", "mixta")]:
        if len(sub) == 0:
            continue
        for m in METRICAS:
            filas.append({"grupo": grupo, "metrica": m, **analizar(sub[m], rng)})
    out = pd.DataFrame(filas)
    out.to_csv(CARPETA / "estadistica_ragas.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    cols = ["grupo", "metrica", "n", "media", "de", "mediana", "ic95_inf", "ic95_sup", "shapiro_p", "wilcoxon_p", "rechaza_H0"]
    print(out[[c for c in cols if c in out.columns]].round(4).to_string(index=False))
    print(f"\nPares evaluados: {len(df)} | Umbral {UMBRAL} | alfa {ALFA} | bootstrap B={B}, semilla {SEMILLA}")


if __name__ == "__main__":
    main()
"""Análisis estadístico de los resultados RAGAS (4.4.3 y 4.4.4).

Para cada métrica (Faithfulness, Answer Relevancy, Context Precision, Context Recall):
- Descriptivo: n, media, desviación estándar, mediana, mínimo y máximo.
- IC 95 % de la media por bootstrap percentil (10 000 remuestreos, semilla 42).
- Prueba de Wilcoxon de una muestra, unilateral: H0: mediana < 0.85 ; H1: mediana >= 0.85 (alfa = 0.05).
  Se usa Wilcoxon porque los puntajes RAGAS están acotados en [0, 1] y no se asume normalidad.
- Se reporta además la prueba de normalidad de Shapiro-Wilk como justificación.

Entrada:  ragas_por_par.csv (de evaluar_ragas.py)
Salida:   estadistica_ragas.csv  + impresión en pantalla

Uso (desde RAGASTESIS/EVALUACION):  python estadistica.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CARPETA = Path(__file__).resolve().parent
METRICAS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
UMBRAL, ALFA, B, SEMILLA = 0.85, 0.05, 10_000, 42


def analizar(x, rng):
    x = np.asarray([v for v in x if v == v], dtype=float)  # sin NaN
    n = len(x)
    fila = {"n": n}
    if n == 0:
        return fila
    boot = rng.choice(x, size=(B, n), replace=True).mean(axis=1)
    fila.update({
        "media": x.mean(), "de": x.std(ddof=1) if n > 1 else np.nan, "mediana": np.median(x),
        "min": x.min(), "max": x.max(),
        "ic95_inf": np.percentile(boot, 2.5), "ic95_sup": np.percentile(boot, 97.5),
    })
    if n >= 3 and np.ptp(x) > 0:
        fila["shapiro_p"] = stats.shapiro(x).pvalue
    d = x - UMBRAL
    if np.any(d != 0):
        res = stats.wilcoxon(d, alternative="greater", zero_method="wilcox")
        fila["wilcoxon_W"] = res.statistic
        fila["wilcoxon_p"] = res.pvalue
        fila["rechaza_H0"] = "sí" if res.pvalue < ALFA else "no"
    return fila


def main():
    ruta = CARPETA / "ragas_por_par.csv"
    if not ruta.exists():
        sys.exit(f"No existe {ruta}")
    df = pd.read_csv(ruta).drop_duplicates("id", keep="last")
    rng = np.random.default_rng(SEMILLA)
    filas = []
    for grupo, sub in [("global", df)] + [(t, df[df.tipo == t]) for t in ("personal", "normativa", "mixta")]:
        if len(sub) == 0:
            continue
        for m in METRICAS:
            filas.append({"grupo": grupo, "metrica": m, **analizar(sub[m], rng)})
    out = pd.DataFrame(filas)
    out.to_csv(CARPETA / "estadistica_ragas.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    cols = ["grupo", "metrica", "n", "media", "de", "mediana", "ic95_inf", "ic95_sup", "shapiro_p", "wilcoxon_p", "rechaza_H0"]
    print(out[[c for c in cols if c in out.columns]].round(4).to_string(index=False))
    print(f"\nPares evaluados: {len(df)} | Umbral {UMBRAL} | alfa {ALFA} | bootstrap B={B}, semilla {SEMILLA}")


if __name__ == "__main__":
    main()