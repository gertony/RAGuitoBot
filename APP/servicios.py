"""Servicios compartidos por las páginas de RAGuitoBot: rutas, agente, métricas y procesos en segundo plano.

La app vive en RAGASTESIS/APP y reutiliza el código de la tesis sin modificarlo:
AGENTE/agente.py (orquestador), TESTBED/consultas.py (datos personales) y CHROMADB2/indexar_normativa.py.
"""
import json
import statistics
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent
for carpeta in ("AGENTE", "TESTBED", "CHROMADB2"):
    if str(RAIZ / carpeta) not in sys.path:
        sys.path.insert(0, str(RAIZ / carpeta))
load_dotenv(RAIZ / ".env")  # claves y LLM_PROVEEDOR solo desde .env (RNF-09, RNF-10)

DB = RAIZ / "DB" / "academico.db"
RESULTADOS = RAIZ / "AGENTE" / "resultados_testbed.jsonl"
RAGAS_POR_PAR = RAIZ / "EVALUACION" / "ragas_por_par.csv"
UMBRAL = 0.85
REFERENCIAS = {"Neupane et al. (2024)": 0.96, "Jimenez Orellana et al. (2026), solo referencial": 0.94}
# nombres posibles de las columnas que escribe RAGAS 0.4.x  [VERIFICAR con ragas_por_par.csv]
METRICAS = {
    "Faithfulness": ("faithfulness",),
    "Answer Relevancy": ("answer_relevancy", "response_relevancy"),
    "Context Precision": ("context_precision",),
    "Context Recall": ("context_recall",),
}


@st.cache_resource(show_spinner="Cargando el agente (modelo de embeddings e índice normativo)...")
def agente():
    """Una sola instancia del agente para toda la app: el modelo de embeddings se carga una vez."""
    from agente import Agente
    return Agente(db=DB)


# ------------------------------------------------------------------ métricas (P-03 y P-05)
def resumen_latencia(ruta=RESULTADOS):
    """Latencia y enrutamiento del testbed, con el mismo cálculo de correr_testbed.py."""
    if not Path(ruta).exists():
        return None
    filas = [json.loads(x) for x in Path(ruta).read_text(encoding="utf-8").splitlines() if x.strip()]
    if not filas:
        return None
    lat = sorted(f["latencia_ms"] for f in filas)
    p95 = lat[min(len(lat) - 1, int(round(0.95 * len(lat))) - 1)]
    return {"pares": len(filas), "media": statistics.mean(lat), "mediana": statistics.median(lat), "p95": p95,
            "ruta_ok": sum(f["ruta"] == f["ruta_esperada"] for f in filas)}


def _columna(df, claves):
    return next((c for c in df.columns if any(k in c.lower() for k in claves)), None)


def tabla_ragas(ruta=RAGAS_POR_PAR):
    """Media, mediana y n por métrica frente al umbral de 0.85. Devuelve None si aún no hay evaluación."""
    if not Path(ruta).exists():
        return None
    df = pd.read_csv(ruta)
    filas = []
    for nombre, claves in METRICAS.items():
        col = _columna(df, claves)
        if col is None:
            continue
        v = pd.to_numeric(df[col], errors="coerce").dropna()
        if v.empty:
            continue
        filas.append({"Métrica": nombre, "n": len(v), "Media": round(v.mean(), 3), "Mediana": round(v.median(), 3),
                      "Umbral": UMBRAL, "Mediana > 0.85": "Sí" if v.median() > UMBRAL else "No"})
    return pd.DataFrame(filas) if filas else None


def ragas_agregado(tabla):
    """Media armónica de las medias de las cuatro métricas (tesis, §4.3.3)."""
    if tabla is None or len(tabla) < 4 or (tabla["Media"] <= 0).any():
        return None
    return round(statistics.harmonic_mean(tabla["Media"].tolist()), 3)


# ------------------------------------------------------------------ procesos largos (testbed, RAGAS)
@st.cache_resource
def _procesos():
    return {}


def lanzar(nombre, script, carpeta, args=()):
    """Ejecuta un script de la tesis en segundo plano y guarda su salida en un .log junto al script."""
    p = _procesos().get(nombre)
    if p and p["proc"].poll() is None:
        return False  # ya está corriendo
    log = Path(carpeta) / f"{nombre}.log"
    fh = log.open("w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-u", script, *args], cwd=carpeta, stdout=fh,  # noqa: S603
                            stderr=subprocess.STDOUT, text=True)
    _procesos()[nombre] = {"proc": proc, "log": log}
    return True


def estado(nombre):
    """('sin ejecutar' | 'en curso' | 'terminado' | 'con error', últimas líneas del log)."""
    p = _procesos().get(nombre)
    if not p:
        return "sin ejecutar", ""
    cola = "\n".join(p["log"].read_text(encoding="utf-8", errors="ignore").splitlines()[-12:])
    codigo = p["proc"].poll()
    return ("en curso" if codigo is None else "terminado" if codigo == 0 else "con error"), cola
