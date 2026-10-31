"""Evaluación RAGAS de las respuestas del agente (Bloque 5: OE2 y OE3).

Métricas (Es et al., 2024): Faithfulness y Answer Relevancy (OE2); Context Precision y Context Recall (OE3).
- LLM evaluador: distinto del generador para evitar el sesgo de autoevaluación. Por defecto Gemini
  (el generador fue openai/gpt-oss-120b vía Groq). Answer Relevancy con strictness=1.
- Embeddings (Answer Relevancy): intfloat/multilingual-e5-small, los mismos del índice, cargados localmente.

Entrada:  ../AGENTE/resultados_testbed.jsonl  (salida de correr_testbed.py)
Salidas (en esta carpeta):
  ragas_por_par.csv   una fila por par con las 4 métricas (se va llenando por lotes; es reanudable)
  ragas_resumen.csv   media, desviación estándar y n por métrica, por tipo de consulta y global, más el
                      RAGAS agregado (media armónica de las 4 medias) y la comparación con 0.85 / 0.96 / 0.94

Uso (desde RAGASTESIS/EVALUACION, con GOOGLE_API_KEY definida):  python evaluar_ragas.py
Respaldo:  $env:EVAL_PROVEEDOR="groq"; $env:EVAL_MODEL="openai/gpt-oss-20b"; python evaluar_ragas.py
"""
import argparse
import json
import os
import statistics
import sys
import types
from pathlib import Path

# Parche de compatibilidad: ragas 0.4.3 importa un módulo que langchain-community 0.4 ya no trae
try:
    import langchain_community.chat_models.vertexai  # noqa: F401
except Exception:
    _m = types.ModuleType("langchain_community.chat_models.vertexai")
    _m.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules[_m.__name__] = _m

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
from langchain_core.embeddings import Embeddings
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

CARPETA = Path(__file__).resolve().parent
RAIZ = CARPETA.parent
METRICAS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
UMBRAL, NEUPANE, JIMENEZ = 0.85, 0.96, 0.94


class EmbeddingsE5(Embeddings):
    """multilingual-e5-small con el prefijo 'query: ' (Answer Relevancy compara preguntas con preguntas)."""

    def __init__(self, modelo="intfloat/multilingual-e5-small"):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(modelo)

    def embed_documents(self, textos):
        return self.m.encode([f"query: {t}" for t in textos], normalize_embeddings=True).tolist()

    def embed_query(self, texto):
        return self.embed_documents([texto])[0]


def llm_evaluador():
    """Evaluador distinto del generador (openai/gpt-oss-120b): por defecto Gemini (otra familia de modelos).

    Respaldo: EVAL_PROVEEDOR=groq y EVAL_MODEL=openai/gpt-oss-20b (misma familia que el generador: declararlo
    como limitación). qwen/qwen3.8-27b se descartó: su límite gratuito es de 1 000 tokens de salida por minuto.
    """
    proveedor = os.environ.get("EVAL_PROVEEDOR", "gemini").lower()
    if proveedor == "groq":
        from langchain_groq import ChatGroq
        modelo = os.environ.get("EVAL_MODEL", "openai/gpt-oss-20b")
        extra = {"reasoning_format": "hidden"} if "qwen" in modelo.lower() else {}  # que el "pensamiento" no rompa el JSON
        return ChatGroq(model=modelo, temperature=0, max_retries=3, timeout=120, **extra)
    from langchain_google_genai import ChatGoogleGenerativeAI
    if not os.environ.get("GOOGLE_API_KEY"):
        sys.exit("Falta GOOGLE_API_KEY para el evaluador Gemini (o usa EVAL_PROVEEDOR=groq).")
    return ChatGoogleGenerativeAI(model=os.environ.get("EVAL_MODEL", "gemini-3.8-flash"), temperature=0,
                                  max_retries=6, timeout=120)


def columna(df, nombre):
    """Los nombres de columna cambian entre versiones de RAGAS: se busca por prefijo."""
    for c in df.columns:
        if c == nombre or c.startswith(nombre):
            return c
    if nombre == "context_precision":
        for c in df.columns:
            if "precision" in c:
                return c
    return None


def resumen(filas):
    salida = []
    grupos = {"global": filas}
    for t in ("personal", "normativa", "mixta"):
        grupos[t] = [f for f in filas if f["tipo"] == t]
    for g, fs in grupos.items():
        medias = {}
        for m in METRICAS:
            vals = [f[m] for f in fs if f.get(m) is not None and f[m] == f[m]]  # descarta NaN
            n = len(vals)
            media = statistics.mean(vals) if vals else float("nan")
            de = statistics.stdev(vals) if n > 1 else float("nan")
            medias[m] = media
            salida.append({"grupo": g, "metrica": m, "n": n, "media": round(media, 4), "de": round(de, 4),
                           "cumple_umbral_0.85": "sí" if media >= UMBRAL else "no"})
        validas = [v for v in medias.values() if v == v and v > 0]
        armonica = len(validas) / sum(1 / v for v in validas) if len(validas) == 4 else float("nan")
        salida.append({"grupo": g, "metrica": "RAGAS_agregado_media_armonica", "n": len(fs),
                       "media": round(armonica, 4), "de": "",
                       "cumple_umbral_0.85": "sí" if armonica >= UMBRAL else "no",
                       "referencia_Neupane_2024": NEUPANE, "referencia_JimenezOrellana_2026": JIMENEZ})
    return pd.DataFrame(salida)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resultados", default=str(RAIZ / "AGENTE" / "resultados_testbed.jsonl"))
    ap.add_argument("--lote", type=int, default=5, help="pares por lote (se guarda al terminar cada lote)")
    ap.add_argument("--solo-resumen", action="store_true")
    args = ap.parse_args()

    por_par = CARPETA / "ragas_por_par.csv"
    filas = [json.loads(l) for l in Path(args.resultados).read_text(encoding="utf-8").splitlines() if l.strip()]
    hechos = set(pd.read_csv(por_par)["id"]) if por_par.exists() else set()
    pendientes = [f for f in filas if f["id"] not in hechos]
    print(f"Respuestas: {len(filas)} | ya evaluadas: {len(hechos)} | por evaluar: {len(pendientes)}")

    if pendientes and not args.solo_resumen:
        llm = LangchainLLMWrapper(llm_evaluador())
        emb = LangchainEmbeddingsWrapper(EmbeddingsE5())
        # strictness=1: una sola pregunta generada por respuesta (las API gratuitas no admiten n > 1)
        metricas = [Faithfulness(), ResponseRelevancy(strictness=1), LLMContextPrecisionWithReference(), LLMContextRecall()]
        conf = RunConfig(max_workers=int(os.environ.get("EVAL_WORKERS", "1")), timeout=240, max_retries=8, max_wait=90)
        print(f"Evaluador: {os.environ.get('EVAL_PROVEEDOR', 'gemini')} / {getattr(llm.langchain_llm, 'model', getattr(llm.langchain_llm, 'model_name', '?'))}")
        for i in range(0, len(pendientes), args.lote):
            lote = pendientes[i:i + args.lote]
            ds = EvaluationDataset([SingleTurnSample(user_input=f["pregunta"], retrieved_contexts=f["contextos"] or ["(sin contexto)"],
                                                     response=f["respuesta"], reference=f["ground_truth"]) for f in lote])
            res = evaluate(ds, metrics=metricas, llm=llm, embeddings=emb, run_config=conf, show_progress=True)
            df = res.to_pandas()
            salida = pd.DataFrame({"id": [f["id"] for f in lote], "tipo": [f["tipo"] for f in lote],
                                   "categoria": [f["categoria"] for f in lote]})
            for m in METRICAS:
                c = columna(df, m)
                salida[m] = df[c].values if c else float("nan")
            # no se guardan los pares sin ninguna métrica (p. ej. por timeouts): quedan pendientes para reintentar
            validos = salida[salida[METRICAS].notna().any(axis=1)]
            if len(validos):
                validos.to_csv(por_par, mode="a", header=not por_par.exists(), index=False, encoding="utf-8")
            print(f"Lote {i // args.lote + 1}: guardados {len(validos)}/{len(salida)} ({', '.join(validos['id'])})")
            if len(validos) == 0:
                sys.exit("Ningún par del lote obtuvo métricas (¿evaluador saturado o sin cuota?). Se detiene para no "
                         "gastar tiempo; prueba otro evaluador o vuelve a intentarlo más tarde.")

    if not por_par.exists():
        sys.exit("Aún no hay pares evaluados.")
    todos = pd.read_csv(por_par).drop_duplicates("id", keep="last").to_dict("records")
    r = resumen(todos)
    r.to_csv(CARPETA / "ragas_resumen.csv", index=False, encoding="utf-8-sig")
    print("\n== Resumen RAGAS")
    print(r[["grupo", "metrica", "n", "media", "de", "cumple_umbral_0.85"]].to_string(index=False))
    nulos = {m: sum(1 for f in todos if f.get(m) != f.get(m)) for m in METRICAS}
    if any(nulos.values()):
        print(f"\nAviso: valores sin calcular (NaN) por métrica: {nulos}. Vuelve a correr para reintentarlos "
              f"borrando esas filas de ragas_por_par.csv, o repórtalos como no evaluados.")


if __name__ == "__main__":
    main()
