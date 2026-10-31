"""Diagnóstico de latencia de Gemini: mide una llamada mínima con distintas configuraciones.

Uso (desde RAGASTESIS/AGENTE, con GOOGLE_API_KEY definida):  python diag_gemini.py
"""
import os
import time

from langchain_google_genai import ChatGoogleGenerativeAI

MODELO = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
PREGUNTA = "Responde solo con la palabra: listo"

PRUEBAS = [
    ("por defecto, sin reintentos", {}),
    ("razonamiento mínimo (thinking_level=low)", {"thinking_level": "low"}),
    ("sin razonamiento (thinking_budget=0)", {"thinking_budget": 0}),
]

for nombre, extra in PRUEBAS:
    try:
        llm = ChatGoogleGenerativeAI(model=MODELO, temperature=0, max_retries=0, **extra)
        t0 = time.perf_counter()
        r = llm.invoke(PREGUNTA)
        print(f"OK   {nombre:<45} {time.perf_counter() - t0:7.1f} s | respuesta: {str(r.content)[:40]!r}")
    except Exception as e:
        print(f"FALLA {nombre:<44} {type(e).__name__}: {str(e)[:160]}")
