"""Prueba mínima de OpenRouter: una pregunta al LLM y una consulta completa del agente.

Uso (desde la raíz del proyecto, con el .env configurado):
    python AGENTE/prueba_openrouter.py
"""
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parent.parent
load_dotenv(RAIZ / ".env")
os.environ["LLM_PROVEEDOR"] = "openrouter"
sys.path.insert(0, str(RAIZ / "AGENTE"))

from agente import Agente, crear_llm  # noqa: E402

print("Modelo:", os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-120b"),
      "| proveedor fijado:", os.environ.get("OPENROUTER_PROVEEDOR", "groq") or "(elige OpenRouter)")

t = time.perf_counter()
r = crear_llm().invoke("Responde solo con la palabra: listo")
print(f"1) LLM directo ({time.perf_counter() - t:.1f} s):", r.content.strip())
uso = r.response_metadata.get("token_usage") or {}
print("   tokens:", uso.get("prompt_tokens"), "de entrada y", uso.get("completion_tokens"), "de salida")

t = time.perf_counter()
agente = Agente(registrar=False)
res = agente.responder("EST0030", "¿Cuál es mi promedio ponderado?")
print(f"2) Agente completo ({time.perf_counter() - t:.1f} s), ruta:", res.get("ruta"))
print("  ", res.get("respuesta"))
