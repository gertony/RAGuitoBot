"""Lista los modelos de chat disponibles en tu cuenta de Groq (necesita GROQ_API_KEY).

Uso: python listar_modelos.py
"""
from groq import Groq

modelos = sorted(m.id for m in Groq().models.list().data)
print("Modelos disponibles en Groq:")
for m in modelos:
    marca = "  <- Llama" if "llama" in m.lower() else ""
    print(f"  {m}{marca}")