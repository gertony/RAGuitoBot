"""RNF-09: el proveedor del LLM se cambia por configuración, sin tocar el código (OpenRouter, 20/11/2026)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "AGENTE"))
pytest.importorskip("langchain_openai")
import agente  # noqa: E402


def test_openrouter_fija_el_proveedor_de_la_linea_base(monkeypatch):
    monkeypatch.setenv("LLM_PROVEEDOR", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "clave-de-prueba")
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_PROVEEDOR", raising=False)
    llm = agente.crear_llm()
    assert type(llm).__name__ == "ChatOpenAI"
    assert llm.model_name == "openai/gpt-oss-120b"
    assert str(llm.openai_api_base).startswith("https://openrouter.ai/api/v1")
    assert llm.temperature == 0
    assert llm.extra_body == {"provider": {"order": ["groq"], "allow_fallbacks": False}}


def test_openrouter_sin_proveedor_fijo_deja_elegir(monkeypatch):
    monkeypatch.setenv("LLM_PROVEEDOR", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "clave-de-prueba")
    monkeypatch.setenv("OPENROUTER_PROVEEDOR", "")
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-oss-20b")
    llm = agente.crear_llm()
    assert llm.model_name == "openai/gpt-oss-20b"
    assert llm.extra_body is None


def test_openrouter_sin_clave_avisa(monkeypatch):
    monkeypatch.setenv("LLM_PROVEEDOR", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        agente.crear_llm()
