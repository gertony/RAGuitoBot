"""Enrutador v2 (ROUTER_MODO=llm, 20/11/2026): el LLM decide la ruta y las reglas aportan intenciones y curso."""
import sys
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "AGENTE"))
import agente  # noqa: E402

if not agente.DB_POR_DEFECTO.exists():
    pytest.skip("requiere academico.db (build_db.py)", allow_module_level=True)


class SinNormativa:
    def buscar(self, *a, **k):
        return []


def enrutar(monkeypatch, salida_llm, pregunta, eid="EST0030"):
    monkeypatch.setenv("ROUTER_MODO", "llm")
    ag = agente.Agente(llm=FakeListChatModel(responses=[salida_llm]), buscador=SinNormativa(), registrar=False)
    return ag._enrutar({"pregunta": pregunta, "estudiante_id": eid})


def test_hipotesis_en_primera_persona_va_a_normativa(monkeypatch):
    # con reglas iba a «personal» por el «me»; el LLM la clasifica y se descartan las intenciones personales
    x = enrutar(monkeypatch, '{"ruta": "normativa"}', "si jalo un curso 3 veces, ¿me botan de la u?")
    assert (x["ruta"], x["router"], x["intenciones"]) == ("normativa", "llm-v2", [])


def test_mixta_sin_dato_concreto_trae_ficha_y_desaprobados(monkeypatch):
    x = enrutar(monkeypatch, '{"ruta": "mixta"}', "¿puedo entrar al orden de mérito de mi promoción?", "EST0037")
    assert x["ruta"] == "mixta" and x["intenciones"] == ["resumen", "desaprobados"]


def test_personal_conserva_las_intenciones_de_las_reglas(monkeypatch):
    x = enrutar(monkeypatch, '{"ruta": "personal"}', "oe, ¿cuánto llevo de promedio?")
    assert x["ruta"] == "personal" and "promedio" in x["intenciones"]


def test_si_el_llm_responde_mal_se_usan_las_reglas(monkeypatch):
    x = enrutar(monkeypatch, "no sé", "¿Cuál es mi promedio ponderado?")
    assert x["ruta"] == "personal" and x["router"].startswith("reglas")


def test_modo_por_defecto_es_la_linea_base(monkeypatch):
    monkeypatch.delenv("ROUTER_MODO", raising=False)
    ag = agente.Agente(llm=FakeListChatModel(responses=["x"]), buscador=SinNormativa(), registrar=False)
    assert ag.modo_router == "reglas"


def test_union_discrepancia_trae_ambas_fuentes(monkeypatch):
    # las reglas ven un dato propio («volviendo a llevar»), el LLM solo norma: se recuperan las dos fuentes
    monkeypatch.setenv("ROUTER_MODO", "union")
    ag = agente.Agente(llm=FakeListChatModel(responses=['{"ruta": "normativa"}']), buscador=SinNormativa(),
                       registrar=False)
    x = ag._enrutar({"pregunta": "Estoy volviendo a llevar Programación, ¿qué pasa si la desapruebo otra vez?",
                     "estudiante_id": "EST0030"})
    assert x["ruta"] == "mixta" and x["router"].startswith("union (reglas=")


def test_union_si_coinciden_usa_esa_ruta(monkeypatch):
    monkeypatch.setenv("ROUTER_MODO", "union")
    ag = agente.Agente(llm=FakeListChatModel(responses=['{"ruta": "personal"}']), buscador=SinNormativa(),
                       registrar=False)
    x = ag._enrutar({"pregunta": "¿Cuál es mi promedio ponderado?", "estudiante_id": "EST0030"})
    assert (x["ruta"], x["router"]) == ("personal", "union (coinciden)")
