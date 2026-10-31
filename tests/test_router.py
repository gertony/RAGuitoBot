"""RF-02 y RF-16: las reglas del orquestador asignan la ruta esperada del testbed.

Cuando las reglas no deciden (ruta None), desempata el LLM; esa parte necesita la API y no se prueba aquí.
"""
import json

import pytest
from conftest import DB_PATH, RAIZ

TESTBED = json.loads((RAIZ / "TESTBED" / "testbed.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def academico():
    from consultas import Academico
    return Academico(DB_PATH)


def test_testbed_tiene_35_pares():
    tipos = [t["tipo"] for t in TESTBED]
    assert (len(tipos), tipos.count("personal"), tipos.count("normativa"), tipos.count("mixta")) == (35, 20, 10, 5)


@pytest.mark.parametrize("par", TESTBED, ids=[t["id"] for t in TESTBED])
def test_reglas_no_contradicen_la_ruta_esperada(academico, par):
    from agente import router_reglas
    ruta = router_reglas(par["pregunta"], academico)["ruta"]
    assert ruta in (None, par["ruta_esperada"])
