"""Recuperación v2 (20/11/2026): sinónimos estudiantiles (D-01), BM25 para la búsqueda híbrida y D-12."""
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "AGENTE"))
sys.path.insert(0, str(RAIZ / "CHROMADB2"))
import agente  # noqa: E402


def test_normalizar_agrega_terminos_sin_borrar_la_pregunta():
    q = agente.normalizar_consulta("si jalo un curso 3 veces, ¿me botan de la u?")
    assert q.startswith("si jalo un curso 3 veces")
    for termino in ("desaprobar", "tres veces", "retiro definitivo", "universidad"):
        assert termino in q


def test_normalizar_no_toca_una_pregunta_formal():
    q = "¿Cuál es la nota mínima para aprobar un curso?"
    assert agente.normalizar_consulta(q) == q


def test_bm25_prioriza_el_documento_con_las_palabras_exactas():
    docs = ["La reserva de matrícula se solicita por un semestre.",
            "La desmatrícula procede dentro de las cuatro primeras semanas del semestre.",
            "El promedio ponderado se calcula con los créditos."]
    assert agente._BM25(docs).ranking("¿cuándo procede la desmatrícula?", 3)[0] == 1


def test_disposiciones_se_separan_del_ultimo_articulo():
    ix = pytest.importorskip("indexar_normativa", reason="requiere las dependencias del indexador")
    unidad = {"documento": "Reglamento General UNE", "articulo": "381", "titulo": "Órgano de control", "fuente": "x",
              "pagina": 1, "texto": "Artículo 381. Órgano de control\nTexto del artículo.\n"
              "DISPOSICIONES COMPLEMENTARIAS, TRANSITORIAS Y FINALES\n"
              "PRIMERA: Plazo de adecuación\nTexto uno.\n"
              "QUINTA: Excepción para la obtención del bachillerato automático\nTexto cinco."}
    salida = ix.separar_disposiciones([unidad])
    assert salida[0]["articulo"] == "381" and "PRIMERA" not in salida[0]["texto"]
    quinta = next(u for u in salida if u["articulo"] == "DCTF-5")
    assert quinta["titulo"].startswith("Quinta disposición complementaria transitoria y final: Excepción")
    assert "Texto cinco." in quinta["texto"] and "Texto uno." not in quinta["texto"]



def test_retiro_de_un_curso_se_traduce_a_desmatricula():
    assert "desmatrícula" in agente.normalizar_consulta("¿Puedo retirarme de un curso después de matricularme?")
    assert "desmatrícula" in agente.normalizar_consulta("Si me retiro de Seminario de Tesis, ¿sigo siendo regular?")
    assert "desmatrícula" not in agente.normalizar_consulta("¿Qué pasa si me retiro de la universidad?")


def test_jalar_conjugaciones():
    for q in ("Jalé física", "me jalaron en cálculo", "si jalo un curso", "estoy jalando", "jalado tres veces"):
        assert "desaprobar" in agente.normalizar_consulta(q), q
    assert "desaprobar" not in agente.normalizar_consulta("jalea de fresa")


def test_intenciones_coloquiales():
    import consultas
    a = consultas.Academico()
    casos = {"¿qué me toca el miércoles?": "horario", "¿a qué hora entro el jueves?": "horario",
             "¿qué cursos llevo ahorita?": "en_curso", "¿puedo llevar circuitos 3 el otro ciclo?": "puede_llevar"}
    for q, intencion in casos.items():
        assert intencion in agente.router_reglas(q, a)["intenciones"], q
    assert a.buscar_curso("circuitos 3") == "TEEI0646" and a.buscar_curso("lenguaje 1") == "ACFB0101"
    r = agente.router_reglas("estoy llevando lenguaje 1 por tercera vez, ¿qué me pasa si no la paso?", a)
    assert r["ruta"] == "mixta" and "desaprobados" in r["intenciones"]
    assert agente.router_reglas("Si desapruebo Física I por tercera vez, ¿qué pasa?", a)["ruta"] == "normativa"


def test_cursos_disponibles_proximo_semestre():
    import consultas
    a = consultas.Academico()
    for q in ("que cursos puedo llevar el siguiente semestre?", "¿qué cursos tengo disponibles para 2024-II?"):
        r = agente.router_reglas(q, a)
        assert "disponibles" in r["intenciones"] and "en_curso" not in r["intenciones"], q
    assert "disponibles" not in agente.router_reglas("¿puedo llevar circuitos 3 el otro ciclo?", a)["intenciones"]
    d = a.cursos_disponibles("EST0030")
    aprobados = a._aprobados("EST0030")
    assert d["datos"]["ya"] and all(a.prereq.get(c, set()) <= aprobados for c in d["datos"]["ya"])
    assert not set(d["datos"]["ya"]) & aprobados
    assert "en que" in agente._plano("¿en qué año estoy?")
    assert "estado" in agente.router_reglas("¿en qué año estoy?", a)["intenciones"]
