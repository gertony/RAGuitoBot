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


def test_jale_es_dato_propio_y_si_jalo_es_hipotesis():
    import consultas
    a = consultas.Academico()
    for q in ("jalé algun curso el semenstre pasado?", "me jalaron en física"):
        assert "desaprobados" in agente.router_reglas(q, a)["intenciones"], q
    assert "desaprobados" not in agente.router_reglas("si jalo un curso 3 veces, ¿me botan de la u?", a)["intenciones"]


def test_jale_en_pasado_es_dato_propio():
    import consultas
    a = consultas.Academico()
    r = agente.router_reglas("jalé algun curso el semenstre pasado?", a)
    assert {"desaprobados", "notas_periodo"} <= set(r["intenciones"])
    assert "desaprobados" in agente.router_reglas("¿reprobé algún curso?", a)["intenciones"]
    # «si jalo» es una hipótesis: no pide su historial
    assert "desaprobados" not in agente.router_reglas("si jalo un curso 3 veces, ¿me botan de la u?", a)["intenciones"]


def test_preguntas_coloquiales_21_11():
    import consultas
    a = consultas.Academico()
    r = lambda q: agente.router_reglas(q, a)  # noqa: E731
    assert r("cuál es mi pps?")["intenciones"] == ["promedio"]
    assert r("que nota saque en mate 2?")["curso"] == "ACFB0213"
    assert r("mis notas del 2023-1")["periodo"] == "2023-I" and r("promedio del 2023 2")["periodo"] == "2023-II"
    assert "avance" in r("cuantos creditos tengo?")["intenciones"]
    assert "en_curso" in r("cuantos creditos tengo matriculados?")["intenciones"]
    assert r("tengo clases el sabado?")["dia"] == "sabado"
    assert "desaprobados" in r("estoy en riesgo de que me boten?")["intenciones"]
    assert r("me falta algún curso del ciclo 3?")["intenciones"] == ["pendientes"] and r("del ciclo 3?")["ciclo"] == 3
    assert "pendientes" in r("qué cursos me faltan?")["intenciones"]
    assert "desaprobados" not in r("si jalo un curso 3 veces, ¿me botan de la u?")["intenciones"]
    p = a.cursos_pendientes("EST0030")
    assert p["datos"]["cursos"] and not set(p["datos"]["cursos"]) & a._aprobados("EST0030")
    assert "2023-I" in a.promedio_ponderado("EST0030", "2023-I")["texto"]


def test_d14_el_agente_respeta_chroma_ruta(monkeypatch, tmp_path):
    # D-14: el Agente pasaba CHROMA_POR_DEFECTO explícito y BuscadorNormativa ignoraba CHROMA_RUTA
    usadas = []

    class Espia:
        def __init__(self, ruta=None, *a, **k):
            usadas.append(ruta)

    monkeypatch.setattr(agente, "BuscadorNormativa", Espia)
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    agente.Agente(llm=FakeListChatModel(responses=["x"]), registrar=False)
    assert usadas == [None]  # None → BuscadorNormativa lee CHROMA_RUTA


def test_v5_consultas_agregadas(monkeypatch):
    import consultas
    monkeypatch.setenv("EXTRAS_V5", "1")
    a = consultas.Academico()
    r = lambda q: agente.router_reglas(q, a)  # noqa: E731
    assert r("cual fue mi mejor nota historica?")["intenciones"] == ["extremos"]
    assert r("que cursos tengo con nota 12?")["intenciones"] == ["notas_valor"] and r("con nota 12")["nota"] == 12
    q = r("cuantos puntos necesito para tener promedio 15 este semestre?")
    assert q["intenciones"] == ["meta_promedio"] and q["objetivo"] == 15
    assert r("que cursos de los que estoy llevando son obligatorios?")["intenciones"] == ["tipo_en_curso"]
    assert r("que cursos no puedo llevar si jalo algun curso?")["intenciones"] == ["bloqueos"]
    assert r("¿Cuál es la nota mínima para aprobar un curso?")["intenciones"] == []  # norma, no «peor nota»
    assert "cursos se desaprueban" in agente.normalizar_consulta("todos los cuross se despruabn con 12?")
    assert agente.corregir_erratas("todos los cuross se despruabn") == "todos los cursos se desaprueban"
    assert agente.corregir_erratas("estoy matriculado") == "estoy matriculado"  # una forma válida no se toca
    ext = a.notas_extremas("EST0026")
    assert ext["datos"] == {"max": 17, "min": 6}
    assert len(a.cursos_con_nota("EST0039", 12)["datos"]["cursos"]) == 18
    assert a.meta_promedio("EST0033", 15)["datos"] == {"creditos": 14.0, "suma_necesaria": 210.0}


def test_v5_apagado_reproduce_la_v4(monkeypatch):
    import consultas
    monkeypatch.setenv("EXTRAS_V5", "0")
    a = consultas.Academico()
    assert agente.router_reglas("cual fue mi mejor nota historica?", a)["intenciones"] == ["resumen"]
    q = "todos los cuross se despruabn con 12?"
    assert agente.normalizar_consulta(q) == q


def test_v5_ajustes_de_la_prueba_en_la_app(monkeypatch):
    # 22/11, prueba manual en la app con EST0001 (primer semestre, sin notas)
    import consultas
    monkeypatch.setenv("EXTRAS_V5", "1")
    a = consultas.Academico()
    q = "que cursos que estoy llevando ahora son prerequisitos de un curso del siguiennde semenstre?"
    assert agente.router_reglas(q, a)["intenciones"] == ["bloqueos"]
    assert "primer semestre" in a.notas_extremas("EST0001")["texto"]
    assert "todavía no tienen nota" in a.notas_extremas("EST0030", semestre_actual=True)["texto"]
