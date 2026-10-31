"""Genera el testbed de evaluación: pares (pregunta, respuesta esperada) para RAGAS.

- 20 PERSONALES: la respuesta esperada (ground truth) se calcula con las mismas funciones de consultas.py
  sobre academico.db. Los estudiantes y cursos se eligen con semilla fija según el caso a probar.
- 10 NORMATIVAS: respuestas redactadas a partir del texto de los artículos (Reglamento General UNE
  consolidado, plan de estudios). Requieren revisión del autor ("revisado": false).
- 5 MIXTAS: combinan un dato del estudiante (calculado) con una regla normativa (redactada). También
  requieren revisión del autor.

Salida: testbed.json (para el pipeline) y testbed_revision.csv (para revisar en Excel).

Uso (desde RAGASTESIS/TESTBED): python generar_testbed.py
  Por defecto lee ../DB/academico.db y escribe testbed.json y testbed_revision.csv en la carpeta TESTBED.
"""
import argparse
import csv
import json
import random
from pathlib import Path

from consultas import DB_POR_DEFECTO, Academico, _f

CARPETA = Path(__file__).resolve().parent

DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]

# Validación de las respuestas esperadas normativas y mixtas: el autor cotejó contra el PDF original los artículos
# que sustentan cada par (Reglamento General UNE, Res. 3255-2022: arts. 282, 283, 288, 289, 290, 292, 125.2;
# Res. 1887-2024: art. 125.1). Plan de estudios y Ley 30220 son PDF con texto (sin OCR).
VALIDACION = "Validado por el autor contra el PDF original (28/10/2026)"
REVISADOS = {f"N{i:02d}" for i in range(1, 11)} | {f"M{i:02d}" for i in range(1, 6)}


def item(i, tipo, cat, eid, pregunta, gt, fuentes, revisado):
    return {"id": i, "tipo": tipo, "categoria": cat, "estudiante_id": eid, "pregunta": pregunta,
            "ground_truth": gt, "fuentes_esperadas": fuentes, "ruta_esperada": tipo, "revisado": revisado,
            "origen_ground_truth": "calculado con consultas.py sobre academico.db" if tipo == "personal"
            else ("redactado a partir de la normativa; revisar contra el documento" if tipo == "normativa"
                  else "dato calculado + regla redactada a partir de la normativa; revisar")}


def personales(a, rng):
    q = a.q
    reg = [r[0] for r in q("SELECT id FROM estudiante WHERE estado='matriculado regular' ORDER BY id")]
    avanzados = [e for e in reg if len(a.promedio_ponderado(e)["datos"]["parciales"]) >= 3]
    por_estado = lambda est: [r[0] for r in q("SELECT id FROM estudiante WHERE estado=? ORDER BY id", est)]
    elegir = lambda lista: rng.choice(lista)
    out, n = [], 0

    def add(cat, eid, preg, gt, fuente):
        nonlocal n
        n += 1
        out.append(item(f"P{n:02d}", "personal", cat, eid, preg, gt, [fuente], True))

    # estado de matrícula (3)
    for est, preg in [("matriculado regular", "¿Estoy matriculado este ciclo?"),
                      ("egresado", "¿Cuál es mi situación académica actual?"),
                      ("retirado", "¿En qué estado está mi matrícula?")]:
        e = elegir(por_estado(est))
        d = a.estado_matricula(e)["datos"]
        gt = {"matriculado regular": f"Sí. Está matriculado como estudiante regular en el periodo {d['periodo']}, en el ciclo {d['ciclo']}.",
              "egresado": f"Es egresado: concluyó el plan de estudios en el periodo {d['periodo']}.",
              "retirado": f"Su matrícula figura como retirada; su última matrícula fue en el periodo {d['periodo']}, en el ciclo {d['ciclo']}."}[est]
        add("estado", e, preg, gt, "estado_matricula")

    # promedio ponderado (3)
    for preg in ["¿Cuál es mi promedio ponderado?", "¿Qué promedio ponderado tuve el semestre pasado?",
                 "¿Cuál es mi promedio acumulado de toda la carrera?"]:
        e = elegir(avanzados)
        d = a.promedio_ponderado(e)["datos"]
        if "semestre pasado" in preg:
            gt = f"Su promedio ponderado del semestre {d['ultimo_semestre']} fue {_f(d['parciales'][d['ultimo_semestre']])}."
        else:
            gt = f"Su promedio ponderado total es {_f(d['total'])}, calculado sobre {len(d['parciales'])} semestres."
        add("promedio", e, preg, gt, "promedio_ponderado")

    # avance curricular (3)
    for preg, campo in [("¿Cuántos créditos tengo aprobados?", "aprobados"),
                        ("¿Qué porcentaje de la carrera llevo avanzado?", "porcentaje"),
                        ("¿Cuántos créditos me faltan para terminar la carrera?", "faltan")]:
        e = elegir(avanzados)
        d = a.avance_curricular(e)["datos"]
        gt = {"aprobados": f"Tiene {d['creditos_aprobados']:.0f} créditos aprobados de {d['creditos_totales']:.0f}.",
              "porcentaje": f"Lleva aprobado el {str(d['porcentaje']).replace('.', ',')} % de la carrera "
                            f"({d['creditos_aprobados']:.0f} de {d['creditos_totales']:.0f} créditos).",
              "faltan": f"Le faltan {d['creditos_totales'] - d['creditos_aprobados']:.0f} créditos "
                        f"(tiene {d['creditos_aprobados']:.0f} de {d['creditos_totales']:.0f})."}[campo]
        add("avance", e, preg, gt, "avance_curricular")

    # notas del último periodo cerrado (2)
    for preg in ["¿Qué notas saqué el ciclo pasado?", "¿Cómo me fue en el último semestre?"]:
        e = elegir(avanzados)
        d = a.notas_periodo(e)["datos"]
        lista = "; ".join(f"{x['curso']}: {x['nota']} ({x['condicion']})" for x in d["notas"])
        add("notas_periodo", e, preg, f"En el periodo {d['periodo']} obtuvo: {lista}.", "notas_periodo")

    # nota de un curso (3): uno aprobado a la primera, uno desaprobado y luego aprobado, uno desaprobado
    casos = {"aprobado": [], "recuperado": [], "desaprobado": []}
    for e in avanzados:
        por_curso = {}
        for m in a._matriculas(e):
            por_curso.setdefault(m["curso"], []).append(m["condicion"])
        for cod, conds in por_curso.items():
            if conds == ["aprobado"]:
                casos["aprobado"].append((e, cod))
            elif conds[-1] == "aprobado" and "desaprobado" in conds:
                casos["recuperado"].append((e, cod))
            elif conds[-1] == "desaprobado":
                casos["desaprobado"].append((e, cod))
    for tipo, plantilla in [("aprobado", "¿Qué nota saqué en {}?"), ("recuperado", "¿Aprobé {}?"),
                            ("desaprobado", "¿Cómo me fue en {}?")]:
        e, cod = elegir(casos[tipo])
        nombre = a.cursos[cod]["nombre"]
        d = a.nota_curso(e, cod)["datos"]["intentos"]
        partes = "; ".join(f"en {x['periodo']} obtuvo {x['nota']} ({x['condicion']})" for x in d)
        veces = "una matrícula" if len(d) == 1 else f"{len(d)} matrículas"
        add("nota_curso", e, plantilla.format(nombre), f"En {nombre} registra {veces}: {partes}.", "nota_curso")

    # cursos en curso (2)
    e = elegir(reg)
    d = a.cursos_en_curso(e)["datos"]
    nombres = "; ".join(a.cursos[c]["nombre"] for c in d["cursos"])
    add("en_curso", e, "¿Qué cursos estoy llevando este ciclo?",
        f"En {d['periodo']} lleva {len(d['cursos'])} asignaturas: {nombres}.", "cursos_en_curso")
    e = elegir(reg)
    d = a.cursos_en_curso(e)["datos"]
    add("en_curso", e, "¿Cuántos créditos llevo matriculados este semestre?",
        f"Lleva {d['creditos']:.0f} créditos en {len(d['cursos'])} asignaturas en el periodo {d['periodo']}.", "cursos_en_curso")

    # horario de un día (1)
    e = elegir(reg)
    ses = a.horario(e)["datos"]["sesiones"]
    dia = rng.choice(sorted({s["dia"] for s in ses}, key=DIAS.index))
    del_dia = [s for s in ses if s["dia"] == dia]
    lista = "; ".join(f"{s['nombre']} de {s['hora_inicio']} a {s['hora_fin']} en el aula {s['aula']}" for s in del_dia)
    add("horario", e, f"¿Qué clases tengo el {dia.lower()}?", f"El {dia.lower()} tiene: {lista}.", "horario")

    # prerrequisito de un curso (1)
    cod = rng.choice(sorted(a.prereq))
    req = ", ".join(a.cursos[r]["nombre"] for r in sorted(a.prereq[cod]))
    add("prerrequisitos", elegir(reg), f"¿Qué requisito tiene el curso {a.cursos[cod]['nombre']}?",
        f"{a.cursos[cod]['nombre']} tiene como prerrequisito: {req}.", "prerrequisitos")

    # ¿puede llevar? (2): un sí y un no por prerrequisito
    si, no = [], []
    for e in reg:
        for cod in a.prereq:
            d = a.puede_llevar(e, cod)["datos"]
            (si if d["puede"] else no if d["faltan"] else []).append((e, cod, d))
    for lista, resp in [(si, True), (no, False)]:
        e, cod, d = elegir(lista)
        nombre = a.cursos[cod]["nombre"]
        req = ", ".join(a.cursos[r]["nombre"] for r in sorted(a.prereq[cod]))
        gt = (f"Sí puede matricularse en {nombre}: ya aprobó su prerrequisito ({req})." if resp else
              f"No puede matricularse en {nombre}: le falta aprobar el prerrequisito "
              f"{', '.join(a.cursos[f]['nombre'] for f in d['faltan'])}.")
        add("puede_llevar", e, f"¿Puedo matricularme en {nombre} el próximo ciclo?", gt, "puede_llevar")
    return out


NORMATIVAS = [
    ("nota_aprobatoria", "¿Cuál es la nota mínima para aprobar un curso en pregrado?",
     "En pregrado la evaluación es vigesimal y la nota mínima aprobatoria es once (11).", ["RG UNE art. 292"]),
    ("repitencia", "¿Qué pasa si desapruebo la misma asignatura tres veces?",
     "Desaprobar la misma materia tres veces da lugar a la separación temporal del estudiante por un año. "
     "Al término de ese plazo solo puede matricularse en la asignatura que desaprobó, para retornar de manera "
     "regular en el ciclo siguiente. Si la desaprueba por cuarta vez, procede su retiro definitivo.",
     ["RG UNE art. 282", "Ley 30220 art. 102"]),
    ("reserva", "¿Puedo reservar mi matrícula y por cuánto tiempo?",
     "Sí. Quien no pueda matricularse conserva sus derechos hasta por tres años consecutivos o alternos, previa "
     "solicitud al Decano de la Facultad. Concluido el plazo puede pedir la actualización de matrícula y debe "
     "adecuarse al plan de estudios vigente.", ["RG UNE art. 289"]),
    ("desmatricula", "¿Puedo retirarme de un curso después de matricularme?",
     "Sí, mediante la desmatrícula de una o más asignaturas, siempre que mantenga como mínimo 12 créditos para "
     "seguir siendo estudiante regular. La autoriza la Facultad por resolución, con informe del Director de la "
     "Escuela Profesional, dentro de los 30 días calendario de iniciadas las clases.", ["RG UNE art. 290"]),
    ("tope_creditos", "¿Cuál es el máximo de créditos en que me puedo matricular por semestre según el reglamento?",
     "Según el Reglamento General, el tope máximo es de veintidós (22) créditos por semestre.", ["RG UNE art. 283"]),
    ("rectificacion", "¿Hasta cuándo puedo hacer una rectificación de matrícula?",
     "La matrícula extemporánea y la rectificación de matrícula se realizan antes del inicio de clases, en las "
     "fechas del calendario académico, y no hay prórroga.", ["RG UNE art. 288"]),
    ("bachiller", "¿Qué requisitos necesito para obtener el grado de bachiller?",
     "Se requiere haber aprobado los estudios de pregrado y acreditar el conocimiento de un idioma extranjero, de "
     "preferencia inglés, o de una lengua nativa. Los estudios de pregrado incluyen un curso de trabajo de "
     "investigación en el último semestre.", ["RG UNE art. 125.1 (mod. Res. 1887-2024-R-UNE)", "Ley 30220 art. 45"]),
    ("titulo", "¿Qué necesito para obtener el título profesional?",
     "Se requiere el grado académico de bachiller y la aprobación de una tesis o de un trabajo de suficiencia "
     "profesional. El título se otorga a quienes obtuvieron el bachiller en la UNE.", ["RG UNE art. 125.2"]),
    ("sumilla", "¿De qué trata el curso de Sistemas Digitales II?",
     "Es un curso teórico-práctico y obligatorio de 4 créditos. Desarrolla la capacidad de implementar sistemas "
     "lógicos secuenciales programados en VHDL y usando tarjetas FPGA. Sus contenidos incluyen lógica secuencial, "
     "flip-flops, contadores, máquinas de estado, registros, memorias, convertidores análogo-digital y "
     "digital-análogo, lenguaje ensamblador y aplicaciones en VHDL.", ["Plan de estudios, sumilla TEEI0539"]),
    ("electivos", "¿Cuántos cursos electivos debo llevar en el séptimo ciclo?",
     "En el VII ciclo el estudiante elige y se matricula en un solo curso electivo de especialidad, de 2 créditos.",
     ["Plan de estudios, reglas de matrícula (VII ciclo)"]),
]


def mixtas(a, rng):
    q = a.q
    out = []
    reg = [r[0] for r in q("SELECT id FROM estudiante WHERE estado='matriculado regular' ORDER BY id")]

    # M01: suspendido por el art. 282
    e = q("SELECT id FROM estudiante WHERE estado='suspendido' ORDER BY id")[0][0]
    f = a.desaprobados(e)["datos"]["cursos"][0]
    out.append(item("M01", "mixta", "repitencia_282", e,
                    f"Desaprobé {f['nombre']} varias veces, ¿qué consecuencia tiene eso para mí?",
                    f"Desaprobó {f['nombre']} {f['veces']} veces. Según el Reglamento General, la desaprobación de una "
                    "misma materia por tres veces da lugar a la separación temporal por un año; al término de ese plazo "
                    "solo podrá matricularse en esa asignatura y, si la desaprueba por cuarta vez, procede su retiro "
                    "definitivo. Su estado actual es suspendido (separado temporalmente).",
                    ["desaprobados", "estado_matricula", "RG UNE art. 282"], False))

    # M02: está repitiendo un curso (lo lleva en curso tras desaprobarlo)
    cand = [(e, c) for e in reg for c in a.desaprobados(e)["datos"]["cursos"] if c["situacion_actual"] == "en curso"]
    e, c = rng.choice(cand)
    restantes = 3 - c["veces"]
    out.append(item("M02", "mixta", "repitencia_riesgo", e,
                    f"Estoy volviendo a llevar {c['nombre']}. ¿Qué pasa si la desapruebo otra vez?",
                    f"Ya desaprobó {c['nombre']} {c['veces']} {'vez' if c['veces'] == 1 else 'veces'} y la está "
                    f"cursando de nuevo. Si la desaprueba otra vez acumularía {c['veces'] + 1}. Según el Reglamento "
                    "General, a la tercera desaprobación de una misma materia corresponde la separación temporal por un "
                    "año, y a la cuarta, el retiro definitivo."
                    + (" Le quedaría una sola desaprobación antes de la separación." if restantes == 2 else ""),
                    ["desaprobados", "cursos_en_curso", "RG UNE art. 282"], False))

    # M03: egresado y bachiller
    e = q("SELECT id FROM estudiante WHERE estado='egresado' ORDER BY id")[0][0]
    d = a.avance_curricular(e)["datos"]
    out.append(item("M03", "mixta", "bachiller_egresado", e, "¿Ya cumplo los requisitos para tramitar el bachiller?",
                    f"Cumple el requisito de haber aprobado los estudios de pregrado: aprobó los {d['creditos_aprobados']:.0f} "
                    f"de {d['creditos_totales']:.0f} créditos del plan. Además, debe acreditar el conocimiento de un idioma "
                    "extranjero, de preferencia inglés, o de una lengua nativa; ese dato no figura en su registro académico.",
                    ["avance_curricular", "RG UNE art. 125.1"], False))

    # M04: desmatrícula de un curso concreto y mínimo de 12 créditos
    e = rng.choice([x for x in reg if a.cursos_en_curso(x)["datos"]["creditos"] >= 14])
    d = a.cursos_en_curso(e)["datos"]
    cod = rng.choice(d["cursos"])
    cr, nombre = a.cursos[cod]["creditos"], a.cursos[cod]["nombre"]
    resto = d["creditos"] - cr
    out.append(item("M04", "mixta", "desmatricula_regular", e,
                    f"Si me retiro de {nombre}, ¿sigo siendo estudiante regular?",
                    f"Lleva {d['creditos']:.0f} créditos; {nombre} tiene {cr:.0f}, así que le quedarían {resto:.0f}. "
                    f"El reglamento exige mantener como mínimo 12 créditos para seguir siendo regular, por lo que "
                    f"{'sí seguiría siendo regular' if resto >= 12 else 'dejaría de ser regular'}. La desmatrícula debe "
                    "solicitarse dentro de los 30 días calendario de iniciadas las clases.",
                    ["cursos_en_curso", "RG UNE art. 290"], False))

    # M05: créditos que lleva frente al tope del reglamento
    e = rng.choice([x for x in reg if a._estudiante(x)["ciclo_actual"] not in (5, 6)])
    d = a.cursos_en_curso(e)["datos"]
    out.append(item("M05", "mixta", "tope_creditos", e,
                    "¿Cuántos créditos estoy llevando y cuál es el máximo que permite el reglamento?",
                    f"Lleva {d['creditos']:.0f} créditos en el periodo {d['periodo']}. El Reglamento General fija un tope "
                    f"máximo de 22 créditos por semestre"
                    + (f", por lo que su carga {'supera' if d['creditos'] > 22 else 'está dentro de'} ese tope." ),
                    ["cursos_en_curso", "RG UNE art. 283"], False))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB_POR_DEFECTO))
    ap.add_argument("--out", default=str(CARPETA / "testbed.json"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    a = Academico(args.db)
    tb = personales(a, rng)
    tb += [item(f"N{i:02d}", "normativa", cat, None, p, gt, f, False) for i, (cat, p, gt, f) in enumerate(NORMATIVAS, 1)]
    tb += mixtas(a, rng)
    for t in tb:
        if t["id"] in REVISADOS:
            t["revisado"] = True
            t["validacion"] = VALIDACION
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(tb, fh, ensure_ascii=False, indent=2)
    with open(args.out.replace(".json", "_revision.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["id", "tipo", "categoria", "estudiante", "pregunta", "ground_truth", "fuentes", "revisado", "comentario_autor"])
        for t in tb:
            w.writerow([t["id"], t["tipo"], t["categoria"], t["estudiante_id"] or "", t["pregunta"], t["ground_truth"],
                        " | ".join(t["fuentes_esperadas"]), "sí" if t["revisado"] else "PENDIENTE", ""])
    por_tipo = {}
    for t in tb:
        por_tipo[t["tipo"]] = por_tipo.get(t["tipo"], 0) + 1
    print(f"Testbed: {len(tb)} pares {por_tipo} | pendientes de revisión: {sum(not t['revisado'] for t in tb)}")
    print(f"Guardado: {args.out} y {args.out.replace('.json', '_revision.csv')}")
