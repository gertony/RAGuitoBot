"""Construye academico.db: base relacional simulada para el agente RAG híbrido.

Fuentes
- UNE (PNDA, 2024): matrículas del programa "Educación con especialidad de Electrónica e Informática",
  régimen REGULAR. Cada fila es una matrícula en un periodo; el UUID NO identifica a la persona en el
  tiempo, por lo que se usa la "foto" de la matrícula (ciclo, cursos, créditos, sexo, promoción, sede).
- Realinho et al. (2022): tasa de aprobación por curso y distribución de notas (escala 0-20) de los
  estudiantes que no abandonaron (Target = Graduate o Enrolled).
- Plan de estudios UNE (Res. N.° 0326-2020-D-FATEC): datos/malla_ei.csv.
- Reglamento General UNE: aprobatoria 11 (art. 292); tres desaprobaciones -> separación (art. 282).

Uso:
  python build_db.py --une une_matriculados.csv --realinho realinho.csv --malla malla_ei.csv --out academico.db
"""
import argparse
import math
import random
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

PROGRAMA = "EDUCACIÓN CON ESPECIALIDAD DE ELECTRÓNICA E INFORMÁTICA"
PERIODO_ACTUAL = 20241          # último periodo del dataset UNE = "hoy" del sistema
NOTA_APROBATORIA = 11           # art. 292
MAX_DESAPROBACIONES = 2         # la 3.ª activa el art. 282; en el historial se modela solo en los casos "suspendido"
NOTA_DESAPROBADA = (5, 10)      # supuesto declarado: Realinho no informa notas de cursos desaprobados
DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
HORA_INICIO, HORA_FIN = 8, 22
EXTRA = {"egresado": 4, "retirado": 4, "suspendido": 2}
TOPE_POR_CICLO = {5: 23, 6: 23}  # plan de estudios: "en la matrícula del V/VI ciclo, no excederse en más de 23 créditos"

ESQUEMA = """
CREATE TABLE meta (clave TEXT PRIMARY KEY, valor TEXT NOT NULL);
CREATE TABLE carrera (id INTEGER PRIMARY KEY, nombre TEXT NOT NULL, institucion TEXT NOT NULL,
    facultad TEXT NOT NULL, plan TEXT NOT NULL, creditos_totales REAL NOT NULL);
CREATE TABLE curso (codigo TEXT PRIMARY KEY, nombre TEXT NOT NULL, ciclo INTEGER NOT NULL,
    creditos REAL NOT NULL, ht INTEGER NOT NULL, hp INTEGER NOT NULL,
    electivo INTEGER NOT NULL, grupo_electivo TEXT);
CREATE TABLE prerrequisito (curso TEXT NOT NULL REFERENCES curso(codigo),
    requiere TEXT NOT NULL REFERENCES curso(codigo), PRIMARY KEY (curso, requiere));
CREATE TABLE estudiante (id TEXT PRIMARY KEY, uuid_une TEXT NOT NULL, sexo TEXT, promocion INTEGER,
    modalidad TEXT, sede TEXT, ciclo_actual INTEGER NOT NULL, periodo_referencia TEXT NOT NULL,
    cursos_matriculados_une INTEGER, creditos_matriculados_une INTEGER,
    estado TEXT NOT NULL CHECK (estado IN ('matriculado regular','egresado','retirado','suspendido')),
    origen_estado TEXT NOT NULL);
CREATE TABLE matricula (id INTEGER PRIMARY KEY, estudiante_id TEXT NOT NULL REFERENCES estudiante(id),
    curso TEXT NOT NULL REFERENCES curso(codigo), periodo TEXT NOT NULL, intento INTEGER NOT NULL,
    nota INTEGER CHECK (nota BETWEEN 0 AND 20),
    condicion TEXT NOT NULL CHECK (condicion IN ('aprobado','desaprobado','en curso')));
CREATE TABLE horario (id INTEGER PRIMARY KEY, curso TEXT NOT NULL REFERENCES curso(codigo),
    periodo TEXT NOT NULL, dia TEXT NOT NULL, hora_inicio TEXT NOT NULL, hora_fin TEXT NOT NULL, aula TEXT NOT NULL);
CREATE TABLE log_consultas (id INTEGER PRIMARY KEY, ts TEXT NOT NULL, estudiante_id TEXT,
    ruta TEXT, pregunta TEXT NOT NULL, respuesta TEXT, contextos TEXT, latencia_ms REAL);
CREATE INDEX ix_matricula_est ON matricula(estudiante_id, periodo);
"""


# ---------------------------------------------------------------- periodos
def etiqueta(p):
    return f"{p // 10}-{'I' if p % 10 == 1 else 'II'}"


def mover(p, n):
    """Desplaza un periodo YYYYS n semestres (n negativo = hacia atrás)."""
    idx = (p // 10) * 2 + (p % 10 - 1) + n
    return (idx // 2) * 10 + idx % 2 + 1


# ---------------------------------------------------------------- carga
def cargar_une(ruta):
    d = pd.read_csv(ruta, sep=";", encoding="cp1252", low_memory=False)
    d = d[(d.PROGRAMA_ESTUDIOS == PROGRAMA) & (d.DETALLE_TIPO_ESTUDIO == "REGULAR")]
    n0 = len(d)
    # criterio de exclusión 3.3.4: campos vacíos o inconsistentes (CICLO vacío o 99, sin cursos/créditos)
    d = d.dropna(subset=["CICLO", "CANTCURMAT", "CANTCREDMAT", "UUID"])
    d = d[(d.CICLO.between(1, 10)) & (d.CANTCURMAT > 0)].copy()
    print(f"UNE: {n0} matrículas REGULAR del programa; excluidas {n0 - len(d)} por campos vacíos o inconsistentes")
    d["CICLO"] = d.CICLO.astype(int)
    return d


def parametros_realinho(ruta):
    r = pd.read_csv(ruta, sep=";")
    r.columns = r.columns.str.strip()
    r = r[r.Target.isin(["Graduate", "Enrolled"])]
    cols = lambda c: r[f"Curricular units 1st sem ({c})"] + r[f"Curricular units 2nd sem ({c})"]
    tasa = cols("approved").sum() / cols("enrolled").sum()
    notas = pd.concat([r["Curricular units 1st sem (grade)"], r["Curricular units 2nd sem (grade)"]])
    notas = notas[notas > 0]
    return {"tasa_aprobacion": float(tasa), "media": float(notas.mean()), "de": float(notas.std()), "n": int(len(notas))}


def cargar_malla(ruta):
    m = pd.read_csv(ruta)
    m["electivo"] = m.electivo.astype(str).eq("True")
    m["grupo"] = m.grupo.fillna("")
    cursos = {r.codigo: r._asdict() for r in m.itertuples(index=False)}
    prereq = defaultdict(set)
    for r in m.itertuples(index=False):
        if isinstance(r.requisito_codigo, str) and r.requisito_codigo:
            prereq[r.codigo].add(r.requisito_codigo)
    grupos = defaultdict(list)
    for c in cursos.values():
        if c["electivo"]:
            grupos[c["grupo"]].append(c["codigo"])
    return cursos, prereq, grupos


# ---------------------------------------------------------------- simulación
class Simulador:
    def __init__(self, cursos, prereq, grupos, params, tope_creditos, rng, franjas=None):
        self.cursos, self.prereq, self.grupos = cursos, prereq, grupos
        self.franjas = franjas or {}  # curso -> [(dia, hora_ini, hora_fin)] del periodo actual
        self.p, self.tope, self.rng = params, tope_creditos, rng

    def nota_aprobada(self):
        n = round(self.rng.gauss(self.p["media"], self.p["de"]))
        return min(20, max(NOTA_APROBATORIA, n))

    def pendientes(self, aprobados, eleccion, hasta_ciclo, en_periodo):
        """Cursos que el estudiante puede llevar: no aprobados, prerrequisitos aprobados antes, ciclo <= hasta_ciclo."""
        cand = []
        for c in self.cursos.values():
            if c["codigo"] in aprobados or c["ciclo"] > hasta_ciclo or c["codigo"] in en_periodo:
                continue
            if c["electivo"]:
                g = c["grupo"]
                if any(o in aprobados for o in self.grupos[g]):
                    continue
                if g in eleccion and eleccion[g] != c["codigo"]:
                    continue
            if self.prereq[c["codigo"]] <= aprobados:
                cand.append(c)
        # primero lo atrasado (ciclo más bajo); dentro de un grupo electivo, una sola opción al azar
        self.rng.shuffle(cand)
        cand.sort(key=lambda c: c["ciclo"])
        vistos, salida = set(), []
        for c in cand:
            if c["electivo"]:
                if c["grupo"] in vistos:
                    continue
                vistos.add(c["grupo"])
            salida.append(c)
        return salida

    def cruza(self, codigo, elegidos):
        return any(d1 == d2 and a1 < b2 and a2 < b1
                   for d1, a1, b1 in self.franjas.get(codigo, [])
                   for e in elegidos for d2, a2, b2 in self.franjas.get(e, []))

    def elegir(self, aprobados, eleccion, hasta_ciclo, n_cursos=None, forzar_todo=False, sin_cruces=False):
        elegidos, creditos = [], 0.0
        for c in self.pendientes(aprobados, eleccion, hasta_ciclo, set()):
            if n_cursos is not None and len(elegidos) >= n_cursos:
                break
            if not forzar_todo and creditos + c["creditos"] > TOPE_POR_CICLO.get(hasta_ciclo, self.tope):
                continue
            if sin_cruces and self.cruza(c["codigo"], elegidos):  # no se permite matricular cursos que se cruzan
                continue
            elegidos.append(c["codigo"])
            creditos += c["creditos"]
            if c["electivo"]:
                eleccion[c["grupo"]] = c["codigo"]
        return elegidos

    def historial(self, ciclo_ref, periodo_ref, estado, n_cursos_actual):
        """Genera matrículas: un periodo por ciclo previo + el periodo de referencia."""
        aprobados, eleccion, intentos, filas = set(), {}, defaultdict(int), []
        for k in range(1, ciclo_ref + 1):
            periodo = mover(periodo_ref, k - ciclo_ref)
            ultimo = k == ciclo_ref
            if ultimo and estado == "matriculado regular":
                for cod in self.elegir(aprobados, eleccion, k, n_cursos=n_cursos_actual, sin_cruces=True):
                    intentos[cod] += 1
                    filas.append((cod, periodo, intentos[cod], None, "en curso"))
                break
            forzar = ultimo and estado == "egresado"  # el egresado cierra el plan en su último periodo
            for cod in self.elegir(aprobados, eleccion, k, forzar_todo=forzar):
                intentos[cod] += 1
                if forzar:
                    aprueba = True
                else:
                    aprueba = self.rng.random() < self.p["tasa_aprobacion"] or intentos[cod] > MAX_DESAPROBACIONES
                nota = self.nota_aprobada() if aprueba else self.rng.randint(*NOTA_DESAPROBADA)
                filas.append((cod, periodo, intentos[cod], nota, "aprobado" if aprueba else "desaprobado"))
                if aprueba:
                    aprobados.add(cod)
            if forzar:  # cursos que no entraron en la ronda anterior por estar bloqueados por prerrequisitos
                while True:
                    resto = self.elegir(aprobados, eleccion, 10, forzar_todo=True)
                    if not resto:
                        break
                    for cod in resto:
                        intentos[cod] += 1
                        filas.append((cod, periodo, intentos[cod], self.nota_aprobada(), "aprobado"))
                        aprobados.add(cod)
        return filas



def generar_suspendido(sim, ciclo_ref, periodo_ref):
    """Caso del art. 282: un curso desaprobado 3 veces, la última en el periodo de referencia.

    Se simula el historial y se toma el primer curso desaprobado al menos dos periodos antes del de referencia;
    sus intentos se reescriben como 3 desaprobaciones y se eliminan los cursos que dependen de él (directa o
    indirectamente), porque no pudieron llevarse sin aprobarlo.
    """
    for _ in range(500):
        filas = sim.historial(ciclo_ref, periodo_ref, "retirado", None)
        limite = mover(periodo_ref, -1)
        candidatos = [f for f in filas if f[4] == "desaprobado" and f[1] < limite]
        for cand in candidatos:
            cod, p0 = cand[0], cand[1]
            dependientes, frontera = set(), {cod}
            while frontera:
                nuevos = {c for c, req in sim.prereq.items() if req & frontera} - dependientes
                dependientes |= nuevos
                frontera = nuevos
            base = [f for f in filas if f[0] != cod and f[0] not in dependientes]
            seq = [p0, mover(p0, 1), periodo_ref]
            carga = defaultdict(float)
            for f in base:
                carga[f[1]] += sim.cursos[f[0]]["creditos"]
            ciclo_de = lambda p: ciclo_ref - (orden_periodo(etiqueta(periodo_ref)) - orden_periodo(etiqueta(p)))
            if all(carga[p] + sim.cursos[cod]["creditos"] <= TOPE_POR_CICLO.get(ciclo_de(p), sim.tope) for p in seq):
                return base + [(cod, p, i + 1, sim.rng.randint(*NOTA_DESAPROBADA), "desaprobado")
                               for i, p in enumerate(seq)], cod
    raise RuntimeError("No se pudo generar un caso suspendido")


# ---------------------------------------------------------------- horarios
def generar_horarios(cursos, periodo, rng):
    """2 sesiones semanales por curso, sin cruces dentro de un mismo ciclo (opciones electivas incluidas)."""
    filas = []
    por_ciclo = defaultdict(list)
    for c in cursos.values():
        por_ciclo[c["ciclo"]].append(c)
    for ciclo, lista in sorted(por_ciclo.items()):
        ocupado = defaultdict(set)  # dia -> horas ocupadas
        rng.shuffle(lista)
        for n, c in enumerate(sorted(lista, key=lambda x: -(x["ht"] + x["hp"]))):
            total = int(c["ht"] + c["hp"])
            sesiones = [math.ceil(total / 2), total // 2] if total > 1 else [total]
            dias = rng.sample(DIAS, len(DIAS))
            usados = []
            for dur in sesiones:
                for dia in dias:
                    if dia in usados:
                        continue
                    libres = [h for h in range(HORA_INICIO, HORA_FIN - dur + 1)
                              if not set(range(h, h + dur)) & ocupado[dia]]
                    if libres:
                        h = rng.choice(libres)
                        ocupado[dia] |= set(range(h, h + dur))
                        usados.append(dia)
                        filas.append((c["codigo"], etiqueta(periodo), dia, f"{h:02d}:00", f"{h + dur:02d}:00",
                                      f"T-{ciclo}{n + 1:02d}"))
                        break
                else:
                    raise RuntimeError(f"Sin hueco para {c['codigo']}")
    return filas


# ---------------------------------------------------------------- principal
def construir(args):
    rng = random.Random(args.seed)
    une = cargar_une(args.une)
    params = parametros_realinho(args.realinho)
    cursos, prereq, grupos = cargar_malla(args.malla)
    tope = float(une.CANTCREDMAT.max())  # tope observado en el programa (art. 283 [VERIFICAR])
    horarios = generar_horarios(cursos, PERIODO_ACTUAL, rng)
    franjas = defaultdict(list)
    for cod, _, dia, ini, fin, _ in horarios:
        franjas[cod].append((dia, int(ini[:2]), int(fin[:2])))
    sim = Simulador(cursos, prereq, grupos, params, tope, rng, franjas)

    # muestra estratificada: matrículas REGULAR del periodo actual, n por ciclo
    actual = une[une.PERIODO == PERIODO_ACTUAL]
    print("Matrículas 2024-I por ciclo:", actual.CICLO.value_counts().sort_index().to_dict())
    muestra = pd.concat([g.sample(min(args.n_por_ciclo, len(g)), random_state=args.seed)
                         for _, g in actual.groupby("CICLO")])
    faltan = args.n_regulares - len(muestra)  # ciclos con pocos estudiantes en 2024-I: se completa al azar
    if faltan > 0:
        muestra = pd.concat([muestra, actual.drop(muestra.index).sample(faltan, random_state=args.seed)])
    previos = une[une.PERIODO.isin([mover(PERIODO_ACTUAL, -1), mover(PERIODO_ACTUAL, -2)])]
    extra = {
        "egresado": previos[previos.CICLO == 10].sample(EXTRA["egresado"], random_state=args.seed),
        "retirado": previos[previos.CICLO.between(2, 8)].sample(EXTRA["retirado"], random_state=args.seed + 1),
        "suspendido": previos[previos.CICLO.between(3, 8)].sample(EXTRA["suspendido"], random_state=args.seed + 2),
    }

    estudiantes, matriculas = [], []
    plan = [("matriculado regular", r, "real: matriculado en 2024-I (UNE)") for r in muestra.itertuples()]
    for estado, df in extra.items():
        plan += [(estado, r, "simulado: asignado a una matrícula UNE de 2023") for r in df.itertuples()]

    for i, (estado, r, origen) in enumerate(plan, start=1):
        eid = f"EST{i:04d}"
        if estado == "suspendido":
            filas, _ = generar_suspendido(sim, r.CICLO, r.PERIODO)
        else:
            filas = sim.historial(r.CICLO, r.PERIODO, estado, int(r.CANTCURMAT))
        estudiantes.append((eid, r.UUID, r.SEXO, int(r.PROMO), r.MODALIDAD, r.SEDE, int(r.CICLO),
                            etiqueta(r.PERIODO), int(r.CANTCURMAT), int(r.CANTCREDMAT), estado, origen))
        matriculas += [(eid, cod, etiqueta(p), it, nota, cond) for cod, p, it, nota, cond in filas]

    out = Path(args.out)
    out.unlink(missing_ok=True)
    con = sqlite3.connect(out)
    con.executescript(ESQUEMA)
    meta = {
        "fecha_construccion": datetime.now().isoformat(timespec="seconds"),
        "semilla": args.seed,
        "periodo_actual": etiqueta(PERIODO_ACTUAL),
        "fuente_estudiantes": "PNDA (2024), matrículas UNE, régimen REGULAR, sin CICLO=99",
        "fuente_notas": "Realinho et al. (2022), Target in {Graduate, Enrolled}",
        "tasa_aprobacion_curso": round(params["tasa_aprobacion"], 4),
        "nota_media": round(params["media"], 2),
        "nota_de": round(params["de"], 2),
        "n_notas_realinho": params["n"],
        "nota_desaprobada_supuesto": f"uniforme entera en {NOTA_DESAPROBADA}",
        "nota_aprobatoria": f"{NOTA_APROBATORIA} (Reglamento General UNE, art. 292)",
        "tope_creditos_periodo": f"{tope:.0f} (máximo observado en UNE; art. 283 [VERIFICAR]); 23 en los ciclos V y VI (plan de estudios)",
        "plan_estudios": "Res. N.° 0326-2020-D-FATEC",
        "horarios": "sintéticos, 2 sesiones/semana, sin cruces dentro de un ciclo",
    }
    con.executemany("INSERT INTO meta VALUES (?,?)", [(k, str(v)) for k, v in meta.items()])
    total = sum(c["creditos"] for c in cursos.values() if not c["electivo"]) + \
        sum(cursos[g[0]]["creditos"] for g in grupos.values())
    con.execute("INSERT INTO carrera VALUES (1,?,?,?,?,?)",
                ("Educación con especialidad de Electrónica e Informática",
                 "Universidad Nacional de Educación Enrique Guzmán y Valle", "Facultad de Tecnología",
                 "Res. N.° 0326-2020-D-FATEC", total))
    con.executemany("INSERT INTO curso VALUES (?,?,?,?,?,?,?,?)",
                    [(c["codigo"], c["nombre"], c["ciclo"], c["creditos"], c["ht"], c["hp"], int(c["electivo"]),
                      c["grupo"] or None) for c in cursos.values()])
    con.executemany("INSERT INTO prerrequisito VALUES (?,?)", [(c, q) for c, req in prereq.items() for q in req])
    con.executemany("INSERT INTO estudiante VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", estudiantes)
    con.executemany("INSERT INTO matricula (estudiante_id,curso,periodo,intento,nota,condicion) VALUES (?,?,?,?,?,?)",
                    matriculas)
    con.executemany("INSERT INTO horario (curso,periodo,dia,hora_inicio,hora_fin,aula) VALUES (?,?,?,?,?,?)",
                    horarios)
    con.commit()
    return con, cursos, prereq, grupos, total


# ---------------------------------------------------------------- verificación
def orden_periodo(et):
    a, s = et.split("-")
    return int(a) * 2 + (0 if s == "I" else 1)


def verificar(con, prereq, grupos, total):
    q = lambda s, *a: con.execute(s, a).fetchall()
    print("\n== Conteos")
    for t in ["curso", "prerrequisito", "estudiante", "matricula", "horario", "log_consultas", "meta"]:
        print(f"  {t:<14} {q(f'SELECT COUNT(*) FROM {t}')[0][0]}")
    print("  estados:", dict(q("SELECT estado, COUNT(*) FROM estudiante GROUP BY estado")))
    print("  estudiantes por ciclo:", dict(q("SELECT ciclo_actual, COUNT(*) FROM estudiante GROUP BY 1")))

    errores = []
    # 1. prerrequisitos aprobados en un periodo anterior
    filas = q("SELECT estudiante_id, curso, periodo, condicion FROM matricula")
    aprob = defaultdict(dict)
    for e, c, p, cond in filas:
        if cond == "aprobado":
            aprob[e][c] = min(orden_periodo(p), aprob[e].get(c, 10**6))
    for e, c, p, cond in filas:
        for req in prereq.get(c, ()):
            if aprob[e].get(req, 10**6) >= orden_periodo(p):
                errores.append(f"{e}: {c} en {p} sin {req} aprobado antes")
    # 2. notas coherentes
    malas = q("SELECT COUNT(*) FROM matricula WHERE (condicion='aprobado' AND nota<11) OR "
              "(condicion='desaprobado' AND nota>=11) OR (condicion='en curso' AND nota IS NOT NULL)")[0][0]
    if malas:
        errores.append(f"{malas} notas incoherentes con su condición")
    # 3. egresados con el plan completo
    for (e,) in q("SELECT id FROM estudiante WHERE estado='egresado'"):
        cr = q("SELECT SUM(c.creditos) FROM (SELECT DISTINCT curso FROM matricula WHERE estudiante_id=? "
               "AND condicion='aprobado') m JOIN curso c ON c.codigo=m.curso", e)[0][0]
        if abs(cr - total) > 1e-6:
            errores.append(f"{e} egresado con {cr} de {total} créditos")
    # 4. 'en curso' solo para matriculados regulares
    if q("SELECT COUNT(*) FROM matricula m JOIN estudiante s ON s.id=m.estudiante_id "
         "WHERE m.condicion='en curso' AND s.estado!='matriculado regular'")[0][0]:
        errores.append("hay cursos en curso en estudiantes no regulares")
    # 5. art. 282: tres desaprobaciones solo en suspendidos
    tres = q("SELECT s.estado, COUNT(*) FROM (SELECT estudiante_id, curso FROM matricula WHERE condicion='desaprobado' "
             "GROUP BY 1,2 HAVING COUNT(*)>=3) x JOIN estudiante s ON s.id=x.estudiante_id GROUP BY 1")
    if any(est != "suspendido" for est, _ in tres):
        errores.append(f"3 desaprobaciones fuera de suspendidos: {tres}")
    # 6. una sola opción por grupo electivo aprobada
    for g, ops in grupos.items():
        marcas = ",".join("?" * len(ops))
        dup = q(f"SELECT estudiante_id FROM matricula WHERE condicion='aprobado' AND curso IN ({marcas}) "
                f"GROUP BY 1 HAVING COUNT(DISTINCT curso)>1", *ops)
        if dup:
            errores.append(f"grupo {g}: más de una opción aprobada en {len(dup)} estudiantes")

    # 7. tope de créditos por ciclo (23 en V y VI según el plan), salvo el cierre de los egresados
    for e, ciclo_act, ref, est in q("SELECT id, ciclo_actual, periodo_referencia, estado FROM estudiante"):
        for p, cr in q("SELECT m.periodo, SUM(c.creditos) FROM matricula m JOIN curso c ON c.codigo=m.curso "
                       "WHERE m.estudiante_id=? GROUP BY m.periodo", e):
            k = ciclo_act - (orden_periodo(ref) - orden_periodo(p))
            if est == "egresado" and p == ref:
                continue
            if cr > TOPE_POR_CICLO.get(k, 26):
                errores.append(f"{e}: {cr} cr en {p} (ciclo {k}) supera el tope")

    print("\n== Indicadores")
    print("  aprobación simulada:", q("SELECT ROUND(AVG(condicion='aprobado'),4) FROM matricula WHERE condicion!='en curso'")[0][0])
    print("  nota media aprobados:", q("SELECT ROUND(AVG(nota),2), MIN(nota), MAX(nota) FROM matricula WHERE condicion='aprobado'")[0])
    cr = q("SELECT MAX(t) FROM (SELECT SUM(c.creditos) t FROM matricula m JOIN curso c ON c.codigo=m.curso "
           "JOIN estudiante s ON s.id=m.estudiante_id WHERE NOT (s.estado='egresado' AND m.periodo=s.periodo_referencia) "
           "GROUP BY m.estudiante_id, m.periodo)")[0][0]
    print("  máx. créditos en un periodo (sin el cierre de egresados):", cr)
    dif = q("SELECT AVG(ABS(n - s.cursos_matriculados_une)) FROM (SELECT estudiante_id, COUNT(*) n FROM matricula "
            "WHERE condicion='en curso' GROUP BY 1) m JOIN estudiante s ON s.id=m.estudiante_id")[0][0]
    print("  diferencia media n.º de cursos en curso vs UNE:", round(dif or 0, 2))
    cruces = 0
    for (e,) in q("SELECT id FROM estudiante WHERE estado='matriculado regular'"):
        h = q("SELECT h.dia, h.hora_inicio, h.hora_fin FROM matricula m JOIN horario h ON h.curso=m.curso "
              "WHERE m.estudiante_id=? AND m.condicion='en curso'", e)
        franjas = defaultdict(list)
        for d, a, b in h:
            franjas[d].append((int(a[:2]), int(b[:2])))
        cruces += any(x[1] > y[0] for v in franjas.values() for x, y in zip(sorted(v), sorted(v)[1:]))
    print("  estudiantes con cruce de horario (por arrastrar cursos de otro ciclo):", cruces)

    print("\n== Verificación:", "OK" if not errores else f"{len(errores)} ERRORES")
    for err in errores[:15]:
        print("  -", err)
    return not errores


def ficha(con, eid):
    q = lambda s, *a: con.execute(s, a).fetchall()
    e = q("SELECT id, estado, ciclo_actual, periodo_referencia, sexo, promocion, sede FROM estudiante WHERE id=?", eid)[0]
    print(f"\n== Ficha {e[0]} | {e[1]} | ciclo {e[2]} | ref. {e[3]} | sexo {e[4]} | promoción {e[5]} | sede {e[6]}")
    pp = q("SELECT SUM(m.nota*c.creditos)/SUM(c.creditos) FROM matricula m JOIN curso c ON c.codigo=m.curso "
           "WHERE m.estudiante_id=? AND m.nota IS NOT NULL", eid)[0][0]
    ap = q("SELECT SUM(c.creditos) FROM (SELECT DISTINCT curso FROM matricula WHERE estudiante_id=? AND condicion='aprobado') m "
           "JOIN curso c ON c.codigo=m.curso", eid)[0][0]
    print(f"  promedio ponderado: {pp:.2f} | créditos aprobados: {ap:.0f}")
    for p, cod, nom, nota, cond, it in q(
            "SELECT m.periodo, m.curso, c.nombre, m.nota, m.condicion, m.intento FROM matricula m JOIN curso c "
            "ON c.codigo=m.curso WHERE m.estudiante_id=? ORDER BY m.periodo DESC, c.ciclo LIMIT 12", eid):
        print(f"  {p:<8} {cod:<9} {nom[:38]:<38} {nota if nota is not None else '-':>3}  {cond}{' (intento ' + str(it) + ')' if it > 1 else ''}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--une", required=True)
    ap.add_argument("--realinho", required=True)
    ap.add_argument("--malla", required=True)
    ap.add_argument("--out", default="academico.db")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-por-ciclo", type=int, default=6)
    ap.add_argument("--n-regulares", type=int, default=60)
    a = ap.parse_args()
    con, cursos, prereq, grupos, total = construir(a)
    ok = verificar(con, prereq, grupos, total)
    ficha(con, "EST0040")
    print(f"\nBase guardada en {a.out}")
    raise SystemExit(0 if ok else 1)