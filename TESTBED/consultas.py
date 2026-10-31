"""Funciones parametrizadas sobre academico.db (capa de datos personales del agente).

Cada función recibe el ID del estudiante, que viene de la sesión y NUNCA de la pregunta (así el modelo no
puede consultar a otro estudiante), y devuelve un diccionario con:
  - "datos": valores estructurados (sirven para calcular el ground truth del testbed)
  - "texto": 1–3 frases en español que se entregan al LLM como contexto recuperado

Reglas aplicadas
- Aprobatoria: 11 (Reglamento General UNE, art. 292).
- Promedio ponderado (art. 293): por semestre = Σ(créditos × nota) / Σ créditos de las asignaturas
  calificadas en el semestre; total = promedio de los promedios semestrales (293.2.2), con dos decimales
  (293.2.3). [VERIFICAR] el texto de 293.2.1: el OCR dice "entre el número de asignaturas", lo que daría
  valores mayores que 20; se usa el número de créditos.
- Periodo actual: 2024-I (último periodo del dataset UNE).
"""
import difflib
import re
import sqlite3
import unicodedata
from collections import defaultdict
from pathlib import Path

# Estructura del proyecto: RAGASTESIS/{DB, CHROMADB2, TESTBED}. Las rutas se resuelven desde la ubicación
# de este archivo, así funciona sin importar desde qué carpeta se ejecute.
RAIZ = Path(__file__).resolve().parent.parent
DB_POR_DEFECTO = RAIZ / "DB" / "academico.db"

APROBATORIA = 11
ROMANOS = {"1": "i", "2": "ii", "3": "iii", "4": "iv", "5": "v", "6": "vi"}


def _norm(t):
    t = unicodedata.normalize("NFKD", t or "").encode("ascii", "ignore").decode().lower()
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    palabras = [ROMANOS.get(w, w) for w in t.split()]  # "sistemas digitales 2" = "sistemas digitales ii"
    return " ".join(w for w in palabras if w not in {"de", "del", "la", "el", "y", "e", "en", "curso", "asignatura"})


def _orden(periodo):
    a, s = periodo.split("-")
    return int(a) * 2 + (0 if s == "I" else 1)


def _f(x):
    return f"{x:.2f}".replace(".", ",")


class Academico:
    def __init__(self, ruta_db=DB_POR_DEFECTO):
        if not Path(ruta_db).exists():
            raise FileNotFoundError(f"No se encontró la base de datos: {ruta_db}")
        self.con = sqlite3.connect(ruta_db, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.cursos = {r["codigo"]: dict(r) for r in self.q("SELECT * FROM curso")}
        self.prereq = defaultdict(set)
        for r in self.q("SELECT curso, requiere FROM prerrequisito"):
            self.prereq[r["curso"]].add(r["requiere"])
        self.total_creditos = self.q("SELECT creditos_totales FROM carrera")[0][0]
        self._indice = {_norm(c["nombre"]): cod for cod, c in self.cursos.items()}

    def q(self, sql, *args):
        return self.con.execute(sql, args).fetchall()

    # ------------------------------------------------------------ utilidades
    def buscar_curso(self, texto):
        """Ubica un curso mencionado en lenguaje natural. Devuelve el código o None."""
        n = _norm(texto)
        if not n:
            return None
        # 1) el nombre normalizado de algún curso aparece completo en el texto (se prefiere el más largo)
        contenidos = [(len(nom), cod) for nom, cod in self._indice.items() if f" {nom} " in f" {n} "]
        if contenidos:
            return max(contenidos)[1]
        # 2) parecido aproximado (errores de tipeo, nombres incompletos)
        cerca = difflib.get_close_matches(n, self._indice.keys(), n=1, cutoff=0.6)
        if cerca:
            return self._indice[cerca[0]]
        # 3) ventana de palabras del tamaño de cada nombre
        palabras = n.split()
        mejor, cod_mejor = 0.0, None
        for nom, cod in self._indice.items():
            k = len(nom.split())
            for i in range(max(1, len(palabras) - k + 1)):
                r = difflib.SequenceMatcher(None, " ".join(palabras[i:i + k]), nom).ratio()
                if r > mejor:
                    mejor, cod_mejor = r, cod
        return cod_mejor if mejor >= 0.8 else None

    def _estudiante(self, eid):
        r = self.q("SELECT * FROM estudiante WHERE id=?", eid)
        if not r:
            raise KeyError(f"Estudiante {eid} no existe")
        return dict(r[0])

    def _matriculas(self, eid):
        return [dict(r) for r in self.q(
            "SELECT m.*, c.nombre, c.creditos, c.ciclo FROM matricula m JOIN curso c ON c.codigo=m.curso "
            "WHERE m.estudiante_id=? ORDER BY m.periodo, c.ciclo, c.codigo", eid)]

    def _aprobados(self, eid):
        return {m["curso"] for m in self._matriculas(eid) if m["condicion"] == "aprobado"}

    def ultimo_periodo_cerrado(self, eid):
        cerrados = sorted({m["periodo"] for m in self._matriculas(eid) if m["condicion"] != "en curso"}, key=_orden)
        return cerrados[-1] if cerrados else None

    # ------------------------------------------------------------ consultas
    def estado_matricula(self, eid):
        e = self._estudiante(eid)
        detalle = {
            "matriculado regular": f"está matriculado como estudiante regular en el periodo {e['periodo_referencia']}, "
                                   f"en el ciclo {e['ciclo_actual']}",
            "egresado": f"es egresado; concluyó el plan de estudios en el periodo {e['periodo_referencia']}",
            "retirado": f"figura como retirado; su última matrícula fue en el periodo {e['periodo_referencia']} "
                        f"(ciclo {e['ciclo_actual']})",
            "suspendido": f"está separado temporalmente (suspendido) desde el periodo {e['periodo_referencia']}",
        }[e["estado"]]
        datos = {"estado": e["estado"], "ciclo": e["ciclo_actual"], "periodo": e["periodo_referencia"]}
        return {"datos": datos, "texto": f"Estado de matrícula del estudiante {eid}: {detalle}."}

    def promedio_ponderado(self, eid):
        por_sem = defaultdict(lambda: [0.0, 0.0])
        for m in self._matriculas(eid):
            if m["nota"] is not None:
                por_sem[m["periodo"]][0] += m["nota"] * m["creditos"]
                por_sem[m["periodo"]][1] += m["creditos"]
        parciales = {p: round(s / c, 2) for p, (s, c) in sorted(por_sem.items(), key=lambda x: _orden(x[0]))}
        total = round(sum(parciales.values()) / len(parciales), 2) if parciales else None
        if total is None:
            return {"datos": {"total": None, "parciales": {}},
                    "texto": f"El estudiante {eid} aún no tiene asignaturas calificadas."}
        ultimo = list(parciales)[-1]
        texto = (f"Promedio ponderado total del estudiante {eid}: {_f(total)} (promedio de {len(parciales)} semestres "
                 f"calificados). Promedio ponderado del último semestre calificado ({ultimo}): {_f(parciales[ultimo])}.")
        return {"datos": {"total": total, "parciales": parciales, "ultimo_semestre": ultimo}, "texto": texto}

    def avance_curricular(self, eid):
        aprob = self._aprobados(eid)
        cred = sum(self.cursos[c]["creditos"] for c in aprob)
        pct = round(100 * cred / self.total_creditos, 1)
        obligatorios = [c for c in self.cursos.values() if not c["electivo"]]
        grupos = {c["grupo_electivo"] for c in self.cursos.values() if c["electivo"]}
        ob_ok = sum(c["codigo"] in aprob for c in obligatorios)
        gr_ok = sum(any(c["codigo"] in aprob for c in self.cursos.values() if c["grupo_electivo"] == g) for g in grupos)
        datos = {"creditos_aprobados": cred, "creditos_totales": self.total_creditos, "porcentaje": pct,
                 "obligatorios_aprobados": ob_ok, "obligatorios_totales": len(obligatorios),
                 "electivos_cumplidos": gr_ok, "electivos_totales": len(grupos)}
        texto = (f"Avance curricular del estudiante {eid}: {cred:.0f} de {self.total_creditos:.0f} créditos aprobados "
                 f"({str(pct).replace('.', ',')} %). Asignaturas obligatorias aprobadas: {ob_ok} de {len(obligatorios)}; "
                 f"grupos electivos cumplidos: {gr_ok} de {len(grupos)}. Faltan {self.total_creditos - cred:.0f} créditos.")
        return {"datos": datos, "texto": texto}

    def notas_periodo(self, eid, periodo=None):
        periodo = periodo or self.ultimo_periodo_cerrado(eid)
        if periodo is None:
            return {"datos": {"periodo": None, "notas": []},
                    "texto": f"El estudiante {eid} aún no tiene periodos académicos concluidos con notas."}
        filas = [m for m in self._matriculas(eid) if m["periodo"] == periodo and m["nota"] is not None]
        if not filas:
            return {"datos": {"periodo": periodo, "notas": []},
                    "texto": f"El estudiante {eid} no tiene notas registradas en el periodo {periodo}."}
        notas = [{"curso": m["nombre"], "codigo": m["curso"], "nota": m["nota"], "condicion": m["condicion"]} for m in filas]
        lista = "; ".join(f"{n['curso']}: {n['nota']} ({n['condicion']})" for n in notas)
        return {"datos": {"periodo": periodo, "notas": notas},
                "texto": f"Notas del estudiante {eid} en el periodo {periodo}: {lista}."}

    def nota_curso(self, eid, codigo):
        c = self.cursos[codigo]
        intentos = [m for m in self._matriculas(eid) if m["curso"] == codigo]
        if not intentos:
            return {"datos": {"codigo": codigo, "intentos": []},
                    "texto": f"El estudiante {eid} no registra matrícula en {c['nombre']} ({codigo})."}
        partes = [f"{m['periodo']}: {m['nota']} ({m['condicion']})" if m["nota"] is not None
                  else f"{m['periodo']}: en curso, sin nota" for m in intentos]
        datos = {"codigo": codigo, "intentos": [{"periodo": m["periodo"], "nota": m["nota"], "condicion": m["condicion"]}
                                                for m in intentos]}
        return {"datos": datos, "texto": f"Registro del estudiante {eid} en {c['nombre']} ({codigo}), "
                                         f"{len(intentos)} matrícula(s): " + "; ".join(partes) + "."}

    def cursos_en_curso(self, eid):
        filas = [m for m in self._matriculas(eid) if m["condicion"] == "en curso"]
        if not filas:
            return {"datos": {"cursos": [], "creditos": 0},
                    "texto": f"El estudiante {eid} no tiene asignaturas en curso."}
        cred = sum(m["creditos"] for m in filas)
        lista = "; ".join(f"{m['nombre']} ({m['creditos']:.0f} cr)" for m in filas)
        return {"datos": {"cursos": [m["curso"] for m in filas], "creditos": cred, "periodo": filas[0]["periodo"]},
                "texto": f"Asignaturas en curso del estudiante {eid} en {filas[0]['periodo']} "
                         f"({len(filas)} asignaturas, {cred:.0f} créditos): {lista}."}

    def horario(self, eid):
        filas = self.q("SELECT c.nombre, h.dia, h.hora_inicio, h.hora_fin, h.aula FROM matricula m "
                       "JOIN horario h ON h.curso=m.curso AND h.periodo=m.periodo JOIN curso c ON c.codigo=m.curso "
                       "WHERE m.estudiante_id=? AND m.condicion='en curso'", eid)
        if not filas:
            return {"datos": {"sesiones": []}, "texto": f"El estudiante {eid} no tiene horario porque no tiene asignaturas en curso."}
        dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
        filas = sorted(filas, key=lambda r: (dias.index(r["dia"]), r["hora_inicio"]))
        lista = "; ".join(f"{r['dia']} {r['hora_inicio']}-{r['hora_fin']}: {r['nombre']} (aula {r['aula']})" for r in filas)
        return {"datos": {"sesiones": [dict(r) for r in filas]}, "texto": f"Horario del estudiante {eid}: {lista}."}

    def prerrequisitos(self, codigo):
        c = self.cursos[codigo]
        req = sorted(self.prereq.get(codigo, ()))
        if not req:
            return {"datos": {"codigo": codigo, "requisitos": []},
                    "texto": f"{c['nombre']} ({codigo}, ciclo {c['ciclo']}, {c['creditos']:.0f} créditos) no tiene prerrequisitos."}
        nombres = ", ".join(f"{self.cursos[r]['nombre']} ({r})" for r in req)
        return {"datos": {"codigo": codigo, "requisitos": req},
                "texto": f"{c['nombre']} ({codigo}, ciclo {c['ciclo']}, {c['creditos']:.0f} créditos) tiene como "
                         f"prerrequisito: {nombres}."}

    def puede_llevar(self, eid, codigo):
        e = self._estudiante(eid)
        c = self.cursos[codigo]
        aprob = self._aprobados(eid)
        en_curso = set(self.cursos_en_curso(eid)["datos"]["cursos"])
        faltan = sorted(self.prereq.get(codigo, set()) - aprob)
        if codigo in aprob:
            veredicto, motivo = False, "ya la aprobó"
        elif codigo in en_curso:
            veredicto, motivo = False, "la está cursando actualmente"
        elif e["estado"] in ("egresado", "retirado", "suspendido"):
            veredicto, motivo = False, f"su estado es '{e['estado']}'"
        elif faltan:
            nombres = ", ".join(self.cursos[f]["nombre"] for f in faltan)
            veredicto, motivo = False, f"le falta aprobar el prerrequisito: {nombres}"
        else:
            veredicto, motivo = True, "cumple los prerrequisitos y no la ha aprobado"
        texto = (f"{'Sí' if veredicto else 'No'} puede matricularse en {c['nombre']} ({codigo}): {motivo}. "
                 + self.prerrequisitos(codigo)["texto"])
        return {"datos": {"codigo": codigo, "puede": veredicto, "motivo": motivo, "faltan": faltan}, "texto": texto}

    def desaprobados(self, eid):
        cuenta, ult = defaultdict(int), {}
        for m in self._matriculas(eid):
            if m["condicion"] == "desaprobado":
                cuenta[m["curso"]] += 1
            ult[m["curso"]] = m["condicion"]
        if not cuenta:
            return {"datos": {"cursos": []}, "texto": f"El estudiante {eid} no tiene asignaturas desaprobadas."}
        filas = [{"codigo": c, "nombre": self.cursos[c]["nombre"], "veces": n, "situacion_actual": ult[c]}
                 for c, n in sorted(cuenta.items(), key=lambda x: -x[1])]
        lista = "; ".join(f"{f['nombre']}: desaprobada {f['veces']} {'vez' if f['veces'] == 1 else 'veces'} "
                          f"(situación actual: {f['situacion_actual']})"
                          for f in filas)
        return {"datos": {"cursos": filas, "max_veces": filas[0]["veces"]},
                "texto": f"Asignaturas desaprobadas del estudiante {eid}: {lista}."}

    def resumen(self, eid):
        """Ficha breve: se usa cuando la pregunta es general ('¿cómo voy?')."""
        partes = [self.estado_matricula(eid)["texto"], self.promedio_ponderado(eid)["texto"],
                  self.avance_curricular(eid)["texto"]]
        return {"datos": {}, "texto": " ".join(partes)}


if __name__ == "__main__":
    import sys
    a = Academico(sys.argv[1] if len(sys.argv) > 1 else DB_POR_DEFECTO)
    eid = sys.argv[2] if len(sys.argv) > 2 else "EST0040"
    for nombre in ["estado_matricula", "promedio_ponderado", "avance_curricular", "notas_periodo",
                   "cursos_en_curso", "horario", "desaprobados"]:
        print(f"\n[{nombre}]\n{getattr(a, nombre)(eid)['texto']}")
    for t in ["sistemas digitales 2", "Cálculo II", "circuitos electronicos III", "inglés 1", "programacion"]:
        cod = a.buscar_curso(t)
        print(f"\nbuscar_curso('{t}') -> {cod} {a.cursos[cod]['nombre'] if cod else ''}")
    cod = a.buscar_curso("Circuitos Electrónicos III")
    print("\n[puede_llevar]\n" + a.puede_llevar(eid, cod)["texto"])