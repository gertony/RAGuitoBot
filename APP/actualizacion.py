"""CU-10 / RF-11: actualizar o reemplazar notas en academico.db sin reconstruir el sistema."""
import re

import pandas as pd

APROBATORIA = 11  # Reglamento General UNE, art. 292


def validar(df, con):
    errores = []
    faltan = {"estudiante_id", "curso", "periodo", "nota"} - set(df.columns)
    if faltan:
        return [f"Faltan columnas: {', '.join(sorted(faltan))}"]
    est = {r[0] for r in con.execute("SELECT id FROM estudiante")}
    cur = {r[0] for r in con.execute("SELECT codigo FROM curso")}
    for i, r in df.iterrows():
        fila = i + 2
        if r.estudiante_id not in est:
            errores.append(f"Fila {fila}: el estudiante {r.estudiante_id} no existe")
        if r.curso not in cur:
            errores.append(f"Fila {fila}: el curso {r.curso} no existe")
        if not re.fullmatch(r"\d{4}-(I|II)", str(r.periodo)):
            errores.append(f"Fila {fila}: periodo inválido ({r.periodo}); formato 2024-I")
        if pd.isna(r.nota) or not float(r.nota).is_integer() or not 0 <= r.nota <= 20:
            errores.append(f"Fila {fila}: la nota debe ser un entero de 0 a 20")
    return errores


def aplicar(df, con):
    nuevos = reemplazados = 0
    for r in df.itertuples():
        nota = int(r.nota)
        condicion = "aprobado" if nota >= APROBATORIA else "desaprobado"
        existe = con.execute("SELECT id FROM matricula WHERE estudiante_id=? AND curso=? AND periodo=?",
                             (r.estudiante_id, r.curso, r.periodo)).fetchone()
        if existe:
            con.execute("UPDATE matricula SET nota=?, condicion=? WHERE id=?", (nota, condicion, existe[0]))
            reemplazados += 1
        else:
            intento = con.execute("SELECT COUNT(*) FROM matricula WHERE estudiante_id=? AND curso=?",
                                  (r.estudiante_id, r.curso)).fetchone()[0] + 1
            con.execute("INSERT INTO matricula (estudiante_id, curso, periodo, intento, nota, condicion) "
                        "VALUES (?,?,?,?,?,?)", (r.estudiante_id, r.curso, r.periodo, intento, nota, condicion))
            nuevos += 1
    con.commit()
    return nuevos, reemplazados
