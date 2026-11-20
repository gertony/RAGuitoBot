"""Respaldo y restauración de RAGuitoBot (P6: política de respaldo; P10: plan de reversión).

Copia en RESPALDOS/<AAAAMMDD_HHMMSS>/:
  academico.db     copia consistente con la API de respaldo de SQLite (sirve aunque la app esté abierta)
  chroma_e5/       copia de la carpeta del índice normativo
  log_consultas.csv  las consultas registradas (RF-12), para conservar la evidencia de la sesión
  manifiesto.json  conteos por tabla y del índice, para validar la restauración

Uso (desde la raíz del proyecto):
  python DB/respaldo.py crear [--etiqueta antes_piloto]
  python DB/respaldo.py listar
  python DB/respaldo.py restaurar <carpeta>   # antes de restaurar, respalda el estado actual (pre_restauracion)
"""
import argparse
import csv
import json
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DB = RAIZ / "DB" / "academico.db"
CHROMA = RAIZ / "CHROMADB2" / "chroma_e5"
RESPALDOS = RAIZ / "RESPALDOS"
TABLAS = ["estudiante", "matricula", "curso", "prerrequisito", "horario", "usuario", "log_consultas"]


def conteos(db, chroma):
    """Filas por tabla e integridad de la base; fragmentos del índice si chromadb puede abrirlo."""
    con = sqlite3.connect(db)
    try:
        existentes = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        info = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLAS if t in existentes}  # noqa: S608
        info["integridad"] = con.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        con.close()
    try:
        import chromadb
        info["fragmentos"] = chromadb.PersistentClient(path=str(chroma)).get_collection("normativa").count()
    except Exception:  # noqa: BLE001  el índice puede no existir o no tener la colección
        info["fragmentos"] = None
    return info


def crear(etiqueta="", db=DB, chroma=CHROMA, destino_base=RESPALDOS):
    if not Path(db).exists():
        sys.exit(f"No existe {db}")
    nombre = datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_{etiqueta}" if etiqueta else "")
    destino = Path(destino_base) / nombre
    destino.mkdir(parents=True)
    origen = sqlite3.connect(db)
    copia = sqlite3.connect(destino / "academico.db")
    try:
        origen.backup(copia)
        filas = origen.execute("SELECT * FROM log_consultas").fetchall() if origen.execute(
            "SELECT 1 FROM sqlite_master WHERE name='log_consultas'").fetchone() else []
        columnas = [d[0] for d in origen.execute("SELECT * FROM log_consultas").description] if filas else []
    finally:
        copia.close()
        origen.close()
    with open(destino / "log_consultas.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if columnas:
            w.writerow(columnas)
        w.writerows(filas)
    if Path(chroma).exists():
        shutil.copytree(chroma, destino / "chroma_e5")
    manifiesto = conteos(destino / "academico.db", destino / "chroma_e5")
    manifiesto["creado"] = datetime.now().isoformat(timespec="seconds")
    (destino / "manifiesto.json").write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8")
    if manifiesto["integridad"] != "ok":
        sys.exit(f"La copia no pasó PRAGMA integrity_check: {manifiesto['integridad']}")
    print(f"Respaldo creado: {destino}")
    print(json.dumps(manifiesto, ensure_ascii=False))
    return destino


def restaurar(carpeta, db=DB, chroma=CHROMA, destino_base=RESPALDOS):
    origen = Path(carpeta)
    if not origen.is_absolute():
        origen = Path(destino_base) / carpeta
    manifiesto = json.loads((origen / "manifiesto.json").read_text(encoding="utf-8"))
    crear("pre_restauracion", db, chroma, destino_base)  # nunca se pierde el estado actual
    fuente = sqlite3.connect(origen / "academico.db")
    actual = sqlite3.connect(db)
    try:
        fuente.backup(actual)
    finally:
        actual.close()
        fuente.close()
    if (origen / "chroma_e5").exists():
        if Path(chroma).exists():
            shutil.rmtree(chroma)
        shutil.copytree(origen / "chroma_e5", chroma)
    ahora = conteos(db, chroma)
    distintos = {k: (manifiesto.get(k), v) for k, v in ahora.items() if manifiesto.get(k) != v}
    if distintos:
        sys.exit(f"La restauración no coincide con el manifiesto: {distintos}")
    print(f"Restaurado desde {origen.name}; conteos verificados: {json.dumps(ahora, ensure_ascii=False)}")


def listar(destino_base=RESPALDOS):
    for d in sorted(Path(destino_base).glob("*/manifiesto.json")):
        m = json.loads(d.read_text(encoding="utf-8"))
        print(d.parent.name, {k: m.get(k) for k in ("matricula", "usuario", "log_consultas", "fragmentos")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="accion", required=True)
    c = sub.add_parser("crear")
    c.add_argument("--etiqueta", default="")
    sub.add_parser("listar")
    r = sub.add_parser("restaurar")
    r.add_argument("carpeta")
    a = ap.parse_args()
    if a.accion == "crear":
        crear(a.etiqueta)
    elif a.accion == "listar":
        listar()
    else:
        restaurar(a.carpeta)
