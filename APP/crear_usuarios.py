"""Crea las cuentas de RAGuitoBot en academico.db (vuelve a ejecutarlo si reconstruyes la base con build_db.py).

- Estudiantes: una cuenta por estudiante (código = su ID, p. ej. EST0001) con una contraseña aleatoria.
  Las contraseñas se escriben en APP/credenciales.csv (está en .gitignore): de ahí entregas una a cada
  participante del test de usuario o del SUS.
- Evaluador: el código y la contraseña se piden por teclado; no quedan escritos en ningún archivo.

Uso (desde RAGASTESIS/APP):  python crear_usuarios.py --evaluador EVAL01
"""
import argparse
import csv
import getpass
import secrets
from pathlib import Path

from auth import MIN_CLAVE, conectar, crear_usuario

RAIZ = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(RAIZ / "DB" / "academico.db"))
    ap.add_argument("--evaluador", help="código de la cuenta del investigador/evaluador, p. ej. EVAL01")
    ap.add_argument("--salida", default=str(Path(__file__).resolve().parent / "credenciales.csv"))
    args = ap.parse_args()

    con = conectar(args.db)
    ids = [r[0] for r in con.execute("SELECT id FROM estudiante ORDER BY id")]
    with open(args.salida, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["codigo", "clave", "estado"])
        for eid in ids:
            clave = secrets.token_urlsafe(6)  # 8 caracteres aleatorios
            crear_usuario(con, eid, clave, "estudiante", eid)
            estado = con.execute("SELECT estado FROM estudiante WHERE id = ?", (eid,)).fetchone()[0]
            w.writerow([eid, clave, estado])
    print(f"{len(ids)} cuentas de estudiante creadas; contraseñas en {args.salida} (no lo subas a GitHub)")

    if args.evaluador:
        clave = getpass.getpass(f"Contraseña para {args.evaluador} (mín. {MIN_CLAVE} caracteres): ")
        if len(clave) < MIN_CLAVE:
            raise SystemExit("Contraseña demasiado corta")
        crear_usuario(con, args.evaluador, clave, "evaluador")
        print(f"Cuenta de evaluador {args.evaluador.upper()} creada")


if __name__ == "__main__":
    main()
