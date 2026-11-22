"""Exporta de log_consultas las consultas de las cuentas usadas en una sesión de prueba (evidencia de RF-12).

Uso: python PRUEBAS/exportar_log.py salida.csv EST0032 EST0013 EST0031 EST0025
"""
import csv
import sqlite3
import sys
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "DB" / "academico.db"


def main():
    salida, cuentas = sys.argv[1], sys.argv[2:]
    con = sqlite3.connect(DB)
    marcas = ",".join("?" * len(cuentas))
    filas = con.execute(f"SELECT * FROM log_consultas WHERE estudiante_id IN ({marcas}) ORDER BY ts", cuentas)  # noqa: S608
    with open(salida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([d[0] for d in filas.description])
        w.writerows(filas)
    print(f"Exportado: {salida}")


if __name__ == "__main__":
    main()
