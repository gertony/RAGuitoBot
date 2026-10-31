"""P-03 Panel del evaluador. RF-11 (CU-10), RF-12, RF-13, RF-14."""
import sqlite3

import pandas as pd
import streamlit as st
from actualizacion import aplicar, validar

import servicios

RAIZ = servicios.RAIZ
st.title("Panel del evaluador")

# ------------------------------------------------------------------ estado del sistema y métricas
lat = servicios.resumen_latencia()
ragas = servicios.tabla_ragas()
c1, c2, c3 = st.columns(3)
c1.metric("Pares del testbed procesados", lat["pares"] if lat else "—")
c2.metric("Latencia p95 (RNF-01 ≤ 10 000 ms)", f"{lat['p95']:.0f} ms" if lat else "—")
c3.metric("Ruta correcta del orquestador", f"{lat['ruta_ok']}/{lat['pares']}" if lat else "—")
if lat:
    st.caption(f"Latencia media {lat['media']:.0f} ms · mediana {lat['mediana']:.0f} ms")

st.subheader("Métricas RAGAS (umbral ≥ 0.85)")
if ragas is None:
    st.info("Aún no hay resultados de RAGAS (EVALUACION/ragas_por_par.csv).")
else:
    st.dataframe(ragas, hide_index=True, width="stretch")
    st.page_link("paginas/reporte.py", label="Ver reporte completo", icon="📊")

# ------------------------------------------------------------------ ejecuciones (RF-13)
st.subheader("Ejecuciones")
a, b = st.columns(2)
if a.button("Ejecutar testbed", width="stretch",
            disabled=not (RAIZ / "TESTBED" / "testbed.json").exists()):
    servicios.lanzar("testbed", "correr_testbed.py", RAIZ / "AGENTE")
if b.button("Evaluar con RAGAS", width="stretch"):
    servicios.lanzar("ragas", "evaluar_ragas.py", RAIZ / "EVALUACION")  # [VERIFICAR] argumentos del script
for nombre in ("testbed", "ragas"):
    est, cola = servicios.estado(nombre)
    if est != "sin ejecutar":
        st.markdown(f"**{nombre}**: {est}")
        st.code(cola or "(sin salida todavía)", language=None)
st.button("Actualizar estado", icon="🔄")

# ------------------------------------------------------------------ CU-10 / RF-11: actualizar la base académica
st.subheader("Actualizar base académica")
st.caption("CSV con columnas estudiante_id, curso, periodo, nota. Si el registro existe se reemplaza la nota; "
           "si no, se agrega como un nuevo intento. No requiere reconstruir el sistema (RF-11).")
archivo = st.file_uploader("Archivo de notas (.csv)", type="csv")


if archivo is not None:
    df = pd.read_csv(archivo, dtype={"estudiante_id": str, "curso": str, "periodo": str})
    st.dataframe(df.head(20), hide_index=True, width="stretch")
    with sqlite3.connect(servicios.DB) as con:
        errores = validar(df, con)
        if errores:
            st.error("No se aplicó ningún cambio:\n\n- " + "\n- ".join(errores[:10]))
        elif st.button("Aplicar cambios", type="primary"):
            nuevos, reemplazados = aplicar(df, con)
            st.success(f"Base actualizada: {nuevos} registros nuevos y {reemplazados} reemplazados. "
                       "Las siguientes consultas ya usan estos datos.")

# ------------------------------------------------------------------ RF-12: últimas consultas registradas
st.subheader("Últimas consultas registradas")
with sqlite3.connect(servicios.DB) as con:
    log = pd.read_sql_query("SELECT ts AS fecha, estudiante_id AS estudiante, ruta, pregunta, "
                            "ROUND(latencia_ms) AS latencia_ms FROM log_consultas ORDER BY id DESC LIMIT 20", con)
st.dataframe(log, hide_index=True, width="stretch") if not log.empty else st.caption("Sin consultas aún.")
