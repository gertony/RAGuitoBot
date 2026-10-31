"""P-05 Reporte de métricas RAGAS. RF-14."""
import streamlit as st

import servicios

st.title("Reporte de métricas RAGAS")
tabla = servicios.tabla_ragas()
if tabla is None:
    st.info("Aún no hay resultados de RAGAS. Ejecuta el testbed y la evaluación desde el panel.")
    st.stop()

st.dataframe(tabla, hide_index=True, width="stretch")
agregado = servicios.ragas_agregado(tabla)
st.subheader("RAGAS agregado (media armónica de las cuatro métricas)")
if agregado is None:
    st.write("—  (se necesitan las cuatro métricas)")
else:
    st.metric("RAGAS agregado", f"{agregado:.3f}")
    for ref, valor in servicios.REFERENCIAS.items():
        st.write(f"- Frente a **{ref}**: {valor:.2f} → diferencia {agregado - valor:+.3f}")
    st.caption("El 0.94 de Jimenez Orellana et al. (2026) no es una media armónica de las cuatro métricas: "
               "la comparación con ese valor es solo referencial.")
lat = servicios.resumen_latencia()
if lat:
    st.write(f"Latencia: media {lat['media']:.0f} ms · mediana {lat['mediana']:.0f} ms · p95 {lat['p95']:.0f} ms")

st.download_button("Exportar CSV", tabla.to_csv(index=False).encode("utf-8-sig"), "reporte_ragas.csv", "text/csv")
st.page_link("paginas/panel.py", label="Volver al panel", icon="↩️")
