"""P-04 Actualizar normatividad (RF-21, CU-13): agrega un PDF con texto al índice de ChromaDB.

Usa las mismas funciones de CHROMADB2/indexar_normativa.py (fragmentos de 800 caracteres, solape de 100,
multilingual-e5-small con prefijo 'passage: '). Los documentos escaneados deben pasar antes por OCR.
"""
import tempfile
from pathlib import Path

import streamlit as st

import servicios  # noqa: F401 - agrega AGENTE/, TESTBED/ y CHROMADB2/ al sys.path

TIPOS = ["Reglamento General de la UNE o resolución modificatoria", "Ley Universitaria N.° 30220",
         "Documento de la Facultad de Tecnología"]
MIN_TEXTO = 200  # caracteres por página: por debajo, el PDF probablemente es escaneado


@st.cache_resource(show_spinner="Cargando el modelo de embeddings...")
def _embedding():
    from indexar_normativa import MODELO, EmbeddingST
    return EmbeddingST(MODELO)


def _unidades(ruta, tipo, nombre):
    from pypdf import PdfReader

    import indexar_normativa as ix
    if tipo.startswith("Ley"):
        return ix.leer_ley(ruta)
    if tipo.startswith("Documento"):
        return ix.leer_diapositivas(ruta)
    paginas = [ix.limpiar(p.extract_text() or "") for p in PdfReader(ruta).pages]
    return [{"documento": "Reglamento General UNE", "articulo": "", "titulo": f"{nombre}, página {i}",
             "fuente": nombre, "pagina": i, "texto": t} for i, t in enumerate(paginas, start=1) if len(t) >= 40]


st.title("Actualizar normatividad")
st.caption("Agrega un documento normativo al índice sin reconstruir el sistema (RF-21).")
archivo = st.file_uploader("Documento (.pdf con texto)", type="pdf")
tipo = st.selectbox("Tipo de documento", TIPOS, index=None, placeholder="Elige el tipo")

if st.button("Cargar e indexar", type="primary", disabled=archivo is None or tipo is None):
    import chromadb

    from agente import CHROMA_POR_DEFECTO
    from indexar_normativa import COLECCION, fragmentar
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(archivo.getvalue())
    with st.status("Indexando...", expanded=True) as estado:
        unidades = _unidades(tmp.name, tipo, Path(archivo.name).stem)
        Path(tmp.name).unlink(missing_ok=True)
        texto = sum(len(u["texto"]) for u in unidades)
        if not unidades or texto / max(len(unidades), 1) < MIN_TEXTO:
            estado.update(label="El PDF casi no tiene texto", state="error")
            st.error("Parece un documento escaneado: aplica OCR (Tesseract, modelo spa) antes de cargarlo.")
            st.stop()
        fragmentos = fragmentar(unidades)
        for f in fragmentos:  # ids propios para no pisar los del índice original
            f["id"] = f"UI|{Path(archivo.name).stem}|{f['id']}"
        col = chromadb.PersistentClient(path=str(CHROMA_POR_DEFECTO)).get_collection(COLECCION)
        antes = col.count()
        vectores = _embedding().documentos([f["texto"] for f in fragmentos])
        col.upsert(ids=[f["id"] for f in fragmentos], documents=[f["texto"] for f in fragmentos],
                   metadatas=[f["meta"] for f in fragmentos], embeddings=vectores)
        estado.update(label="Documento indexado", state="complete")
    st.success(f"«{archivo.name}»: {len(fragmentos)} fragmentos indexados. El índice pasó de {antes} "
               f"a {col.count()} fragmentos.")
    st.warning("Después de agregar normatividad, vuelve a ejecutar el testbed (criterio de reindexación, II.4.5).")
