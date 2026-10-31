"""Indexa el corpus normativo en ChromaDB (capa de recuperación semántica del agente).

Corpus
- Reglamento General UNE consolidado (normativa_une_consolidada.jsonl, un artículo por línea).
- Ley Universitaria N.° 30220 (PDF con texto): se parte por artículo.
- Documentos de la Facultad de Tecnología (PDF de diapositivas): se parte por página.
- Plan de estudios de Electrónica e Informática (Res. N.° 0326-2020-D-FATEC), solo sus partes de texto:
  sumillas (una unidad por curso), reglas de matrícula marcadas con (*), perfil del egresado y certificación
  intermedia. La tabla de la malla NO se indexa: cursos, créditos y prerrequisitos viven en SQLite.

Fragmentación: cada artículo es la unidad; si supera CHUNK caracteres se divide con
RecursiveCharacterTextSplitter (solape SOLAPE). Cada fragmento lleva una cabecera con documento y
artículo, para que el LLM pueda citar la fuente y para que el embedding conserve el contexto.

Uso:
  python indexar_normativa.py --consolidado normativa_une_consolidada.jsonl --ley ley_universitaria.pdf \
      --fatec bachiller.pdf titulo.pdf --db chroma_normativa
  python indexar_normativa.py ... --solo-probar      # no reindexa; solo corre las preguntas de prueba
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

MODELO = "intfloat/multilingual-e5-small"  # elegido por comparación: 7/7 frente a 6/7 de paraphrase-multilingual-MiniLM-L12-v2
COLECCION = "normativa"
CHUNK, SOLAPE = 800, 100
RE_ART = re.compile(r"(?m)^\s*Art[ií]culo\s+(\d+[A-Z]?(?:-[A-Z])?)\.?\s*(.*)$")

# Preguntas con respuesta conocida (verificación de la recuperación, captura para 4.3.1)
# Cada prueba acepta cualquiera de las fuentes listadas (documento, artículo); artículo "" = cualquier página del documento
RG, LEY = "Reglamento General UNE", "Ley Universitaria N.° 30220"
PLAN = "Plan de estudios EI (Res. N.° 0326-2020-D-FATEC)"
BACH = "Procedimientos para la obtención del grado de bachiller (FATEC)"
PRUEBAS = [
    ("¿Cuál es la nota mínima para aprobar un curso en pregrado?", [(RG, "292")]),
    ("¿Qué pasa si desapruebo la misma asignatura tres veces?", [(RG, "282"), (LEY, "102")]),
    ("¿Cómo se calcula el promedio ponderado?", [(RG, "293")]),
    ("¿Puedo reservar mi matrícula y por cuánto tiempo?", [(RG, "289")]),
    ("¿Qué requisitos necesito para obtener el grado de bachiller?", [(RG, "125"), (LEY, "45"), (BACH, "")]),
    ("¿De qué trata el curso de Sistemas Digitales II?", [(PLAN, "TEEI0539")]),
    ("¿Cuántos créditos como máximo puedo matricular en el quinto ciclo?", [(PLAN, "")]),
]


def nfc(t):
    return unicodedata.normalize("NFC", t or "")


def limpiar(t):
    t = nfc(t)
    t = re.sub(r"-\n(?=[a-záéíóúñ])", "", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# ---------------------------------------------------------------- lectura de fuentes
def leer_consolidado(ruta):
    unidades = []
    for linea in Path(ruta).read_text(encoding="utf-8").splitlines():
        a = json.loads(linea)
        unidades.append({
            "documento": "Reglamento General UNE",
            "articulo": str(a["articulo"]),
            "titulo": a["titulo"],
            "fuente": a["fuente"],
            "pagina": a["pagina"] or 0,
            "texto": limpiar(a["texto"]),
        })
    return unidades


def leer_ley(ruta):
    lector = PdfReader(ruta)
    paginas = [(i + 1, nfc(p.extract_text() or "")) for i, p in enumerate(lector.pages)]
    texto, inicio_pag = "", []
    for n, t in paginas:
        inicio_pag.append((len(texto), n))
        texto += t + "\n"
    marcas = list(RE_ART.finditer(texto))
    unidades = []
    for k, m in enumerate(marcas):
        fin = marcas[k + 1].start() if k + 1 < len(marcas) else len(texto)
        pagina = max(n for pos, n in inicio_pag if pos <= m.start())
        unidades.append({
            "documento": "Ley Universitaria N.° 30220",
            "articulo": m.group(1),
            "titulo": m.group(2).strip(" .*")[:120],
            "fuente": "Ley N.° 30220 (2014) y modificatorias",
            "pagina": pagina,
            "texto": limpiar(texto[m.start():fin]),
        })
    return unidades


def leer_diapositivas(ruta):
    nombre = Path(ruta).stem
    titulo = ("Procedimientos para la obtención del grado de bachiller (FATEC)" if "BACHILLER" in nombre.upper()
              else "Lineamientos para la obtención del título profesional (FATEC)")
    unidades = []
    for i, p in enumerate(PdfReader(ruta).pages, start=1):
        t = limpiar(p.extract_text())
        if len(t) < 40:  # portadas o diapositivas casi vacías
            continue
        unidades.append({"documento": titulo, "articulo": "", "titulo": f"Diapositiva {i}",
                         "fuente": "Facultad de Tecnología, UNE", "pagina": i, "texto": t})
    return unidades


ROMANOS = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}


def _norm(t):
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()
    vacias = {"de", "del", "la", "el", "los", "las", "y", "e", "n", "en"}  # "Desarrollo del Software" = "Desarrollo de Software"
    return [w for w in re.sub(r"[^a-z0-9 ]", " ", re.sub(r"\s+", " ", t)).split() if w not in vacias]


def leer_plan(ruta, malla=None):
    paginas = [nfc(p.extract_text() or "") for p in PdfReader(ruta).pages]
    texto, inicio = "", []
    for n, t in enumerate(paginas, start=1):
        inicio.append((len(texto), n))
        texto += t + "\n"
    pag = lambda pos: max(n for i, n in inicio if i <= pos)
    unidades = []

    def seccion(desde, hasta, titulo):
        a, b = texto.find(desde), texto.find(hasta)
        if a < 0 or b <= a:
            print(f"AVISO: no se encontró la sección '{titulo}' del plan")
            return
        unidades.append({"documento": PLAN, "articulo": "", "titulo": titulo, "fuente": PLAN,
                         "pagina": pag(a), "texto": limpiar(texto[a:b])})

    seccion("PERFIL DEL EGRESADO", "MAPA FUNCIONAL", "Perfil del egresado")
    seccion("CERTIFICACIÓN PROFESIONAL INTERMEDIA", "SUMILLAS", "Certificación profesional intermedia al VI ciclo")

    # reglas de matrícula: notas (*) / (**) dentro de la tabla del plan, con su ciclo
    reglas, ciclo = [], None
    for linea in texto[:texto.find("SUMILLAS")].splitlines():
        m = re.match(r"^\s*(I|II|III|IV|V|VI|VII|VIII|IX|X)\s+CICLO\b", linea)
        if m:
            ciclo = m.group(1)
        if re.search(r"\(\*+\)\s*(EN LA MATR|EL ESTUDIANTE|CURSOS ELECTIVOS)", linea):
            regla = re.sub(r"\s+", " ", re.sub(r"\(\*+\)", "", linea)).strip().capitalize()
            regla = re.sub(r"\b(i|ii|iii|iv|v|vi|vii|viii|ix|x)\b(?= ciclo)", lambda m: m.group(1).upper(), regla)
            entrada = f"Ciclo {ciclo}: {regla}" if ciclo else regla
            if entrada not in reglas:
                reglas.append(entrada)
    if reglas:
        unidades.append({"documento": PLAN, "articulo": "", "titulo": "Reglas de matrícula del plan de estudios",
                         "fuente": PLAN, "pagina": pag(texto.find("CICLO")),
                         "texto": "Reglas de matrícula del plan de estudios de Educación con especialidad de "
                                  "Electrónica e Informática:\n- " + "\n- ".join(reglas)})

    # sumillas: un bloque por "ASIGNATURA:"
    codigos = {}
    if malla:
        import csv
        with open(malla, encoding="utf-8") as fh:
                filas_malla = list(csv.DictReader(fh))
        codigos = {" ".join(_norm(r["nombre"])): r["codigo"] for r in filas_malla}
        creditos_malla = {r["codigo"]: float(r["creditos"]) for r in filas_malla}
    marcas = [m.start() for m in re.finditer(r"ASIGNATURA\s*:", texto)]
    for k, a in enumerate(marcas):
        b = marcas[k + 1] if k + 1 < len(marcas) else len(texto)
        bloque = re.split(r"\n\s*(?:I|II|III|IV|V|VI|VII|VIII|IX|X)\s+CICLO\s*\n|\n\s*\d+\.\s+SUMILLAS", texto[a:b])[0]
        nombre = re.match(r"ASIGNATURA\s*:\s*(.+)", bloque).group(1).strip()
        clave = " ".join(_norm(nombre))
        codigo = codigos.get(clave) or next((c for n, c in codigos.items() if n.startswith(clave) or clave.startswith(n)), "")
        unidades.append({"documento": PLAN, "articulo": codigo, "titulo": f"Sumilla: {nombre.title()}",
                         "fuente": PLAN, "pagina": pag(a), "texto": limpiar(bloque)})

    # El PDF repite algunas sumillas (p. ej., "Desarrollo del Software" con 3 y con 5 créditos):
    # se conserva la versión cuyos créditos coinciden con la malla y se informa la exclusión.
    if codigos:
        por_codigo = {}
        for u in unidades:
            if u["titulo"].startswith("Sumilla") and u["articulo"]:
                por_codigo.setdefault(u["articulo"], []).append(u)
        for cod, versiones in por_codigo.items():
            if len(versiones) < 2:
                continue
            cr = lambda u: float(re.search(r"CR[EÉ]DITOS\s*:\s*(\d+)", u["texto"]).group(1)) if re.search(r"CR[EÉ]DITOS\s*:\s*(\d+)", u["texto"]) else -1
            buena = next((v for v in versiones if cr(v) == creditos_malla.get(cod)), versiones[-1])
            for v in versiones:
                if v is not buena:
                    unidades.remove(v)
                    print(f"AVISO: sumilla duplicada de {cod} (p. {v['pagina']}, {cr(v):.0f} cr) excluida; "
                          f"se conserva la de p. {buena['pagina']} ({cr(buena):.0f} cr)")
    return unidades


# ---------------------------------------------------------------- fragmentación
def fragmentar(unidades):
    divisor = RecursiveCharacterTextSplitter(chunk_size=CHUNK, chunk_overlap=SOLAPE,
                                             separators=["\n\n", "\n", ". ", " ", ""])
    fragmentos = []
    for u in unidades:
        if u["documento"] == PLAN:
            cab = f"[{u['documento']} | {u['titulo']}" + (f" ({u['articulo']})" if u["articulo"] else "") + "]"
        else:
            cab = f"[{u['documento']}" + (f" | Art. {u['articulo']}. {u['titulo']}" if u["articulo"] else f" | {u['titulo']}") + "]"
        partes = divisor.split_text(u["texto"]) if len(u["texto"]) > CHUNK else [u["texto"]]
        for j, parte in enumerate(partes, start=1):
            ident = f"{u['documento'][:12]}|{u['articulo'] or u['pagina']}|{j}"
            fragmentos.append({
                "id": re.sub(r"[^A-Za-z0-9|]", "_", ident),
                "texto": f"{cab}\n{parte}",
                "meta": {"documento": u["documento"], "articulo": u["articulo"], "titulo": u["titulo"],
                         "fuente": u["fuente"], "pagina": int(u["pagina"]), "parte": j, "partes": len(partes)},
            })
    # ids únicos aunque el OCR repita un número de artículo
    vistos = {}
    for f in fragmentos:
        vistos[f["id"]] = vistos.get(f["id"], 0) + 1
        if vistos[f["id"]] > 1:
            f["id"] += f"#{vistos[f['id']]}"
    return fragmentos


# ---------------------------------------------------------------- embeddings
class EmbeddingST:
    """sentence-transformers. Los modelos e5 requieren los prefijos 'query: ' y 'passage: '."""

    def __init__(self, modelo):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(modelo)
        self.e5 = "e5" in modelo.lower()

    def documentos(self, textos):
        textos = [f"passage: {t}" for t in textos] if self.e5 else textos
        return self.m.encode(textos, batch_size=32, normalize_embeddings=True, show_progress_bar=True).tolist()

    def consulta(self, texto):
        texto = f"query: {texto}" if self.e5 else texto
        return self.m.encode([texto], normalize_embeddings=True).tolist()[0]


class EmbeddingPrueba:
    """SOLO para probar el código sin descargar el modelo: bolsa de palabras con hashing. No usar en la tesis."""

    def _v(self, t, dim=512):
        import hashlib
        v = [0.0] * dim
        for w in re.findall(r"\w{3,}", nfc(t).lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    def documentos(self, textos):
        return [self._v(t) for t in textos]

    def consulta(self, texto):
        return self._v(texto)


# ---------------------------------------------------------------- índice y búsqueda
def indexar(fragmentos, emb, ruta_db, modelo):
    cliente = chromadb.PersistentClient(path=str(ruta_db))
    try:
        cliente.delete_collection(COLECCION)
    except Exception:
        pass
    col = cliente.create_collection(COLECCION, metadata={"hnsw:space": "cosine", "modelo": modelo,
                                                         "chunk": CHUNK, "solape": SOLAPE})
    vectores = emb.documentos([f["texto"] for f in fragmentos])
    for i in range(0, len(fragmentos), 500):
        lote = fragmentos[i:i + 500]
        col.add(ids=[f["id"] for f in lote], documents=[f["texto"] for f in lote],
                metadatas=[f["meta"] for f in lote], embeddings=vectores[i:i + 500])
    return col


def buscar(col, emb, pregunta, k=4):
    """Devuelve [(texto, metadatos, distancia)]. Es la función que usará el orquestador (Bloque 4)."""
    r = col.query(query_embeddings=[emb.consulta(pregunta)], n_results=k)
    return list(zip(r["documents"][0], r["metadatas"][0], r["distances"][0]))


def probar(col, emb, k):
    print(f"\n== Prueba de recuperación (k={k})")
    aciertos = 0
    for pregunta, esperadas in PRUEBAS:
        res = buscar(col, emb, pregunta, k)
        ok = any(m["documento"] == doc and (art == "" or m["articulo"] == art)
                 for _, m, _ in res for doc, art in esperadas)
        aciertos += ok
        texto_esp = " o ".join((f"{d.split(' (')[0]} {a}" if d == PLAN else f"{d.split(' (')[0]} art. {a}") if a
                               else d.split(" (")[0] for d, a in esperadas)
        print(f"\n{'✓' if ok else '✗'} {pregunta}\n   esperado: {texto_esp}")
        for pos, (_, m, d) in enumerate(res, start=1):
            ref = (m["articulo"] or "-") if m["documento"] == PLAN else (f"Art. {m['articulo']}" if m["articulo"] else m["titulo"])
            print(f"   {pos}. {m['documento'][:34]:<34} {ref:<16} {m['titulo'][:38]:<38} dist={d:.3f}")
    print(f"\nAciertos en top-{k}: {aciertos}/{len(PRUEBAS)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--consolidado", required=True)
    ap.add_argument("--ley", required=True)
    ap.add_argument("--fatec", nargs="*", default=[])
    ap.add_argument("--plan", help="PDF del plan de estudios (sumillas, reglas de matrícula, perfil, certificación)")
    ap.add_argument("--malla", help="malla_ei.csv, para enlazar cada sumilla con el código del curso")
    ap.add_argument("--db", default="chroma_normativa")
    ap.add_argument("--modelo", default=MODELO)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--solo-probar", action="store_true")
    ap.add_argument("--embedding-prueba", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()

    emb = EmbeddingPrueba() if a.embedding_prueba else EmbeddingST(a.modelo)
    if a.solo_probar:
        col = chromadb.PersistentClient(path=a.db).get_collection(COLECCION)
    else:
        unidades = leer_consolidado(a.consolidado) + leer_ley(a.ley)
        for f in a.fatec:
            unidades += leer_diapositivas(f)
        if a.plan:
            unidades += leer_plan(a.plan, a.malla)
        fragmentos = fragmentar(unidades)
        por_doc = {}
        for f in fragmentos:
            por_doc[f["meta"]["documento"]] = por_doc.get(f["meta"]["documento"], 0) + 1
        print("Fragmentos por documento:")
        for d, n in por_doc.items():
            print(f"  {d:<62} {n}")
        largos = [len(f["texto"]) for f in fragmentos]
        print(f"Total: {len(fragmentos)} fragmentos | longitud media {sum(largos) / len(largos):.0f} car. | máx {max(largos)}")
        Path(a.db).mkdir(exist_ok=True)
        with open(Path(a.db) / "fragmentos.jsonl", "w", encoding="utf-8") as fh:  # evidencia inspeccionable
            for f in fragmentos:
                fh.write(json.dumps(f, ensure_ascii=False) + "\n")
        col = indexar(fragmentos, emb, a.db, a.modelo if not a.embedding_prueba else "prueba-hashing")
        print(f"Colección '{COLECCION}' en {a.db}: {col.count()} vectores")
    probar(col, emb, a.k)
