"""Agente conversacional RAG híbrido: orquestador en LangChain + SQLite + ChromaDB + Gemini.

Flujo (cadena LCEL de LangChain):
    pregunta ─► router ─► recuperación ─► prompt aumentado ─► LLM ─► respuesta
                 │            ├─ personal : funciones parametrizadas de consultas.py (SQLite)
                 │            ├─ normativa: búsqueda semántica en ChromaDB (multilingual-e5-small)
                 │            └─ mixta    : ambas
                 └─ reglas por palabras clave; el LLM solo desempata cuando las reglas no deciden

Requisitos funcionales cubiertos: RF-02 (fuente según la consulta), RF-03 (responde solo con el contexto),
RF-04…RF-09 (consultas), RF-10 (avisa cuando no hay información), RF-12 (registro con latencia).

Configuración (variables de entorno):
    LLM_PROVEEDOR    gemini (por defecto) | groq (respaldo)
    GROQ_API_KEY / GROQ_MODEL   si se usa Groq
    GOOGLE_API_KEY   clave de Google AI Studio
    GEMINI_MODEL     nombre del modelo (por defecto gemini-3.8-flash, el que indicó la API el 28/10/2026)

Ubicación: RAGASTESIS/AGENTE (importa consultas.py desde RAGASTESIS/TESTBED).

Uso rápido:
    python agente.py EST0040 "¿Cuál es mi promedio ponderado?"
    python agente.py EST0069 "Desaprobé Lenguaje y Comunicación I varias veces, ¿qué me pasa?"
"""
import json
import os
import re
import sys
import threading
import time
import unicodedata
from datetime import datetime
from pathlib import Path

# Estructura: RAGASTESIS/{DB, CHROMADB2, TESTBED, AGENTE}. consultas.py vive en TESTBED.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "TESTBED"))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from consultas import DB_POR_DEFECTO, RAIZ, Academico

CHROMA_POR_DEFECTO = RAIZ / "CHROMADB2" / "chroma_e5"
MODELO_EMB = "intfloat/multilingual-e5-small"
K_NORMATIVA = 4

# ------------------------------------------------------------------ router por reglas
DIAS = ["lunes", "martes", "miercoles", "jueves", "viernes"]
PERSONAL = r"\b(mi|mis|me|tengo|llevo|llevando|saque|aprobe|desaprobe|estoy|voy|puedo matricularme|puedo llevar|" \
           r"me falta|me faltan|me retiro|cumplo|sigo siendo|volviendo a llevar)\b"
NORMATIVA = r"\b(reglamento|que pasa si|que me pasa si|si no (la|lo) paso|requisitos?|nota minima|reserv\w*|desmatricul\w*|retirarme|tope|maximo|" \
            r"bachiller|titulo|de que trata|electivo\w*|rectificacion|extemporane\w*|separa\w*|consecuencia|" \
            r"permite|segun|norma\w*|tres veces|tercera vez|cuarta vez|regular|prorroga|se aprueba|aprobar un|" \
            r"que temas|contenidos?|sumilla|que se ve|que ve|que se necesita|que necesito)\b"
SUMILLA = r"de que trata|que temas|contenidos?|sumilla|que se ve|que ve"
INTENCIONES = [  # (intención, patrón); se evalúan todas y se acumulan
    ("promedio", r"promedio"),
    ("avance", r"creditos .*aprobados|porcentaje|avanzad|me faltan? .*terminar|avance|cuanto me falta"),
    ("notas_periodo", r"notas|ciclo pasado|semestre pasado|ultimo semestre|como me fue en el ultimo"),
    ("en_curso", r"estoy llevando|llevo (matriculad|este)|creditos (estoy )?llevando|cursos .*este (ciclo|semestre)|"
                 r"que llevo|que cursos tengo|me retiro de|creditos llevo|mi carga|carga de creditos|"
                 r"cursos llevo|llevo (ahorita|ahora|actualmente)|estoy cursando"),
    ("horario", r"horario|clases|me toca|a que hora|hora (entro|salgo)"),
    ("puede_llevar", r"puedo (matricularme|llevar)"),
    ("disponibles", r"(que|cuales) (cursos|asignaturas|materias) (puedo|me toca|tengo disponibles?|hay disponibles?)|"
                    r"(cursos|asignaturas|materias) disponibles|puedo llevar (el|en el) (siguiente|proximo|otro)"),
    ("desaprobados", r"desaprob|repit|volviendo a llevar|varias veces|otra vez"),  # "si desapruebo" (hipótesis) no es dato propio
    ("estado", r"matriculad[oa] este|estado|situacion|en que (ciclo|semestre|ano)\b"),
    ("avance_bachiller", r"cumplo .*bachiller|tramitar el bachiller"),
]


def _plano(t):
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", " ", t)


def router_reglas(pregunta, academico):
    p = _plano(pregunta)
    curso = academico.buscar_curso(pregunta)
    intenciones = [n for n, pat in INTENCIONES if re.search(pat, p)]
    # una pregunta personal que nombra un día de la semana pide el horario ("¿qué me toca el miércoles?")
    if "horario" not in intenciones and re.search(PERSONAL, p) and any(re.search(rf"\b{d}\b", p) for d in DIAS):
        intenciones.append("horario")
    if curso and not intenciones:
        if re.search(r"requisito", p) and not re.search(PERSONAL, p):
            intenciones = ["prerrequisitos"]
        elif re.search(SUMILLA, p):
            intenciones = []  # contenido del curso: está en la sumilla (ChromaDB), no en la base
        else:
            intenciones = ["nota_curso"]
    if curso and "disponibles" in intenciones:
        intenciones.remove("disponibles")  # pregunta por un curso concreto: basta puede_llevar
    if "disponibles" in intenciones and "en_curso" in intenciones:
        intenciones.remove("en_curso")  # «¿qué cursos tengo disponibles?» no pide los que lleva ahora
    if "avance" in intenciones and "en_curso" in intenciones and re.search(r"aprobados", p):
        intenciones.remove("en_curso")
    if "desaprobados" in intenciones and "estado" not in intenciones:
        intenciones.append("estado")  # la repitencia (art. 282) puede haber cambiado su estado de matrícula
    if "avance_bachiller" in intenciones:
        intenciones = [i for i in intenciones if i != "avance_bachiller"] + ["avance"]
    es_normativa = bool(re.search(NORMATIVA, p))
    # "mi/me" por sí solos no bastan: "¿puedo reservar mi matrícula?" es normativa si no pide un dato propio
    es_personal = bool(intenciones) or (bool(re.search(PERSONAL, p)) and not es_normativa)
    # una pregunta personal "pura" sobre un dato no necesita normativa aunque contenga 'regular' o 'segun'
    if es_personal and es_normativa and intenciones and not re.search(
            r"reglamento|que pasa si|que me pasa si|si no (la|lo) paso|consecuencia|requisitos|permite|tope|maximo|sigo siendo|bachiller|titulo|separa", p):
        es_normativa = False
    if es_personal and es_normativa:
        ruta = "mixta"
        if re.search(r"\bveces\b|\bvez\b", p) and "desaprobados" not in intenciones:
            intenciones += ["desaprobados", "estado"]  # repitencia (art. 282): hace falta su historial del curso
    elif es_personal:
        ruta = "personal"
    elif es_normativa:
        ruta = "normativa"
    else:
        ruta = None  # las reglas no deciden: desempata el LLM
    return {"ruta": ruta, "intenciones": intenciones or (["resumen"] if ruta in ("personal", "mixta") else []),
            "curso": curso, "dia": next((d for d in DIAS if re.search(rf"\b{d}\b", p)), None)}


PROMPT_ROUTER = ChatPromptTemplate.from_messages([
    ("system", "Clasifica la consulta de un estudiante universitario. Responde SOLO un JSON con la clave \"ruta\": "
               "\"personal\" si pide sus datos académicos (notas, promedio, créditos, horario, matrícula), "
               "\"normativa\" si pregunta por reglas, reglamentos, requisitos o contenidos de cursos, "
               "\"mixta\" si necesita ambas cosas."),
    ("human", "{pregunta}"),
])

# Enrutador v2 (20/11/2026): el LLM clasifica todas las consultas. Motivo: las reglas acertaban 35/35 en el testbed
# con el que se calibraron, pero 5/15 en un conjunto nuevo escrito antes de este cambio (sobreajuste). Los ejemplos
# del prompt son propios: no salen del testbed ni del conjunto de retención.
PROMPT_ROUTER_V2 = ChatPromptTemplate.from_messages([
    ("system",
     "Clasificas la consulta de un estudiante universitario según de dónde sale la respuesta. Responde SOLO un JSON "
     "con la clave \"ruta\".\n"
     "- \"personal\": la respuesta es un dato propio del estudiante en el registro académico (sus notas, promedio, "
     "créditos, avance, horario, cursos, estado de matrícula o si cumple un prerrequisito del plan).\n"
     "- \"normativa\": la respuesta está en el reglamento, la ley universitaria o el plan de estudios, y es la misma "
     "para cualquier estudiante. Las preguntas en primera persona que plantean una situación hipotética (\"si me "
     "pasa X, ¿qué ocurre?\") o piden requisitos generales son normativas.\n"
     "- \"mixta\": para responder hay que comparar los datos del estudiante con una norma (¿cumplo?, ¿me alcanza?, "
     "¿qué me pasa a mí con lo que tengo?), o explicar un dato suyo con una regla.\n\n"
     "Ejemplos:\n"
     "\"¿en qué aula tengo clase el jueves?\" -> {{\"ruta\": \"personal\"}}\n"
     "\"¿cuánto saqué en el examen de física?\" -> {{\"ruta\": \"personal\"}}\n"
     "\"si me enfermo, ¿puedo justificar una falta?\" -> {{\"ruta\": \"normativa\"}}\n"
     "\"¿cada cuánto se eligen los delegados estudiantiles?\" -> {{\"ruta\": \"normativa\"}}\n"
     "\"con mis notas, ¿me alcanza para pedir una beca?\" -> {{\"ruta\": \"mixta\"}}\n"
     "\"ya acabé todos mis cursos, ¿qué me falta para titularme?\" -> {{\"ruta\": \"mixta\"}}"),
    ("human", "{pregunta}"),
])

# ------------------------------------------------------------------ prompt de generación
PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Eres un asistente académico de la Universidad Nacional de Educación Enrique Guzmán y Valle que ayuda a un "
     "estudiante a interpretar su información académica. Reglas obligatorias:\n"
     "1. Responde ÚNICAMENTE con la información del CONTEXTO. No uses conocimiento propio.\n"
     "2. Si el contexto no contiene la información pedida, responde exactamente: \"No tengo esa información en los "
     "registros disponibles.\" No inventes datos.\n"
     "3. Copia los números tal como aparecen (notas, promedios, créditos, fechas, plazos).\n"
     "4. Si usas una norma, menciona su fuente (por ejemplo, \"según el art. 282 del Reglamento General\").\n"
     "5. Responde en español, en segunda persona (tú), de forma clara y breve (máximo 5 oraciones).\n"
     "6. Si la pregunta combina datos del estudiante y normas, relaciona ambos para interpretar su situación."),
    ("human", "CONTEXTO:\n{contexto}\n\nPREGUNTA DEL ESTUDIANTE: {pregunta}"),
])


# ------------------------------------------------------------------ componentes
# Expresiones estudiantiles que el índice no conoce (D-01). Origen: frases de la prueba de usuarios del 31/10/2026
# («jalé un curso», «cómo voy en la u», «3 veces»), no las preguntas del testbed. Se AGREGAN términos a la búsqueda.
SINONIMOS = [
    (r"\bjal(?:o|ó|e|é|a|as|an|en|ar|aste|aron|ado|ada|ados|adas|amos|ando|ara|aran)\b", "desaprobar desaprobación"),
    (r"\bbot(an|aron|ar|en|aran)\b", "separado retiro definitivo"),
    (r"\bla u\b", "universidad"),
    (r"\bprofe\b", "docente"),
    (r"\bciclo\b", "semestre académico"),
    # Término institucional: retirarse de una asignatura ya matriculada se llama «desmatrícula» (RG UNE, art. 290).
    # Ajuste dirigido (20/11/2026) a partir del análisis de errores de N04 y M04 (D-03); se declara como tal.
    (r"\bretir(?:o|arme|arse|arte|e|ar)\s+(?:de\s+)?(?:un|una|el|la|mi)?\s*(?:curso|asignatura|materia)", "desmatrícula"),
    (r"\bme\s+retiro\s+de\s+(?!la\s+universidad|la\s+u\b|los\s+estudios)", "desmatrícula"),
]
CIFRAS = {"1": "una", "2": "dos", "3": "tres", "4": "cuatro", "5": "cinco"}
VACIAS = set("a al como con de del el en es la las lo los me mi mis o para por que se si su sus te tu un una y "
             "cual cuales cuando donde puedo tengo hay pasa".split())


def normalizar_consulta(pregunta):
    p = _plano(pregunta)
    extra = [t for pat, t in SINONIMOS if re.search(pat, p)]
    extra += [f"{CIFRAS[n]} veces" for n in re.findall(r"\b([1-5])\s*(?:veces|vez)\b", p)]
    return pregunta + (" " + " ".join(extra) if extra else "")


def _tokens(texto):
    return [w for w in _plano(texto).split() if len(w) > 1 and w not in VACIAS]


class _BM25:
    """BM25 (Robertson y Zaragoza, 2009) sobre los mismos fragmentos de ChromaDB, sin dependencias nuevas."""

    def __init__(self, documentos, k1=1.5, b=0.75):
        import math
        from collections import Counter
        self.k1, self.b = k1, b
        self.tf = [Counter(_tokens(d)) for d in documentos]
        self.largo = [sum(t.values()) for t in self.tf]
        self.medio = sum(self.largo) / max(len(self.largo), 1)
        df = Counter(w for t in self.tf for w in t)
        n = len(documentos)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def ranking(self, consulta, n):
        q = _tokens(consulta)
        puntajes = []
        for i, tf in enumerate(self.tf):
            s = sum(self.idf.get(w, 0) * tf[w] * (self.k1 + 1) /
                    (tf[w] + self.k1 * (1 - self.b + self.b * self.largo[i] / self.medio)) for w in q if w in tf)
            if s > 0:
                puntajes.append((s, i))
        return [i for _, i in sorted(puntajes, reverse=True)[:n]]


class BuscadorNormativa:
    """Recuperación normativa. Por defecto, la de la tesis: búsqueda densa e5, k = 4.

    Opciones (20/11/2026, se miden con EVALUACION/evaluar_recuperacion.py antes de adoptarlas):
      RECUPERACION=hibrido  densa + BM25 fusionadas con Reciprocal Rank Fusion (Cormack et al., 2009)
      NORMALIZAR=1          agrega sinónimos estudiantiles y cifras en palabras a la consulta (D-01)
      K_NORMATIVA=6         fragmentos que se entregan al LLM
      CHROMA_RUTA=...       índice alternativo (p. ej. CHROMADB2/chroma_e5_v2, con el D-12 corregido)
    """

    def __init__(self, ruta_chroma=None, modelo=MODELO_EMB, embedding=None, modo=None, normalizar=None, k=None):
        import chromadb
        ruta_chroma = Path(ruta_chroma or os.environ.get("CHROMA_RUTA") or CHROMA_POR_DEFECTO)
        if not ruta_chroma.is_absolute() and not ruta_chroma.exists():
            ruta_chroma = RAIZ / ruta_chroma
        if not ruta_chroma.exists():
            raise FileNotFoundError(f"No se encontró el índice de ChromaDB: {ruta_chroma}")
        self.col = chromadb.PersistentClient(path=str(ruta_chroma)).get_collection("normativa")
        if embedding is None:
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(modelo)
            pref = "query: " if "e5" in modelo.lower() else ""
            embedding = lambda t: m.encode([pref + t], normalize_embeddings=True).tolist()[0]  # noqa: E731
        self.embedding = embedding
        self.modo = (modo or os.environ.get("RECUPERACION", "denso")).lower()
        self.normalizar = normalizar if normalizar is not None else os.environ.get("NORMALIZAR", "0") == "1"
        self.k = int(k or os.environ.get("K_NORMATIVA", K_NORMATIVA))
        self._bm25 = None

    def _corpus(self):
        if self._bm25 is None:
            todo = self.col.get(include=["documents", "metadatas"])
            self._docs, self._metas = todo["documents"], todo["metadatas"]
            self._bm25 = _BM25(self._docs)
        return self._bm25

    def buscar_con_meta(self, pregunta, k=None):
        k = k or self.k
        consulta = normalizar_consulta(pregunta) if self.normalizar else pregunta
        n = k if self.modo != "hibrido" else max(4 * k, 20)
        r = self.col.query(query_embeddings=[self.embedding(consulta)], n_results=n)
        densos = list(zip(r["documents"][0], r["metadatas"][0]))
        if self.modo != "hibrido":
            return densos[:k]
        bm25 = self._corpus()
        lexicos = [(self._docs[i], self._metas[i]) for i in bm25.ranking(consulta, n)]
        puntaje = {}
        for lista in (densos, lexicos):
            for pos, (doc, meta) in enumerate(lista):
                d, _ = puntaje.get(doc, (0.0, meta))
                puntaje[doc] = (d + 1 / (60 + pos + 1), meta)
        return [(doc, meta) for doc, (_, meta) in sorted(puntaje.items(), key=lambda x: -x[1][0])[:k]]

    def buscar(self, pregunta, k=None):
        return [doc for doc, _ in self.buscar_con_meta(pregunta, k)]


def _limitador():
    """Limitador de solicitudes al LLM (D-06, P9): LLM_RPM = solicitudes por minuto permitidas.

    Sin LLM_RPM no se limita nada (comportamiento de la tesis). Con varias sesiones a la vez, el
    limitador hace esperar las consultas en vez de que el proveedor las rechace con el error 429.
    """
    rpm = float(os.environ.get("LLM_RPM", "0"))
    if rpm <= 0:
        return None
    from langchain_core.rate_limiters import InMemoryRateLimiter
    return InMemoryRateLimiter(requests_per_second=rpm / 60, check_every_n_seconds=0.1, max_bucket_size=1)


def chat_openrouter(modelo, reintentos=2, timeout=60, rate_limiter=None, max_tokens=None):
    """Modelo servido por OpenRouter (API compatible con OpenAI, pago por uso).

    OPENROUTER_PROVEEDOR fija qué proveedor atiende la solicitud (por defecto «groq», el mismo de la línea base),
    sin pasar a otro si falla, para que los resultados sigan siendo comparables. Vacío = que OpenRouter elija.
    """
    from langchain_openai import ChatOpenAI
    clave = os.environ.get("OPENROUTER_API_KEY")
    if not clave:
        raise RuntimeError("Falta la variable de entorno OPENROUTER_API_KEY (clave de openrouter.ai)")
    orden = [p.strip() for p in os.environ.get("OPENROUTER_PROVEEDOR", "groq").split(",") if p.strip()]
    extra = {"provider": {"order": orden, "allow_fallbacks": False}} if orden else None
    return ChatOpenAI(model=modelo, base_url="https://openrouter.ai/api/v1", api_key=clave, temperature=0,
                      max_retries=reintentos, timeout=timeout, rate_limiter=rate_limiter, max_tokens=max_tokens,
                      extra_body=extra)


def crear_llm(modelo=None):
    """LLM generador. LLM_PROVEEDOR=gemini (por defecto), groq u openrouter (Tabla 6 de la tesis).

    Se agregó Groq el 28/10/2026 porque gemini-3.8-flash devolvía 503 (servicio saturado) en el plan gratuito.
    Se agregó OpenRouter el 20/11/2026 porque el plan de pago de Groq no admitía nuevas altas y el plan
    gratuito (8 000 tokens por minuto) limitaba la carga a unos 4 usuarios y RAGAS a una cuota diaria.
    """
    proveedor = os.environ.get("LLM_PROVEEDOR", "gemini").lower()
    reintentos = int(os.environ.get("LLM_REINTENTOS", "2"))
    timeout = float(os.environ.get("LLM_TIMEOUT", "60"))
    limitador = _limitador()
    if proveedor == "groq":
        from langchain_groq import ChatGroq
        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError("Falta la variable de entorno GROQ_API_KEY (clave de console.groq.com)")
        nombre = modelo or os.environ.get("GROQ_MODEL")
        if not nombre:
            raise RuntimeError("Define GROQ_MODEL con un modelo de tu cuenta (lístalos con: python listar_modelos.py)")
        return ChatGroq(model=nombre, temperature=0, max_retries=reintentos, timeout=timeout, rate_limiter=limitador)
    if proveedor == "openrouter":
        return chat_openrouter(modelo or os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-120b"),
                               reintentos=reintentos, timeout=timeout, rate_limiter=limitador)
    from langchain_google_genai import ChatGoogleGenerativeAI
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("Falta la variable de entorno GOOGLE_API_KEY (clave de Google AI Studio)")
    # timeout y pocos reintentos: si el servicio está saturado (503), se falla rápido en vez de esperar minutos
    return ChatGoogleGenerativeAI(model=modelo or os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"),
                                  temperature=0, max_retries=reintentos, timeout=timeout,
                                  rate_limiter=limitador)


class Agente:
    def __init__(self, db=DB_POR_DEFECTO, chroma=CHROMA_POR_DEFECTO, llm=None, buscador=None, registrar=True):
        self.a = Academico(db)
        self.buscador = buscador or BuscadorNormativa(chroma)
        self.llm = llm or crear_llm()
        self.registrar = registrar
        self.cadena_router = PROMPT_ROUTER | self.llm | StrOutputParser()
        self.cadena_router_v2 = PROMPT_ROUTER_V2 | self.llm | StrOutputParser()
        # ROUTER_MODO=reglas (línea base de la tesis: reglas + desempate con el LLM), llm (v2: decide el LLM) o union
        # (v2: si reglas y LLM discrepan, ruta mixta para recuperar ambas fuentes). Los dos últimos, del 20/11/2026.
        self.modo_router = os.environ.get("ROUTER_MODO", "reglas").lower()
        # Orquestación LCEL: router -> recuperación -> prompt -> LLM -> texto
        self.recuperacion = RunnableLambda(self._enrutar) | RunnableLambda(self._recuperar)
        self.generacion = PROMPT | self.llm | StrOutputParser()
        self.cadena = self.recuperacion | RunnableLambda(self._preparar) | self.generacion
        # La app comparte una sola instancia entre sesiones (un hilo por sesión): la conexión SQLite se usa
        # de a un hilo a la vez y el estado de cada consulta viaja en la cadena, no en el objeto (D-02, P9).
        self._lock = threading.Lock()

    # -- pasos de la cadena
    def _ruta_llm(self, pregunta):
        salida = self.cadena_router_v2.invoke({"pregunta": pregunta})
        ruta = json.loads(re.search(r"\{.*\}", salida, re.S).group())["ruta"]
        if ruta not in ("personal", "normativa", "mixta"):
            raise ValueError(f"ruta desconocida: {ruta}")
        return ruta

    def _enrutar(self, x):
        r = router_reglas(x["pregunta"], self.a)
        if self.modo_router in ("llm", "union"):  # v2: el LLM decide; las reglas aportan intenciones, curso y día
            try:
                ruta_llm = self._ruta_llm(x["pregunta"])
                concreta = [i for i in r["intenciones"] if i != "resumen"]
                if self.modo_router == "union" and r["ruta"] not in (None, ruta_llm) and concreta:
                    # unión de fuentes: si discrepan y las reglas vieron un dato propio concreto, se recuperan ambas
                    r["ruta"], r["router"] = "mixta", f"union (reglas={r['ruta']}, llm={ruta_llm})"
                elif self.modo_router == "union" and r["ruta"] not in (None, ruta_llm):
                    # discrepan pero las reglas solo vieron pronombres («mi», «me»): decide el LLM. Así una norma
                    # preguntada en primera persona no arrastra la ficha del estudiante al contexto (ruido que bajaba
                    # Context Precision en la corrida del 20/11; D-13)
                    r["ruta"], r["router"] = ruta_llm, f"union (llm decide; reglas={r['ruta']} sin dato concreto)"
                else:
                    r["ruta"], r["router"] = ruta_llm, ("union (coinciden)" if self.modo_router == "union" else "llm-v2")
            except Exception:
                r["router"] = "reglas (llm-v2 fallido)"
                if r["ruta"] is None:
                    r["ruta"] = "normativa"
            if r["ruta"] == "normativa":
                r["intenciones"] = []
            elif not r["intenciones"] or r["intenciones"] == ["resumen"]:
                # sin un dato concreto pedido: ficha general; en las mixtas, también su historial de desaprobados
                r["intenciones"] = ["resumen"] + (["desaprobados"] if r["ruta"] == "mixta" else [])
        elif r["ruta"] is None:  # desempate con el LLM
            try:
                salida = self.cadena_router.invoke({"pregunta": x["pregunta"]})
                r["ruta"] = json.loads(re.search(r"\{.*\}", salida, re.S).group())["ruta"]
                r["router"] = "llm"
            except Exception:
                r["ruta"] = "normativa"
                r["router"] = "llm-fallido"
            if r["ruta"] in ("personal", "mixta") and not r["intenciones"]:
                r["intenciones"] = ["resumen"]
        else:
            r["router"] = "reglas"
        if x.get("estudiante_id") is None and r["ruta"] in ("personal", "mixta"):
            r["ruta"] = "normativa"  # sin sesión de estudiante no hay datos personales que consultar
        x.update(r)
        return x

    def _recuperar(self, x):
        eid, ctx = x.get("estudiante_id"), []
        if x["ruta"] in ("personal", "mixta"):
            for i in x["intenciones"]:
                if i == "nota_curso" and x["curso"]:
                    ctx.append(self.a.nota_curso(eid, x["curso"])["texto"])
                elif i == "prerrequisitos" and x["curso"]:
                    ctx.append(self.a.prerrequisitos(x["curso"])["texto"])
                elif i == "puede_llevar" and x["curso"]:
                    ctx.append(self.a.puede_llevar(eid, x["curso"])["texto"])
                elif i == "horario":
                    texto = self.a.horario(eid)["texto"]
                    if x["dia"]:  # solo las sesiones del día preguntado
                        dia = {"miercoles": "Miércoles"}.get(x["dia"], x["dia"].capitalize())
                        partes = [s for s in texto.split(": ", 1)[-1].split("; ") if s.startswith(dia)]
                        texto = f"Horario del estudiante {eid} el {dia.lower()}: " + ("; ".join(partes) or "sin clases") + "."
                    ctx.append(texto)
                else:
                    f = {"promedio": self.a.promedio_ponderado, "avance": self.a.avance_curricular,
                         "notas_periodo": self.a.notas_periodo, "en_curso": self.a.cursos_en_curso,
                         "disponibles": self.a.cursos_disponibles,
                         "desaprobados": self.a.desaprobados, "estado": self.a.estado_matricula,
                         "resumen": self.a.resumen}.get(i)
                    if f:
                        ctx.append(f(eid)["texto"])
            if x["curso"] and "en_curso" in x["intenciones"]:  # p. ej. "si me retiro de X": créditos del curso
                c = self.a.cursos[x["curso"]]
                ctx.append(f"{c['nombre']} ({c['codigo']}) tiene {c['creditos']:.0f} créditos.")
        if x["ruta"] in ("normativa", "mixta"):
            ctx += self.buscador.buscar(x["pregunta"])
        x["contextos"] = list(dict.fromkeys(ctx))  # sin duplicados, en orden
        return x

    def _preparar(self, x):
        return {"pregunta": x["pregunta"], "contexto": "\n\n".join(x["contextos"]) or "(sin contexto)"}

    # -- interfaz pública
    def responder(self, estudiante_id, pregunta):
        t0 = time.perf_counter()
        with self._lock:  # recuperación (SQLite y ChromaDB) de una consulta a la vez
            x = self.recuperacion.invoke({"estudiante_id": estudiante_id, "pregunta": pregunta})
        respuesta = self.generacion.invoke(self._preparar(x))  # el LLM sí atiende consultas en paralelo
        latencia = (time.perf_counter() - t0) * 1000
        if self.registrar:  # RF-12
            with self._lock:
                self._registrar(x, estudiante_id, pregunta, respuesta, latencia)
        return {"respuesta": respuesta.strip(), "contextos": x["contextos"], "ruta": x["ruta"],
                "intenciones": x["intenciones"], "router": x["router"], "latencia_ms": round(latencia, 1)}

    def _registrar(self, x, estudiante_id, pregunta, respuesta, latencia):
        self.a.con.execute(
            "INSERT INTO log_consultas (ts, estudiante_id, ruta, pregunta, respuesta, contextos, latencia_ms) "
            "VALUES (?,?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), estudiante_id, x["ruta"], pregunta, respuesta,
             json.dumps(x["contextos"], ensure_ascii=False), latencia))
        self.a.con.commit()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit('Uso: python agente.py EST0040 "¿Cuál es mi promedio ponderado?"')
    eid = None if sys.argv[1].lower() in ("-", "ninguno") else sys.argv[1]
    r = Agente().responder(eid, sys.argv[2])
    print(f"\nLLM: {os.environ.get('LLM_PROVEEDOR', 'gemini')}")
    print(f"Ruta: {r['ruta']} ({r['router']}) | intenciones: {r['intenciones']} | {r['latencia_ms']} ms")
    print("\nContexto recuperado:")
    for c in r["contextos"]:
        print("  -", c[:220].replace("\n", " "))
    print(f"\nRespuesta:\n{r['respuesta']}")
