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
NORMATIVA = r"\b(reglamento|que pasa si|requisitos?|nota minima|reserv\w*|desmatricul\w*|retirarme|tope|maximo|" \
            r"bachiller|titulo|de que trata|electivo\w*|rectificacion|extemporane\w*|separa\w*|consecuencia|" \
            r"permite|segun|norma\w*|tres veces|cuarta vez|regular|prorroga|se aprueba|aprobar un|" \
            r"que temas|contenidos?|sumilla|que se ve|que ve|que se necesita|que necesito)\b"
SUMILLA = r"de que trata|que temas|contenidos?|sumilla|que se ve|que ve"
INTENCIONES = [  # (intención, patrón); se evalúan todas y se acumulan
    ("promedio", r"promedio"),
    ("avance", r"creditos .*aprobados|porcentaje|avanzad|me faltan? .*terminar|avance|cuanto me falta"),
    ("notas_periodo", r"notas|ciclo pasado|semestre pasado|ultimo semestre|como me fue en el ultimo"),
    ("en_curso", r"estoy llevando|llevo (matriculad|este)|creditos (estoy )?llevando|cursos .*este (ciclo|semestre)|"
                 r"que llevo|que cursos tengo|me retiro de|creditos llevo|mi carga|carga de creditos"),
    ("horario", r"horario|clases"),
    ("puede_llevar", r"puedo (matricularme|llevar)"),
    ("desaprobados", r"desaprob|repit|volviendo a llevar|varias veces|otra vez"),  # "si desapruebo" (hipótesis) no es dato propio
    ("estado", r"matriculad[oa] este|estado|situacion|en que ciclo"),
    ("avance_bachiller", r"cumplo .*bachiller|tramitar el bachiller"),
]


def _plano(t):
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", " ", t)


def router_reglas(pregunta, academico):
    p = _plano(pregunta)
    curso = academico.buscar_curso(pregunta)
    intenciones = [n for n, pat in INTENCIONES if re.search(pat, p)]
    if curso and not intenciones:
        if re.search(r"requisito", p) and not re.search(PERSONAL, p):
            intenciones = ["prerrequisitos"]
        elif re.search(SUMILLA, p):
            intenciones = []  # contenido del curso: está en la sumilla (ChromaDB), no en la base
        else:
            intenciones = ["nota_curso"]
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
            r"reglamento|que pasa si|consecuencia|requisitos|permite|tope|maximo|sigo siendo|bachiller|titulo|separa", p):
        es_normativa = False
    if es_personal and es_normativa:
        ruta = "mixta"
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
class BuscadorNormativa:
    def __init__(self, ruta_chroma=CHROMA_POR_DEFECTO, modelo=MODELO_EMB, embedding=None):
        import chromadb
        if not Path(ruta_chroma).exists():
            raise FileNotFoundError(f"No se encontró el índice de ChromaDB: {ruta_chroma}")
        self.col = chromadb.PersistentClient(path=str(ruta_chroma)).get_collection("normativa")
        if embedding is None:
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(modelo)
            pref = "query: " if "e5" in modelo.lower() else ""
            embedding = lambda t: m.encode([pref + t], normalize_embeddings=True).tolist()[0]
        self.embedding = embedding

    def buscar(self, pregunta, k=K_NORMATIVA):
        r = self.col.query(query_embeddings=[self.embedding(pregunta)], n_results=k)
        return [doc for doc in r["documents"][0]]


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


def crear_llm(modelo=None):
    """LLM generador. LLM_PROVEEDOR=gemini (por defecto) o groq (respaldo: modelos Llama, Tabla 6 de la tesis).

    Se agregó Groq el 28/10/2026 porque gemini-3.8-flash devolvía 503 (servicio saturado) en el plan gratuito.
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
        # Orquestación LCEL: router -> recuperación -> prompt -> LLM -> texto
        self.recuperacion = RunnableLambda(self._enrutar) | RunnableLambda(self._recuperar)
        self.generacion = PROMPT | self.llm | StrOutputParser()
        self.cadena = self.recuperacion | RunnableLambda(self._preparar) | self.generacion
        # La app comparte una sola instancia entre sesiones (un hilo por sesión): la conexión SQLite se usa
        # de a un hilo a la vez y el estado de cada consulta viaja en la cadena, no en el objeto (D-02, P9).
        self._lock = threading.Lock()

    # -- pasos de la cadena
    def _enrutar(self, x):
        r = router_reglas(x["pregunta"], self.a)
        if r["ruta"] is None:  # desempate con el LLM
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
