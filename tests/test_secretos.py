"""RNF-10: ninguna clave de API en el código ni fuera de .env (P8 · II.4.7)."""
import pathlib
import re

RAIZ = pathlib.Path(__file__).resolve().parents[1]
PATRONES = {
    "Groq": re.compile(r"gsk_[A-Za-z0-9]{20,}"),
    "Google/Gemini": re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    "OpenAI": re.compile(r"sk-[A-Za-z0-9]{20,}"),
}
EXTENSIONES = {".py", ".toml", ".yml", ".yaml", ".json", ".md", ".ipynb"}
IGNORAR = {
    ".git", ".venv", "venv", ".idea", "__pycache__",
    "chroma_e5", "chroma_normativa", "CHROMADM",
}


def _archivos():
    for p in RAIZ.rglob("*"):
        if p.is_file() and not IGNORAR & set(p.parts) and p.suffix in EXTENSIONES:
            yield p


def test_sin_claves_en_el_repositorio():
    hallazgos = [
        (str(p.relative_to(RAIZ)), nombre)
        for p in _archivos()
        for nombre, rx in PATRONES.items()
        if rx.search(p.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert hallazgos == []


def test_env_ignorado_por_git():
    gitignore = RAIZ / ".gitignore"
    assert gitignore.exists()
    lineas = gitignore.read_text().splitlines()
    assert any(linea.strip() in {".env", "*.env", "/.env"} for linea in lineas)
