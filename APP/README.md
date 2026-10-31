# RAGuitoBot: interfaz Streamlit

## Primera vez
1. Instala Streamlit en tu entorno: `pip install streamlit==1.66.1` y actualiza `requirements.txt` con `pip freeze > requirements.txt`.
2. Crea `.env` en la raíz (copia `.env.example`) con `LLM_PROVEEDOR`, `GROQ_API_KEY`, `GROQ_MODEL` o `GOOGLE_API_KEY`.
3. Crea las cuentas (desde `APP/`): `python crear_usuarios.py --evaluador EVAL01`.
   - Te pide la contraseña del evaluador.
   - Las contraseñas de los 70 estudiantes quedan en `APP/credenciales.csv` (no se sube a GitHub).
   - Si reconstruyes `academico.db` con `build_db.py`, vuelve a ejecutar este paso.

## Ejecutar
Desde la raíz `RAGASTESIS`: `streamlit run APP/app.py`

| Rol | Pantallas |
|---|---|
| Estudiante | P-02 Chat |
| Investigador/Evaluador | P-03 Panel · P-04 Actualizar normatividad · P-05 Reporte RAGAS |

## Archivos
- `app.py`: P-01 inicio de sesión, navegación por rol y cierre de sesión.
- `auth.py`: tabla `usuario`, bcrypt y validaciones (RF-19, RF-20, RNF-11).
- `servicios.py`: agente en caché, métricas del testbed y RAGAS, procesos en segundo plano.
- `actualizacion.py`: CU-10 / RF-11, actualizar notas desde un CSV (`estudiante_id, curso, periodo, nota`).
- `paginas/`: `chat.py`, `panel.py`, `normatividad.py` y `reporte.py`.
