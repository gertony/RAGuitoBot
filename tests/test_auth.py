"""RF-19, RF-20 y RNF-11: inicio de sesión con contraseña cifrada (bcrypt) y rol."""
import pytest


@pytest.fixture()
def con(tmp_path):
    import auth
    c = auth.conectar(tmp_path / "u.db")
    auth.crear_usuario(c, "EST0001", "clave123", "estudiante", "EST0001")
    auth.crear_usuario(c, "EVAL01", "evalua01", "evaluador")
    return c


def test_no_guarda_la_clave_en_texto_plano(con):
    (h,) = con.execute("SELECT hash FROM usuario WHERE codigo = 'EST0001'").fetchone()
    assert "clave123" not in h and h.startswith("$2")


def test_login_correcto_devuelve_rol_e_id(con):
    import auth
    assert auth.autenticar(con, "est0001", "clave123") == {
        "codigo": "EST0001", "rol": "estudiante", "estudiante_id": "EST0001"}
    assert auth.autenticar(con, "EVAL01", "evalua01")["rol"] == "evaluador"


@pytest.mark.parametrize(("codigo", "clave"), [("EST0001", "otra-clave"), ("EST0099", "clave123"), ("", "")])
def test_login_incorrecto(con, codigo, clave):
    import auth
    assert auth.autenticar(con, codigo, clave) is None


@pytest.mark.parametrize(("codigo", "clave", "valido"), [
    ("EST0001", "clave123", True), ("EST12", "clave123", False), ("EST0001", "123", False), ("", "x", False)])
def test_validacion_del_formulario(codigo, clave, valido):
    import auth
    assert (auth.validar_formulario(codigo, clave) is None) == valido
