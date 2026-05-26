def test_dsge_compatibility_import():
    from src.dsge import DSGE

    assert DSGE.__name__ == "DSGE"


def test_dsge_latex_repr_has_readable_signature():
    import sympy as sp
    from src.dsge import DSGE

    x_t, eps_t = sp.symbols("x_t eps_t")
    model = DSGE([sp.Eq(x_t, eps_t)], [x_t], eps_t=[eps_t])

    latex = model._repr_latex_()

    assert "1 equation | 1 state | 0 leads | 0 lags | 1 shock" in latex
    assert "1eqs" not in latex
