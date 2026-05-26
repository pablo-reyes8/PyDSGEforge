def test_dsge_compatibility_import():
    from src.dsge import DSGE

    assert DSGE.__name__ == "DSGE"
