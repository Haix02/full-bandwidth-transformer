from app import glu_cross


def test_zero_hidden_forces_zero_input():
    dim = 3
    ident = [[1.0 if i == j else 0.0 for j in range(dim)] for i in range(dim)]
    out = glu_cross([0.0, 0.0, 0.0], [1.2, -0.4, 0.8], ident, ident)
    assert out == [0.0, 0.0, 0.0]
