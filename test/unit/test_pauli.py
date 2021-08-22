import numpy as np
from simulator.pauli import multiply_paulis, pauli_to_matrix
from utils import bitstrings


def test_multiply_paulis_simple():
    p = [0, 1, 0, 1]
    q = [1, 0, 0, 1]
    assert multiply_paulis([p, q]) == ([1, 1, 0, 0], 3)


def test_multiply_paulis():
    n = 2
    for p in bitstrings(2 * n):
        for q in bitstrings(2 * n):
            r, e = multiply_paulis([p, q])
            lhs = pauli_to_matrix(p) @ pauli_to_matrix(q)
            rhs = 1j ** e * pauli_to_matrix(r)
            assert np.allclose(lhs, rhs)
