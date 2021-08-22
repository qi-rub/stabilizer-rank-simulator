import numpy as np
import pytest
from simulator.linalg import solve_mod_2
from utils import bitstrings


def test_solve_mod_2_easy():
    A = [[0, 1], [1, 1]]
    b = [1, 1]
    assert np.array_equal(solve_mod_2(A, b), [0, 1])

    A = [[1], [1]]
    b = [1, 1]
    assert np.array_equal(solve_mod_2(A, b), [1])

    b = [1, 0]
    assert solve_mod_2(A, b) is None


def test_solve_mod_2_many():
    n = 5
    for a1 in bitstrings(n):
        for a2 in bitstrings(n):
            # skip linearly dependent case
            a1 = np.array(a1)
            a2 = np.array(a2)
            if not (a1.any() and a2.any() and (a1 != a2).any()):
                continue

            # solve full rank linear system Ax = b (of size 5x2) for all possible x
            A = np.array([a1, a2]).T
            for x in bitstrings(2):
                b = (A @ x) % 2
                assert np.array_equal(solve_mod_2(A, b), x)
