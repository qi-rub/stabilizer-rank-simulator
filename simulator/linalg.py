import numpy as np


def solve_mod_2(A, b):
    """
    Solve linear system modulo two using Gaussian elimination.
    Returns `None` if system has no solution.

    This function assumes that the matrix has full column rank.

    Attributes
    ==========
    * A --  will most likely be a 2-d array of size (2n x n)
    * b -- a 1-d array of length 2n
    """
    A = np.array(A)
    b = np.array(b)
    assert np.all(A ** 2 == A) and np.all(b ** 2 == b), "Arguments should be binary."
    num_rows, num_cols = A.shape

    # convert system to reduced row echelon form
    col = 0
    for row in range(num_rows):
        for i in range(row, num_rows):
            if A[i, col]:
                break
        else:
            assert False
        if i != row:
            A[[i, row]] = A[[row, i]]
            b[[i, row]] = b[[row, i]]
        for i in range(num_rows):
            if i != row and A[i][col]:
                A[i] = (A[i] - A[row]) % 2
                b[i] = (b[i] - b[row]) % 2
        col += 1
        if col == num_cols:
            break

    assert np.all(A[:num_cols] == np.eye(num_cols))
    assert not np.any(A[num_cols:])

    if np.any(b[num_cols:]):
        return None
    return b[:num_cols]
