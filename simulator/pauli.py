"""
This module provides various functions for manipulating Pauli operators.

N-qubit Pauli operators are always encoded by bitstrings of length 2n of the from

  [z|x] = [z_0 ... z_{n-1} | x_0 ... x_{n-1}]

where z and x are bitstrings of length n. The Pauli operator correponding to such a bitstring is

  P(z, x) = P(z_0, x_0) otimes ... otimes P(z_{n-1}, x_{n-1})

where

  P(0, 0) = I,
  P(0, 1) = X,
  P(1, 0) = Z,
  P(1, 1) = Y.

Throughout, `pauli` always refers to an n-qubit Pauli operator encoded in the way described above.
"""
import numpy as np


# single-qubit Pauli operators
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

PAULI = {(0, 0): I, (0, 1): X, (1, 1): Y, (1, 0): Z}


def is_pauli(pauli):
    """Return True if `pauli` is a bitstring of even length."""
    pauli = np.asarray(pauli)
    return np.all(pauli ** 2 == pauli) and len(pauli) % 2 == 0


def symplectic_form(n):
    """Return 2n x 2n symplectic matrix that guides the commutation relations of Pauli operators."""
    zero = np.zeros((n, n), dtype=int)
    eye = np.eye(n, dtype=int)
    return np.block([[zero, eye], [-eye, zero]])


def pauli_to_matrix(pauli):
    """Return matrix representation of given Pauli operator."""
    assert is_pauli(pauli)
    num_qubits = len(pauli) // 2

    m = 1
    for (z, x) in zip(pauli[:num_qubits], pauli[num_qubits:]):
        m = np.kron(m, PAULI[z, x])
    return m


def pauli_to_binary_str(pauli):
    """
    Return human-readable binary string representation of given Pauli operator.
    The format looks like "z_0z_1... x_0x_1...".
    """
    assert is_pauli(pauli)
    num_qubits = len(pauli) // 2
    z, x = pauli[:num_qubits], pauli[num_qubits:]
    return "".join(map(str, z)) + " " + "".join(map(str, x))


def multiply_paulis(paulis):
    """
    Return product of a collection of Pauli operators.

    Since a product of Pauli operators is always equal to a Pauli operator
    times an overall phase that is a power of $i = sqrt(-1)$, the function
    returns a pair
        (pauli, exp_mod_4)
    where `pauli` is a bitstring representation the Pauli operator
    and `exp_mod_4` is the exponent of $i$ modulo 4.
    """
    paulis = np.array(paulis)
    assert np.all(paulis ** 2 == paulis)
    num_qubits = len(paulis[0]) // 2

    # multiply paulis (resulting z, x need not be in {0,1})
    z, x = paulis[0][:num_qubits], paulis[0][num_qubits:]
    exp_mod_4 = 0
    for p in paulis[1:]:
        z_, x_ = p[:num_qubits], p[num_qubits:]
        exp_mod_4 += z @ x_ - z_ @ x
        z += z_
        x += x_

    # normalize z, x back to {0,1}
    z_norm = z % 2
    x_norm = x % 2
    exp_mod_4 += z_norm @ x_norm - z @ x

    return list(z_norm) + list(x_norm), exp_mod_4 % 4
