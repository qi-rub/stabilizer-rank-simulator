import numpy as np
from simulator.circuit import Circuit
from simulator.backend.chtableau import CHState, run
from simulator.backend.statevector import run as run_statevector
from simulator.samplingch.overlap import overlap_given_basis as overlap
from simulator.randomcircuit import random_circuit


def test_basis_states():
    # <00|00> = 1
    basis_state = np.array([0, 0])
    state = CHState.basis_state([0, 0])
    assert overlap(basis_state, state) == 1

    # <00|11> = 0
    basis_state = np.array([0, 0])
    state = CHState.basis_state([1, 1])
    assert overlap(basis_state, state) == 0

    # <01|01> = 1
    basis_state = np.array([0, 1])
    state = CHState.basis_state([0, 1])
    assert overlap(basis_state, state) == 1

    # <11|11> = 1
    basis_state = np.array([1, 1])
    state = CHState.basis_state([1, 1])
    assert overlap(basis_state, state) == 1

    # <10|01> = 0
    basis_state = np.array([1, 0])
    state = CHState.basis_state([0, 1])
    assert overlap(basis_state, state) == 0

    # <00|11> = 0
    basis_state = np.array([0, 0])
    state = CHState.basis_state([1, 1])
    assert overlap(basis_state, state) == 0


def test_had_basis():
    # <0|+> = 1/sqrt(2)
    basis_state = np.array([0])
    circ = Circuit(1)
    circ.h(0)
    state = run(circ)

    assert np.isclose(overlap(basis_state, state), 1 / np.sqrt(2))

    # < 0 | - > = 1/sqrt(2)
    basis_state = np.array([0])
    circ = Circuit(1)
    circ.h(0)
    circ.s(0)
    circ.s(0)
    state = run(circ)

    assert np.isclose(overlap(basis_state, state), 1 / np.sqrt(2))

    # < 1 | - > = - 1/sqrt(2)
    basis_state = np.array([1])
    circ = Circuit(1)
    circ.h(0)
    circ.s(0)
    circ.s(0)
    state = run(circ)

    assert np.isclose(overlap(basis_state, state), -1 / np.sqrt(2))

    # < 11 | -- > = 1/sqrt(4)
    basis_state = np.array([1, 1])
    circ = Circuit(2)
    circ.h(0)
    circ.s(0)
    circ.s(0)
    circ.h(1)
    circ.s(1)
    circ.s(1)
    state = run(circ)

    assert np.isclose(overlap(basis_state, state), 1 / np.sqrt(4))

    # < 10 | -- > = -1/sqrt(4)
    basis_state = np.array([1, 0])
    circ = Circuit(2)
    circ.h(0)
    circ.s(0)
    circ.s(0)
    circ.h(1)
    circ.s(1)
    circ.s(1)
    state = run(circ)

    assert np.isclose(overlap(basis_state, state), -1 / np.sqrt(4))


def test_bell_states():
    # < 00| phi + > = 1/sqrt(2)
    basis_state = np.array([0, 0])
    circ = Circuit(2)
    circ.h(0)
    circ.cx(0, 1)
    state = run(circ)
    assert np.isclose(overlap(basis_state, state), 1 / np.sqrt(2))

    # < 11 | phi - > = -1/sqrt(2)
    basis_state = np.array([1, 1])
    circ = Circuit(2)
    circ.h(0)
    circ.cx(0, 1)
    circ.s(0)
    circ.s(0)
    state = run(circ)

    assert np.isclose(overlap(basis_state, state), -1 / np.sqrt(2))

    # < 01 | psi - > = 1/sqrt(2)
    basis_state = np.array([0, 1])
    circ = Circuit(2)
    circ.h(0)
    circ.cx(0, 1)
    circ.s(0)
    circ.s(0)
    circ.h(1)
    circ.s(1)
    circ.s(1)
    circ.h(1)
    state = run(circ)

    assert np.isclose(overlap(basis_state, state), 1 / np.sqrt(2))

    # < 10 | psi - > = -1/sqrt(2)
    basis_state = np.array([1, 0])
    circ = Circuit(2)
    circ.h(0)
    circ.cx(0, 1)
    circ.s(0)
    circ.s(0)
    circ.h(1)
    circ.s(1)
    circ.s(1)
    circ.h(1)
    state = run(circ)

    assert np.isclose(overlap(basis_state, state), -1 / np.sqrt(2))


def test_random_circ():
    circ = random_circuit(5, 16)
    state = run(circ)
    statevector = run_statevector(circ).psi
    basis_state = [1, 0, 1, 0, 1]
    pos = int("".join(str(i) for i in basis_state), 2)
    assert np.isclose(overlap(basis_state, state), statevector[pos])


# NEED IMPROVED TESTS
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# def test_given_pauli_basis_states():
#     # <01|01> = 1
#     state = CHState.basis_state([0, 1])
#     basis_state = np.array([0, 0])
#     pauli, pauli_phase = overlap_given_basis(basis_state, state)[1:]
#     y = np.array([0, 1])
#     assert np.isclose(
#         overlap_given_pauli((pauli, pauli_phase), 1, state)[0],
#         overlap_given_basis(y, state)[0],
#     )

#     # <00|11> = 0
#     state = CHState.basis_state([1, 1])
#     basis_state = np.array([0, 1])
#     pauli, pauli_phase = overlap_given_basis(basis_state, state)[1:]
#     y = np.array([0, 0])
#     assert np.isclose(
#         overlap_given_pauli((pauli, pauli_phase), 1, state)[0],
#         overlap_given_basis(y, state)[0],
#     )


# def test_given_pauli_had_basis():
#     # < 1 | - > = - 1/sqrt(2)
#     basis_state = np.array([0])
#     circ = Circuit(1)
#     circ.h(0)
#     circ.s(0)
#     circ.s(0)
#     state = run(circ)

#     pauli, pauli_phase = overlap_given_basis(basis_state, state)[1:]
#     y = np.array([1])
#     assert np.isclose(
#         overlap_given_pauli((pauli, pauli_phase), 0, state)[0],
#         overlap_given_basis(y, state)[0],
#     )

#     # < 10 | -- > = -1/sqrt(4)
#     basis_state = np.array([1, 1])
#     circ = Circuit(2)
#     circ.h(0)
#     circ.s(0)
#     circ.s(0)
#     circ.h(1)
#     circ.s(1)
#     circ.s(1)
#     state = run(circ)

#     pauli, pauli_phase = overlap_given_basis(basis_state, state)[1:]
#     y = np.array([1, 0])
#     assert np.isclose(
#         overlap_given_pauli((pauli, pauli_phase), 1, state)[0],
#         overlap_given_basis(y, state)[0],
#     )


# def test_given_pauli_bell_states():
#     # < 11 | phi - > = -1/sqrt(2)
#     basis_state = np.array([1, 0])
#     circ = Circuit(2)
#     circ.h(0)
#     circ.cx(0, 1)
#     circ.s(0)
#     circ.s(0)
#     state = run(circ)

#     pauli, pauli_phase = overlap_given_basis(basis_state, state)[1:]
#     y = np.array([1, 1])
#     assert np.isclose(
#         overlap_given_pauli((pauli, pauli_phase), 1, state)[0],
#         overlap_given_basis(y, state)[0],
#     )

#     # < 10 | -- > = -1/sqrt(4)
#     basis_state = np.array([0, 0])
#     circ = Circuit(2)
#     circ.h(0)
#     circ.s(0)
#     circ.s(0)
#     circ.h(1)
#     circ.s(1)
#     circ.s(1)
#     state = run(circ)

#     pauli, pauli_phase = overlap_given_basis(basis_state, state)[1:]
#     y = np.array([1, 0])
#     assert np.isclose(
#         overlap_given_pauli((pauli, pauli_phase), 0, state)[0],
#         overlap_given_basis(y, state)[0],
#     )


# def test_simple_given_pauli():
#     circ = Circuit(1)
#     circ.h(0)
#     circ.s(0)
#     state = run(circ)

#     basis_state = [0]
#     pauli, pauli_phase = overlap_given_basis(basis_state, state)[1:]
#     y = np.array([1])
#     assert np.isclose(
#         overlap_given_pauli((pauli, pauli_phase), 0, state)[0],
#         overlap_given_basis(y, state)[0],
#     )


# def test_complex_given_pauli():
#     circ = Circuit(2)
#     circ.h(1)
#     circ.cx(0, 1)
#     circ.s(0)
#     circ.cz(1, 0)
#     state = run(circ)

#     basis_state = [1, 0]
#     pauli, pauli_phase = overlap_given_basis(basis_state, state)[1:]
#     y = np.array([1, 1])
#     assert np.isclose(
#         overlap_given_pauli((pauli, pauli_phase), 1, state)[0],
#         overlap_given_basis(y, state)[0],
#     )


# def test_random_given_pauli():
#     circ = random_circuit(5, 18)
#     state = run(circ)

#     basis_state = [1, 0, 0, 1, 1]
#     pauli, pauli_phase = overlap_given_basis(basis_state, state)[1:]
#     y = np.array([1, 0, 1, 1, 1])
#     assert np.isclose(
#         overlap_given_pauli((pauli, pauli_phase), 2, state)[0],
#         overlap_given_basis(y, state)[0],
#     )
