import pytest
import numpy as np
from sympy import Matrix
from simulator.backend.tableau import Tableau, run
from simulator.circuit import Circuit


def test_init():
    # We check that the Tableau object intializes correctly. We do so for different types of input, and check that the when the input is invalid
    # it does throw an error message.
    paulis = [[1, 0, 1, 1], [1, 1, 0, 0], [1, 0, 0, 1], [1, 1, 1, 0]]
    phases = [0, 0, 0, 0]
    tableau = Tableau(paulis, phases)
    assert np.array_equal(tableau.tableau, paulis)
    assert np.array_equal(tableau.phases, phases)
    assert tableau.num_qubits == 2

    # All of the following initializations should create errors.
    with pytest.raises(Exception):
        Tableau([[1, 0, 1], [1, 0, 1]])
    with pytest.raises(Exception):
        Tableau([[1, 0, 1, 0], [2, 1, 0, 1], [0, 1, 1, 0], [0, 0, 0, 1]])
    with pytest.raises(Exception):
        Tableau([[1, 0, 1, 1], [0, 0, 0, 1], [0, 1, 1, 0], [0, 0, 0, 1]], [0])
    with pytest.raises(Exception):
        Tableau([[0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1]], [2, 0])


def test_basis_state():
    # check we obtain the correct tableau and phases for basis state |010>.
    paulis = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )
    state = Tableau.basis_state([0, 1, 0])
    assert np.array_equal(state.tableau, paulis)
    assert np.array_equal(state.phases, [0, 1, 0, 0, 0, 0])


def test_apply_h():
    # |0> -> |+> -> |0>
    tab = Tableau.basis_state([0])
    tab.apply_h(0)
    assert tab == Tableau([[0, 1], [1, 0]], [0, 0])
    tab.apply_h(0)
    assert tab == Tableau.basis_state([0])

    # |1> -> |-> -> |1>
    tab = Tableau.basis_state([1])
    tab.apply_h(0)
    assert tab == Tableau([[0, 1], [1, 0]], [1, 0])
    tab.apply_h(0)
    assert tab == Tableau.basis_state([1])

    # |L> -> |R> -> |L>
    tab = Tableau([[1, 1], [0, 1]], [0, 0])
    tab.apply_h(0)
    assert tab == Tableau([[1, 1], [0, 1]], [1, 0])
    tab.apply_h(0)
    assert tab == Tableau([[1, 1], [0, 1]], [0, 0])

    # apply a Hadamard gate to the last qubit of |00> and check that the result is |0+> and not |+0>
    tab = Tableau.basis_state([0, 0])
    tab.apply_h(1)
    assert np.array_equal(
        tab.tableau, [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]
    )
    assert np.array_equal(tab.phases, [0, 0, 0, 0])
    plus = [[1 / 2, 1 / 2], [1 / 2, 1 / 2]]
    zero = [[1, 0], [0, 0]]
    assert np.allclose(tab.density_matrix(), np.kron(zero, plus))


def test_apply_s():
    # |0> -> |0>
    tab = Tableau.basis_state([0])
    tab.apply_s(0)
    assert np.array_equal(tab.tableau, [[1, 0], [1, 1]])
    assert np.array_equal(tab.phases, [0, 0])

    # |1> -> |1>
    tab = Tableau.basis_state([1])
    tab.apply_s(0)
    assert np.array_equal(tab.tableau, [[1, 0], [1, 1]])
    assert np.array_equal(tab.phases, [1, 0])

    # |+> -> |L> -> |-> -> |R> -> |+>
    tab = Tableau([[0, 1], [1, 0]], [0, 0])
    tab.apply_s(0)
    assert tab == Tableau([[1, 1], [1, 0]], [0, 0])
    tab.apply_s(0)
    assert tab == Tableau([[0, 1], [1, 0]], [1, 0])
    tab.apply_s(0)
    assert tab == Tableau([[1, 1], [1, 0]], [1, 0])
    tab.apply_s(0)
    assert tab == Tableau([[0, 1], [1, 0]], [0, 0])


def test_cx_on_basis_states():
    # |00> -> |00>
    tab = Tableau.basis_state([0, 0])
    tab.apply_cx(0, 1)
    assert tab == Tableau.basis_state([0, 0])

    # |01> -> |01>
    tab = Tableau.basis_state([0, 1])
    tab.apply_cx(0, 1)
    assert tab == Tableau.basis_state([0, 1])

    # |10> -> |11> -> |10>
    tab = Tableau.basis_state([1, 0])
    tab.apply_cx(0, 1)
    assert tab == Tableau.basis_state([1, 1])
    tab.apply_cx(0, 1)
    assert tab == Tableau.basis_state([1, 0])


def test_bell_states():
    # We check that the Bell circuit (Hadamard on the first qubit and then a
    # CNOT controlled on the first qubit) produces the Bell basis states.
    # |00> + |11>
    tab = Tableau.basis_state([0, 0])
    tab.apply_h(0)
    tab.apply_cx(0, 1)
    assert tab == Tableau(
        [[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 0]], [0, 0, 0, 0]
    )

    # |00> - |11>
    tab = Tableau.basis_state([1, 0])
    tab.apply_h(0)
    tab.apply_cx(0, 1)
    assert tab == Tableau(
        [[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 0]], [0, 1, 0, 0]
    )

    # |01> + |10>
    tab = Tableau.basis_state([0, 1])
    tab.apply_h(0)
    tab.apply_cx(0, 1)
    assert tab == Tableau(
        [[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 0]], [1, 0, 0, 0]
    )

    # |01> - |10>
    tab = Tableau.basis_state([1, 1])
    tab.apply_h(0)
    tab.apply_cx(0, 1)
    assert tab == Tableau(
        [[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 0]], [1, 1, 0, 0]
    )


def test_density_matrix():
    state = Tableau(
        [[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 0]], [0, 0, 0, 0]
    )
    density_mat = state.density_matrix()
    result = np.array([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]])
    assert np.array_equal(density_mat, result)


def test_pauli_expectation():
    state = Tableau(
        np.array(
            [
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
            ]
        ),
        [0, 0, 0, 0],
    )
    assert state.pauli_expectation([1, 1, 0, 0]) == 1
    assert state.pauli_expectation([0, 0, 1, 1]) == 1
    assert state.pauli_expectation([1, 1, 1, 1]) == -1


def test_measure_pauli():
    # Measurements of paulis that commute with all generators. Has no post-measurement state.
    state = Tableau(
        [[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 0]], [0, 0, 0, 0]
    )
    assert state.measure_pauli([1, 1, 0, 0]) == 1
    assert state.measure_pauli([0, 0, 1, 1]) == 1
    assert state.measure_pauli([1, 1, 1, 1]) == -1
    assert state == Tableau(
        [[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 0]], [0, 0, 0, 0]
    )

    # Measurement of a pauli that anti-commutes with one generator. Has a post-measurement state.
    state.measure_pauli([1, 0, 0, 0], random_bit=0)
    post_measurement_state = Tableau(
        [[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1]], [0, 0, 0, 0]
    )
    assert state == post_measurement_state

    # Important test! Helped catch a bug.
    state.measure_pauli([1, 1, 1, 1], random_bit=0)
    post_measurement_state = Tableau(
        [[1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 0]], [0, 0, 0, 0]
    )
    assert state == post_measurement_state

    # Measurement of a pauli that anti-commutes with both generators. Has a post-measurement state.
    state.measure_pauli([0, 0, 0, 1], random_bit=1)
    post_measurement_state = Tableau(
        np.array([[0, 0, 0, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]]), [1, 1, 0, 0]
    )
    assert state == post_measurement_state


def test_apply_pauli():
    # Z(X)Z = -X
    state = Tableau(np.array([[0, 1], [1, 0]]), [0, 0])
    state.apply_pauli([1, 0], 0)
    assert state == Tableau([[0, 1], [1, 0]], [1, 0])

    # X(-X)X = -X
    state = Tableau(np.array([[0, 1], [1, 0]]), [1, 0])
    state.apply_pauli([0, 1], 0)
    assert state == Tableau([[0, 1], [1, 0]], [1, 1])

    # Y(-Z)Y = Z
    state = Tableau(np.array([[1, 0], [0, 1]]), [1, 1])
    state.apply_pauli([1, 1], 0)
    assert state == Tableau([[1, 0], [0, 1]], [0, 0])

    # (Y x Z x X )|000> = |101>
    state = Tableau.basis_state([0, 0, 0])
    state.apply_pauli([1, 1, 0, 1, 0, 1])
    assert state == Tableau(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        [1, 0, 1, 1, 1, 0],
    )


def test_run():
    circ = Circuit(2)
    circ.h(0)
    circ.cx(0, 1)
    tab = run(circ)
    assert tab == Tableau(
        [[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 0]], [0, 0, 0, 0]
    )
