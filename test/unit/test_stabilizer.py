import pytest
import numpy as np
from sympy import Matrix
from simulator.backend.stabilizer import StabilizerState, run
from simulator.circuit import Circuit


def test_init():
    # We check that stabilizer state objects are intialized correctly. We do so for different types of input, and check that the when the input is invalid
    # it does throw an error message.
    paulis = [[1, 0, 1, 1, 0, 1], [1, 1, 0, 0, 0, 1], [1, 1, 1, 0, 1, 0]]
    phases = [0, 0, 0]
    stab = StabilizerState(paulis, phases)
    assert stab.num_qubits == 3
    assert np.array_equal(stab.paulis, paulis)
    assert np.array_equal(stab.phases, phases)

    paulis = [[1, 0, 1, 1], [1, 1, 0, 1]]
    phases = [1, 0]
    stab = StabilizerState(paulis, phases)
    assert np.array_equal(stab.paulis, paulis)
    assert np.array_equal(stab.phases, [1, 0])

    # All of the following intializations should create errors.
    with pytest.raises(Exception):
        StabilizerState([[1, 0, 1], [1, 0, 1]])
    with pytest.raises(Exception):
        StabilizerState([[1, 0, 1, 0], [2, 1, 0, 1]])
    with pytest.raises(Exception):
        StabilizerState([[1, 0, 1, 1], [0, 0, 0, 1]], [0])
    with pytest.raises(Exception):
        StabilizerState([[0, 1, 0, 1], [0, 1, 1, 0]], [2, 0])
    with pytest.raises(Exception):
        StabilizerState([[1, 0, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0], [0, 1, 1, 0, 1, 1]])


def test_apply_h():
    # |0> -> |+> -> |0>
    stab = StabilizerState.basis_state([0])
    stab.apply_h(0)
    assert stab == StabilizerState([[0, 1]], [0])
    stab.apply_h(0)
    assert stab == StabilizerState.basis_state([0])

    # |1> -> |-> -> |1>
    stab = StabilizerState.basis_state([1])
    stab.apply_h(0)
    assert stab == StabilizerState([[0, 1]], [1])
    stab.apply_h(0)
    assert stab == StabilizerState.basis_state([1])

    # |L> -> |R> -> |L>
    stab = StabilizerState([[1, 1]], [0])
    stab.apply_h(0)
    assert stab == StabilizerState([[1, 1]], [1])
    stab.apply_h(0)
    assert stab == StabilizerState([[1, 1]], [0])

    # apply a Hadamard gate to the last qubit of |00> and check that the result is |0+> and not |+0>
    stab = StabilizerState.basis_state([0, 0])
    stab.apply_h(1)
    assert np.array_equal(
        stab.paulis,
        [
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ],
    )
    assert np.array_equal(stab.phases, [0, 0])
    plus = [[1 / 2, 1 / 2], [1 / 2, 1 / 2]]
    zero = [[1, 0], [0, 0]]
    assert np.allclose(stab.density_matrix(), np.kron(zero, plus))


def test_apply_s():
    # |0> -> |0>
    stab = StabilizerState.basis_state([0])
    stab.apply_s(0)
    assert np.array_equal(stab.paulis, [[1, 0]])
    assert np.array_equal(stab.phases, [0])

    # |1> -> |1>
    stab = StabilizerState.basis_state([1])
    stab.apply_s(0)
    assert np.array_equal(stab.paulis, [[1, 0]])
    assert np.array_equal(stab.phases, [1])

    # |+> -> |L> -> |-> -> |R> -> |+>
    stab = StabilizerState([[0, 1]], [0])
    stab.apply_s(0)
    assert stab == StabilizerState([[1, 1]], [0])
    stab.apply_s(0)
    assert stab == StabilizerState([[0, 1]], [1])
    stab.apply_s(0)
    assert stab == StabilizerState([[1, 1]], [1])
    stab.apply_s(0)
    assert stab == StabilizerState([[0, 1]], [0])


def test_cx_on_basis_states():
    # |00> -> |00>
    stab = StabilizerState.basis_state([0, 0])
    stab.apply_cx(0, 1)
    assert stab == StabilizerState.basis_state([0, 0])

    # |01> -> |01>
    stab = StabilizerState.basis_state([0, 1])
    stab.apply_cx(0, 1)
    assert stab == StabilizerState.basis_state([0, 1])

    # |10> -> |11> -> |10>
    stab = StabilizerState.basis_state([1, 0])
    stab.apply_cx(0, 1)
    assert stab == StabilizerState.basis_state([1, 1])
    stab.apply_cx(0, 1)
    assert stab == StabilizerState.basis_state([1, 0])


def test_bell_states():
    # We check that the Bell circuit (Hadamard on the first qubit and then a
    # CNOT controlled on the first qubit) produces the Bell basis states.
    # |00> + |11>
    stab = StabilizerState.basis_state([0, 0])
    stab.apply_h(0)
    stab.apply_cx(0, 1)
    assert stab == StabilizerState(np.array([[1, 1, 0, 0], [0, 0, 1, 1]]), [0, 0])

    # |00> - |11>
    stab = StabilizerState.basis_state([1, 0])
    stab.apply_h(0)
    stab.apply_cx(0, 1)
    assert stab == StabilizerState(np.array([[1, 1, 0, 0], [0, 0, 1, 1]]), [0, 1])

    # |01> + |10>
    stab = StabilizerState.basis_state([0, 1])
    stab.apply_h(0)
    stab.apply_cx(0, 1)
    assert stab == StabilizerState(np.array([[1, 1, 0, 0], [0, 0, 1, 1]]), [1, 0])

    # |01> - |10>
    stab = StabilizerState.basis_state([1, 1])
    stab.apply_h(0)
    stab.apply_cx(0, 1)
    assert stab == StabilizerState(np.array([[1, 1, 0, 0], [0, 0, 1, 1]]), [1, 1])


def test_density_matrix():
    # Check that we can retrieve the correct density matrix from the StabilizerState object and its generators.
    state = StabilizerState(np.array([[1, 1, 0, 0], [0, 0, 1, 1]]), [0, 0])
    density_mat = state.density_matrix()
    result = np.array([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]])
    assert np.array_equal(density_mat, result)


def test_pauli_expectation():
    state = StabilizerState(np.array([[1, 1, 0, 0], [0, 0, 1, 1]]), [0, 0])
    assert state.pauli_expectation([1, 1, 0, 0]) == 1
    assert state.pauli_expectation([0, 0, 1, 1]) == 1
    assert state.pauli_expectation([1, 1, 1, 1]) == -1


def test_measure_pauli():
    # Measurements of paulis that commute with all generators. State does not change.
    state = StabilizerState(np.array([[1, 1, 0, 0], [0, 0, 1, 1]]), [0, 0])
    assert state.measure_pauli([1, 1, 0, 0]) == 1
    assert state.measure_pauli([0, 0, 1, 1]) == 1
    assert state.measure_pauli([1, 1, 1, 1]) == -1
    assert state == StabilizerState(np.array([[1, 1, 0, 0], [0, 0, 1, 1]]), [0, 0])

    # Measurement of a pauli that anti-commutes with one generator. Has a post-measurement state.
    state.measure_pauli([1, 0, 0, 0], random_bit=0)
    post_measurement_state = StabilizerState(
        np.array([[1, 1, 0, 0], [1, 0, 0, 0]]), [0, 0]
    )
    assert state == post_measurement_state

    # Important test! Helped catch a bug.
    state.measure_pauli([1, 1, 1, 1], random_bit=0)
    post_measurement_state = StabilizerState(
        np.array([[1, 1, 0, 0], [1, 1, 1, 1]]), [0, 0]
    )
    assert state == post_measurement_state

    # Measurement of a pauli that anti-commutes with both generators. Has a post-measurement state.
    state.measure_pauli([0, 0, 0, 1], random_bit=1)
    post_measurement_state = StabilizerState(
        np.array([[0, 0, 0, 1], [0, 0, 1, 1]]), [1, 1]
    )
    assert state == post_measurement_state


def test_apply_pauli():
    # Z(X)Z = -X
    state = StabilizerState(np.array([[0, 1]]), [0])
    state.apply_pauli([1, 0], 0)
    assert state == StabilizerState([[0, 1]], [1])

    # X(-X)X = -X
    state = StabilizerState(np.array([[0, 1]]), [1])
    state.apply_pauli([0, 1], 0)
    assert state == StabilizerState([[0, 1]], [1])

    # Y(-Z)Y = Z
    state = StabilizerState(np.array([[1, 0]]), [1])
    state.apply_pauli([1, 1], 0)
    assert state == StabilizerState([[1, 0]], [0])

    # (Y x Z x X )|000> = |101>
    state = StabilizerState.basis_state([0, 0, 0])
    state.apply_pauli([1, 1, 0, 1, 0, 1])
    assert state == StabilizerState(
        [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], [1, 0, 1]
    )


def test_run():
    # We test a very simple circuit
    circ = Circuit(2)
    circ.h(0)
    circ.cx(0, 1)
    state = run(circ)
    assert state == StabilizerState([[1, 1, 0, 0], [0, 0, 1, 1]], [0, 0])


def test_run2():
    circ = Circuit(2)
    circ.h(0)
    circ.cx(0, 1)
    # |00> + |11> -> |00> - |11>
    circ.cz(0, 1)
    state = run(circ)

    assert np.allclose(
        state.density_matrix(),
        np.array([[0.5, 0, 0, -0.5], [0, 0, 0, 0], [0, 0, 0, 0], [-0.5, 0, 0, 0.5]]),
    )
