"""This test file runs entire circuits using our stabilizer simulators."""

import pytest
import numpy as np
from simulator.circuit import Circuit
from simulator.backend.stabilizer import run as run_stabilizer
from simulator.backend.statevector import run as run_statevector
from simulator.backend.tableau import run as run_tableau
from simulator.backend.chtableau import run as run_chtableau
from simulator.randomcircuit import random_circuit

# The idea is that we are testing complicated random circuits, and obtaining the state's density matrix.
# We simulate the same circuit with Qiskit, and check that the two density matrices are in fact the same.
@pytest.mark.parametrize(
    "num_qubits, cliffords", [(6, 4), (5, 9), (3, 20), (4, 13), (2, 40)]
)
@pytest.mark.parametrize(
    "run", [run_statevector, run_stabilizer, run_tableau, run_chtableau]
)
def test_clifford_vs_state_vector(run, num_qubits, cliffords):
    circ = random_circuit(num_qubits, cliffords)
    state_got = run(circ)
    state_want = run_statevector(circ)
    assert np.allclose(state_got.density_matrix(), state_want.density_matrix())


# For a very simple circuit we ask to find the expected value of different paulis.
@pytest.mark.parametrize(
    "run", [run_statevector, run_stabilizer, run_tableau, run_chtableau]
)
def test_pauli_expectation(run):
    # create a maximally entangled state
    circ = Circuit(2)
    circ.h(0)
    circ.cx(0, 1)
    state = run(circ)

    result = state.pauli_expectation([1, 1, 1, 1])
    assert np.isclose(result, -1)

    result = state.pauli_expectation([1, 0, 0, 1])
    assert np.isclose(result, 0)

    result = state.pauli_expectation([1, 1, 0, 0])
    assert np.isclose(result, 1)


# For a very simple circuit we compute the density matrix
@pytest.mark.parametrize(
    "run", [run_statevector, run_stabilizer, run_tableau, run_chtableau]
)
def test_density_matrix(run):
    # create a maximally entangled state
    circ = Circuit(2)
    circ.h(0)
    circ.cx(0, 1)
    state = run(circ)

    rho = state.density_matrix()
    assert np.allclose(
        rho, [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]]
    )
