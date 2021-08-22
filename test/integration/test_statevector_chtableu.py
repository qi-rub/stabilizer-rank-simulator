import numpy as np
import pytest
from simulator.circuit import Circuit
from simulator.randomcircuit import random_circuit
from simulator.backend.chtableau import run as run_chtableau
from simulator.backend.statevector import run as run_statevector

# Compare the statevector of the chtableau to that of qiskit. In doing so, we are also testing the global phase procedure of the chtableau and its overlap function.


def test_simple():
    # -1 * <basis_state| - >
    circ = Circuit(1)
    circ.h(0)
    circ.s(0)
    circ.s(0)
    circ.h(0)
    circ.s(0)
    circ.s(0)
    circ.h(0)

    assert np.allclose(run_chtableau(circ).statevector(), run_statevector(circ).psi)


def test_2():
    circ = Circuit(1)
    circ.s(0)
    circ.s(0)
    circ.h(0)
    circ.s(0)
    assert np.allclose(run_chtableau(circ).statevector(), run_statevector(circ).psi)


def test_3():
    circ = Circuit(3)
    circ.s(1)
    circ.s(2)
    circ.h(0)
    circ.cx(0, 2)
    circ.s(1)
    assert np.allclose(run_chtableau(circ).statevector(), run_statevector(circ).psi)


def test_4():
    circ = Circuit(2)
    circ.s(0)
    circ.cx(1, 0)
    circ.cz(1, 0)
    circ.cx(1, 0)
    circ.h(1)
    circ.cx(1, 0)
    assert np.allclose(run_chtableau(circ).statevector(), run_statevector(circ).psi)


def test_5():
    circ = Circuit(2)
    circ.h(1)
    circ.s(1)
    circ.h(0)
    assert np.allclose(run_chtableau(circ).statevector(), run_statevector(circ).psi)


def test_6():
    circ = Circuit(2)
    circ.h(1)
    circ.cz(0, 1)
    circ.s(0)
    circ.cx(1, 0)
    circ.h(0)
    circ.s(1)
    circ.h(1)
    assert np.allclose(run_chtableau(circ).statevector(), run_statevector(circ).psi)


@pytest.mark.parametrize(
    "num_qubits, cliffords",
    [(1, 20), (1, 27), (1, 35), (1, 16), (2, 19), (2, 14), (3, 20), (3, 14), (4, 13)],
)
def test_random_circuits(num_qubits, cliffords):
    circ = random_circuit(num_qubits, cliffords)
    assert np.allclose(run_chtableau(circ).statevector(), run_statevector(circ).psi)
