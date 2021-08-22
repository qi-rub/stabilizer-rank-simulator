import pytest
import numpy as np
from simulator.circuit import Circuit
from simulator.backend.statevector import run as run_statevector
from simulator.backend.stabilizer import run as run_stabilizer
from simulator.backend.tableau import run as run_tableau


@pytest.mark.parametrize("run", [run_stabilizer, run_tableau])
def test_teleportation_protocol(run):
    circ = Circuit(3)
    # create state 1/sqrt(2) |0> - |1> for qubit 0. Retrieve its density matrix.
    circ.h(0)
    circ.s(0)

    psi = run(circ)
    dens_mat_q0 = np.trace(psi.density_matrix().reshape((2, 4, 2, 4)), axis1=1, axis2=3)

    # create maximally entangled state between qubits 1 and 2.
    circ.h(1)
    circ.cx(1, 2)
    # start teleportation protocol
    circ.cx(0, 1)
    circ.h(0)

    state = run(circ)

    # measure qubit 0 and qubit 1
    outcomeq0 = state.measure_pauli([1, 0, 0, 0, 0, 0])
    outcomeq1 = state.measure_pauli([0, 1, 0, 0, 0, 0])

    # fix phases
    if outcomeq0 == 1 and outcomeq1 == 1:
        pass
    elif outcomeq0 == -1 and outcomeq1 == 1:
        state.apply_pauli([1, 0], 2)
    elif outcomeq0 == 1 and outcomeq1 == -1:
        state.apply_pauli([0, 1], 2)
    else:
        state.apply_pauli([0, 1], 2)
        state.apply_pauli([1, 0], 2)

    # obtain density matrix of state and trace out the first two systems.
    dens_mat_q2 = np.trace(state.density_matrix().reshape(4, 2, 4, 2), axis1=0, axis2=2)
    assert np.allclose(dens_mat_q0, dens_mat_q2)
