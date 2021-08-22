import numpy as np
from simulator.circuit import Circuit
from simulator.backend.statevector import run

# We test a very simple circuit
def test_run():
    circ = Circuit(2)
    circ.h(0)
    circ.cx(0, 1)
    rho = run(circ).density_matrix()
    assert np.allclose(
        rho, [[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]]
    )
