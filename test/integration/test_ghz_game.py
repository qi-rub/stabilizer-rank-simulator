import numpy as np
import pytest
from simulator.circuit import Circuit
from simulator.backend.stabilizer import run as run_stabilizer
from simulator.backend.statevector import run as run_statevector
from simulator.backend.tableau import run as run_tableau
from simulator.backend.chtableau import run as run_chtableau


@pytest.mark.parametrize(
    "run", [run_statevector, run_stabilizer, run_tableau, run_chtableau]
)
def test_ghz_measurement(run):
    # test all possible inputs (questions)
    for a, b, c in [(0, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1)]:
        # create |000>-|011>-|101>-|110>
        circ = Circuit(3)
        circ.h(0)
        circ.h(1)
        circ.cx(0, 2)
        circ.cx(1, 2)
        circ.s(0)
        circ.s(1)
        circ.s(2)

        # perform strategy corresponding to given questions
        if a:
            circ.h(0)
        if b:
            circ.h(1)
        if c:
            circ.h(2)
        state = run(circ)
        want = 1 if a == b == c == 0 else -1
        assert np.isclose(state.pauli_expectation([1, 1, 1, 0, 0, 0]), want)
