import pytest
import random
from simulator.randomcircuit import random_circuit
from simulator.backend.statevector import run as run_statevector
from simulator.backend.stabilizer import run as run_stabilizer
from simulator.backend.tableau import run as run_tableau


def test_create(benchmark):
    benchmark(random_circuit, num_qubits=100, cliffords=1000)


def make_id(val):
    if callable(val):
        mod = val.__module__
        return mod[mod.rfind(".") + 1 :]


@pytest.mark.parametrize(
    "run, num_qubits, cliffords",
    [
        (run_statevector, 10, 1000),
        (run_statevector, 20, 1000),
        (run_stabilizer, 10, 10),
        (run_stabilizer, 10, 1000),
        (run_stabilizer, 100, 10),
        (run_stabilizer, 100, 1000),
        (run_stabilizer, 100, 10000),
        (run_stabilizer, 250, 10),
        (run_tableau, 10, 10),
        (run_tableau, 10, 1000),
        (run_tableau, 100, 10),
        (run_tableau, 100, 1000),
        (run_tableau, 100, 10000),
        (run_tableau, 250, 10),
    ],
    ids=make_id,
)
def test_run(run, num_qubits, cliffords, benchmark):
    random.seed(42)
    # run a random circuit
    circ = random_circuit(num_qubits, cliffords)
    state = benchmark(run, circ)
