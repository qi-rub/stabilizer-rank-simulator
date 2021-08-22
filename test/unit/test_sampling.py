import os
import numpy as np
import pickle as pk
from simulator.randomcircuit import random_circuit
from simulator.samplingch.parallelsampling import obtain_samples
from simulator.backend.statevector import run
from simulator.samplingch.statedecomposer import stabilizer_decomposition
from simulator.circuit import Circuit
from cliffords.decompositionsobj import DecompositionTerm
from simulator.counter import BinCounter

CLIFF_DIR = os.path.dirname(os.path.abspath("ncdecompositions.p"))

# load the decomposition object
gate_decompositions = pk.load(
    open(os.path.join(CLIFF_DIR, "cliffords/ncdecompositions.p"), "rb")
)

np.random.seed(16)


def test_cliffordstate():
    # State: |psi> = | phi+ >

    num_qubits = 2
    # Define circuit
    circ = Circuit(num_qubits)
    circ.h(0)
    circ.cx(0, 1)

    # Compute prob
    statevector = run(circ).psi.flatten()
    theoretic_pr = abs(statevector) ** 2

    # Sample elements
    num_samples = 500
    statescoeff = stabilizer_decomposition(
        circ, delta=0.1, nc_decompositions=gate_decompositions
    )
    samples_l = obtain_samples(num_samples, statescoeff, x=[0, 1], iters=100, cores=4)

    # Create the count dictionary
    c = BinCounter(num_qubits, all=True)
    c.update(samples_l)

    # Compute prob from sampled elements
    outcome_pr = []
    for item, val in c.counts.items():
        outcome_pr.append(val / num_samples)

    # as long as the two digits after the decimal point match we accept it
    assert np.allclose(theoretic_pr, outcome_pr, atol=10e-2)


def test_simple():
    # This tests is funny because if the number of iterations is even, the chain always returns to the place it started, and so does not give the proper statistics. We obtain the correct answer when the initial state of the chain is chosen at random.

    num_qubits = 1
    # Define circuit
    circ = Circuit(num_qubits)
    circ.h(0)
    circ.t(0)

    # Compute prob
    statevector = run(circ).psi.flatten()
    theoretic_pr = abs(statevector) ** 2

    # Sample elements
    num_samples = 200
    statescoeff = stabilizer_decomposition(
        circ, delta=0.1, nc_decompositions=gate_decompositions
    )
    samples_l = obtain_samples(num_samples, statescoeff, iters=100, cores=4)

    # Create the count dictionary
    c = BinCounter(num_qubits, all=True)
    c.update(samples_l)

    # Compute prob from sampled elements
    outcome_pr = []
    for item, val in c.counts.items():
        outcome_pr.append(val / num_samples)

    assert np.allclose(theoretic_pr, outcome_pr, atol=10e-2)


def test_random_circ1():
    num_qubits = 2
    circ = random_circuit(num_qubits, 5, 2)

    # Compute prob
    statevector = run(circ).psi.flatten()
    theoretic_pr = abs(statevector) ** 2

    # Sample elements
    num_samples = 200
    statescoeff = stabilizer_decomposition(
        circ, delta=0.1, nc_decompositions=gate_decompositions
    )
    samples_l = obtain_samples(num_samples, statescoeff, iters=100, cores=4)

    # Create the count dictionary
    c = BinCounter(num_qubits, all=True)
    c.update(samples_l)

    # Compute prob from sampled elements
    outcome_pr = []
    for item, val in c.counts.items():
        outcome_pr.append(val / num_samples)

    assert np.allclose(theoretic_pr, outcome_pr, atol=10e-2)


def test_random_circ2():
    num_qubits = 4
    circ = random_circuit(num_qubits, 14, 4)

    # Compute prob
    statevector = run(circ).psi.flatten()
    theoretic_pr = abs(statevector) ** 2

    # Sample elements
    num_samples = 200
    statescoeff = stabilizer_decomposition(
        circ, delta=0.1, nc_decompositions=gate_decompositions
    )
    samples_l = obtain_samples(num_samples, statescoeff, iters=100, cores=4)

    # Create the count dictionary
    c = BinCounter(num_qubits, all=True)
    c.update(samples_l)

    # Compute prob from sampled elements
    outcome_pr = []
    for item, val in c.counts.items():
        outcome_pr.append(val / num_samples)

    assert np.allclose(theoretic_pr, outcome_pr, atol=10e-2)