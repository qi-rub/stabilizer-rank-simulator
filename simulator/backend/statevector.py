import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from ..state import State
from ..pauli import pauli_to_matrix


class VectorState(State):
    """State represented by a NumPy (column) vector."""

    def __init__(self, psi):
        """Initialize given a one-dimensional numpy array."""
        assert len(psi.shape) == 1
        self.psi = np.array([psi]).T

    def __repr__(self):
        return repr(self.psi)

    def density_matrix(self):
        return self.psi @ self.psi.T.conj()

    def pauli_expectation(self, pauli):
        m = pauli_to_matrix(pauli)
        result = self.psi.T.conj() @ m @ self.psi
        assert len(result) == 1
        return result[0][0]


def run(circuit):
    """Run Circuit using qiskit and return a VectorState."""
    # declare the backend before the "save_statevector()" line.
    backend = Aer.get_backend("aer_simulator_statevector")
    # run using qiskit
    qiskit_circuit = QuantumCircuit.from_qasm_str(circuit.qasm)
    qiskit_circuit.save_statevector()
    psi = execute(qiskit_circuit, backend).result().get_statevector()

    # qiskit uses qubit order |q_{n-1} ... q_0> while we use |q_0 ... q_{n-1}>,
    # so we need to reverse the order.
    psi = psi.reshape([2] * circuit.num_qubits).transpose().reshape(-1)

    return VectorState(psi)
