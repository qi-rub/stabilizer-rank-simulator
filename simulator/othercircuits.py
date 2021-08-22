from simulator.circuit import Circuit


def hthcirc(num_qubits: int) -> Circuit:
    circ = Circuit(num_qubits)
    for i in range(num_qubits):
        circ.h(i)
        circ.t(i)
        circ.h(i)
    return circ
