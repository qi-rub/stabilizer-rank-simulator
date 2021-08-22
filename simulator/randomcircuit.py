import numpy as np
from .circuit import Circuit


def random_circuit(num_qubits: int, cliffords: int, noncliffords=0) -> Circuit:
    """
    Create a random Clifford or non-Clifford circuit of given size

    Attributes
    ==========
    * num_qubits - number of qubits that circuit will consist of.
    * cliffords - number of Clifford gates in the circuit.
    * noncliffords - number of non-Clifford gates in the circuit.
    """

    if num_qubits == 1:
        c_gates = ["h", "s"]
        nc_gates = ["t"]
    else:
        c_gates = ["h", "s", "cx", "cz"]
        nc_gates = ["t"]

    # lists of gates the circuit will consist of.
    c_gates_list = np.random.choice(c_gates, cliffords)
    nc_gates_list = np.random.choice(nc_gates, noncliffords)

    # indices of the locations where we will insert the non-Clifford gates.
    loc_list = np.random.choice(cliffords, noncliffords)

    # list of randomly chosen qubit indices.
    single_gate_qubit_list = list(map(int, np.random.choice(num_qubits, cliffords)))

    instructions = []
    for i in range(cliffords):
        if c_gates_list[i] == "cx" or c_gates_list[i] == "cz":
            qubs = list(map(int, np.random.choice(num_qubits, 2, replace=False)))
            control, target = qubs[0], qubs[1]
            instructions.append((c_gates_list[i], control, target))
        else:
            instructions.append((c_gates_list[i], single_gate_qubit_list[i]))
    for i in range(noncliffords):
        instructions.insert(loc_list[i], (nc_gates_list[i], single_gate_qubit_list[i]))

    return Circuit(num_qubits, instructions)
