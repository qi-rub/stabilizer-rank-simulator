import re
import numpy as np
from qiskit import QuantumCircuit


class Circuit:
    """
    Quantum circuit.

    The "instructions" property contains a list of instructions which can be of the following form:
    - Single qubit gates:
        ("z", qubit)
    - Single qubit gates with argument
        ("rphi", angle, qubit)
    - Control two-qubit gates
        ("cx", control, target)
    - Unordered two-qubit gates
        ("swap", qubit1, qubit2)
    - Unordered two-qubit gates with argument
        ("crphi", angle, control, target)

    * Note: some of the gates included in this simulator are not standard and so have no standard designated name in QASM. For this reason we declare them in terms of native gates that Qiskit id familiar with.
    """

    def __init__(self, num_qubits, instructions=None):
        """
        Initialize a quantum circuit.

        The function only requires one input, the number of qubits that our circuit should contain.
        The other argument is used internally by the classmethods "from_qasm" and "from_qasm_file" outlined below.
        """
        assert isinstance(num_qubits, int) or isinstance(
            num_qubits, np.int64
        )  # The number of qubits has to be an integer.
        assert num_qubits > 0  # The number of qubits has to be greater than zero.

        self.num_qubits = num_qubits
        self.instructions = instructions if instructions is not None else []

    def __repr__(self):
        return repr(self.instructions)

    def __len__(self):
        return len(self.instructions)

    def draw(self):
        new_instructions = []
        for instruction in self.instructions:
            l = list(instruction)
            if instruction[0] in [
                "cx",
                "cz",
                "x",
                "y",
                "z",
                "t",
                "h",
                "s",
                "sx",
                "sy",
                "sw",
                "barrier",
                "id",
            ]:
                pass
            elif len(instruction) == 3:
                l[0] = "cz"
            elif len(instruction) == 4:  # change crphi to a normal cz
                new_name = "cz"
                l = [new_name] + l[2:]
            instruction = tuple(l)
            new_instructions.append(instruction)
        new_circ = Circuit(self.num_qubits, new_instructions)
        qasm = new_circ.qasm
        circuit = QuantumCircuit.from_qasm_str(qasm)
        return circuit.draw(plot_barriers=False)

    @property
    def qasm(self):
        # headlines of the qasm file.
        # define the supremacy gates. sx is already used by Qiskit so no need to define here.
        lines = [
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            "gate sy p0 {ry(pi/2) p0;}",  # sqrty gate
            "gate sw p0 {u3(pi/2,pi/2,0) p0;}",  # sqrtw gate
        ]

        # add to the qasm file depending on the circuit instructions.
        lines.append(f"qreg q[{self.num_qubits}];")
        for ins in self.instructions:
            gate = ins[0]
            if gate == "cx":
                lines.append(f"cx q[{ins[1]}], q[{ins[2]}];")
            elif gate == "cz":
                lines.append(f"cz q[{ins[1]}], q[{ins[2]}];")
            elif gate == "h":
                lines.append(f"h q[{ins[1]}];")
            elif gate == "s":
                lines.append(f"s q[{ins[1]}];")
            elif gate == "z":
                lines.append(f"z q[{ins[1]}];")
            elif gate == "x":
                lines.append(f"x q[{ins[1]}];")
            elif gate == "y":
                lines.append(f"y q[{ins[1]}];")
            elif gate == "t":
                lines.append(f"t q[{ins[1]}];")
            elif gate == "sqrtx":
                lines.append(f"sx q[{ins[1]}];")
            elif gate == "sqrty":
                lines.append(f"sy q[{ins[1]}];")
            elif gate == "sqrtw":
                lines.append(f"sw q[{ins[1]}];")
            elif gate == "fsim":  # is an iswap followed by a cphase rotation.
                # since qiskit does not recognize "iswap" we decompose it.
                lines.append(f"s q[{ins[1]}];")
                lines.append(f"s q[{ins[2]}];")
                lines.append(f"h q[{ins[1]}];")
                lines.append(f"cx q[{ins[1]}], q[{ins[2]}];")
                lines.append(f"cx q[{ins[2]}], q[{ins[1]}];")
                lines.append(f"h q[{ins[2]}];")

                lines.append(
                    f"cp(pi/6) q[{ins[1]}], q[{ins[2]}];"
                )  # doesn't work in quantum nodes version
            elif gate == "id":
                lines.append(f"id q[{ins[1]}];")
            elif gate == "barrier":
                lines.append(
                    "barrier "
                    + "".join(f"q[{i}]," for i in range(self.num_qubits - 1))
                    + f"q[{self.num_qubits - 1}];"
                )
            else:
                raise Exception(f"Unexpected gate {ins}")
        return "\n".join(lines)

    @staticmethod
    def from_qasm_str(s):
        """Create a circuit object given a QASM string."""
        num_qubits = None
        instructions = []
        for line in s.splitlines():
            line = line.strip()
            if not line:
                continue

            # skip header and includes"
            if line.startswith("OPENQASM 2.0;") or line.startswith("include"):
                continue

            # qubit register (we only support a single one called "q")
            m = re.match(r"qreg (\w+)\[(\d+)\];", line)
            if m:
                assert m.group(1) == "q", "Only supporting qubit registers called 'q'."
                assert num_qubits is None, "Should have exactly one qreg statement."
                num_qubits = int(m.group(2))
                continue

            # single-qubit gate
            m = re.match(r"(\w+) q\[(\d+)\];", line)
            if m:
                gate, qubit = m.group(1), int(m.group(2))
                assert gate in [
                    "s",
                    "h",
                    "z",
                    "x",
                    "y",
                    "t",
                ], "Gate not found in gate set."
                instructions.append((gate, qubit))
                continue

            # two-qubit gates
            m = re.match(r"(\w+) q\[(\d+)\],\s*q\[(\d+)\];", line)
            if m:
                gate, control, target = m.group(1), int(m.group(2)), int(m.group(3))
                assert gate in ["cx", "cz"], "Gate not found in gate set."
                instructions.append((gate, control, target))
                continue

            raise Exception(f"Unexpected QASM code: {line}")

        assert num_qubits is not None, "Missing qreg statement."
        return Circuit(num_qubits, instructions)

    @staticmethod
    def from_qasm_file(f):
        """Create a circuit object given a QASM file name or object."""
        if isinstance(f, str):
            f = open(f)
        return Circuit.from_qasm_str(f.read())

    #  Begin defining circuit tools and clifford gates - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def barrier(self):
        """"Create a barrier to separate gates in a circuit. Acts on all qubits at once."""
        self.instructions.append(("barrier",))

    def cx(self, control, target):
        """Add a CNOT gate to the circuit. The gate acts on a control and target qubit."""
        if control >= self.num_qubits or target >= self.num_qubits:
            raise Exception("Circuit does not contain enough qubits.")
        if control == target:
            raise Exception("Control should not be same as target.")
        self.instructions.append(("cx", control, target))

    def cz(self, control, target):
        """Add a CZ gate to the circuit. The gate acts on a control and target qubit."""
        if control >= self.num_qubits or target >= self.num_qubits:
            raise Exception("Circuit does not contain enough qubits.")
        if control == target:
            raise Exception("Control should not be same as target.")
        self.instructions.append(("cz", control, target))

    def h(self, qubit):
        """Add a Hadamard gate to the circuit. The gate acts on a qubit."""
        if qubit >= self.num_qubits:
            raise Exception("Circuit does not contain enough qubits.")
        self.instructions.append(("h", qubit))

    def s(self, qubit):
        """Add a Phase gate to the circuit. The gate acts on a qubit."""
        if qubit >= self.num_qubits:
            raise Exception("Circuit does not contain enough qubits.")
        self.instructions.append(("s", qubit))

    def z(self, qubit):
        """Add a Pauli Z gate to the circuit. The gate acts on a qubit."""
        if qubit >= self.num_qubits:
            raise Exception("Circuit does not contain enough qubits.")
        self.instructions.append(("z", qubit))

    def x(self, qubit):
        """Add a Pauli X gate to the circuit. The gate acts on a qubit."""
        if qubit >= self.num_qubits:
            raise Exception("Circuit does not contain enough qubits.")
        self.instructions.append(("x", qubit))

    def y(self, qubit):
        """Add a Pauli Y gate to the circuit. The gate acts on a qubit."""
        if qubit >= self.num_qubits:
            raise Exception("Circuit does not contain enough qubits.")
        self.instructions.append(("y", qubit))

    def sqrtx(self, qubit):
        """Add a sqrt(X) gate to the circuit. Supremacy (clifford) gate."""
        if qubit >= self.num_qubits:
            raise Exception("Circuit does not contain enough qubits.")
        self.instructions.append(("sqrtx", qubit))

    def sqrty(self, qubit):
        """Add a sqrt(Y) gate to the circuit. Supremacy (clifford) gate."""
        if qubit >= self.num_qubits:
            raise Exception("Circuit does not contain enough qubits.")
        self.instructions.append(("sqrty", qubit))

    # Begin adding non-clifford gates - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def t(self, qubit):
        """Add a T gate to the circuit. The gate acts on a qubit."""
        if qubit >= self.num_qubits:
            raise Exception("Circuit does not contain enough qubits.")
        self.instructions.append(("t", qubit))

    def sqrtw(self, qubit):
        """Add a sqrt(W) gate to the circuit, where W = 1/sqrt(2) * (X+Y). Supremacy (non-clifford) gate."""
        if qubit >= self.num_qubits:
            raise Exception("Circuit does not contain enough qubits.")
        self.instructions.append(("sqrtw", qubit))

    def fsim(self, qubit1, qubit2):
        if qubit1 >= self.num_qubits or qubit2 >= self.num_qubits:
            raise Exception("Circuit does not contain enough qubits.")
        if qubit1 == qubit2:
            raise Exception("The two qubits cannot be the same.")
        self.instructions.append(("fsim", qubit1, qubit2))

    # Gates that may or not be clifford gates depending on the value of a parameter - - - - - - - - - - - - - - - - - - - -

    def rphi(self, phi, qubit):
        if qubit >= self.num_qubits:
            raise Exception("Circuit does not contain enough qubits.")
        self.instructions.append(("rphi", phi, qubit))

    def crphi(self, phi, control, target):
        """Add a CR_phi gate to the circuit. Makes use of an angle phi in radians."""
        if control >= self.num_qubits or target >= self.num_qubits:
            raise Exception("Circuit does not contain enough qubits.")
        if control == target:
            raise Exception("Control should not be same as target.")
        self.instructions.append(("crphi", phi, control, target))
