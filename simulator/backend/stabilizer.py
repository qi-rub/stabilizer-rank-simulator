import numpy as np
from ..state import State
from ..pauli import *
from ..linalg import solve_mod_2


class StabilizerState(State):
    """
    This class represents a stabilizer state of an n-qubit stabilizer state in
    terms of the generators of the corresponding stabilizer group.

    Each generator is a Pauli operator times a sign. As usual, we represent the
    Pauli operator by bitstring of length 2n (see the `pauli` module). The sign
    is represented by a bit, called the "phase" (times pi). Thus, the generator
    corresponding to a pair (pauli, phase) is

        g = (-1)**phase * pauli.

    There are n pairwise commuting generators.

    Attributes
    ==========
    * num_qubits -- n, the number of qubits
    * paulis -- (n x 2n)-matrix representing the Pauli operators of the generators
    * phases -- n-vector representing the phases of the generators
    * sf -- symplectic matrix of size 2num_qubits x 2num_qubits that guides the commutation relations of Pauli operators.
    """

    def __init__(self, paulis, phases):
        """
        Initialize stabilizer state. See above for the meaning of the parameters `pauli` and `phases`.
        """
        self.num_qubits = len(paulis)
        self.paulis = np.array(paulis)
        self.phases = np.array(phases)
        self.sf = symplectic_form(self.num_qubits)

        # check shapes
        assert self.paulis.shape == (self.num_qubits, 2 * self.num_qubits)
        assert self.phases.shape == (self.num_qubits,)

        # check contents
        assert np.all(self.paulis ** 2 == self.paulis)
        assert np.all(self.phases ** 2 == self.phases)

        # check that the rows are pairwise orthogonal
        ips = (self.paulis @ self.sf @ self.paulis.T) % 2
        assert np.all(ips == 0), "Generators do not pairwise commute."

    @staticmethod
    def basis_state(bits):
        """Return computational basis state |bits> as a StabilizerState."""
        n = len(bits)
        paulis = np.hstack([np.eye(n, dtype=int), np.zeros((n, n), dtype=int)])
        return StabilizerState(paulis, bits)

    @property
    def generators(self):
        """Return pairs (pauli,phase), one for each generator."""
        return zip(self.paulis, self.phases)

    def __repr__(self):
        return f"StabilizerState(paulis={self.paulis}, phases={self.phases})"

    def __str__(self):
        return "\n".join(
            f"[{pauli_to_binary_str(pauli)}|{phase}]"
            for pauli, phase in self.generators
        )

    def __eq__(self, other):
        """Compare two stabilizer states."""
        # NOTE: we cannot simply compare the Paulis and the phases, since the representation is not unique.
        # instead we check that `self` is stabilized by the generators of `other`!
        return all(
            self.pauli_expectation(pauli) == (-1) ** phase
            for pauli, phase in other.generators
        )

    def density_matrix(self):
        """Returns the density matrix representation of the state."""
        identity = np.eye(2 ** self.num_qubits)
        rho = identity
        for pauli, phase in self.generators:
            gen = (-1) ** phase * pauli_to_matrix(pauli)
            rho = rho @ (identity + gen) / 2
        return rho

    def pauli_expectation(self, pauli):
        """Return expectation value of a given Pauli operator."""
        # write pauli bitstring as linear combination of generator bitstrings
        x = solve_mod_2(self.paulis.T, pauli)

        # if impossible, expectation value is 0. Also means that the pauli anticommutes with at least with one generator
        if x is None:
            return 0

        # determine paulis that multiply to given Pauli (up to sign)
        prod_pauli, prod_exp_mod_4 = multiply_paulis(self.paulis[x == 1])
        assert np.array_equal(prod_pauli, pauli)
        assert prod_exp_mod_4 % 2 == 0

        # determine phase
        phase = np.sum(self.phases[x == 1])
        phase += prod_exp_mod_4 // 2
        return (-1) ** phase

    def measure_pauli(self, pauli, random_bit=None):
        """
        Performs a measurement of the given Pauli operator.

        Attributes
        ===========
        * random_bit -- Argument that sets the seed in order to get a deterministic measurement outcome in the anti-commuting case. It is only used for testing.
        """
        assert random_bit in [None, 0, 1]

        # write pauli bitstring as linear combination of generator bitstrings
        x = solve_mod_2(self.paulis.T, pauli)

        # the pauli commutes with all of the generators. The post-measurement state does not change
        if x is not None:
            # determine paulis that multiply to given Pauli (up to sign)
            prod_pauli, prod_exp_mod_4 = multiply_paulis(self.paulis[x == 1])
            assert np.array_equal(prod_pauli, pauli)
            assert prod_exp_mod_4 % 2 == 0

            # determine phase
            phase = np.sum(self.phases[x == 1])
            phase += prod_exp_mod_4 // 2
            return (-1) ** phase
        else:
            # reduce to only one anti-commuting generator
            commutations = (self.paulis @ self.sf @ pauli) % 2
            anticommuting_i = np.where(commutations != 0)[0]
            p_i = anticommuting_i[0]
            for i in anticommuting_i[1:]:
                self.paulis[i], prod_exp_mod_4 = multiply_paulis(
                    [self.paulis[i], self.paulis[p_i]]
                )
                self.phases[i] = (
                    prod_exp_mod_4 // 2 + self.phases[i] + self.phases[p_i]
                ) % 2
            # replace the first anticommuting generator with pauli
            self.paulis[p_i] = pauli
            np.random.seed(random_bit)
            outcome = np.random.choice([1, -1])
            self.phases[p_i] = 0 if outcome == 1 else 1
            return outcome

    def apply_pauli(self, pauli, qubit=None):
        """
        Apply a Pauli unitary to the state.
        The pauli is given by the 2n bitstring we are familiar with.
        However, if a qubit is specified, then we expect a single-qubit Pauli represented by 2 bits.
        """
        if qubit is None:
            new_phases = (self.paulis @ self.sf @ pauli) % 2
            self.phases = (self.phases + new_phases) % 2
        else:
            assert qubit < self.num_qubits
            assert len(pauli) == 2
            # Symplectic product between relevant qubits and single qubit pauli for all rows.
            new_phases = (
                self.paulis[:, [qubit, qubit + self.num_qubits]] @ pauli[::-1]
            ) % 2
            self.phases = (self.phases + new_phases) % 2

    def apply_h(self, qubit):
        """Apply Hadamard gate to the given qubit."""
        assert qubit < self.num_qubits
        zq, xq = qubit, qubit + self.num_qubits

        # flip phase if Y at qubit
        self.phases = (self.phases + self.paulis[:, xq] * self.paulis[:, zq]) % 2

        # swap X and Z columns corresponding to the qubit
        self.paulis[:, (zq, xq)] = self.paulis[:, (xq, zq)]

    def apply_s(self, qubit):
        """Applies phase gate (S gate) to the given qubit."""
        assert qubit < self.num_qubits
        zq, xq = qubit, qubit + self.num_qubits

        # flip phase if Y at qubit
        self.phases = (self.phases + self.paulis[:, xq] * self.paulis[:, zq]) % 2

        # add X column to the Z column corresponding to the qubit
        self.paulis[:, zq] = (self.paulis[:, zq] + self.paulis[:, xq]) % 2

    def apply_cx(self, control, target):
        """Applies CX gate (CNOT gate) to the given control and target qubit."""
        assert control < self.num_qubits and target < self.num_qubits
        zc, xc = control, control + self.num_qubits
        zt, xt = target, target + self.num_qubits

        # flip phase if (X at control and Z at target) and again if (Y at control and Y at target)
        # this can be simplified as follows:
        self.phases = (
            self.phases
            + self.paulis[:, xc]
            * self.paulis[:, zt]
            * (self.paulis[:, zc] + self.paulis[:, xt] + 1)
        ) % 2

        # add target Z => control Z and control X => target X
        self.paulis[:, zc] = (self.paulis[:, zc] + self.paulis[:, zt]) % 2
        self.paulis[:, xt] = (self.paulis[:, xc] + self.paulis[:, xt]) % 2

    def apply_cz(self, control, target):
        """Applies CZ gate to the given control and target qubit."""
        assert control < self.num_qubits and target < self.num_qubits
        zc, xc = control, control + self.num_qubits
        zt, xt = target, target + self.num_qubits

        # flip the phase if both control and target are either X's or Y's
        self.phases = (
            self.phases
            + self.paulis[:, xc]
            * self.paulis[:, xt]
            * (self.paulis[:, zc] + self.paulis[:, zt])
        ) % 2

        # add target X => control Z and control X => target Z
        self.paulis[:, zc] = (self.paulis[:, zc] + self.paulis[:, xt]) % 2
        self.paulis[:, zt] = (self.paulis[:, zt] + self.paulis[:, xc]) % 2


def run(circuit):
    """Runs given quantum circuit starting with a |0...0> state and returns the resulting stabilizer state."""
    state = StabilizerState.basis_state([0] * circuit.num_qubits)

    for instruction in circuit.instructions:
        gate = instruction[0]
        if len(instruction) == 3:
            control = int(instruction[1])
            target = int(instruction[2])

            # CX gate
            if gate == ("cx"):
                state.apply_cx(control, target)
            # CZ gate
            elif gate == ("cz"):
                state.apply_cz(control, target)
            else:
                raise Exception("Unrecognized gate.")

        else:
            qubit = int(instruction[1])

            # H gate
            if gate == "h":
                state.apply_h(qubit)
            # S gate
            elif gate == "s":
                state.apply_s(qubit)
            # Pauli Z gate
            elif gate == "z":
                state.apply_pauli([1, 0], qubit)
            # Pauli X gate
            elif gate == "x":
                state.apply_pauli([0, 1], qubit)
            # Pauli Y gate
            elif gate == "y":
                state.apply_pauli([1, 1], qubit)
            else:
                raise Exception("Unrecognized gate.")
    return state
