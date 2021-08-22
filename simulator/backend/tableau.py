import numpy as np
from ..state import State
from ..pauli import *


class Tableau:

    """
    This class represent a state in terms of the n generators of its stabilizer group, plus another n conjugate generators.
    This object is stored as a 2n x 2n NumPy array, plus an extra list of phases of length 2n. One phase per generator or
    conjugate generator.
    The generators all pairwise commute with themselves, and similarly for the conjugate generators. However, for a generator
    on row i (0<i<n), it will anticommute with the conjugate generator in row i+n.

    Each generator and conjugate generator consists of a Pauli operator times a sign. As usual, we represent the
    Pauli operator by bitstring of length 2n (see the `pauli` module for more information). The sign
    is represented by a bit, called the "phase" (times pi). Thus, the generator
    corresponding to a pair (pauli, phase) is

        g = (-1)**phase * pauli.

    Attributes
    ============
    * num_qubits -- n, the number of qubits
    * tableau -- (2n x 2n)-matrix representing the generators and conjugate generators. No phases included
    * phases -- n-vector representing the phases of the generators
    * sf -- symplectic matrix of size 2num_qubits x 2num_qubits that guides the commutation relations of Pauli operators
    """

    def __init__(self, tableau, phases):
        """
        Initialize stabilizer state. See above for the meaning of the parameters `tableau` and `phases`.
        """
        self.num_qubits = len(tableau) // 2
        self.tableau = np.array(tableau)
        self.phases = np.array(phases)
        self.sf = symplectic_form(self.num_qubits)

        # check shapes
        assert self.tableau.shape == (2 * self.num_qubits, 2 * self.num_qubits)
        assert self.phases.shape == (2 * self.num_qubits,)

        # check contents
        assert np.all(self.tableau ** 2 == self.tableau)
        assert np.all(self.phases ** 2 == self.phases)

        # check that the rows of the paulis are pairwise orthogonal
        ips = (
            self.tableau[: self.num_qubits, :]
            @ self.sf
            @ self.tableau[: self.num_qubits, :].T
        ) % 2
        assert np.all(ips == 0), "Generators do not pairwise commute."
        # check that the rows of the conjugate paulis are pairwise orthogonal
        ips_conj = (
            self.tableau[self.num_qubits, :]
            @ self.sf
            @ self.tableau[self.num_qubits :, :].T
        ) % 2
        assert np.all(ips_conj == 0), "Conjugate generators do not pairwise commute."

    @staticmethod
    def basis_state(bits):
        """Return computational basis state |bits> as a Tableau."""
        n = len(bits)
        tableau = np.eye(2 * n, dtype=int)
        return Tableau(tableau, bits + ([0] * n))

    @property
    def gen_and_conjugate_gen(self):
        """
        Return pairs of (pauli, phase) one for each pauli in the tableau. Includes both
        generators and conjugate generators.
        """
        return zip(self.tableau, self.phases)

    @property
    def generators(self):
        """Return pairs (pauli,phase), one for each generator."""
        return zip(self.tableau[: self.num_qubits, :], self.phases[: self.num_qubits])

    def __repr__(self):
        return f"Tableau(tableau={self.tableau}, phases={self.phases})"

    def __str__(self):
        return "\n".join(
            f"[{pauli_to_binary_str(pauli)} | {phase}]"
            for pauli, phase in self.gen_and_conjugate_gen
        )

    def __eq__(self, other):
        """Compare two tableau states."""
        # NOTE: we cannot simply compare the Paulis and the phases, since the representation is not unique.
        # instead we check that `self` is stabilized by the generators of `other`! No need to check the
        # conjugate generators.
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
        # We check if the pauli commuted with all the generators in the tableau
        if np.any((self.tableau[: self.num_qubits] @ self.sf @ pauli) % 2 == 1):
            return 0

        # write pauli bitstring as linear combination of generator bitstrings
        x = (self.tableau[self.num_qubits :] @ self.sf @ pauli) % 2

        # determine paulis that multiply to given Pauli (up to sign)
        paulis = self.tableau[: self.num_qubits][x == 1]
        prod_pauli, prod_exp_mod_4 = multiply_paulis(paulis)
        assert np.array_equal(prod_pauli, pauli)
        assert prod_exp_mod_4 % 2 == 0

        # determine phase
        phases = self.phases[: self.num_qubits][x == 1]
        phase = np.sum(phases)
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

        # We check if the pauli commuted with all the generators in the tableau
        acomm_gen = (self.tableau[: self.num_qubits] @ self.sf @ pauli) % 2
        if np.all(acomm_gen == 0):
            # write pauli bitstring as linear combination of generator bitstrings
            x = (self.tableau[self.num_qubits :] @ self.sf @ pauli) % 2

            # determine paulis that multiply to given Pauli (up to sign)
            paulis = self.tableau[: self.num_qubits][x == 1]
            prod_pauli, prod_exp_mod_4 = multiply_paulis(paulis)
            assert np.array_equal(prod_pauli, pauli)
            assert prod_exp_mod_4 % 2 == 0

            # determine phase
            phases = self.phases[: self.num_qubits][x == 1]
            phase = np.sum(phases)
            phase += prod_exp_mod_4 // 2
            return (-1) ** phase
        else:
            acomm_con_gen = (self.tableau[self.num_qubits :] @ self.sf @ pauli) % 2
            acomm_i = np.where(np.append(acomm_gen, acomm_con_gen) == 1)[0]
            p_i = acomm_i[0]  # index where the pauli will be placed
            # reduce to only one anti-commuting generator
            for i in acomm_i[1:]:
                self.tableau[i], prod_exp_mod_4 = multiply_paulis(
                    [self.tableau[i], self.tableau[p_i]]
                )
                self.phases[i] = (
                    prod_exp_mod_4 // 2 + self.phases[i] + self.phases[p_i]
                ) % 2
            # update the tableau with the pauli
            self.tableau[p_i + self.num_qubits] = self.tableau[p_i]
            self.phases[p_i + self.num_qubits] = self.phases[p_i]
            self.tableau[p_i] = pauli
            np.random.seed(random_bit)
            outcome = np.random.choice([1, -1])
            self.phases[p_i] = 0 if outcome == 1 else 1
            return outcome

    def apply_pauli(self, pauli, qubit=None):
        """
        Apply a Pauli unitary to the state
        The pauli is given by the 2n bitstring we are familiar with.
        However, if a qubit is specified, then we expect a single-qubit Pauli represented by 2 bits.
        """
        if qubit is None:
            new_phases_gen = (self.tableau[: self.num_qubits] @ self.sf @ pauli) % 2
            new_phases_conj_gen = (
                self.tableau[self.num_qubits :] @ self.sf @ pauli
            ) % 2
            new_phases = np.concatenate((new_phases_gen, new_phases_conj_gen))
            self.phases = (self.phases + new_phases) % 2
        else:
            assert len(pauli) == 2
            assert qubit < self.num_qubits
            # Symplectic product between relevant qubits and single qubit pauli for all rows.
            new_phases = (
                self.tableau[:, [qubit, qubit + self.num_qubits]] @ pauli[::-1]
            ) % 2
            self.phases = (self.phases + new_phases) % 2

    def apply_h(self, qubit):
        """Apply Hadamard gate to the given qubit."""
        assert qubit < self.num_qubits
        zq, xq = qubit, qubit + self.num_qubits

        # flip phase if Y at qubit
        self.phases = (self.phases + self.tableau[:, xq] * self.tableau[:, zq]) % 2

        # swap X and Z columns corresponding to the qubit
        self.tableau[:, (zq, xq)] = self.tableau[:, (xq, zq)]

    def apply_s(self, qubit):
        """Applies phase gate (S gate) to the given qubit."""
        assert qubit < self.num_qubits
        zq, xq = qubit, qubit + self.num_qubits

        # flip phase if Y at qubit
        self.phases = (self.phases + self.tableau[:, xq] * self.tableau[:, zq]) % 2

        # add X column to the Z column corresponding to the qubit
        self.tableau[:, zq] = (self.tableau[:, zq] + self.tableau[:, xq]) % 2

    def apply_cx(self, control, target):
        """Applies CX gate (CNOT gate) to the given control and target qubit."""
        assert control < self.num_qubits and target < self.num_qubits
        zc, xc = control, control + self.num_qubits
        zt, xt = target, target + self.num_qubits

        # flip phase if (X at control and Z at target) and again if (Y at control and Y at target)
        # this can be simplified as follows:
        self.phases = (
            self.phases
            + self.tableau[:, xc]
            * self.tableau[:, zt]
            * (self.tableau[:, zc] + self.tableau[:, xt] + 1)
        ) % 2

        # add target Z => control Z and control X => target X
        self.tableau[:, zc] = (self.tableau[:, zc] + self.tableau[:, zt]) % 2
        self.tableau[:, xt] = (self.tableau[:, xc] + self.tableau[:, xt]) % 2

    def apply_cz(self, control, target):
        """Applies CZ gate to the given control and target qubit."""
        assert control < self.num_qubits and target < self.num_qubits
        zc, xc = control, control + self.num_qubits
        zt, xt = target, target + self.num_qubits

        # flip the phase if both control and target are either X's or Y's
        self.phases = (
            self.phases
            + self.tableau[:, xc]
            * self.tableau[:, xt]
            * (self.tableau[:, zc] + self.tableau[:, zt])
        ) % 2

        # add target X => control Z and control X => target Z
        self.tableau[:, zc] = (self.tableau[:, zc] + self.tableau[:, xt]) % 2
        self.tableau[:, zt] = (self.tableau[:, zt] + self.tableau[:, xc]) % 2


def run(circuit):
    """Runs given quantum circuit starting with a |0...0> state and returns the resulting stabilizer state."""
    # We create the generators for the initial state of the circuit.
    state = Tableau.basis_state([0] * circuit.num_qubits)

    for instruction in circuit.instructions:
        gate = instruction[0]
        if len(instruction) == 3:
            control = int(instruction[1])
            target = int(instruction[2])

            # CX gate
            if gate == ("cx"):
                state.apply_cx(control, target)
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
