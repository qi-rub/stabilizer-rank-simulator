import numpy as np
import math
from ..state import State
from ..pauli import *
from ..samplingch.overlap import overlap_given_basis as overlap


class CHState(State):
    """
    This class contains all the parameters necessary to represent the C-H form of a given stabilizer state.
    We can represent any stabilizer state |phi> as |phi> = w U_C U_H |s>, where "w" is a global complex phase, U_C is the product of S,CZ, and CX gates,
    U_H is the product of Hadamard gates, and s is a basis state.

    U_C can be represented by two "tableaus", (n x n)-matrices containing Pauli Z and X literals (symbols). The first tableau only contains Z literals and no local phases,
    while the second tableau contains both X and Z literals and does require and array of length n for the phases. We use binary matrices (G,F,M) to keep track of the
    positions of such literals.

    Attributes
    ===========
    * num_qubits -- n, number of qubits
    * G -- binary (n x n)-matrix representing the positions of the Z literals in the first tableau
    * F -- binary (n x n)-matrix representing the positions of the X literals in the second tableau
    * M -- binary (n x n)-matrix representing the positions of the Z literals in the second tableau
    * l_phases -- array of length n that contains the phases of the second tableau
    * g_phase -- global complex phase of the state.
    * h_vector -- binary array of length n that indicates if there are Hadamard gates applied to the qubits
    * b_state -- bitstring representing the current basis state
    """

    def __init__(self, G, F, M, l_phases, g_phase, h_vector, b_state):
        """Initialize the C-H form of a stabilizer state. See above for the meaning of all parameters."""
        self.num_qubits = len(h_vector)

        assert all(np.array_equal(i ** 2, i) for i in G) and G.shape[0] == G.shape[1]
        assert all(np.array_equal(i ** 2, i) for i in F) and F.shape[0] == F.shape[1]
        assert all(np.array_equal(i ** 2, i) for i in M) and M.shape[0] == M.shape[1]
        assert len(l_phases) == self.num_qubits
        # Assert for the global phase?
        assert all(h_vector ** 2 == h_vector) and len(h_vector) == self.num_qubits
        assert all(b_state ** 2 == b_state) and len(b_state) == self.num_qubits

        self.G = np.array(G)
        self.F = np.array(F)
        self.M = np.array(M)
        self.l_phases = np.array(l_phases) % 4
        self.g_phase = g_phase
        self.h_vector = np.array(h_vector)
        self.b_state = np.array(b_state)

    def __str__(self):
        return f"[G = {self.G}, F = {self.F}, M =  {self.M}, gam =  {self.l_phases}, ome = {self.g_phase}, v = {self.h_vector}, b = {self.b_state}]"

    def __eq__(self, other):
        """Compare two CHState objects."""
        # NOTE: we cannot compare all seven parameters that define a CHState since the representation is not unique.
        # Instead we compare the density matrices. (Is there a better way of doing this?).
        return np.array_equal(self.density_matrix(), other.density_matrix())

    @staticmethod
    def trivial_state(num_qubits):
        """Create the C-H state for the state |00...0>."""
        I = np.eye(num_qubits)
        mat_z = np.zeros((num_qubits, num_qubits))
        array_z = np.zeros(num_qubits)
        return CHState(I, I, mat_z, array_z, 1, array_z, array_z)

    @staticmethod
    def basis_state(bits):
        """ Creates the C-H state of a given basis state. "Bits" is a list of bits."""
        state = CHState.trivial_state(len(bits))
        state.b_state = np.array(bits)
        return state

    def density_matrix(self):
        """
        Obtain the density matrix of the C-H state.

        The formula for the density matrix makes use of a different ("tilde") tableu, as we require U P U_dag instead of U_dag P U.
        """
        identity = np.eye(2 ** self.num_qubits)
        rho = identity

        # relations between normal and tilde tableaus.
        Gtil = self.F.T
        Ftil = self.G.T
        Mtil = (self.G.T @ self.M @ self.F.T) % 2

        # calculate gamma tilde.
        gamma_1 = self.l_phases % 2
        gamma_0 = self.l_phases // 2
        com_phase = np.zeros(self.num_qubits)
        for p in range(self.num_qubits):
            temp = 0
            for l in range(self.num_qubits):
                temp_med = 0
                for r in range(self.num_qubits)[1:]:
                    temp_smaller = 0
                    for g in range(r):
                        temp_smaller += (
                            Mtil[g, l] * self.F[p, g] + Gtil[g, l] * self.M[p, g]
                        )
                    temp_med += Ftil[r, l] * self.F[p, r] * temp_smaller
                temp += temp_med
            com_phase[p] = temp
        gamma_til_1 = (self.G.T @ gamma_1) % 2
        gamma_1_sum = gamma_1 + self.F @ gamma_til_1
        assert np.all(gamma_1_sum % 2 == 0)
        gamma_til_0 = (self.G.T @ (gamma_0 + com_phase + gamma_1_sum // 2)) % 2
        assert np.all(
            (
                (2 * gamma_0 + gamma_1)
                + 2 * com_phase
                + self.F @ (2 * gamma_til_0 + gamma_til_1)
            )
            % 4
            == 0
        ), "Equation should be zero"
        gamma_til = 2 * gamma_til_0 + gamma_til_1

        # compute the density matrix.
        for i in range(self.num_qubits):
            if self.h_vector[i] == 1:
                z = Mtil[i]
                x = Ftil[i]
                gen = (1j) ** (gamma_til[i] - (z @ x)) * pauli_to_matrix(
                    np.append(z, x)
                )
            else:
                z = Gtil[i]
                gen = pauli_to_matrix(np.append(z, np.zeros(self.num_qubits)))
            rho = rho @ (identity + (-1) ** self.b_state[i] * gen) / 2

        assert np.array_equal(
            rho, rho.conjugate().transpose()
        ), "The matrix is not hermitian."

        return rho

    def pauli_expectation(self, pauli):
        """Find the expectation value of a Pauli operator."""
        assert len(pauli) == self.num_qubits * 2

        z, x = np.array(pauli[: self.num_qubits]), np.array(pauli[self.num_qubits :])
        z_col = np.reshape(z, (self.num_qubits, 1))
        x_col = np.reshape(x, (self.num_qubits, 1))
        pauli_phase_exp = (-1) * (z @ x)

        # defnitions.
        K_mat = self.F * x_col
        L_mat = (self.G * z_col + self.M * x_col) % 2
        X_mat = (K_mat * self.h_vector + K_mat + L_mat * self.h_vector) % 2
        Z_mat = (L_mat * self.h_vector + L_mat + K_mat * self.h_vector) % 2
        tot_phase = (
            pauli_phase_exp
            + np.sum(np.reshape(self.l_phases, (self.num_qubits, 1)) * x_col)
            + np.sum(2 * (self.G * K_mat * z_col + K_mat * L_mat * self.h_vector))
        )

        # collapse to 1d array.
        Xs_on_basis = np.sum(X_mat, axis=0)
        Zs_on_basis = np.sum(Z_mat, axis=0)
        # compute the sign change when commuting Pauli operators, back to normal form.
        com_phase = np.zeros(self.num_qubits)
        for j in range(self.num_qubits):
            for r in range(self.num_qubits)[1:]:
                com_phase[j] += X_mat[r, j] * np.sum(Z_mat[:r, j])
        exp_phase_of_Q = tot_phase + 2 * np.sum(com_phase)  # total phase mod 4.

        # evaluate
        if np.all((Xs_on_basis % 2 == 0)):
            total_phase = 1
            for i in range(self.num_qubits):
                if self.b_state[i] == 1 and Zs_on_basis[i] != 0:
                    total_phase = total_phase * (-1) ** Zs_on_basis[i]
            if total_phase == (1j) ** exp_phase_of_Q:
                return 1
            else:
                return -1
        else:
            return 0

    def statevector(self):
        """Function that computes the statevector from the C-H form of the state."""
        statevector = []
        for num in range(2 ** self.num_qubits):
            binary_array = [int(x) for x in bin(num)[2:]]
            basis_state = [0] * (self.num_qubits - len(binary_array)) + binary_array
            statevector.append(overlap(basis_state, self))
        return np.reshape(np.array(statevector), (2 ** self.num_qubits, 1))  # 2d array.

    def apply_s_l(self, qubit):
        """Apply S from the left."""
        self.M[qubit] = (self.M[qubit] + self.G[qubit]) % 2
        self.l_phases[qubit] = (self.l_phases[qubit] - 1) % 4

    def apply_s_r(self, qubit):
        """Apply S from the right."""
        self.M[:, qubit] = (self.M[:, qubit] + self.F[:, qubit]) % 2
        self.l_phases = (self.l_phases - self.F[:, qubit]) % 4

    def apply_cz_l(self, control, target):
        """Apply CZ from the left."""
        self.M[control] = (self.M[control] + self.G[target]) % 2
        self.M[target] = (self.M[target] + self.G[control]) % 2

    def apply_cz_r(self, control, target):
        """Apply CZ from the right."""
        self.M[:, control] = (self.M[:, control] + self.F[:, target]) % 2
        self.M[:, target] = (self.M[:, target] + self.F[:, control]) % 2
        self.l_phases = (self.l_phases + 2 * self.F[:, control] * self.F[:, target]) % 4

    def apply_cx_l(self, control, target):
        """Apply CNOT from the left."""
        self.l_phases[control] = (
            self.l_phases[control]
            + self.l_phases[target]
            + 2 * (self.M @ self.F.T)[control, target]
        ) % 4
        self.G[target] = (self.G[target] + self.G[control]) % 2
        self.F[control] = (self.F[control] + self.F[target]) % 2
        self.M[control] = (self.M[control] + self.M[target]) % 2

    def apply_cx_r(self, control, target):
        """Apply CNOT from the right."""
        self.G[:, control] = (self.G[:, control] + self.G[:, target]) % 2
        self.F[:, target] = (self.F[:, target] + self.F[:, control]) % 2
        self.M[:, control] = (self.M[:, control] + self.M[:, target]) % 2

    def apply_h(self, qubit):
        """Apply a Hadamard gate."""
        # definitions.
        t = (self.b_state + self.G[qubit] * self.h_vector) % 2
        u = (
            self.b_state
            + self.F[qubit] * (1 - self.h_vector)
            + self.M[qubit] * self.h_vector
        ) % 2
        alpha = sum(self.G[qubit] * (1 - self.h_vector) * self.b_state)
        beta = sum(
            self.M[qubit] * (1 - self.h_vector) * self.b_state
            + self.F[qubit] * self.h_vector * (self.M[qubit] + self.b_state)
        )

        # compare t and u.
        if all(t == u):
            self.b_state = t
            self.g_phase = (
                (1 / np.sqrt(2))
                * ((-1) ** alpha + (1j) ** self.l_phases[qubit] * (-1) ** beta)
                * self.g_phase
            )
        else:
            # definitions.
            delta = (self.l_phases[qubit] + 2 * (alpha + beta)) % 4
            V_0 = [
                i
                for i in range(self.num_qubits)
                if t[i] != u[i] and self.h_vector[i] == 0
            ]
            V_1 = [
                i
                for i in range(self.num_qubits)
                if t[i] != u[i] and self.h_vector[i] == 1
            ]
            ind, origin = (V_0[0], 0) if len(V_0) > 0 else (V_1[0], 1)  # tuple.

            # compute the strings y and z so they differ only on one qubit. Technically, we only need y.
            if t[ind] == 1:
                y = u
                y[ind] = (y[ind] + 1) % 2
            else:
                y = t

            # calculations.
            a = delta % 2
            b = (self.h_vector[ind] * (delta + 1) + 1) % 2
            if y[ind] == 0:
                c = math.floor((delta + self.h_vector[ind]) / 2) % 2
                omega = math.sqrt(2) * ((self.h_vector[ind] * delta + 1) % 2) + (
                    1 + (-1) ** math.floor(delta / 2) * 1j
                ) * ((self.h_vector[ind] * delta) % 2)
            else:
                c = math.ceil((delta - self.h_vector[ind]) / 2) % 2
                omega = ((1j) ** delta) * math.sqrt(2) * (
                    (self.h_vector[ind] * delta + 1) % 2
                ) + (1 + (-1) ** math.floor(delta / 2) * 1j) * (
                    (self.h_vector[ind] * delta) % 2
                )

            # update tableaus, vectors, and phases.
            self.h_vector[ind] = b
            self.g_phase = (1 / math.sqrt(2)) * (-1) ** alpha * omega * self.g_phase
            self.b_state = y
            self.b_state[ind] = c

            # update U_C.
            if origin == 0:
                for target in V_0[1:]:
                    self.apply_cx_r(ind, target)
                for target in V_1:
                    self.apply_cz_r(ind, target)
            else:
                for target in V_1[1:]:
                    self.apply_cx_r(target, ind)
            if a == 1:
                self.apply_s_r(ind)


def run(circuit) -> CHState:
    """Runs given quantum circuit starting with a |0...0> state and returns the resulting CH-state."""
    state = CHState.trivial_state(circuit.num_qubits)

    for instruction in circuit.instructions:
        gate = instruction[0]
        if len(instruction) == 3:
            control = int(instruction[1])
            target = int(instruction[2])

            # CX gate.
            if gate == "cx":
                state.apply_cx_l(control, target)
            # CZ gate.
            elif gate == "cz":
                state.apply_cz_l(control, target)
            else:
                raise Exception("Unrecognized gate.")
        elif len(instruction) == 2:
            qubit = int(instruction[1])

            # H gate.
            if gate == "h":
                state.apply_h(qubit)
            # S gate
            elif gate == "s":
                state.apply_s_l(qubit)
            # SQRT_X gate. Interpreted as SQRT_X = H*S*H.
            elif gate == "sqrtx":
                state.apply_h(qubit)
                state.apply_s_l(qubit)
                state.apply_h(qubit)
            # SQRT_Y gate. Interpreted as SQRT_Y = H*S*S.
            elif gate == "sqrty":
                state.apply_h(qubit)
                state.apply_s_l(qubit)
                state.apply_s_l(qubit)
            elif gate == "id":  # ignore.
                pass
            else:
                raise Exception(f"Unrecognized gate: {gate}.")
        else:
            if gate == "barrier":  # ignore.
                pass
    return state
