class State:
    def density_matrix(self):
        """Return density matrix (might be slow)."""
        raise NotImplementedError

    def pauli_expectation(self, pauli):
        """Return expectation value of Pauli operator (represented as binary string)."""
        raise NotImplementedError

    def measure_pauli(self, pauli, random_bit=None):
        """
        Measure Pauli operator.

        Return measurement outcome and update state with post-measurement state.
        The outcome is either deterministic or uniformly random (in the latter case,
        use `random_bit` if provided).
        """
        raise NotImplementedError

    def apply_pauli(self, pauli):
        """Apply a unitary pauli gate to the state."""
        raise NotImplementedError
