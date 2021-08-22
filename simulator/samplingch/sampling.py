import numpy as np
from .overlap import overlap_given_basis_sampling, overlap_given_pauli_sampling


def obtain_samples(samples: int, stabdecos, x=None, iters=300) -> list:
    """
    Function that performs the MCMC sampling of the probability distribution P(x) = <x|U_nC |0> where U_nC is a non-Clifford circuit.
    Since this is very costly to compute exactly, we allow some error delta and instead compute P(x) = <x|psi'> where |psi'> is a linear combination of k stabilizer states and ||U_nC|0> - |psi'>|| < delta.

    Returns an n-bit string.

    Attributes
    ==========
    * iters -- number of iterations that the chain will do.
    * stabdecos -- StabDecos (namedtuple) objects. For more information refere to the "stabilizer_decomposition" function docstring.
    """
    assert isinstance(samples, int) or isinstance(
        samples, np.int64
    ), "The number of samples should be an integer."
    assert isinstance(iters, int) or isinstance(
        iters, np.int64
    ), "The number of iterations should be an integer."

    x = [] if x is None else x
    x = np.array(x)
    assert all(x ** 2 == x)
    original_x = x.copy()

    num_qubits = stabdecos.num_qubits

    # create the random bits needed for the sampling all at once.
    initial_bits = np.random.choice([0, 1], size=num_qubits * samples)

    samples_l = []
    for s in range(samples):
        x = initial_bits[num_qubits * s : num_qubits * (s + 1)] if len(x) == 0 else x

        # Make it so we can give the whole array to "overlap_given_basis"! - - - - - - - - - -
        # obtain P(x_in) up to a normalization constant. Uses the stabilizer decomposition.
        overlap_x, paulis_used = overlap_given_basis_sampling(x, stabdecos)
        p_x = abs(overlap_x) ** 2

        # locations in which we will apply bit flips to the string x.
        locs = np.random.choice(len(x), size=iters)

        # now we do the metropolis steps.
        for iter in range(iters):
            ind_to_change = locs[iter]
            y = x.copy()
            y[ind_to_change] = (y[ind_to_change] + 1) % 2  # y = x + e_j.

            # find p(y).
            overlap_y, paulis_used_y = overlap_given_pauli_sampling(
                paulis_used, ind_to_change, stabdecos
            )
            p_y = abs(overlap_y) ** 2

            # now we compare ratios and perform the metropolis steps.
            ratio = 1 if p_x == 0 else p_y / p_x
            decision_bit = (
                np.random.choice([0, 1], p=[1 - ratio, ratio]) if ratio <= 1 else 1
            )
            if ratio >= 1 or decision_bit == 1:
                x = y
                p_x = p_y
                paulis_used = paulis_used_y

        samples_l.append("".join(map(str, x)))

        x = original_x  # reset x.

    return samples_l
