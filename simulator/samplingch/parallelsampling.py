import numpy as np
from joblib import Parallel, delayed
from .overlap import overlap_given_basis_sampling, overlap_given_pauli_sampling


def obtain_samples(
    samples: int, stabdecos, x=None, iters=300, cores=1, loc=None
) -> list:
    """
    Function that performs the MCMC sampling of the probability distribution P(x) = <x|U_nC |0> where U_nC is a non-Clifford circuit.
    Since this is very costly to compute exactly, we allow some error delta and instead compute P(x) = <x|psi'> where |psi'> is a linear combination of k stabilizer states and ||U_nC|0> - |psi'>|| < delta.

    Since each individual sample is independent of the others, this function is parallelized. Each core will sample one bitsring and append it to a list, this process continues until the total number of samples desired has been met.

    Returns a list of the sampled bitstrings.

    Note: We have to create new random number generators with different seeds for each of the samples. If this was not the case then each of the child processes would get the same seed and produce the same data.

    Attributes
    ==========
    * samples -- number of samples required.
    * stabdecos -- StabDecos (namedtuple) objects. For more information refere to the "stabilizer_decomposition" function docstring.
    * iters -- number of iterations that the chain will do.
    * cores -- number of cores that we want to use for the parallel sampling.
    * loc -- path were we might want to store the samples we have collected so far in case the program is terminated. (Useful when using LISA.)
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

    # fix the seeds for the random number generators that will be used in the child processes of the parallel sampling function.
    random_seeds = np.random.randint(np.iinfo(np.int32).max, size=samples)

    num_qubits = stabdecos.num_qubits

    return (
        Parallel(n_jobs=cores)(
            delayed(sample)(num_qubits, x, stabdecos, iters, random_seeds[i])
            for i in range(samples)
        )
        if loc is None
        else Parallel(n_jobs=cores)(
            delayed(sample_remember)(
                num_qubits, x, stabdecos, iters, random_seeds[i], loc
            )
            for i in range(samples)
        )
    )


def sample(num_qubits, x, stabdecos, iters, random_state):

    # seed new random number generator object.
    rng = np.random.RandomState(random_state)

    # create the initial state of the Markov chain.
    x = rng.choice([0, 1], size=num_qubits) if len(x) == 0 else x

    # obtain P(x_in) up to a normalization constant.
    overlap_x, paulis_used = overlap_given_basis_sampling(x, stabdecos)
    p_x = abs(overlap_x) ** 2

    # random locations for which we will apply bit flips to the string x.
    locs = rng.choice(len(x), size=iters)

    # metropolis steps.
    for iter in range(iters):
        ind_to_change = locs[iter]

        y = (x + np.identity(num_qubits, dtype=int)[ind_to_change]) % 2  # y = x + e_j

        # find p(y).
        overlap_y, paulis_used_y = overlap_given_pauli_sampling(
            paulis_used, ind_to_change, stabdecos
        )
        p_y = abs(overlap_y) ** 2

        # compare ratios and update accordingly.
        ratio = 1 if p_x == 0 else p_y / p_x
        decision_bit = rng.choice([0, 1], p=[1 - ratio, ratio]) if ratio <= 1 else 1
        if ratio >= 1 or decision_bit == 1:
            x = y
            p_x = p_y
            paulis_used = paulis_used_y

    return "".join(map(str, x))


def sample_remember(num_qubits, x, stabdecos, iters, random_state, loc=None):

    # seed new random number generator object.
    rng = np.random.RandomState(random_state)

    # create the initial state of the Markov chain.
    x = rng.choice([0, 1], size=num_qubits) if len(x) == 0 else x

    # obtain P(x_in) up to a normalization constant.
    overlap_x, paulis_used = overlap_given_basis_sampling(x, stabdecos)
    p_x = abs(overlap_x) ** 2

    # random locations for which we will apply bit flips to the string x.
    locs = rng.choice(len(x), size=iters)

    # metropolis steps.
    for iter in range(iters):
        ind_to_change = locs[iter]

        y = (x + np.identity(num_qubits, dtype=int)[ind_to_change]) % 2  # y = x + e_j

        # find p(y).
        overlap_y, paulis_used_y = overlap_given_pauli_sampling(
            paulis_used, ind_to_change, stabdecos
        )
        p_y = abs(overlap_y) ** 2

        # compare ratios and update accordingly.
        ratio = 1 if p_x == 0 else p_y / p_x
        decision_bit = rng.choice([0, 1], p=[1 - ratio, ratio]) if ratio <= 1 else 1
        if ratio >= 1 or decision_bit == 1:
            x = y
            p_x = p_y
            paulis_used = paulis_used_y

    bin_str = "".join(map(str, x))
    if loc is not None:
        with open(loc, "a") as f:
            f.write(bin_str + ",")

    return bin_str
