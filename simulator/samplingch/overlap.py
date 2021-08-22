import numpy as np
from simulator.pauli import multiply_paulis
from collections import namedtuple

Paulis = namedtuple("Paulis", "z_parts x_parts phases")


def overlap_given_basis(basis_state, chstate):
    """
    Function that calculates the overlap between a basis state and a stabilizer state given in its C-H form. <basis_state|phi>.
    Returns the value of the overlap.
    """
    basis_state = np.array(basis_state)
    assert len(basis_state) == chstate.num_qubits
    assert np.all(basis_state ** 2 == basis_state)

    z, x, tot_phase_exp = find_pauli(basis_state, chstate)

    # compute the product associated with 0 and 1 of the h_vector at the same time.
    prod_01 = np.prod(
        np.where(
            chstate.h_vector,
            (-1) ** (x * chstate.b_state),
            (x + chstate.b_state + 1) % 2,
        )
    )
    result = (
        2 ** (-1 * np.sum(chstate.h_vector) / 2)
        * (1j) ** tot_phase_exp
        * prod_01
        * chstate.g_phase
    )
    return result


def find_pauli(basis_state: np.array, chstate):
    """Compute the product of the pauli operators that are marked (value 1) by the basis state array."""
    inds = np.where(basis_state == 1)[0]

    # calculate product of paulis or return identity operator.
    if len(inds) > 1:
        z, x = chstate.M[inds, :], chstate.F[inds, :]
        com_phase = np.sum(z * x)
        final_pauli, phase_exp = multiply_paulis(
            [np.append(z[i, :], x[i, :]) for i in range(np.shape(z)[0])]
        )
        z_, x_ = np.array(final_pauli[: chstate.num_qubits]), np.array(
            final_pauli[chstate.num_qubits :]
        )
        tot_phase_exp = (
            np.sum(chstate.l_phases[inds]) + phase_exp + com_phase + z_ @ x_
        ) % 4
    elif len(inds) == 1:
        z_, x_ = chstate.M[inds[0], :], chstate.F[inds[0], :]
        com_phase = np.sum(z_ * x_)
        tot_phase_exp = (chstate.l_phases[inds[0]] + 2 * com_phase) % 4
    else:
        z_, x_ = np.array([0] * len(basis_state)), np.array([0] * len(basis_state))
        tot_phase_exp = 0
    return z_, x_, tot_phase_exp


def overlap_given_pauli_sampling(qxs: Paulis, flipped_bit: int, stabdecos):
    """
    Function that calculates the overlap  <0| Q_y |s> = <0| (U_C^dag * X_(flipped_bit) * U_C) * Q_x |s> for all stabilizers in a stabilizer decomposition.

    Returns the value of the overlap, and a Paulis named tuple object. See "overlap_given_pauli_sampling" for more details.

    Attributes
    ==========
    * qxs -- Paulis (namedtuple) object.
    * flipped_bit -- index of the bit that is flipped.
    * stabdecos -- StabDecos (namedtuple) object. For more information refere to the "stabilizer_decomposition" function docstring.
    """
    num_qubits = stabdecos.num_qubits
    k = len(stabdecos.coeffs)

    a = np.arange(flipped_bit, num_qubits * k, num_qubits)
    F_block = stabdecos.F_block[a, :]
    M_block = stabdecos.M_block[a, :]
    l_block = stabdecos.l_block[a, :]

    old_pauli_phases = np.reshape(qxs.phases, (k, -1))
    z_new = (M_block + qxs.z_parts) % 2
    x_new = (F_block + qxs.x_parts) % 2
    c = 2 * np.sum(
        z_new * x_new - qxs.z_parts * qxs.x_parts - qxs.z_parts * F_block,
        axis=1,
        keepdims=True,
    )
    qy_phases = (c + old_pauli_phases + l_block) % 4

    prod_01 = np.prod(
        np.where(
            stabdecos.h_block,
            (-1) ** (x_new * stabdecos.b_block),
            (x_new + stabdecos.b_block + 1) % 2,
        ),
        axis=1,
    )

    result = (
        2 ** (-1 * np.sum(stabdecos.h_block, axis=1) / 2)
        * stabdecos.g_block
        * (1j) ** qy_phases.flatten()
        * prod_01
    )

    return (
        result @ stabdecos.coeffs,
        Paulis(z_new, x_new, qy_phases),
    )


def overlap_given_basis_sampling(basis_state, stabdecos):
    """
    Function that calculates the overlap between a basis state and a stabilizer decomposition.

    Returns a Paulis namedtuple:
    * Paulis.z_parts -- 2d array refering to the first n bits of the pauli strings.
    * Paulis.x_parts -- 2d array refering to the last n bits of the pauli strings.
    * Paulis.phases -- 1d array of the phases associated with each pauli string.
    ​
    Attributes
    ==========
    * basis_state -- 1d array of zeros and ones.
    * stabdecos -- StabDecos (namedtuple) objects. For more information refere to the "stabilizer_decomposition" function docstring.
    """
    basis_state = np.array(basis_state)
    assert len(basis_state) == stabdecos.num_qubits
    assert np.all(basis_state ** 2 == basis_state)

    paulis_z, paulis_x, tot_phases_exp = find_paulis(basis_state, stabdecos)

    # should be a 1d array of length k.
    prod_01 = np.prod(
        np.where(
            stabdecos.h_block,
            (-1) ** (paulis_x * stabdecos.b_block),
            (paulis_x + stabdecos.b_block + 1) % 2,
        ),
        axis=1,
    )

    # also a 1d vector.
    result = (
        2 ** (-1 * np.sum(stabdecos.h_block, axis=1) / 2)
        * (1j) ** tot_phases_exp
        * prod_01
        * stabdecos.g_block
    )

    overlap = result @ stabdecos.coeffs

    return overlap, Paulis(paulis_z, paulis_x, tot_phases_exp)


def find_paulis(basis_state, stabdecos):
    """
    Finds all k paulis (divided in their 'z' and 'x' parts) simultaneously. The final "z" and "x" arrays have size (k x num_qubits), while the array of phases is 1d and of length k.
    """
    num_qubits = stabdecos.num_qubits
    k = len(stabdecos.coeffs)
    inds = np.where(basis_state == 1)[0]
    lines_pblock = len(inds)

    if lines_pblock > 1:
        # obtain the relevant "z" and "x" lines from each block.
        relevant_inds_blocks = np.add.outer(
            inds, np.arange(0, k, 1) * num_qubits
        ).ravel(order="F")
        zs = stabdecos.M_block[relevant_inds_blocks, :]
        xs = stabdecos.F_block[relevant_inds_blocks, :]

        # multiply the paulis together in each block and compute the final phases.
        block_divs = np.arange(0, len(zs), lines_pblock)
        zs_, xs_, phases_exp = mult_paulis(zs, xs, k, lines_pblock)
        com_phases = np.add.reduceat(zs * xs, block_divs, axis=0)
        com_phases = np.sum(com_phases, axis=1)
        l_array = stabdecos.l_block[relevant_inds_blocks, :]
        tot_phases_exp = (
            np.add.reduceat(l_array, block_divs, axis=0).flatten()
            + phases_exp
            + com_phases
            + np.sum(zs_ * xs_, axis=1)
        ) % 4
    elif lines_pblock == 1:
        relevant_inds_blocks = inds[0] + np.arange(0, k, 1) * num_qubits
        zs_ = stabdecos.M_block[relevant_inds_blocks, :]
        xs_ = stabdecos.F_block[relevant_inds_blocks, :]
        com_phases = np.sum(zs_ * xs_, axis=1)
        tot_phases_exp = (
            stabdecos.l_block[relevant_inds_blocks, :].flatten() + 2 * com_phases
        ) % 4
    else:
        zs_ = np.zeros((k, num_qubits))
        xs_ = np.zeros((k, num_qubits))
        tot_phases_exp = np.zeros(k)
    return zs_, xs_, tot_phases_exp


def mult_paulis(zs, xs, k, lines_pblock):
    """
    Multiplies multiple Pauli operators in their binary form simultaneously. Returns the new Pauli operators split into its 'z' and 'x' parts, of size (k, n) each, as well as their respective phases.

    Attributes
    ===========
    * zs - "z" parts of the relevant Pauli operators found on "find_paulis". Size is (k * lines_pblock, n).
    * xs - "x" parts of the relevant Pauli operators found on "find_paulis". Size is (k * lines_pblock, n).
    """
    # grab the first z and x of every block and then delete these from the bigger array.
    first_row_of_each = np.arange(0, k, 1) * lines_pblock
    original_zs = zs[first_row_of_each, :]
    original_xs = xs[first_row_of_each, :]
    zs = np.delete(zs, first_row_of_each, axis=0)
    xs = np.delete(xs, first_row_of_each, axis=0)

    # sum together all z's and x's in a block.
    block_divs = np.arange(0, len(zs), lines_pblock - 1)
    sum_zs = np.add.reduceat(zs, block_divs, axis=0)
    sum_xs = np.add.reduceat(xs, block_divs, axis=0)
    exps_mod_4 = np.sum(original_zs * sum_xs, axis=1) - np.sum(
        original_xs * sum_zs, axis=1
    )

    # normalize and determine the final phase.
    zs_nonorm, xs_nonorm = original_zs + sum_zs, original_xs + sum_xs
    zs_norm, xs_norm = zs_nonorm % 2, xs_nonorm % 2
    exps_mod_4 += np.sum(zs_norm * xs_norm, axis=1) - np.sum(
        zs_nonorm * xs_nonorm, axis=1
    )

    return zs_norm, xs_norm, exps_mod_4 % 4