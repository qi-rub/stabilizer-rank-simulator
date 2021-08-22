import numpy as np
from ..circuit import Circuit
from simulator.backend.chtableau import run
from collections import namedtuple

StabDecos = namedtuple(
    "StabDecos",
    "F_block M_block l_block g_block h_block b_block coeffs num_qubits",
)


def stabilizer_decomposition(
    circuit: Circuit, delta: float, nc_decompositions: dict, alpha=1
) -> StabDecos:
    """
    The given circuit specifies a state |psi>. This function produces a state |psi'> with || |psi> - |psi'> || < delta, where |psi'> = sum(c_i |phi_i>) for i = 0, . . ., k.

    The function returns a StabDecos object which basically contains the 6 parameters that describe the states |phi_i> and the coefficients c_i. We do not include the "G" parameter as this is not needed for future calculations. The StabDecos object is as follows:
    * F_block -- the F matrices of each chstate stacked on top of each other. Size is (k * n , n).
    * M_block -- the M matrices of each chstate stacked. Size is (k * n , n).
    * l_block -- the vector of local phases of each chstate stacked. Size is (k * n , 1).
    * g_block -- the global phase of each chstate stacked. Size is (k , 1).
    * h_block -- the array of Hadamards for each chstate stacked. Size is (k * n , 1).
    * b_block -- the array of basis states for each chstate stacked. Size is (k * n , 1).
    * coeffs -- the array of coefficients of each chstate. Size is (k,).
    * num_qubits -- number of qubits of the chstates.

    Attributes
    ==========
    * circuit - circuit that constructs the state |psi> = circuit |0^n>.
    * delta - approximation constant.
    * alpha - constant that allows for manual scaling of k.
    """
    assert delta > 0 and delta < 1

    num_qubits = circuit.num_qubits
    coeffs = []

    # compute k. If the circuit is made entirely out of Cliffords we want k=1.
    circ_norm = compute_norm(circuit, nc_decompositions)
    k = 1 if circ_norm == 1 else int((circ_norm * alpha) / delta ** 2)

    F_block = []
    M_block = []
    l_block = []
    g_block = []
    h_block = []
    b_block = []
    for _ in range(k):

        # sample a Cifford circuit using the sparsification procedure.
        new_circ, coeff = sample_cliff_circ(circuit, nc_decompositions)
        coeffs.append(coeff)

        # evaluate the circuit to create a CH state and add to the list.
        state = run(new_circ)

        # append to the large blocks.
        F_block.append(state.F)
        M_block.append(state.M)
        l_block.append(np.reshape(state.l_phases, (circuit.num_qubits, 1)))
        g_block.append(state.g_phase)
        h_block.append(state.h_vector)
        b_block.append(state.b_state)

    return StabDecos(
        np.vstack(F_block),
        np.vstack(M_block),
        np.vstack(l_block),
        np.array(g_block),
        np.array(h_block),
        np.array(b_block),
        coeffs,
        num_qubits,
    )


def sample_cliff_circ(circuit: Circuit, nc_decompositions: dict):
    """ Obtains a Clifford circuit using the sparsification technique. Returns the circuit, and the coefficient associated with the circuit."""
    output_circ = []
    total_coeff = 1
    for instruction in circuit.instructions:
        if instruction[0] in nc_decompositions:
            # res is a namedtuple object: a Clifford gate and its coefficient.
            # you cannot pass lists of objects to np.random.choice.
            res = nc_decompositions[instruction[0]]["decompositions"][
                np.random.choice(
                    len(nc_decompositions[instruction[0]]["decompositions"]),
                    p=nc_decompositions[instruction[0]]["probabilities"],
                )
            ]
            total_coeff *= res.coeff
            # since the chosen decomposition might be a product of clifford gates we have to split them into different instructions.
            if len(instruction) == 2:  # single-qubit gate.
                for cliff_gate in res.gate_name.split(","):
                    output_circ.append((cliff_gate, instruction[1]))
            elif len(instruction) == 3:  # two-qubit gate
                for cliff_gate in res.gate_name.split(","):
                    # check if the second character is 0 or 1. These could be single qubit gates tensored with the identity.
                    if cliff_gate[1] in ["0", "1"]:
                        output_circ.append(
                            (cliff_gate[0], instruction[int(cliff_gate[1]) + 1])
                        )
                    else:
                        output_circ.append((cliff_gate, instruction[1], instruction[2]))
            else:
                raise Exception(
                    "Decompositions of other types of gates have not been implemented."
                )
        else:
            output_circ.append(instruction)

    return Circuit(circuit.num_qubits, output_circ), total_coeff


def compute_norm(circuit, nc_decompositions):
    """Computes the l-1 norm (squared!) of the Clifford decomposition of a non-Clifford circuit."""
    coeff = 1
    for instruction in circuit.instructions:
        if instruction[0] in nc_decompositions:
            coeff *= (nc_decompositions[instruction[0]]["normalization"]) ** 2
    return coeff
