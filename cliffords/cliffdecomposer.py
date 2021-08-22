import numpy as np
import cvxpy as cp
import pickle as pk
import math
import json
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
cliffords_1q = os.path.join(THIS_DIR, "1q_cliffords.p")
cliffords_2q = os.path.join(THIS_DIR, "2q_cliffords.p")
decompositions_file = os.path.join(THIS_DIR, "decomposition.json")


def calc_min_decomposition(target_mat: np.array, gate_name: str):
    """
    Function that computes the optimal (minimum value of ||c||_1) Clifford decomposition of the 2x2 or 4x4 target matrix. Returns the vector of coefficients that minimize the l-1 norm of the decomposition.
    """
    assert target_mat.shape == (2, 2) or target_mat.shape == (
        4,
        4,
    ), "The matrix must be 2x2 or 4x4."

    dim = target_mat.shape[0]
    cliff_dict, num_cliffs = load_dict(dim)

    # create A, the complex matrix of constraints.
    A = np.zeros((dim ** 2, num_cliffs), dtype=complex)
    for i in range(dim ** 2):
        for ii in range(num_cliffs):
            mat = cliff_dict[ii]["matrix"]
            A[i][ii] = mat[math.floor(i / dim), i % dim]
    target = target_mat.flatten()

    # calculate the minimum decomposition.
    x = cp.Variable(num_cliffs, complex=True)
    objective = cp.Minimize(cp.norm(x, p=1))
    problem = cp.Problem(objective, [A @ x == target])
    problem.solve()

    # extract relevant info.
    non_zero_elements = np.where(np.isclose(x.value, 0) == False)
    coefficients = x.value[non_zero_elements]
    gate_in_generators = []
    for i in non_zero_elements[0]:
        gate_in_generators.append(cliff_dict[i]["decomposition"])

    # write to the decompositions JSON file
    write_to_dict(coefficients, gate_in_generators, gate_name)

    return problem.value, coefficients, gate_in_generators


def load_dict(dim: int):
    """Load all Clifford matrices into a list."""
    if dim == 2:
        group = pk.load(open(cliffords_1q, "rb"))
        return group, 24
    else:
        group = pk.load(open(cliffords_2q, "rb"))
        return group, 11520


def write_to_dict(coefficients: list, gate_decompositions: list, gate_name: str):
    """ Write the decomposition found to the json file."""
    # create the new dictionary of the decomposition.
    assert len(coefficients) == len(
        gate_decompositions
    ), "The sizes of the lists do not match."

    # create the new dictionary to be added.
    decomposition = []
    for i in range(len(coefficients)):
        decomposition_dict = {}
        decomposition_dict["gate"] = gate_decompositions[i]
        decomposition_dict["coeff_real"] = np.real(coefficients[i])
        decomposition_dict["coeff_imag"] = np.imag(coefficients[i])
        decomposition.append(decomposition_dict)

    gate = {gate_name: {"decomposition": decomposition}}

    # save to the JSON file.
    with open(decompositions_file) as f:
        decos = json.load(f)
        decos.update(gate)
    with open(decompositions_file, "w") as f:
        json.dump(decos, f)
