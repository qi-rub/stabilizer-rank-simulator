import os
import json
import pickle as pk
from collections import namedtuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
decomposition_json = os.path.join(THIS_DIR, "decomposition.json")

DecompositionTerm = namedtuple("DecompositionTerm", "gate_name coeff")


def create_decomposition_pickle():
    """Creates a new dictionary of the non-Clifford decompositions from the JSON file and stores this object as a binary stream file. This new dictionary is easier for Python to work with."""
    non_c_gates = {}
    with open(decomposition_json) as f:
        dct = json.load(f)
        for gate, decs in dct.items():
            decompositions = {}
            terms = []
            probabilities = []
            total_coeff = 0
            for dec in decs["decomposition"]:
                deco_name = dec["gate"]
                coeff = complex(dec["coeff_real"], dec["coeff_imag"])
                total_coeff += abs(coeff)
                term = DecompositionTerm(
                    deco_name,
                    coeff,
                )
                probabilities.append(abs(coeff))
                terms.append(term)
            probabilities = [x / total_coeff for x in probabilities]
            decompositions["decompositions"] = terms
            non_c_gates[gate] = decompositions
            non_c_gates[gate]["probabilities"] = probabilities
            non_c_gates[gate]["normalization"] = total_coeff
    pk.dump(non_c_gates, open(os.path.join(THIS_DIR, "ncdecompositions.p"), "wb"))


create_decomposition_pickle()
