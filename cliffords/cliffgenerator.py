import numpy as np
import networkx as nx
import pickle as pk
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
file1 = os.path.join(THIS_DIR, "1q_cliffords.p")
file2 = os.path.join(THIS_DIR, "2q_cliffords.p")


def generate_group(generators, generator_weights, dim):
    """
    (Jonas provided this code. I've performed slight modifications to suit our needs.)
    This function computes all elements of a group given a set of generators. In addition, it also computes the relations between the elements as well as optimal decompositions in terms of generators.

    Returns: 1) A dictionary of the form {"index of group element : matrix form of group element} (The identity is given index 0)
             2) A dictionary of the form {"index of group element: Decomposition of group element into generators}.
    The decomposition is given as a string of the form "gen_1,gen_2,gen3,...,gen_k".

    Attributes:
    ==============
    * generators -- A dictionary of generators taking the form {"name of generator" : matrix form of generator as numpy array}
    * generator_weights --
    * dim -- The dimension of the matrices considered
    """

    dim_loc = dim
    nodes = {
        0: np.eye(dim_loc)
    }  # create a dictionary of nodes with one initial node (associated to the identity).
    edges = {}  # create an empty dictionary of edges.

    current_node = 0  # sets the current node being iterated over.

    while (
        len(nodes) > current_node
    ):  # loop over all nodes until we run out (nodes will be created inside the loop).
        for gen_key in generators:  # loop over all possible generators.

            A = (
                nodes[current_node] * generators[gen_key]
            )  # create group element A by multiplying the current_node element and a generator.
            node_exists = False
            for node_key in nodes:

                # Check if A is already present in one of the nodes (up to a phase).
                if abs(abs(np.trace(A.H * nodes[node_key])) - dim_loc) < 0.01:
                    # print('found!')
                    node_exists = (
                        True  # signify that A is already present in the graph.
                    )

                    # create an edge from current_node to the node containing A (node_key).
                    edges[(current_node, node_key)] = gen_key

                    # break out of loop over all nodes.
                    break

            if (
                node_exists == False
            ):  # check if the node A exists (signified by node_exists).
                # add node A to the graph.
                nodes[len(nodes)] = A
                # add an edge between current_node and A.
                edges[(current_node, len(nodes) - 1)] = gen_key

        if (current_node % 100) == 0 and current_node != 0:
            print(
                f"Total group elements so far: {len(nodes)},",
                f"Currently checking group element: {current_node}",
            )

        current_node = current_node + 1  # go to next node in nodes.
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for edge_key in edges:
        G.add_edge(*edge_key, weight=generator_weights[edges[edge_key]])

    # compute the decompositions in terms of the generators.
    paths_dict = nx.single_source_dijkstra_path(G, 0)

    decomposition_dict = {i: "" for i in range(0, len(nodes))}

    separator = ","
    for group_element in range(1, len(paths_dict)):
        for i in range(0, len(paths_dict[group_element]) - 1):
            if (
                paths_dict[group_element][i],
                paths_dict[group_element][i + 1],
            ) in edges:
                decomposition_dict[group_element] = separator.join(
                    [
                        decomposition_dict[group_element],
                        edges[
                            (
                                paths_dict[group_element][i],
                                paths_dict[group_element][i + 1],
                            )
                        ],
                    ]
                )
            else:
                decomposition_dict[group_element] = separator.join(
                    [
                        decomposition_dict[group_element],
                        edges[
                            (
                                paths_dict[group_element][i + 1],
                                paths_dict[group_element][i],
                            )
                        ],
                    ]
                )
        decomposition_dict[group_element] = decomposition_dict[group_element][1:]

    print("Done!")
    return nodes, decomposition_dict, G


def cliff_group_1():
    """Create the Clifford group on one qubit"""
    h = np.matrix([[1, 1], [1, -1]]) / np.sqrt(2)
    s = np.matrix([[1, 0], [0, 1j]])

    # define the group and its weights.
    single_qubit_group = {"h": h, "s": s}
    single_qubit_group_weights = {"h": 1, "s": 1}

    # compute the group
    (
        group_nodes_martinis,
        group_decomposition_martinis,
        group_graph_martinis,
    ) = generate_group(single_qubit_group, single_qubit_group_weights, 2)

    # write the group as a dictionary, output a pickle file.
    group = {}
    for i in range(len(group_nodes_martinis)):
        group[i] = {
            "matrix": group_nodes_martinis[i],
            "decomposition": group_decomposition_martinis[i],
        }
    pk.dump(group, open(file1, "wb"))


def cliff_group_2():
    """Create the Clifford group on two qubits"""
    h = np.matrix([[1, 1], [1, -1]]) / np.sqrt(2)
    s = np.matrix([[1, 0], [0, 1j]])
    cz = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

    one_gens = {"h": h, "s": s}

    # create the initial dictionary of all combinations of single qubit gates with the identity and the cz gate.
    two_qubit_group = {
        j + "1": np.matrix(np.kron(np.eye(2), one_gens[j])) for j in one_gens
    }
    two_qubit_group.update(
        {i + "0": np.matrix(np.kron(one_gens[i], np.eye(2))) for i in one_gens}
    )
    two_qubit_group["cz"] = cz

    # define new weights.
    two_qubit_group_weights = {j + "1": 0.01 for j in one_gens}
    two_qubit_group_weights.update({i + "0": 0.01 for i in one_gens})
    two_qubit_group_weights["cz"] = 1

    # compute the group.
    (
        group_nodes_martinis,
        group_decomposition_martinis,
        group_graph_martinis,
    ) = generate_group(two_qubit_group, two_qubit_group_weights, 4)

    # write the group as a dictionary, output a pickle file.
    group = {}
    for i in range(len(group_nodes_martinis)):
        group[i] = {
            "matrix": group_nodes_martinis[i],
            "decomposition": group_decomposition_martinis[i],
        }
    pk.dump(group, open(file2, "wb"))


cliff_group_2()
