import numpy as np
import math


class BinCounter:
    """
    Class that helps with the counting of the binary strings sampled by the Monte Carlo simulator.

    Presents a couple of advantages over the "Counter" object from the "collections" module:
        * This class presents the strings in the order in which they were created, and doesn't rearrange them by count order. (If all = True).
        * If given a string, it considers the string as a whole and not the individual characters in it. Does not consider strings as iterables.

    If all = False, it initializes an empty dictionary, and works just like a normal counter.
    """

    def __init__(self, num_qubits, all=False):
        """
        By using the number of qubits in the system under consideration, the function creates a dictionary containing every the strings indexing the computational basis states.
        Thus, if n is the number of qubits, it creates a dictionary with 2^n binary strings.
        """
        assert num_qubits > 0

        self.counts = self.create_dict(num_qubits) if all == True else {}

    def __str__(self):
        return str(self.counts)

    @staticmethod
    def create_dict(num_qubits):
        """Function that creates the ordered dictionary of 2^n strings, starting at 0...0 followed by 0...1, and so on. Initializes the counter of at 0."""

        bin_dict = {}
        for i in range(2 ** num_qubits):
            bin_dict[format(i, "0" + str(num_qubits) + "b")] = 0
        return bin_dict

    def update(self, object_to_count):
        """Given an individual string or list of strings, it increases the counter (or creates an entry) for that specific string"""

        # we have to differentiate between a single string or lists of strings, as we do not want to accidentally iterate over the string
        if isinstance(object_to_count, str):
            if object_to_count in self.counts:
                self.counts[object_to_count] += 1
            else:
                self.counts[object_to_count] = 1
        # if the object is a list
        else:
            for string in object_to_count:
                if string in self.counts:
                    self.counts[string] += 1
                else:
                    self.counts[string] = 1


def reorder_qiskit_counter(num_qubits, count_dict):
    """
    Qiskit's count dictionary uses a different qubit order so this function converts their count dictionary and converts it to our convention. It also sorts the binary strings in increasing order.
    """

    ordered_dict = {}
    for i in range(2 ** num_qubits):
        bin_str = format(i, "0" + str(num_qubits) + "b")
        try:
            ordered_dict[bin_str] = count_dict[bin_str[::-1]]
        except KeyError:
            ordered_dict[bin_str] = 0
    return ordered_dict