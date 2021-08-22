import itertools


def bitstrings(k):
    return itertools.product([0, 1], repeat=k)
