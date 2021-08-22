import numpy as np
from simulator.backend.chtableau import CHState
from simulator.backend.chtableau import run
from simulator.backend.stabilizer import run as run_stab
from simulator.circuit import Circuit


def test_init():
    state = CHState.trivial_state(5)
    assert np.array_equal(state.G, np.eye(state.num_qubits))
    assert np.array_equal(state.F, np.eye(state.num_qubits))
    assert np.array_equal(state.M, np.zeros((state.num_qubits, state.num_qubits)))
    assert np.array_equal(state.l_phases, np.zeros(state.num_qubits))
    assert state.g_phase == 1
    assert np.array_equal(state.h_vector, np.zeros(state.num_qubits))
    assert np.array_equal(state.b_state, np.zeros(state.num_qubits))


def test_basis_state():

    state = CHState.basis_state([0, 1])
    assert np.array_equal(
        state.density_matrix(),
        np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
    )

    state = CHState.basis_state([1, 1])
    assert np.array_equal(
        state.density_matrix(),
        np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]]),
    )

    state = CHState.basis_state([1, 0, 1])
    assert np.array_equal(
        state.density_matrix(),
        np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
    )

    state = CHState.basis_state([1])
    assert state == CHState(
        np.eye(1),
        np.eye(1),
        np.zeros((1, 1)),
        np.zeros(1),
        1,
        np.zeros(1),
        np.array([1]),
    )


def test_apply_h():
    # test a trivial case
    state = CHState.trivial_state(1)
    state.apply_h(0)
    state.apply_h(0)
    assert np.array_equal(state.density_matrix(), np.array([[1, 0], [0, 0]]))

    # |0> -> |+> -> |0>
    state = CHState.basis_state([0])
    state.apply_h(0)
    state.apply_h(0)
    assert state == CHState.basis_state([0])

    # |1> -> |-> -> |1>
    state = CHState.basis_state([1])
    state.apply_h(0)
    state.apply_h(0)
    assert state == CHState.basis_state([1])

    # apply a Hadamard gate to the last qubit of |00> and check that the result is |0+> and not |+0>
    state = CHState.basis_state([0, 0])
    state.apply_h(1)
    plus = [[1 / 2, 1 / 2], [1 / 2, 1 / 2]]
    zero = [[1, 0], [0, 0]]
    assert np.allclose(state.density_matrix(), np.kron(zero, plus))


def test_apply_s_l():
    # test a trivial case
    state = CHState.trivial_state(1)
    state.apply_s_l(0)
    assert np.array_equal(state.density_matrix(), np.array([[1, 0], [0, 0]]))

    state = CHState.basis_state([1])
    state.apply_s_l(0)
    assert np.array_equal(state.density_matrix(), np.array([[0, 0], [0, 1]]))


def test_cx_l_on_basis_states():
    # |00> -> |00>
    state = CHState.basis_state([0, 0])
    state.apply_cx_l(0, 1)
    assert state == CHState.basis_state([0, 0])
    state.apply_cx_l(1, 0)
    assert state == CHState.basis_state([0, 0])

    # |01> -> |01>
    state = CHState.basis_state([0, 1])
    state.apply_cx_l(0, 1)
    assert state == CHState.basis_state([0, 1])

    # |10> -> |11> -> |10>
    state = CHState.basis_state([1, 0])
    state.apply_cx_l(0, 1)
    assert state == CHState.basis_state([1, 1])
    state.apply_cx_l(0, 1)
    assert state == CHState.basis_state([1, 0])


def test_pauli_expectation():
    # check some expected values for state |00>
    state = CHState.trivial_state(2)
    assert state.pauli_expectation([1, 0, 0, 0]) == 1
    assert state.pauli_expectation([0, 1, 0, 0]) == 1
    assert state.pauli_expectation([0, 0, 1, 0]) == 0
    assert state.pauli_expectation([0, 0, 0, 1]) == 0
    assert state.pauli_expectation([1, 1, 0, 0]) == 1
    assert state.pauli_expectation([0, 0, 1, 1]) == 0
    assert state.pauli_expectation([1, 1, 0, 1]) == 0
    assert state.pauli_expectation([1, 1, 1, 1]) == 0
    assert state.pauli_expectation([0, 0, 0, 0]) == 1

    # check expected values for some other basis states
    state = CHState.basis_state([1, 1])
    assert state.pauli_expectation([0, 0, 1, 0]) == 0
    assert state.pauli_expectation([1, 0, 0, 0]) == -1
    assert state.pauli_expectation([1, 0, 1, 0]) == 0
    assert state.pauli_expectation([1, 1, 0, 1]) == 0
    assert state.pauli_expectation([1, 1, 0, 0]) == 1

    # check some expected values for the max. entangled state
    circ = Circuit(2)
    circ.h(0)
    circ.cx(0, 1)
    state = run(circ)

    assert state.pauli_expectation([1, 1, 0, 0]) == 1
    assert state.pauli_expectation([0, 0, 1, 1]) == 1
    assert state.pauli_expectation([1, 1, 1, 1]) == -1


def test_cz_rules_both():
    state = CHState.trivial_state(2)
    state.apply_h(0)
    state.apply_h(1)
    # now we apply cz from the left
    state.apply_cz_l(1, 0)
    # this will be the U_c
    state.apply_s_l(1)
    state.apply_cx_l(0, 1)
    state.apply_s_l(1)
    state.apply_s_l(0)
    state.apply_cx_l(0, 1)

    state1 = CHState.trivial_state(2)
    state1.apply_h(0)
    state1.apply_h(1)
    # this will be the U_c
    state1.apply_s_l(1)
    state1.apply_cx_l(0, 1)
    state1.apply_s_l(1)
    state1.apply_s_l(0)
    state1.apply_cx_l(0, 1)
    # now we apply cz from the right
    state1.apply_cz_r(1, 0)

    assert np.array_equal(state.density_matrix(), state1.density_matrix())


def test_cx_rules_both():
    state = CHState.trivial_state(2)
    state.apply_h(0)
    state.apply_h(1)
    # now we apply cx from the left
    state.apply_cx_l(1, 0)
    # this will be the U_c
    state.apply_s_l(1)
    state.apply_cz_l(0, 1)
    state.apply_s_l(0)
    state.apply_cz_l(0, 1)

    state1 = CHState.trivial_state(2)
    state1.apply_h(0)
    state1.apply_h(1)
    # this will be the U_c
    state1.apply_s_l(1)
    state1.apply_cz_l(0, 1)
    state1.apply_s_l(0)
    state1.apply_cz_l(0, 1)
    # now we apply cz from the right
    state1.apply_cx_r(1, 0)

    assert np.array_equal(state.density_matrix(), state1.density_matrix())


def test_s_rules_both():
    state = CHState.trivial_state(2)
    state.apply_h(0)
    state.apply_h(1)
    # now we apply s from the left
    state.apply_s_l(0)
    # this will be the U_c
    state.apply_cz_l(0, 1)
    state.apply_cx_l(1, 0)
    state.apply_cz_l(0, 1)

    state1 = CHState.trivial_state(2)
    state1.apply_h(0)
    state1.apply_h(1)
    # this will be the U_c
    state1.apply_cz_l(0, 1)
    state1.apply_cx_l(1, 0)
    state1.apply_cz_l(0, 1)
    # now we apply s from the right
    state1.apply_s_r(0)

    assert np.array_equal(state.density_matrix(), state1.density_matrix())


def test_run():
    circ = Circuit(2)
    circ.h(0)
    circ.cx(0, 1)
    state = run(circ)
    assert np.array_equal(
        state.density_matrix(),
        np.array([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]]),
    )

    circ = Circuit(1)
    circ.h(0)
    circ.s(0)
    state = run(circ)

    assert state.pauli_expectation([1, 1]) == 1
    assert state.pauli_expectation([0, 1]) == 0
    assert state.pauli_expectation([1, 0]) == 0

    assert np.allclose(
        state.density_matrix(),
        np.array(
            [
                [0.5, -0.5j],
                [0.5j, 0.5],
            ]
        ),
    )

    circ = Circuit(2)
    circ.s(1)
    circ.s(0)
    circ.h(0)

    circ = Circuit(2)
    state_init = run(circ)
    circ.cx(0, 1)
    state_1 = run(circ)
    circ.h(0)

    state = run(circ)

    assert np.array_equal(
        state.density_matrix(),
        np.array(
            [
                [0.5, 0, 0.5, 0],
                [0, 0, 0, 0],
                [0.5, 0, 0.5, 0],
                [0, 0, 0, 0],
            ]
        ),
    )

    circ = Circuit(2)
    state_init = run(circ)
    circ.cx(1, 0)
    state_1 = run(circ)
    circ.h(1)

    state = run(circ)

    assert np.array_equal(
        state.density_matrix(),
        np.array(
            [
                [0.5, 0.5, 0, 0],
                [0.5, 0.5, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        ),
    )

    circ = Circuit(2)
    circ.s(1)
    circ.h(1)
    state = run(circ)

    assert np.array_equal(
        state.density_matrix(),
        np.array(
            [
                [0.5, 0.5, 0, 0],
                [0.5, 0.5, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        ),
    )


def test_run2():
    circ = Circuit(1)
    circ.s(0)
    circ.h(0)
    state = run(circ)

    assert state.pauli_expectation([0, 1]) == 1
    assert state.pauli_expectation([1, 1]) == 0
    assert state.pauli_expectation([1, 0]) == 0

    assert np.array_equal(state.density_matrix(), np.array([[0.5, 0.5], [0.5, 0.5]]))


def test_run3():
    circ = Circuit(1)
    circ.h(0)
    circ.s(0)
    circ.h(0)
    state = run(circ)

    assert state.pauli_expectation([1, 0]) == 0
    assert state.pauli_expectation([0, 1]) == 0
    assert state.pauli_expectation([1, 1]) == -1

    assert np.array_equal(state.density_matrix(), np.array([[0.5, 0.5j], [-0.5j, 0.5]]))


def test_run4():
    circ = Circuit(2)
    circ.cx(1, 0)
    circ.s(0)
    circ.h(0)
    circ.h(1)
    state = run(circ)

    assert state.pauli_expectation([1, 0, 0, 1]) == 0
    assert state.pauli_expectation([0, 1, 1, 0]) == 0
    assert state.pauli_expectation([0, 0, 0, 1]) == 1
    assert state.pauli_expectation([0, 0, 1, 0]) == 1
    assert state.pauli_expectation([0, 1, 0, 0]) == 0
    assert state.pauli_expectation([1, 0, 0, 0]) == 0
    assert state.pauli_expectation([1, 1, 1, 1]) == 0
    assert state.pauli_expectation([1, 0, 1, 1]) == 0
    assert state.pauli_expectation([1, 1, 0, 1]) == 0
    assert state.pauli_expectation([1, 1, 0, 0]) == 0
    assert state.pauli_expectation([0, 0, 1, 1]) == 1

    assert np.array_equal(
        state.density_matrix(),
        np.array(
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
            ]
        ),
    )


def test_run5():
    circ = Circuit(2)
    circ.h(1)
    circ.h(0)
    circ.s(1)
    circ.cx(0, 1)
    state = run(circ)
    state_stab = run_stab(circ)

    assert state.pauli_expectation([1, 0, 0, 1]) == 0
    assert state.pauli_expectation([0, 1, 1, 0]) == 0
    assert state.pauli_expectation([0, 0, 1, 0]) == 0
    assert state.pauli_expectation([0, 1, 0, 0]) == 0
    assert state.pauli_expectation([1, 0, 0, 0]) == 0
    assert state.pauli_expectation([1, 1, 1, 1]) == 0
    assert state.pauli_expectation([1, 0, 1, 1]) == 0
    assert state.pauli_expectation([1, 1, 0, 1]) == 1
    assert state.pauli_expectation([1, 1, 0, 0]) == 0
    assert state.pauli_expectation([0, 0, 1, 1]) == 1

    # assert np.array_equal(
    #     state.density_matrix(),
    #     np.array(
    #         [
    #             [0.25, -0.25j, -0.25j, 0.25],
    #             [0.25j, 0.25, 0.25, 0.25j],
    #             [0.25j, 0.25, 0.25, 0.25j],
    #             [0.25, -0.25j, -0.25j, 0.25],
    #         ]
    #     ),
    # )

    assert np.allclose(state.density_matrix(), state_stab.density_matrix())


# Passes both.
def test_run6():
    circ = Circuit(2)
    circ.cx(1, 0)
    circ.h(1)
    circ.h(0)
    circ.h(1)
    state = run(circ)

    assert state.pauli_expectation([1, 0, 0, 1]) == 0
    assert state.pauli_expectation([0, 0, 1, 0]) == 1
    assert state.pauli_expectation([1, 0, 0, 0]) == 0
    assert state.pauli_expectation([1, 1, 1, 1]) == 0
    assert state.pauli_expectation([1, 0, 1, 1]) == 0
    assert state.pauli_expectation([1, 1, 0, 1]) == 0
    assert state.pauli_expectation([1, 1, 0, 0]) == 0
    assert state.pauli_expectation([0, 0, 1, 1]) == 0
    assert state.pauli_expectation([0, 1, 0, 0]) == 1
    assert state.pauli_expectation([0, 1, 1, 0]) == 1

    assert np.array_equal(
        state.density_matrix(),
        np.array([[0.5, 0, 0.5, 0], [0, 0, 0, 0], [0.5, 0, 0.5, 0], [0, 0, 0, 0]]),
    )


def test_run7():
    circ = Circuit(2)
    circ.h(0)
    circ.cx(0, 1)
    circ.h(1)
    state = run(circ)
    state_stab = run_stab(circ)

    assert state.pauli_expectation([1, 0, 0, 1]) == 1
    assert state.pauli_expectation([0, 1, 1, 0]) == 1
    assert state.pauli_expectation([0, 0, 1, 0]) == 0
    assert state.pauli_expectation([0, 1, 0, 0]) == 0
    assert state.pauli_expectation([1, 0, 0, 0]) == 0
    assert state.pauli_expectation([1, 1, 1, 1]) == 1
    assert state.pauli_expectation([1, 0, 1, 1]) == 0
    assert state.pauli_expectation([1, 1, 0, 1]) == 0
    assert state.pauli_expectation([1, 1, 0, 0]) == 0
    assert state.pauli_expectation([0, 0, 1, 1]) == 0

    # assert np.array_equal(
    #     state.density_matrix(),
    #     np.array(
    #         [
    #             [0.25, 0.25, 0.25, -0.25],
    #             [0.25, 0.25, 0.25, -0.25],
    #             [0.25, 0.25, 0.25, -0.25],
    #             [-0.25, -0.25, -0.25, 0.25],
    #         ]
    #     ),
    # )

    assert np.allclose(state.density_matrix(), state_stab.density_matrix())


def test_run8():
    circ = Circuit(2)
    circ.cx(0, 1)
    circ.s(1)
    circ.h(1)
    circ.cx(1, 0)
    state = run(circ)

    assert state.pauli_expectation([1, 0, 0, 1]) == 0
    assert state.pauli_expectation([0, 1, 1, 0]) == 0
    assert state.pauli_expectation([0, 0, 1, 0]) == 0
    assert state.pauli_expectation([0, 1, 0, 0]) == 0
    assert state.pauli_expectation([1, 0, 0, 0]) == 0
    assert state.pauli_expectation([1, 1, 1, 1]) == -1
    assert state.pauli_expectation([1, 0, 1, 1]) == 0
    assert state.pauli_expectation([1, 1, 0, 1]) == 0
    assert state.pauli_expectation([1, 1, 0, 0]) == 1
    assert state.pauli_expectation([0, 0, 1, 1]) == 1

    assert np.array_equal(
        state.density_matrix(),
        np.array(
            [
                [0.5, 0, 0, 0.5],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0.5, 0, 0, 0.5],
            ]
        ),
    )


def test_run9():
    circ = Circuit(2)
    circ.h(1)
    circ.s(0)
    circ.s(1)
    circ.cx(1, 0)
    state = run(circ)
    state_stab = run_stab(circ)

    assert state.pauli_expectation([1, 0, 0, 1]) == 0
    assert state.pauli_expectation([0, 1, 1, 0]) == 0
    assert state.pauli_expectation([0, 0, 1, 0]) == 0
    assert state.pauli_expectation([0, 1, 0, 0]) == 0
    assert state.pauli_expectation([1, 0, 0, 0]) == 0
    assert state.pauli_expectation([1, 1, 1, 1]) == 0
    assert state.pauli_expectation([1, 0, 1, 1]) == 1
    assert state.pauli_expectation([1, 1, 0, 1]) == 0
    assert state.pauli_expectation([1, 1, 0, 0]) == 1
    assert state.pauli_expectation([0, 0, 1, 1]) == 0

    assert np.allclose(state.density_matrix(), state_stab.density_matrix())


def test_run10():
    circ = Circuit(2)
    circ.s(1)
    circ.cx(1, 0)
    circ.s(0)
    circ.s(1)
    circ.h(1)
    circ.s(1)
    circ.cx(1, 0)
    circ.s(0)
    circ.h(0)
    state_stab = run_stab(circ)
    state = run(circ)

    assert state.pauli_expectation([0, 1, 1, 0]) == 1
    assert state.pauli_expectation([1, 0, 0, 1]) == -1
    assert state.pauli_expectation([0, 1, 0, 0]) == 0
    assert state.pauli_expectation([0, 0, 1, 0]) == 0
    assert state.pauli_expectation([1, 0, 0, 0]) == 0
    assert state.pauli_expectation([1, 1, 1, 1]) == -1
    assert state.pauli_expectation([1, 0, 1, 1]) == 0
    assert state.pauli_expectation([1, 1, 0, 1]) == 0
    assert state.pauli_expectation([1, 1, 0, 0]) == 0
    assert state.pauli_expectation([0, 0, 1, 1]) == 0

    assert np.allclose(state.density_matrix(), state_stab.density_matrix())
