from simulator.counter import reorder_qiskit_counter


def test_reorder():
    num_qubits = 2
    c = {"00": 2, "11": 4, "01": 20, "10": 83}
    ordered_c = reorder_qiskit_counter(num_qubits, c)

    expected_c = {"00": 2, "01": 83, "10": 20, "11": 4}
    assert ordered_c == expected_c
