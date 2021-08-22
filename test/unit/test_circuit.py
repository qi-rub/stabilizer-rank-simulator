import pytest
import os
import textwrap
from simulator.circuit import Circuit

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def test_read_from_txt():
    my_data_path = os.path.join(THIS_DIR, "testqasmdata.txt")
    circ = Circuit.from_qasm_file(my_data_path)
    assert circ.instructions[0] == ("h", 2)
    assert circ.instructions[1] == ("s", 3)
    assert circ.instructions[2] == ("cx", 3, 1)
    assert circ.instructions[3] == ("z", 2)
    assert circ.instructions[4] == ("x", 1)
    assert circ.instructions[5] == ("y", 0)


# We check that when a gate that is not Clifford is added to the qasm data, when inmporting it will yield an error.
def test_incorrect_gate():
    my_data_path = os.path.join(THIS_DIR, "testincorrectgate.txt")
    with pytest.raises(Exception):
        circ = Circuit.from_qasm_file(my_data_path)


# We test that when manual input gates are given, the simulator reads these correctly and creates the respective tuples.
def test_add_gates():
    circ = Circuit(3)
    circ.h(1)
    circ.cx(2, 0)
    circ.s(1)
    circ.z(2)
    circ.x(1)
    circ.y(1)
    assert circ.instructions == [
        ("h", 1),
        ("cx", 2, 0),
        ("s", 1),
        ("z", 2),
        ("x", 1),
        ("y", 1),
    ]

    qasm_str = """\
        OPENQASM 2.0;
        include "qelib1.inc";
        gate sy p0 {ry(pi/2) p0;}
        gate sw p0 {u3(pi/2,pi/2,0) p0;}
        qreg q[3];
        h q[1];
        cx q[2], q[0];
        s q[1];
        z q[2];
        x q[1];
        y q[1];
    """
    assert circ.qasm == textwrap.dedent(qasm_str.rstrip())


# We test that when an instruction with a greater number of qubits than the ones we have available is given, we get an error.
def test_not_enough_qubits():
    circ = Circuit(3)
    with pytest.raises(Exception):
        circ.h(6)
    with pytest.raises(Exception):
        circ.s(3)
    with pytest.raises(Exception):
        circ.cx(4, 1)
    with pytest.raises(Exception):
        circ.z(3)
    with pytest.raises(Exception):
        circ.x(11)
    with pytest.raises(Exception):
        circ.y(5)


def test_control_vs_target():
    circ = Circuit(1)
    with pytest.raises(Exception):
        circ.cx(0, 0)
