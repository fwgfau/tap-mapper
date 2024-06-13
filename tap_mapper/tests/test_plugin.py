import unittest

from qiskit import QuantumCircuit
from qiskit.compiler.transpiler import transpile
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.preset_passmanagers.plugin import list_stage_plugins


class TestBIPMapping(unittest.TestCase):
    """Tests the BIPMapping plugin."""

    def test_plugin_exists(self):
        """Test bip plugin is installed."""
        self.assertIn("tap", list_stage_plugins("routing"))

    def test_plugin_usage(self):

        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])

        qc = QuantumCircuit(4)
        qc.cx(1, 0)
        qc.cx(1, 2)
        qc.cx(1, 3)

        qct = transpile(qc, coupling_map=coupling, routing_method="tap", optimization_level=0)
        self.assertTrue(qct.count_ops().get("swap", 0) > 0)


if __name__ == "__main__":

    unittest.main()
