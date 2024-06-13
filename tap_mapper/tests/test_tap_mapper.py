import unittest
from qiskit.circuit.library import QuantumVolume
from qiskit.circuit import QuantumCircuit
import numpy as np
from qiskit.converters import circuit_to_dag, dag_to_circuit
import networkx as nx
from qiskit_aer.backends import UnitarySimulator
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes.utils import CheckMap
from qiskit.transpiler.passes.layout.trivial_layout import TrivialLayout
from tap_mapper.tap_pass_grb import TAPMapping


class TestTAPMapping(unittest.TestCase):

    def setUp(self):
        els = [
            list(nx.cycle_graph(5).edges()),
            list(nx.path_graph(5).edges()),
            list(nx.complete_graph(5).edges()),
            list(nx.convert_node_labels_to_integers(nx.grid_2d_graph(2, 3)).edges()),
            list(nx.random_regular_graph(3, 6, seed=1).edges()),
        ]

        cms = []
        qcs = []
        dagctaps = []
        qcts = []
        tap_passes = []
        for k, el in enumerate(els):
            cm = CouplingMap(el)
            cms.append(cm)

            seed = 222 + k
            d = 5
            nq = 5
            qc = QuantumVolume(nq, d, seed=seed)
            qc = qc.decompose()
            qc.measure_all()
            qcs.append(qc)

            tap_pass = TAPMapping(cm)

            dag = circuit_to_dag(qc)

            dagctap = tap_pass.run(dag, time_limit=5)
            dagctaps.append(dagctap)
            qct = dag_to_circuit(dagctap)
            qcts.append(qct)

            tap_passes.append(tap_pass)
        self.qcts = qcts
        self.qcs = qcs
        self.cms = cms
        self.dagctaps = dagctaps
        self.tap_passes = tap_passes

    def test_connectivity(self):
        for k, cm in enumerate(self.cms):
            with self.subTest(i=k):
                checkmap = CheckMap(cm)
                checkmap.run(self.dagctaps[k])
                self.assertTrue(checkmap.property_set["is_swap_mapped"])

    def test_unitary(self):
        def is_permutation_matrix(u):
            for irow in range(u.shape[0]):
                unique, counts = np.unique(np.isclose(u[irow, :], 1), return_counts=True)
                occ = dict(zip(unique, counts))
                if occ[True] != 1:
                    return False
                unique, counts = np.unique(np.isclose(u[irow, :], 0), return_counts=True)
                occ = dict(zip(unique, counts))
                if occ[True] != u.shape[0] - 1:
                    return False
            for icol in range(u.shape[1]):
                unique, counts = np.unique(np.isclose(u[:, icol], 1), return_counts=True)
                occ = dict(zip(unique, counts))
                if occ[True] != 1:
                    return False
                unique, counts = np.unique(np.isclose(u[:, icol], 0), return_counts=True)
                occ = dict(zip(unique, counts))
                if occ[True] != u.shape[0] - 1:
                    return False
            return True

        backend = UnitarySimulator()
        for k, cm in enumerate(self.cms):
            with self.subTest(i=k):
                qct = self.qcts[k]
                qct.remove_final_measurements()

                qc = QuantumCircuit(cm.size())
                qc = qc.compose(self.qcs[k])
                qc.remove_final_measurements()

                qcc = qct.compose(
                    qc.inverse(),
                    qubits=[self.tap_passes[k].placement[-1].index(q) for q in range(qc.num_qubits)],
                )
                res = backend.run(qcc).result()
                u_out = res.get_unitary(qcc, 4).to_matrix()

                self.assertTrue(is_permutation_matrix(u_out))

    def test_cx_error(self):
        cm = self.cms[0]
        d = 1
        nqubits = 4
        qc = QuantumVolume(nqubits, d, seed=1)
        qc = qc.decompose()
        cxerrors = {(0, 1): 1}
        tap_pass = TAPMapping(cm, cxerrors=cxerrors)
        dag = circuit_to_dag(qc)
        dagctap = tap_pass.run(dag, time_limit=20, lam=0.9)
        qct = dag_to_circuit(dagctap)
        cxs = []
        for d in qct.data:
            if len(d.qubits) == 2:
                cxs.append([qct.find_bit(qb).index for qb in d.qubits])

        self.assertFalse(([0, 1] in cxs) or ([1, 0] in cxs))

        cxerrors = {(1, 2): 1}
        tap_pass = TAPMapping(cm, cxerrors=cxerrors)
        dagctap = tap_pass.run(dag, time_limit=20, lam=0.9)
        qct = dag_to_circuit(dagctap)
        cxs = []
        for d in qct.data:
            if len(d.qubits) == 2:
                cxs.append([qct.find_bit(qb).index for qb in d.qubits])

        self.assertFalse(([1, 2] in cxs) or ([2, 1] in cxs))

    def test_sq_error(self):
        cm = self.cms[0]
        d = 1
        nqubits = 4
        qc = QuantumVolume(nqubits, d, seed=1)
        qc = qc.decompose()
        qerrors = [1] + [0] * (cm.size() - 1)
        tap_pass = TAPMapping(cm, qerrors=qerrors)
        dag = circuit_to_dag(qc)
        dagctap = tap_pass.run(dag, time_limit=20, lam=0.9)
        qct = dag_to_circuit(dagctap)
        qubits = []
        nq = 0
        ind = 0
        while nq < nqubits:
            d = qct.data[ind]
            for qb in d.qubits:
                if qct.find_bit(qb).index not in qubits:
                    qubits.append(qct.find_bit(qb).index)
                    nq += 1
            ind += 1

        self.assertFalse(0 in qubits)

        qerrors = [0] * (cm.size() - 1) + [1]
        tap_pass = TAPMapping(cm, qerrors=qerrors)
        dag = circuit_to_dag(qc)
        dagctap = tap_pass.run(dag, time_limit=20, lam=0.9)
        qct = dag_to_circuit(dagctap)
        qubits = []
        nq = 0
        ind = 0
        while nq < nqubits:
            d = qct.data[ind]
            for qb in d.qubits:
                if qct.find_bit(qb).index not in qubits:
                    qubits.append(qct.find_bit(qb).index)
                    nq += 1
            ind += 1

        self.assertFalse((cm.size() - 1) in qubits)

    def test_xt_error(self):
        cm = self.cms[0]
        d = 3
        nqubits = 4
        qc = QuantumVolume(nqubits, d, seed=1)
        qc = qc.decompose()
        crosstalk = {(2, (0, 1)): 1}
        tap_pass = TAPMapping(cm, crosstalk=crosstalk)
        dag = circuit_to_dag(qc)
        dagctap = tap_pass.run(dag, time_limit=20, lam=0.9)
        qct = dag_to_circuit(dagctap)
        cxs = []
        for d in qct.data:
            if len(d.qubits) == 2:
                cxs.append([qct.find_bit(qb).index for qb in d.qubits])
        qubits = []
        nq = 0
        ind = 0
        while nq < nqubits:
            d = qct.data[ind]
            for qb in d.qubits:
                if qct.find_bit(qb).index not in qubits:
                    qubits.append(qct.find_bit(qb).index)
                    nq += 1
            ind += 1

        self.assertFalse(2 in qubits and (([0, 1] in cxs) or ([1, 0] in cxs)))


if __name__ == "__main__":

    unittest.main()
