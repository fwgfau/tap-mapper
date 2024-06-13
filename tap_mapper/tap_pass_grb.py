import copy
import networkx as nx
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import TransformationPass
from qiskit.transpiler.layout import Layout
from tap_mapper.tap_model import get_tap_model
from tap_mapper.ts_algos import approx_ts


class TAPMapping(TransformationPass):
    """Maps a circuit on a given coupling map by inserting swaps."""

    def __init__(self, cm, qerrors=None, cxerrors=None, crosstalk=None):
        """

        Parameters
        ----------
        cm : qiskit.transpiler.CouplingMap or List
              Coupling map of hardware. Nodes need to be consecutive, starting with 0.
        qerrors : List, optional
              Specifies relative error weights for each hardware qubit in the coupling map.
              The required form is [error_i for i in cm.nodes].
              The default is None.
        cxerrors: Dict, optional
              relative cx error weight for hardware graph edges,
              in the form {(i,j):error_ij for ij in cm.edges}.
              The default is None.
        crosstalk : Dict, optional
                Relative crosstalk strength.
                Dictonary of the form {(i,(j,k)):v}, where i is in cm.nodes and (j,k) is in cm.edges
                and v is the absolute increase in single qubit error weight of qubit i when driving (j,k).
                The default is None.
        """
        super().__init__()

        if isinstance(cm, list):
            edgelist = cm
        else:
            edgelist = list(cm.get_edges())
        hardware_graph = nx.DiGraph((nx.Graph(edgelist)))
        self.hardware_graph = hardware_graph

        pair_distances = list(nx.all_pairs_dijkstra_path_length(hardware_graph))
        self.pair_distances = pair_distances

        if qerrors is None:
            qerrors = [0 for i in hardware_graph.nodes]
        self.qerrors = qerrors

        cxerrors_padded = {(i, j): 0 for (i, j) in hardware_graph.edges()}
        if cxerrors:
            cxerrors_padded.update((k, v) for k, v in cxerrors.items())
            cxerrors_padded.update(((k[1], k[0]), v) for k, v in cxerrors.items())
        self.cxerrors = cxerrors_padded

        self.crosstalk = crosstalk or {}
        self.placement = None

    def run(
        self,
        dag,
        less_log_than_phys=True,
        time_limit=100,
        gap=0.01,
        log_level=0,
        max_dist=3,
        mip_focus=1,
        bqp=True,
        heuristic=0.05,
        lam=0.5,
        n_trials_swap=1,
        log_file=None,
    ) -> DAGCircuit:
        """

        Parameters
        ----------
        dag : DAG
            DAG representaion of circuit to be routed.
        less_log_than_phys : Boolean, optional
            If False, inactive qubits are added to the dag to match the physcial qubit number.
            False heavily increases runtime but might deliver better results in some special cases.
            The default is True.
        time_limit : float, optional
            Time limit for mip solver in seconds. The default is 100s.
        nLookAheadLayers : int, optional
            Number of layers to be routed at once.
            Controls quality runtime tradeoff.
            Best results are achieved if all layers are routed in a single step.
            The default is 100.
        gap : foat, optional
            Relative gap for mip solver.
            The default is 0.1.
        log_level : int, optional
            If set to 1, output of mip solver is printed to console.
            The default is 0.
        max_dist : int, optional
            Limits the distance a logical qubit may move on the hardware topology
            between subsequent layers through swaps.
            Controls quality runtime tradeoff.
            Good results are achieved even for quite small values.
            The default is 3.
        mip_focus : int, optional
            Controls mip solver focus.
            If 0, less focus is put on finding good solutions quickly but rather on proofing optimality.
            The default is 1.
        bqp : bool, optional
            If False, a larger but equivalent mip model is constructed.
            False increases runtime by a factor of 2 on average.
            The default is True.
        log_file : String, optional
            Log filepath for mip solver.

        Returns
        -------
        DAGCircuit
            mapped and routed circuit.

        """

        gate_groups, two_qubit_gate_layers = self._group_gates(dag)

        if not gate_groups:
            print("no 2q gates")
            return dag

        n_layers = len(gate_groups)

        mdl = get_tap_model(
            gate_groups,
            self.hardware_graph,
            self.qerrors,
            self.cxerrors,
            self.crosstalk,
            lam,
            max_dist,
            log_level,
            heuristic,
            time_limit,
            mip_focus,
            gap,
            log_file,
            less_log_than_phys,
            bqp,
        )

        mdl.optimize()

        if mdl.SolCount == 0:
            print("No feasible TAP solution found. Aborting routing.")
            return dag

        placement = [[] for t in range(n_layers)]
        for t in range(n_layers):
            placement[t] = [-1 for pq in self.hardware_graph.nodes()]
            for q in mdl._logical_qubits:
                for physical_qubit in self.hardware_graph.nodes:
                    if mdl._w[(t, q, physical_qubit)].X >= 0.5:
                        placement[t][physical_qubit] = q
            for nq in range(len(mdl._logical_qubits), self.hardware_graph.number_of_nodes()):
                for physical_qubit in self.hardware_graph.nodes():
                    if placement[t][physical_qubit] == -1:
                        placement[t][physical_qubit] = nq
                        break

        self.placement = placement

        canonical_register = QuantumRegister(self.hardware_graph.number_of_nodes(), "q")
        mapped_dag = self._create_empty_dag(dag, canonical_register)

        for t in range(n_layers - 1):
            initial_placement = placement[t]
            finall_placement = placement[t + 1]
            swap_count_list = []
            for _ in range(n_trials_swap):
                ml, sl = approx_ts(self.hardware_graph, initial_placement, finall_placement)
                swap_count_list.append(((ml, sl), len(sl)))
            minind = min(
                (k for k in range(n_trials_swap)),
                key=lambda x, scl=swap_count_list: scl[x][1],
            )
            (ml, sl), _ = swap_count_list[minind]

            if t == 0:
                first_layer_ind = 0
            else:
                first_layer_ind = two_qubit_gate_layers[t - 1] + 1

            last_layer_index = two_qubit_gate_layers[t]
            for layer in [l for k, l in enumerate(dag.layers()) if first_layer_ind <= k <= last_layer_index]:
                for node in layer["graph"].op_nodes(include_directives=True):
                    mapped_dag.apply_operation_back(
                        op=copy.deepcopy(node.op),
                        qargs=[canonical_register[placement[t].index(dag.find_bit(q).index)] for q in node.qargs],
                        cargs=node.cargs,
                    )

            for sw in sl:
                mapped_dag.apply_operation_back(
                    op=SwapGate(),
                    qargs=[
                        canonical_register[sw[0]],
                        canonical_register[sw[1]],
                    ],
                )

        t = n_layers - 1
        if t == 0:
            first_layer_ind = 0
        else:
            first_layer_ind = two_qubit_gate_layers[t - 1] + 1
        last_layer_index = len(list(dag.layers()))
        for layer in [l for k, l in enumerate(dag.layers()) if first_layer_ind <= k <= last_layer_index]:
            for node in layer["graph"].op_nodes(include_directives=True):
                mapped_dag.apply_operation_back(
                    op=copy.deepcopy(node.op),
                    qargs=[canonical_register[placement[t].index(dag.find_bit(q).index)] for q in node.qargs],
                    cargs=node.cargs,
                )

        return mapped_dag

    @staticmethod
    def _create_empty_dag(source_dag, canonical_qreg):
        target_dag = DAGCircuit()
        target_dag.name = source_dag.name
        target_dag._global_phase = source_dag._global_phase
        target_dag.metadata = source_dag.metadata

        target_dag.add_qreg(canonical_qreg)
        for creg in source_dag.cregs.values():
            target_dag.add_creg(creg)

        return target_dag

    @staticmethod
    def _group_gates(dag):
        two_qubit_gate_layers = []
        gate_groups = []
        for k, lay in enumerate(dag.layers()):
            group = []
            for node in lay["graph"].two_qubit_ops():
                group.append((dag.find_bit(node.qargs[0]).index, dag.find_bit(node.qargs[1]).index))
            if group:
                two_qubit_gate_layers.append(k)
                gate_groups.append(group)

        return gate_groups, two_qubit_gate_layers
