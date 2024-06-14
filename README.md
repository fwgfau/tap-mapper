# TAP-mapper
An algorithm for qubit routing by swap insertion, implemented as a Qiskit transpiler stage.
## Description
The allgorithm decomposes the routing problem into two sub-problems, of which the first one is solved by integer programming.
The second subproblem is solved by an efficient approximation algorithm, see https://arxiv.org/abs/1602.05150.
For a detailed description of the decomposition algorithm, see https://arxiv.org/abs/2206.01294.
The algorithm can consider single qubit errors, two-qubit errors as well as crosstalk errors.
For a detailed description of the error incorporation, see https://arxiv.org/abs/2401.06423.

A Gurobi license is required.
A free trial license will be installed automatically.
Gurobi offers free full licences for academic users (https://www.gurobi.com/academia/academic-program-and-licenses/).

## Installation
1. Clone this repo into path/to/tap-mapper.
2. In your local environment, run
```bash
pip install path/to/tap-mapper
```
## Usage

As a transformation pass:

```python
from qiskit.circuit.library import QuantumVolume
from qiskit.transpiler import CouplingMap
from qiskit.converters import circuit_to_dag, dag_to_circuit
from tap_mapper.tap_pass_grb import TAPMapping


cm = CouplingMap([[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]])
cm.make_symmetric()

d = 5
nq = 5
qc = QuantumVolume(nq, d)
qc = qc.decompose()


tap_pass = TAPMapping(cm)
dag = circuit_to_dag(qc)
dagt = tap_pass.run(dag)
qct = dag_to_circuit(dagt)
n_swaps_tap = qct.count_ops().get("swap", 0)

print(n_swaps_tap)
```
As a routing stage plugin:

```python
from qiskit import transpile

qct = transpile(qc, coupling_map=cm, routing_method="tap", optimization_level=0)
n_swaps_tap = qct.count_ops().get("swap", 0)

print(n_swaps_tap)
```

## Citation

If you use this transpiler in your research, please cite
https://arxiv.org/abs/2401.06423.
