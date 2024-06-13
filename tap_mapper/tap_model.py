import gurobipy as gp
from networkx import all_pairs_dijkstra_path_length


def get_tap_model(
    gate_groups,
    hardware_graph,
    qerrors,
    cxerrors,
    crosstalk,
    lam=0.5,
    max_dist=3,
    log_level=0,
    heuristic=0.05,
    time_limit=30,
    mip_focus=0,
    gap=0.01,
    log_file=None,
    less_log_than_phys=True,
    bqp=True,
) -> gp.Model:
    """
    Builds a Gurobi model for the token allocation problem.


    Returns
    -------
    mdl : gurobipy.Model
        TAP model.

    """

    num_layers = len(gate_groups)
    if less_log_than_phys:
        logical_qubits = set([Qb for Group in gate_groups for Gate in Group for Qb in Gate])
    else:
        logical_qubits = hardware_graph.nodes
    physical_qubits = hardware_graph.nodes()
    pair_distances = [d for d in all_pairs_dijkstra_path_length(hardware_graph)]

    mdl = gp.Model()
    mdl.params.OutPutFlag = log_level
    mdl.params.Heuristics = heuristic
    mdl.params.TimeLimit = time_limit
    mdl.params.MIPFocus = mip_focus
    mdl.params.MIPGap = gap
    if log_file:
        mdl.Params.LogToConsole = 0
        mdl.Params.OutPutFlag = 1
        mdl.params.log_file = log_file

    w = {}
    for t in range(num_layers):
        for q in logical_qubits:
            for j in physical_qubits:
                w[t, q, j] = mdl.addVar(vtype="B", name=f"w_{t}_{q}_{j}")

    y = {}
    for t in range(num_layers):
        for p, q in gate_groups[t]:
            for i, j in hardware_graph.edges():
                y[t, p, q, i, j] = mdl.addVar(vtype="B", name=f"y_{t}_{p}_{q}_{i}_{j}")

    x = {}
    for t in range(num_layers - 1):
        for q in logical_qubits:
            for i in physical_qubits:
                for j in physical_qubits:
                    if pair_distances[i][1][j] <= max_dist:
                        x[t, q, i, j] = mdl.addVar(vtype="B", name=f"x_{t}_{q}_{i}_{j}")

    mdl.update()

    for t in range(num_layers):
        for q in logical_qubits:
            mdl.addConstr(
                sum(w[t, q, j] for j in physical_qubits) == 1,
                name=f"assignment_lqubits_{q}_at_{t}",
            )
    if less_log_than_phys:
        for t in range(1):
            for j in physical_qubits:
                mdl.addConstr(
                    sum(w[t, q, j] for q in logical_qubits) <= 1,
                    name=f"assignment_pqubits_{j}_at_{t}",
                )
        for t in range(1, num_layers):
            for j in physical_qubits:
                mdl.addConstr(
                    sum(w[t, q, j] for q in logical_qubits) == sum(w[t - 1, q, j] for q in logical_qubits),
                    name=f"assignment_pqubits_{j}_at_{t}",
                )
    else:
        for t in range(num_layers):
            for j in physical_qubits:
                mdl.addConstr(
                    sum(w[t, q, j] for q in logical_qubits) == 1,
                    name=f"assignment_pqubits_{j}_at_{t}",
                )

    for t in range(num_layers):
        for p, q in gate_groups[t]:
            mdl.addConstr(
                sum(y[t, p, q, i, j] for (i, j) in hardware_graph.edges()) == 1,
                name=f"implement_gate_{p}_{q}_{i}_at_{i}_{j}",
            )

    if bqp:
        for t in range(num_layers):
            for p, q in gate_groups[t]:
                for i in hardware_graph.nodes():
                    mdl.addConstr(
                        sum(y[t, p, q, i, j] for j in hardware_graph.neighbors(i)) == w[t, p, i],
                        name="bipartite QP",
                    )
                    mdl.addConstr(
                        sum(y[t, p, q, j, i] for j in hardware_graph.neighbors(i)) == w[t, q, i],
                        name="bipartite QP",
                    )

    else:
        for t in range(num_layers):
            for p, q in gate_groups[t]:
                for i, j in hardware_graph.edges():
                    mdl.addConstr(
                        y[t, p, q, i, j] <= w[t, p, i],
                        name="McCormickUB1_{p}_{q}_{i}_{j}_at_{t}",
                    )

                    mdl.addConstr(
                        y[t, p, q, i, j] <= w[t, q, j],
                        name="McCormickUB2_{p}_{q}_{i}_{j}_at_{t}",
                    )

    for t in range(num_layers - 1):
        for q in logical_qubits:
            for i in physical_qubits:
                mdl.addConstr(
                    w[t, q, i] == sum(x[t, q, i, j] for j in hardware_graph.nodes() if pair_distances[i][1][j] <= max_dist),
                    name=f"flow_out_{q}_{i}_at_{t}",
                )

    for t in range(1, num_layers):
        for q in logical_qubits:
            for i in physical_qubits:
                mdl.addConstr(
                    w[t, q, i] == sum(x[t - 1, q, j, i] for j in hardware_graph.nodes if pair_distances[i][1][j] <= max_dist),
                    name=f"flow_in_{q}_{i}_at_{t}",
                )

    c_swap = sum(
        [
            int(pair_distances[i][1][j]) * x[t, q, i, j]
            for i in physical_qubits
            for j in physical_qubits
            for t in range(num_layers - 1)
            for q in logical_qubits
            if pair_distances[i][1][j] <= max_dist
        ]
    )
    c_noise = (
        sum([num_layers * qerrors[i] * w[0, q, i] for q in logical_qubits for i in physical_qubits])
        + sum(sum(cxerrors[i, j] * y[t, p, q, i, j] for (p, q) in gate_groups[t] for (i, j) in hardware_graph.edges()) for t in range(num_layers))
        + sum(
            v
            * sum(w[0, q, i] for q in logical_qubits)
            * sum(sum(y[t, p, q, j, k] + y[t, p, q, k, j] for (p, q) in gate_groups[t]) for t in range(num_layers))
            for (i, (j, k)), v in crosstalk.items()
        )
    )

    mdl.setObjective((1 - lam) * c_swap + (lam) * c_noise, gp.GRB.MINIMIZE)

    mdl.update()

    mdl._w = w
    mdl._x = x
    mdl._y = y
    mdl._logical_qubits = logical_qubits

    return mdl
