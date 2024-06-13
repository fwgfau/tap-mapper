import numpy as np
import networkx as nx


def approx_ts(graph, initial_mapping, final_mapping):
    """
    Modified version of the token swapping approximation algorithm proposed in https://arxiv.org/abs/1602.05150.

    Parameters
    ----------
    graph : networkx.Graph
        Connectivity.
    initial_mapping : List
        Elements are tokens, indices are vertices.
        [1, 2, 0] means that vertex 0 holds token 1, vertex 1 holds token 2 and vertex 2 holds token 0.
    final_mapping : List
        Target mapping.

    Returns
    -------
    mapping_list : List[List]
        Sequence of mappings wich transform initial into final mapping.
    swap_list : List[Tuple]
        Sequence of swaps transforming consecutive mappings.

    """
    np.random.seed()

    mapping = list(initial_mapping)
    mapping_list = list()
    swap_list = list()
    mapping_list.append(initial_mapping)
    visited = []

    not_satisfied_vertices = [v for v in np.nonzero(np.array(initial_mapping) - np.array(final_mapping))[0]]
    while not_satisfied_vertices:
        preferred_starting_vertices = [v for v in not_satisfied_vertices if v not in visited]
        if preferred_starting_vertices:
            vertex = preferred_starting_vertices[int(np.random.randint(len(preferred_starting_vertices)))]
        else:
            vertex = not_satisfied_vertices[int(np.random.randint(len(not_satisfied_vertices)))]
        visited = [vertex]
        while 1:
            directed_neighbors = list()
            for j in graph.nodes:
                if j in graph.neighbors(vertex):

                    dist_is = distance(graph, mapping[vertex], mapping, final_mapping)
                    swapped_mapping = list(mapping)
                    swapped_mapping[vertex] = mapping[j]
                    swapped_mapping[j] = mapping[vertex]
                    dist_swapped = distance(graph, mapping[vertex], swapped_mapping, final_mapping)
                    if dist_is > dist_swapped:
                        directed_neighbors.append(j)
            if set(directed_neighbors) & set(visited):
                loops = [x for x in visited if x in directed_neighbors]
                vertex = loops[-1]
                chain = visited[visited.index(vertex) :]
                for i in range(len(chain) - 2, -1, -1):
                    swapped_mapping = list(mapping)
                    swapped_mapping[chain[i]] = mapping[chain[i + 1]]
                    swapped_mapping[chain[i + 1]] = mapping[chain[i]]
                    mapping = list(swapped_mapping)
                    mapping_list.append(mapping)
                break
            found_no_dead_end = 0
            for nb in directed_neighbors:
                for j in graph.nodes:
                    if j in graph.neighbors(nb):

                        dist_is = distance(graph, mapping[nb], mapping, final_mapping)
                        swapped_mapping = list(mapping)
                        swapped_mapping[nb] = mapping[j]
                        swapped_mapping[j] = mapping[nb]
                        dist_swapped = distance(graph, mapping[nb], swapped_mapping, final_mapping)
                        if dist_is > dist_swapped:
                            found_no_dead_end = 1
                            break
                if found_no_dead_end:
                    vertex = nb
                    visited.append(vertex)
                    break
            if not found_no_dead_end:
                vertex = directed_neighbors[0]
                swapped_mapping = list(mapping)
                swapped_mapping[vertex] = mapping[visited[-1]]
                swapped_mapping[visited[-1]] = mapping[vertex]
                mapping = list(swapped_mapping)
                mapping_list.append(mapping)
                break

        not_satisfied_vertices = [v for v in np.nonzero(np.array(mapping) - np.array(final_mapping))[0]]
    for i in range(len((mapping_list)) - 1):
        swap_list.append(np.nonzero(np.array(mapping_list[i]) - np.array(mapping_list[i + 1]))[0])
    return mapping_list, swap_list


def distance(graph, token, placement, target_placement):
    """
    Computes distance from token to its target vertex.

    Parameters
    ----------
    graph : netwrokx.Graph
        Connectivity.
    token : int
        token.
    placement : list
        Current placement.
    targetplacement : list
        Target placement.

    Returns
    -------
    int
        distance from token to its target.

    """
    is_vertex = placement.index(token)
    target_vertex = target_placement.index(token)
    return nx.shortest_path_length(graph, is_vertex, target_vertex)
