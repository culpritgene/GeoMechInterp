import networkx as nx
import numpy as np
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash


def dag_to_nx_graph(dag: list[tuple[str, list[str]]]) -> nx.Graph:
    G = nx.DiGraph()
    for node, controls in dag:
        G.add_node(node)
        # add node name as node attribute
        G.nodes[node]["name"] = node
        for control in controls:
            G.add_edge(control, node)
            G.nodes[control]["name"] = control
    return G


def dag_to_adjacency_matrix(dag: list[tuple[str, list[str]]]) -> np.ndarray:
    G = dag_to_nx_graph(dag)
    return nx.to_numpy_matrix(G)


def wl_hash(graph: nx.Graph, strip_node_labels=True, iterations=5) -> str:
    """
    Hash a NetworkX graph using the Weisfeiler-Lehman (WL) graph hashing algorithm.

    Parameters:
    - graph (nx.Graph): A NetworkX graph object.
    - iterations (int): The number of WL iterations to perform (default is 2).

    Returns:
    - str: A hash string representing the graph's isomorphism equivalence class.
    """
    if strip_node_labels:
        return weisfeiler_lehman_graph_hash(graph, iterations=iterations)
    return weisfeiler_lehman_graph_hash(graph, node_attr="name", iterations=iterations)


def dag_to_wl_hash(dag: list[tuple[str, list[str]]], strip_node_labels=True) -> str:
    """
    Hash a DisplayChain DAG using the Weisfeiler-Lehman (WL) graph hashing algorithm.
    """
    graph = dag_to_nx_graph(dag)
    return wl_hash(graph, strip_node_labels=strip_node_labels)


if __name__ == "__main__":
    dag1 = [("a", ["b", "c"]), ("b", ["d"]), ("c", ["d"]), ("d", [])]
    dag2 = [("a", ["b", "c"]), ("c", ["d"]), ("d", []), ("b", ["d"])]
    dag3 = [("a", ["b", "c"]), ("b", ["e"]), ("c", ["e"]), ("e", [])]
    assert dag_to_wl_hash(dag1) == dag_to_wl_hash(dag2)
    assert dag_to_wl_hash(dag1, False) == dag_to_wl_hash(dag2, False)
    assert dag_to_wl_hash(dag1, False) != dag_to_wl_hash(dag3, False)
