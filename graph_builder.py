"""builds the network topologies.

IMPORTANT: build_graph is seeded. same seed = same graph every time.
this is the whole reason the `none` control and an intervention run are
comparable — at seed 0 they're operating on the exact same network, so
the only thing that differs between conditions is the surgery itself.
mess with the seed and you're comparing different graphs, which is wrong.
"""

import networkx as nx
import config


def build_graph(n: int = None, graph_type: str = None, seed: int = 0) -> nx.Graph:
    n = n or config.N_NODES
    graph_type = graph_type or config.GRAPH_TYPE

    if graph_type == "watts_strogatz":
        g = nx.watts_strogatz_graph(n, k=config.WS_K, p=config.WS_P, seed=seed)
    elif graph_type == "barabasi_albert":
        g = nx.barabasi_albert_graph(n, m=config.BA_M, seed=seed)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    # WS occasionally generates disconnected components with low p.
    # just stitch them together with a single edge — not realistic but
    # an isolated node that never hears anything is more annoying to deal with
    if not nx.is_connected(g):
        components = list(nx.connected_components(g))
        for i in range(len(components) - 1):
            u = next(iter(components[i]))
            v = next(iter(components[i + 1]))
            g.add_edge(u, v)

    return g


def graph_summary(g: nx.Graph) -> dict:
    """structural snapshot — gets logged with each experiment run.
    useful for sanity checking that pre/post surgery graphs look right.
    """
    summary = {
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "n_components": nx.number_connected_components(g),
    }
    if g.number_of_nodes() > 0:
        summary["avg_degree"] = sum(dict(g.degree()).values()) / g.number_of_nodes()
    if nx.is_connected(g):
        # these are slow on big graphs but n=30 so it's fine
        summary["avg_clustering"] = nx.average_clustering(g)
        summary["avg_path_length"] = nx.average_shortest_path_length(g)
    return summary
