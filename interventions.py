"""the four network surgeries + the control condition.

each function takes a graph + agents dict, mutates them in place,
and returns a log dict describing what it did.

theory behind each one:
- bridge_removal: granovetter — bridges are the weak ties connecting communities,
  cut them and you slow cross-community spread
- hub_immunization: epidemic theory — high-degree nodes drive R0, convert the
  biggest hub into a fact-checker and watch the cascade slow down
- community_sealing: modularity — just cut EVERY inter-community edge, more
  aggressive version of bridge removal basically
- random_edge_removal: negative control!! same number of edges removed as
  bridge_removal but chosen randomly. if bridge_removal doesn't beat this
  then the whole paper falls apart so fingers crossed
"""

import networkx as nx
import random


def no_intervention(graph, agents, **kwargs):
    return {"name": "no_intervention", "details": "control condition"}


def bridge_removal(graph, agents, fraction=0.1, **kwargs):
    """remove top-k edges by edge betweenness centrality.
    these are the bridges between communities — granovetter's weak ties.
    """
    eb = nx.edge_betweenness_centrality(graph)
    sorted_edges = sorted(eb.items(), key=lambda x: -x[1])
    k = max(1, int(fraction * graph.number_of_edges()))
    removed = []
    for (u, v), _ in sorted_edges[:k]:
        if graph.has_edge(u, v):
            graph.remove_edge(u, v)
            removed.append((u, v))
    return {"name": "bridge_removal", "removed_edges": removed,
            "n_removed": len(removed)}


def hub_immunization(graph, agents, k=1, **kwargs):
    """convert the highest-degree node(s) into fact-checkers.
    changes their persona + resets their belief to 0.
    topology stays the same — they're still connected, just now actively
    pushing back on the rumor instead of spreading it.
    """
    sorted_nodes = sorted(graph.nodes(), key=lambda n: -graph.degree(n))
    immunized = []
    for node_id in sorted_nodes[:k]:
        agents[node_id].persona = {
            "name": agents[node_id].persona["name"],
            "description": (
                "a careful, calm fact-checker who actively debunks unverified "
                "claims when they hear them. They politely but firmly point out "
                "that claims need evidence and explain why the rumor is unreliable."
            ),
        }
        agents[node_id].belief = 0.0
        immunized.append(node_id)
    return {"name": "hub_immunization", "immunized_nodes": immunized}


def random_edge_removal(graph, agents, fraction=0.1, seed=0, **kwargs):
    """NEGATIVE CONTROL. same number of edges as bridge_removal, just random ones.
    if bridge removal doesn't outperform this, we have a problem.
    seeded so results are reproducible.
    """
    rng = random.Random(seed)
    edges = list(graph.edges())
    k = max(1, int(fraction * len(edges)))
    to_remove = rng.sample(edges, min(k, len(edges)))
    for u, v in to_remove:
        graph.remove_edge(u, v)
    return {"name": "random_edge_removal", "removed_edges": to_remove,
            "n_removed": len(to_remove)}


def community_sealing(graph, agents, **kwargs):
    """detect communities and cut ALL edges between them.
    more aggressive than bridge removal — removes every inter-community edge,
    not just the high-betweenness ones. basically turns the graph into islands.
    """
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(graph))
    except Exception as e:
        return {"name": "community_sealing", "error": str(e)}

    node_to_comm = {}
    for i, comm in enumerate(communities):
        for n in comm:
            node_to_comm[n] = i

    removed = []
    for u, v in list(graph.edges()):
        if node_to_comm.get(u) != node_to_comm.get(v):
            graph.remove_edge(u, v)
            removed.append((u, v))
    return {"name": "community_sealing", "removed_edges": removed,
            "n_removed": len(removed), "n_communities": len(communities)}


# pass --intervention <key> to run_experiment.py to use one of these
INTERVENTIONS = {
    "none": no_intervention,
    "bridge_removal": bridge_removal,
    "hub_immunization": hub_immunization,
    "random_edge_removal": random_edge_removal,
    "community_sealing": community_sealing,
}
