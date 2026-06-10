
"""export network snapshots from existing result files for gephi / poster.

standalone script, doesn't touch the simulation at all. reads the json
files you already have, rebuilds the graph, slaps belief values on the
nodes, exports gexf + png at four timesteps.

timesteps:
  t=0   clean graph before anything spreads
  t=4   just before surgery — cascade is growing
  t=5   just after surgery — you can see the cut edges
  t=14  final state

surgery gets replayed from the intervention_log so t=5 and t=14 actually
show the correct post-surgery topology (not just the original graph).

usage:
    python export_networks.py                      # all runs in results/
    python export_networks.py results/run.json     # one specific file
    python export_networks.py --seed 0             # only seed-0 runs
    python export_networks.py --gexf-only          # skip png rendering
    python export_networks.py --png-only           # skip gexf

needs networkx and matplotlib (already in requirements).
graph_builder.py has to be importable (same folder).
"""

import json
import sys
import os
import glob
import argparse

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from graph_builder import build_graph
import config


SNAPSHOT_TIMESTEPS = [0, 4, 5, 14]

# node colors - keep these consistent with the poster
COLOR_SEED       = "#A32D2D"   # where the rumor started
COLOR_BELIEVER   = "#E8A838"   # belief >= 0.5
COLOR_PARTIAL    = "#FAC775"   # 0.3 <= belief < 0.5, believes a bit
COLOR_SUSCEPT    = "#D3D1C7"   # hasn't really bought in
COLOR_IMMUNIZED  = "#1B7F8E"   # fact-checker, converted by hub_immunization
COLOR_CUT_EDGE   = "#A32D2D"   # edges removed by surgery (unused rn but keeping it)


def node_color(node_id, belief, seed_node, immunized_set):
    if node_id in immunized_set:
        return COLOR_IMMUNIZED
    if node_id == seed_node:
        return COLOR_SEED
    if belief >= 0.5:
        return COLOR_BELIEVER
    if belief >= 0.3:
        return COLOR_PARTIAL
    return COLOR_SUSCEPT


def rebuild_graph_for_run(run):
    """rebuild the exact graph from its seed. deterministic so it's always right."""
    seed = run["condition"]["seed"]
    n = run["pre_graph"]["n_nodes"]
    return build_graph(n=n, seed=seed)


def apply_surgery(graph, intervention_log):
    """replay surgery on the graph. returns (graph, set of immunized nodes).

    edge-cutting interventions: remove the logged edges.
    hub immunization: topology unchanged, just track which nodes got converted.
    """
    immunized = set()
    if not intervention_log:
        return graph, immunized

    if "removed_edges" in intervention_log:
        for edge in intervention_log["removed_edges"]:
            u, v = edge[0], edge[1]   # json round-trip gives lists not tuples
            if graph.has_edge(u, v):
                graph.remove_edge(u, v)

    if "immunized_nodes" in intervention_log:
        immunized = set(intervention_log["immunized_nodes"])

    return graph, immunized


def get_beliefs_at(run, t):
    """get {node_id: belief} dict from snapshot at timestep t.

    json keys come back as strings, convert to int.
    clamps t if it's past the end.
    """
    snapshots = run["snapshots"]
    if not snapshots:
        return {}
    t = min(t, len(snapshots) - 1)
    raw = snapshots[t]["beliefs"]
    return {int(k): v for k, v in raw.items()}


def export_run(run, out_dir, do_gexf=True, do_png=True):
    """export all four snapshots for one run."""
    cond = run["condition"]
    intervention = cond["intervention"]
    rumor = cond["rumor_label"]
    seed = cond["seed"]
    seed_node = cond["seed_node"]
    intervention_log = run.get("intervention_log")

    tag = f"{intervention}_{rumor}_seed{seed}"

    for t in SNAPSHOT_TIMESTEPS:
        graph = rebuild_graph_for_run(run)   # fresh graph each time

        # surgery happened at t=5 so snapshots at t>=5 need the cuts applied
        immunized = set()
        if t >= config.INTERVENTION_TIMESTEP:
            graph, immunized = apply_surgery(graph, intervention_log)

        beliefs = get_beliefs_at(run, t)

        # attach attributes so gephi can color/size by them
        for node in graph.nodes():
            b = beliefs.get(node, 0.0)
            graph.nodes[node]["belief"] = float(b)
            graph.nodes[node]["is_seed"] = (node == seed_node)
            graph.nodes[node]["is_immunized"] = (node in immunized)
            if node in immunized:
                state = "immunized"
            elif node == seed_node:
                state = "seed"
            elif b >= 0.5:
                state = "believer"
            elif b >= 0.3:
                state = "partial"
            else:
                state = "susceptible"
            graph.nodes[node]["state"] = state

        base = os.path.join(out_dir, f"{tag}_t{t:02d}")

        if do_gexf:
            nx.write_gexf(graph, base + ".gexf")

        if do_png:
            render_png(graph, seed_node, immunized, base + ".png",
                       title=f"{intervention} · {rumor} · seed {seed} · t={t}")

    return tag


def render_png(graph, seed_node, immunized, out_path, title):
    fig, ax = plt.subplots(figsize=(6, 6))

    # fixed seed so the same graph looks the same across all four timesteps
    pos = nx.spring_layout(graph, seed=42, k=0.6)

    colors = []
    sizes = []
    for node in graph.nodes():
        b = graph.nodes[node].get("belief", 0.0)
        colors.append(node_color(node, b, seed_node, immunized))
        if node == seed_node or node in immunized:
            sizes.append(550)   # slightly bigger for important nodes
        else:
            sizes.append(380)

    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#9A998F",
                           width=1.0, alpha=0.6)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=colors,
                           node_size=sizes, edgecolors="#2C2C2A",
                           linewidths=0.8)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8,
                            font_color="#2C2C2A")

    ax.set_title(title, fontsize=11, pad=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_runs(path_arg):
    """load from a specific file or glob the whole results/ folder."""
    if path_arg and path_arg.endswith(".json"):
        with open(path_arg) as f:
            return json.load(f), [path_arg]

    folder = path_arg if path_arg else "results"
    files = sorted(glob.glob(os.path.join(folder, "results_*.json")))
    if not files:
        print(f"No results files found in {folder}/")
        sys.exit(1)

    runs = []
    for fp in files:
        with open(fp) as f:
            runs.extend(json.load(f))
    return runs, files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=None,
                        help="A results .json file, or a folder (default: results/)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Only export runs with this seed")
    parser.add_argument("--gexf-only", action="store_true",
                        help="Export GEXF files only, skip PNGs")
    parser.add_argument("--png-only", action="store_true",
                        help="Render PNGs only, skip GEXF")
    args = parser.parse_args()

    do_gexf = not args.png_only
    do_png = not args.gexf_only

    runs, files = load_runs(args.path)
    print(f"Loaded {len(runs)} runs from {len(files)} file(s)")

    if args.seed is not None:
        runs = [r for r in runs if r["condition"]["seed"] == args.seed]
        print(f"Filtered to seed={args.seed}: {len(runs)} runs")

    out_dir = "networks"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Exporting snapshots at t={SNAPSHOT_TIMESTEPS} "
          f"({'GEXF' if do_gexf else ''}{' + ' if do_gexf and do_png else ''}"
          f"{'PNG' if do_png else ''})...\n")

    for run in runs:
        tag = export_run(run, out_dir, do_gexf=do_gexf, do_png=do_png)
        print(f"  {tag}: 4 snapshots exported")

    n_files = len(runs) * len(SNAPSHOT_TIMESTEPS) * (do_gexf + do_png)
    print(f"\nDone. ~{n_files} files written to {out_dir}/")
    if do_gexf:
        print("  GEXF: open in Gephi, apply a layout, color nodes by 'state' or 'belief'")
    if do_png:
        print("  PNG: for the poster, use t04 + t05 of the same run as your before/after pair")


if __name__ == "__main__":
    main()
