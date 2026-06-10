"""run ONE intervention at a time. each invocation does one surgery across
all rumors + seeds, writes its own result file.

usage:
    python run_experiment.py --intervention none
    python run_experiment.py --intervention hub_immunization
    python run_experiment.py --intervention bridge_removal
    python run_experiment.py --intervention community_sealing
    python run_experiment.py --intervention random_edge_removal

each writes results/results_<intervention>_<timestamp>.json

run `none` once and it becomes the shared control for everything else.
you can run these on different days or split across machines — completed
runs are never lost or overwritten.

optional flags:
    --mvp        small pilot params (N=10, 1 rumor, 3 seeds) for sanity checks
    --rumors     comma-separated subset, e.g. --rumors mundane,consequential
"""

import argparse
import json
import os
import time

from graph_builder import build_graph, graph_summary
from personas import assign_personas
from agent import Agent
from simulation import run_simulation
from metrics import compute_metrics
from interventions import INTERVENTIONS
import config


RUMORS = {
    "mundane": "The new café on Atatürk Street is using old, recycled coffee beans "
               "and that's why so many customers have been getting sick.",
    "consequential": "The deputy mayor has been quietly approving construction permits "
                     "for his cousin's company in exchange for a share of the profits.",
    "unfalsifiable": "There's a private group of wealthy families in the city who "
                     "secretly decide which businesses get to succeed and which fail.",
}


def run_single_condition(intervention_name, rumor_label, rumor_text,
                         seed, verbose=False):
    """run one (intervention, rumor, seed) combination."""
    t0 = time.time()

    # seeded graph + personas: same seed = same network + same persona placement
    # so `none` control and intervention are directly comparable
    graph = build_graph(seed=seed)
    personas = assign_personas(config.N_NODES, seed=seed)
    agents = {i: Agent(node_id=i, persona=personas[i])
              for i in range(config.N_NODES)}

    # seed node = highest degree node (most connected = most realistic patient zero)
    seed_node = max(graph.nodes(), key=lambda n: graph.degree(n))

    intervention_fn = INTERVENTIONS[intervention_name]
    # random_edge_removal needs the seed for reproducibility
    intervention_kwargs = {"seed": seed} if intervention_name == "random_edge_removal" else {}

    pre_graph = graph_summary(graph)

    trace = run_simulation(
        graph=graph,
        agents=agents,
        rumor=rumor_text,
        seed_node=seed_node,
        intervention_fn=intervention_fn,
        intervention_kwargs=intervention_kwargs,
        verbose=verbose,
    )

    post_graph = graph_summary(graph) if graph.number_of_edges() > 0 else {}
    metrics = compute_metrics(trace, intervention_timestep=config.INTERVENTION_TIMESTEP)
    elapsed = time.time() - t0

    return {
        "condition": {
            "intervention": intervention_name,
            "rumor_label": rumor_label,
            "seed": seed,
            "seed_node": seed_node,
        },
        "pre_graph": pre_graph,
        "post_graph": post_graph,
        "metrics": metrics,
        "intervention_log": trace["intervention_log"],
        "belief_calls_skipped": trace.get("belief_calls_skipped", 0),
        "messages": trace["messages"],
        "snapshots": trace["snapshots"],
        "elapsed_sec": elapsed,
    }


def run_experiment(intervention_name, mvp=False, rumor_subset=None):
    if intervention_name not in INTERVENTIONS:
        raise ValueError(f"Unknown intervention '{intervention_name}'. "
                         f"Choose from: {list(INTERVENTIONS.keys())}")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    if mvp:
        # quick pilot — use this to check nothing is broken before a real run
        config.N_NODES = 10
        config.N_SEEDS = 3
        rumors_to_test = ["mundane"]
    else:
        rumors_to_test = list(RUMORS.keys())

    if rumor_subset:
        rumors_to_test = [r for r in rumor_subset if r in RUMORS]

    n_seeds = config.N_SEEDS
    total = len(rumors_to_test) * n_seeds

    print(f"Intervention: {intervention_name}")
    print(f"  rumors: {rumors_to_test}")
    print(f"  seeds:  {list(range(n_seeds))}")
    print(f"  N={config.N_NODES}, timesteps={config.N_TIMESTEPS}")
    print(f"  total conditions: {total}\n")

    results = []
    run_num = 0
    for rumor_label in rumors_to_test:
        for seed in range(n_seeds):
            run_num += 1
            print(f"[{run_num}/{total}] {intervention_name} "
                  f"rumor={rumor_label} seed={seed}")
            try:
                result = run_single_condition(
                    intervention_name, rumor_label, RUMORS[rumor_label],
                    seed, verbose=config.LOG_VERBOSE)
                results.append(result)
                m = result["metrics"]
                print(f"  -> final={m.get('final_infection_size', '?')} "
                      f"peak={m.get('peak_infection', '?')} "
                      f"skipped_calls={result['belief_calls_skipped']} "
                      f"({result['elapsed_sec']:.1f}s)")
            except Exception as e:
                print(f"  !! FAILED: {e}")
                import traceback
                traceback.print_exc()

    tag = "mvp" if mvp else "full"
    out_path = os.path.join(
        config.RESULTS_DIR,
        f"results_{intervention_name}_{tag}_{int(time.time())}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nWrote {out_path}")
    _print_summary(results, intervention_name)


def _print_summary(results, intervention_name):
    if not results:
        print("No successful runs.")
        return
    finals = [r["metrics"].get("final_infection_size", 0) for r in results]
    peaks = [r["metrics"].get("peak_infection", 0) for r in results]
    n = len(finals)
    total_skipped = sum(r.get("belief_calls_skipped", 0) for r in results)
    print(f"\n=== SUMMARY: {intervention_name} ===")
    print(f"  runs: {n}")
    print(f"  final_infected: mean={sum(finals)/n:.2f}")
    print(f"  peak_infected:  mean={sum(peaks)/n:.2f}")
    print(f"  belief-update calls skipped (opt #5): {total_skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--intervention", required=True,
                        choices=list(INTERVENTIONS.keys()),
                        help="Which single surgery to run")
    parser.add_argument("--mvp", action="store_true",
                        help="Use small pilot parameters")
    parser.add_argument("--rumors", default=None,
                        help="Comma-separated subset of rumors")
    args = parser.parse_args()

    rumor_subset = args.rumors.split(",") if args.rumors else None
    run_experiment(args.intervention, mvp=args.mvp, rumor_subset=rumor_subset)
