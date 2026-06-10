"""reads all the results json files and makes figures for the poster.

globs everything in results/ and merges it, so it doesn't matter that
the interventions were run on different days. just throw all the jsons in
there and run this.

the `none` runs are the control for everything else.

spits out:
  figures/infection_curves.png      <- the main one, goes on the poster
  figures/final_infection_bar.png   <- bar chart, easier to explain in 3min
  figures/semantic_drift.png        <- supplementary, for the paper version
  figures/summary.txt               <- numbers with % reductions vs control

usage:
    python analyze_results.py            # just run it
    python analyze_results.py results/   # same thing, explicit folder
"""

import json
import sys
import os
import glob
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


# colors stolen from the poster palette — don't change these or the figures
# won't match the network visualizations
PALETTE = {
    "none": "#A32D2D", "no_intervention": "#A32D2D",
    "random_edge_removal": "#888780",
    "bridge_removal": "#1B7F8E",
    "hub_immunization": "#E8A838",
    "community_sealing": "#534AB7",
}
LABELS = {
    "none": "no intervention", "no_intervention": "no intervention",
    "random_edge_removal": "random control",
    "bridge_removal": "bridge removal",
    "hub_immunization": "hub immunization",
    "community_sealing": "community sealing",
}


def load_all_results(folder="results") -> list:
    """grab every results_*.json and smoosh them together"""
    files = sorted(glob.glob(os.path.join(folder, "results_*.json")))
    if not files:
        print(f"No results files found in {folder}/")
        sys.exit(1)

    print(f"Merging {len(files)} result file(s):")
    merged = []
    for path in files:
        with open(path) as f:
            runs = json.load(f)
        merged.extend(runs)
        print(f"  {os.path.basename(path)} - {len(runs)} runs")
    print(f"Total: {len(merged)} runs\n")
    return merged


def group_by_intervention(results: list) -> dict:
    grouped = defaultdict(list)
    for r in results:
        grouped[r["condition"]["intervention"]].append(r)
    return grouped


def _baseline_key(grouped):
    # handle both key names because i kept changing it
    for k in ("none", "no_intervention"):
        if k in grouped:
            return k
    return None


def plot_infection_curves(results, out_path, intervention_t=5):
    grouped = group_by_intervention(results)
    fig, ax = plt.subplots(figsize=(8, 5))

    for intervention, runs in grouped.items():
        curves = [r["metrics"].get("infection_curve", []) for r in runs]
        curves = [c for c in curves if c]
        if not curves:
            continue
        max_len = max(len(c) for c in curves)
        # pad shorter curves with their last value so averaging works
        padded = np.array([c + [c[-1]] * (max_len - len(c)) for c in curves])
        mean, std = padded.mean(axis=0), padded.std(axis=0)
        x = np.arange(max_len)
        color = PALETTE.get(intervention, "#444441")
        ax.plot(x, mean, color=color, linewidth=2.2,
                label=LABELS.get(intervention, intervention))
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

    ax.axvline(intervention_t, color="#A32D2D", linestyle="--",
               linewidth=1, alpha=0.6)
    ax.text(intervention_t + 0.2, ax.get_ylim()[1] * 0.95,
            "surgery (t=5)", color="#A32D2D", fontsize=10)
    ax.set_xlabel("timestep", fontsize=12)
    ax.set_ylabel("number of believers", fontsize=12)
    ax.set_title("Infection curves by intervention condition", fontsize=13, pad=12)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_final_infection_bar(results, out_path):
    grouped = group_by_intervention(results)
    interventions = list(grouped.keys())
    means = [np.mean([r["metrics"]["final_infection_size"] for r in grouped[i]])
             for i in interventions]
    stds = [np.std([r["metrics"]["final_infection_size"] for r in grouped[i]])
            for i in interventions]
    colors = [PALETTE.get(i, "#444441") for i in interventions]
    labels = [LABELS.get(i, i) for i in interventions]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, means, yerr=stds, color=colors,
                  capsize=4, edgecolor="white", linewidth=0.5)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{m:.1f}", ha="center", fontsize=10, color="#2C2C2A")
    ax.set_ylabel("final believers (mean +/- std)", fontsize=12)
    ax.set_title("Final cascade size by intervention", fontsize=13, pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_semantic_drift(results, out_path):
    # how much does the rumor mutate before vs after surgery?
    grouped = group_by_intervention(results)
    interventions = list(grouped.keys())
    pre = [np.mean([r["metrics"].get("semantic_drift_pre", 0) for r in grouped[i]])
           for i in interventions]
    post = [np.mean([r["metrics"].get("semantic_drift_post", 0) for r in grouped[i]])
            for i in interventions]
    x = np.arange(len(interventions))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, pre, width, label="pre-surgery", color="#B4B2A9")
    ax.bar(x + width / 2, post, width, label="post-surgery", color="#1B7F8E")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(i, i) for i in interventions],
                       rotation=15, ha="right")
    ax.set_ylabel("semantic drift (1 - preservation)", fontsize=12)
    ax.set_title("Rumor mutation before vs. after surgery", fontsize=13, pad=12)
    ax.legend(frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def write_summary(results, out_path):
    grouped = group_by_intervention(results)
    lines = ["=" * 60, "EXPERIMENT SUMMARY (merged across all runs)", "=" * 60,
             f"Total runs: {len(results)}", f"Interventions: {len(grouped)}", ""]

    header = f"{'intervention':<22} {'final':>12} {'peak':>12} {'contain':>10}"
    lines += ["Per-intervention means:", "-" * 60, header, "-" * 60]
    for intervention, runs in grouped.items():
        finals = [r["metrics"]["final_infection_size"] for r in runs]
        peaks = [r["metrics"]["peak_infection"] for r in runs]
        contains = [r["metrics"]["containment_index"] for r in runs]
        lines.append(
            f"{LABELS.get(intervention, intervention):<22} "
            f"{np.mean(finals):>7.1f}+-{np.std(finals):<3.1f} "
            f"{np.mean(peaks):>7.1f}+-{np.std(peaks):<3.1f} "
            f"{np.mean(contains):>9.2f}")

    base = _baseline_key(grouped)
    if base:
        base_final = np.mean([r["metrics"]["final_infection_size"]
                              for r in grouped[base]])
        lines += ["", "Reduction vs. control:", "-" * 60,
                  f"Control (no intervention) final infection: {base_final:.1f}"]
        for intervention, runs in grouped.items():
            if intervention == base:
                continue
            mean_final = np.mean([r["metrics"]["final_infection_size"]
                                  for r in runs])
            reduction = (base_final - mean_final) / max(base_final, 1) * 100
            lines.append(f"  {LABELS.get(intervention, intervention):<22} "
                         f"{reduction:+5.1f}%")
    else:
        # you forgot to run the control lol
        lines += ["", "[!] No 'none' control found - run "
                  "`python run_experiment.py --intervention none`"]

    text = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(text)
    print(f"  wrote {out_path}\n")
    print(text)


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "results"
    results = load_all_results(folder)
    os.makedirs("figures", exist_ok=True)
    print("Generating figures and summary...")
    plot_infection_curves(results, "figures/infection_curves.png")
    plot_final_infection_bar(results, "figures/final_infection_bar.png")
    plot_semantic_drift(results, "figures/semantic_drift.png")
    write_summary(results, "figures/summary.txt")
    print("\nDone.")


if __name__ == "__main__":
    main()
