# Network Surgery on LLM Rumor Propagation

A controlled causal-intervention pipeline for studying how classical network-theoretic interventions perform when nodes are LLM agents that reason and paraphrase.

This repository contains the pipeline, the calibration study results, and the LaTeX source for the accompanying paper.

**Authors:** Çınar Efe Yetişen (34245), Yasemin Meryem Koç (34440)
**Course:** CS414 Network Science, Sabancı University

## What this project does

Classical network science predicts that cutting bridges, immunizing hubs, or sealing communities will contain rumor cascades. These predictions assume stochastic agents with scalar belief states. LLM-driven agents reason, paraphrase, and update beliefs in natural language, which may violate those assumptions.

We built a pipeline in which:

1. Each node in a synthetic network is an LLM agent with a persona-conditioned system prompt
2. A rumor is seeded at one node and propagates through belief updates and paraphrasing
3. At a designated timestep, a network surgery is applied (hub immunization, bridge removal, community sealing, or random edge removal as a structural null)
4. Post-surgery cascade dynamics are measured and compared against a no-intervention baseline

The pipeline supports five conditions, three rumor categories, and arbitrary seeds, with deterministic graph and persona construction so that conditions can be compared causally rather than merely correlationally.

## Requirements

**System:**
- Python 3.10 or later
- An NVIDIA GPU with at least 8 GB VRAM (the study used a single consumer GPU)
- [Ollama](https://ollama.com) for local LLM serving

**Python packages:**
```
pip install ollama networkx numpy matplotlib sentence-transformers
```

**LLM models (pulled via Ollama):**
```
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
```

The agent model and evaluator model are deliberately from different families (Llama vs. Qwen) to avoid in-family stylistic bias when the evaluator scores agent-generated messages.

## Reproducing the study

The full study consists of five experiment commands, one per intervention. Each command takes several hours on a single consumer GPU and writes its own timestamped JSON file in `results/`.

```bash
# Step 1: compute the shared control (run once, reused by every comparison)
python run_experiment.py --intervention none

# Step 2: run each surgery
python run_experiment.py --intervention hub_immunization
python run_experiment.py --intervention bridge_removal
python run_experiment.py --intervention community_sealing
python run_experiment.py --intervention random_edge_removal
```

The five commands can be run on separate days, different machines, or interrupted and resumed. Each one is independent. As long as `config.py` is not modified between commands, the resulting JSON files are directly comparable because graph construction and persona assignment are deterministic in their seed argument.

For a quick pilot test, add `--mvp` to any command. This uses N = 10, three seeds, and only the mundane rumor.

## Producing figures and tables

After all five intervention commands have completed:

```bash
python analyze_results.py
```

This globs every `results/results_*.json` file, merges them, and writes:

- `figures/infection_curves.png`
- `figures/final_infection_bar.png`
- `figures/semantic_drift.png`
- `figures/summary.txt`

For the before-and-after network snapshots used in the paper:

```bash
python export_networks.py
```

This reads the existing result files, replays each surgery onto its corresponding graph (using the seeded reconstruction guaranteed by `graph_builder.py`), and writes both `.gexf` files (for Gephi) and `.png` snapshots at t = 0, 4, 5, and 14. Optional flags: `--seed N`, `--gexf-only`, `--png-only`.

## Configuration

All experiment knobs live in `config.py`. The defaults match the calibration study reported in the paper:

| Parameter | Value | Meaning |
|---|---|---|
| `N_NODES` | 30 | Network size |
| `GRAPH_TYPE` | watts_strogatz | Topology family |
| `WS_K`, `WS_P` | 4, 0.2 | Mean degree, rewiring probability |
| `N_TIMESTEPS` | 15 | Simulation length |
| `INTERVENTION_TIMESTEP` | 5 | When surgery is applied |
| `N_SEEDS` | 5 | Number of random seeds per condition |
| `TEMPERATURE` | 0.8 | LLM sampling temperature |
| `BELIEF_SHARE_THRESHOLD` | 0.3 | Minimum belief to consider sharing |
| `BELIEF_SATURATION` | 0.85 | Above this, skip belief-update LLM call |
| `MAX_TOKENS_BELIEF` | 10 | Tokens cap on belief-score calls |
| `MAX_TOKENS_SHARE` | 90 | Tokens cap on share-decision JSON calls |
| `MAX_TOKENS_EVAL` | 30 | Tokens cap on evaluator calls |

**Important:** do not modify `config.py` between intervention runs. Persona placement, graph structure, and outcome metrics all depend on these values being identical across the five commands; the comparison validity rests on this.

## Output format

Each run produces a JSON file with one entry per (intervention, rumor, seed) condition. Each entry contains:

```
condition         intervention name, rumor label, seed, seed node
pre_graph         structural summary before surgery
post_graph        structural summary after surgery
intervention_log  which nodes or edges were affected
metrics           final cascade size, peak, containment, drift, endorsement
messages          every message generated, with evaluator scores
snapshots         per-timestep belief vectors for every node
belief_calls_skipped  bookkeeping for the saturation optimisation
elapsed_sec       wall-clock time
```

The metrics in the paper are computed directly from these JSON files, so all reported numbers are reproducible from the logged data without re-running any simulation.

## The paper

The accompanying paper (`paper/main.tex`) documents the pipeline design, the calibration study, the headline findings, and four calibration requirements for a v2 protocol. It is formatted as an IEEE conference submission using `IEEEtran.cls`.

Compile in Overleaf or locally with:

```bash
cd paper
pdflatex main.tex
pdflatex main.tex
```

Two passes are needed for cross-references to resolve.

The figures expected by `main.tex` are produced by `analyze_results.py` (the three summary charts) and `export_networks.py` (the four before-and-after pairs), plus a hand-drawn system diagram (`system_diagram.pdf`) and a TikZ pipeline strip embedded in the source. All figures live in `paper/figures/`.

### Key findings

- Across the conditions tested, community sealing was the only intervention to produce a discernible reduction in final cascade size
- Bridge removal was statistically indistinguishable from random edge removal, though at N = 30 neither surgery actually disconnected the graph, so the test could not fully fire
- Hub immunization showed complete suppression in some seeds but only because the seed node and the immunization target coincided (a contamination flagged at first mention in Results)
- Semantic drift remained low and stable (around 13–16%) across all conditions, indicating that LLM agents paraphrase rumors through their personas without substantially mutating the core claim. This is an LLM-specific empirical regularity worth noting independently of any intervention effect.

The paper frames these as calibration results and identifies four requirements for a full v2 protocol: state-triggered surgery timing, enforced seed–hub separation, increased network scale and seed count, and stationary rather than final-timestep outcome measures.

## Speed optimisations

The pipeline applies three speed optimisations documented in the paper:

1. **Post-hoc batch evaluation.** The evaluator runs once after the simulation, not inline. This keeps the agent model resident throughout the run and loads the evaluator exactly once at the end.
2. **Capped output lengths.** Each call type uses an aggressive `num_predict` ceiling, since generation time scales roughly with tokens produced.
3. **Saturated-belief skip.** Agents whose belief is already above 0.85 skip the belief-update LLM call on subsequent message receipt.

Together these reduce a single MVP-scale condition from several hours to roughly half an hour, and make the full study feasible on a single consumer GPU over a few days of wall-clock time.

## Limitations

This is a calibration study at N = 30 with five seeds. The paper's findings should be read as calibration results rather than definitive conclusions about LLM-driven cascade dynamics. The most important limitations:

- At N = 30, bridge removal does not actually disconnect the network, so the Granovetter structural-hole hypothesis cannot be directly tested
- The seed node and the hub-immunization target frequently coincide, confounding the structural hub effect with the direct effect of neutralizing the rumor's origin
- The containment index is unstable when pre-surgery cascade sizes are small; stationary outcome measures would be more robust
- Persona assignment is a primary determinant of whether a cascade forms at all, and five seeds is not enough to average over this effect
- Evaluator reliability on Turkish-inflected agent outputs has not been independently validated

Section VII of the paper discusses each of these in detail.

## License and citation

This is a course project. If you build on the pipeline, please cite the paper. The code is released under no specific license; copy or adapt as useful for your own research.

For questions, contact `cinar.yetisen@sabanciuniv.edu` or `yasemin.koc@sabanciuniv.edu`.
