# Network Surgery on LLM Rumor Propagation: A Pilot Study

A controlled, causal-intervention pipeline to test whether classical network-science strategies (like bridge removal or hub immunization) can contain rumor cascades when the spreading nodes are reasoning LLM agents with distinct personas. 

This framework transitions multi-agent rumor simulation from purely observational studies to **causal inference**. By running pairs of simulations that are completely identical up until a fixed surgery timestep ($t_s=5$), you can isolate and measure the exact cause-and-effect of structural network interventions on AI-driven misinformation.

---

## Pipeline Architecture

Unlike traditional epidemiological models (e.g., SIR) where agents are uniform, memoryless units with fixed scalar transmission rates, this pipeline replaces nodes with reasoning language models. 
* **Belief-Sensitive Transmission:** Agents process incoming text, update an internal scalar belief using a weighted average, and decide whether to pass the rumor along based on their assigned persona's skepticism, reactivity, or gossip tendencies.
* **Semantic Drift:** Rumors naturally mutate as they traverse the network because agents paraphrase messages in their own unique voices.
* **Dual-Model Safety:** To prevent shared lexical or stylistic biases, one model family acts as the living agents while an entirely distinct model family evaluates the results.

---

## Files

- `personas.py` — The trait-diverse pool of 10 character profiles used to seed agent behaviors.
- `graph_builder.py` — Generates deterministic synthetic networks (primarily Watts-Strogatz small-world topologies).
- `agent.py` — Wraps the local LLM node operations (`RECEIVE` for belief updates and `DECIDE-SHARE` for persona-mediated paraphrasing).
- `simulation.py` — Manages the 4-phase propagation loop (Seed, Free Spread, Network Surgery, and Post-Op Measurement).
- `interventions.py` — Implements the 5 mutation operators applied at $t_s$ (None, Hub Immunization, Bridge Removal, Community Sealing, and Random Edge Removal).
- `evaluator.py` — Handles the batch post-hoc pass to score text for factual preservation and endorsement.
- `metrics.py` — Computes final cascade size, peak infection, containment index, semantic drift, and endorsement stability.
- `run_experiment.py` — Orchestrates execution across separate, reproducible command invocations.
- `config.py` — Central system hyperparameters (temperatures, sharing thresholds, token caps).

---

## Setup

The pipeline is designed to run entirely locally on a single consumer GPU via **Ollama**, eliminating external API fees and ensuring full reproducibility.

```bash
pip install ollama networkx numpy pandas matplotlib
ollama pull llama3.1:8b  # The Agent Model
ollama pull qwen2.5:7b   # The Independent Evaluator Model
```

> **Performance Note:** Evaluation is handled in a single batch pass *after* the simulation completes. This keeps the agent model resident in GPU memory throughout the active propagation steps, cutting execution time significantly by avoiding constant model swaps.

---

## Running Experiments

### 1. Run the MVP (Debug Mode)
To quickly verify that your local environment, model parsing, and fallback JSON handlers are working properly:

```bash
python run_experiment.py --mode mvp
```
*Runs a scaled-down 10-node sandbox with 1 rumor category and 3 seeds. Expect it to finish in 10–20 minutes on a modern laptop GPU.*

### 2. Run the Full Calibration Study
To replicate the 75-run benchmark protocol highlighted in the paper (crossing 5 intervention conditions, 3 rumor categories, and 5 random seeds):

```bash
python run_experiment.py --mode full
```
*Configured for 30-node Watts-Strogatz networks over 15 timesteps with surgery firing at $t=5$. Because this executes thousands of model inferences, plan for a multi-hour to full-day compute budget depending on your hardware.*

---

## Key Baseline Findings

If you are expanding on this repository for a **v2 protocol**, keep these empirical regularities from the pilot study in mind:

* **The Reality of Reasoning Nodes:** Classic interventions largely fail to contain LLM cascades once they pick up steam. Converting a network hub into a fact-checker does not easily flip the beliefs of neighbors who have already built up prior reinforcement states.
* **Structural Severance Wins:** **Community Sealing** (cutting all inter-community edges) was the only surgery that reliably restricted final cascade sizes, operating as a strict structural barrier. 
* **The Bridge Fallacy:** Removing high-betweenness bridges was statistically indistinguishable from cutting random edges, meaning the structural-hole hypothesis did not hold at this pilot scale.
* **Extreme Persona Sensitivity:** Conspiratorial/unfalsifiable rumors yield highly bimodal cascades. If the initial seed node is assigned a highly skeptical persona, the cascade will completely stall before ever reaching the surgery timestep.
