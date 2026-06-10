"""compute outcome metrics from a simulation trace."""

import numpy as np


def compute_metrics(trace: dict, intervention_timestep: int = 5) -> dict:
    snapshots = trace["snapshots"]
    messages = trace["messages"]

    if not snapshots:
        return {}

    n_believing = [s["n_believing"] for s in snapshots]
    mean_belief = [s["mean_belief"] for s in snapshots]

    # containment: did the surgery actually stop the spread from growing?
    # positive = fewer believers after surgery than before, negative = backfired
    pre = n_believing[intervention_timestep - 1] if intervention_timestep > 0 else 0
    post_slice = n_believing[intervention_timestep:]
    post_max = max(post_slice) if post_slice else 0
    containment = (pre - post_max) / max(pre, 1)

    if messages:
        early = [m for m in messages if m["t"] < intervention_timestep]
        late = [m for m in messages if m["t"] >= intervention_timestep]
        early_pres = np.mean([m.get("preserved", 0) for m in early]) if early else 0.0
        late_pres = np.mean([m.get("preserved", 0) for m in late]) if late else 0.0
        early_end = np.mean([m.get("endorsement", 0) for m in early]) if early else 0.0
        late_end = np.mean([m.get("endorsement", 0) for m in late]) if late else 0.0
    else:
        early_pres = late_pres = early_end = late_end = 0.0

    return {
        "final_infection_size": n_believing[-1],
        "peak_infection": max(n_believing),
        "time_to_peak": int(np.argmax(n_believing)),
        "final_mean_belief": mean_belief[-1],
        "containment_index": containment,
        "n_messages_total": len(messages),
        "n_messages_pre": len([m for m in messages if m["t"] < intervention_timestep]),
        "n_messages_post": len([m for m in messages if m["t"] >= intervention_timestep]),
        "preservation_pre": early_pres,
        "preservation_post": late_pres,
        "semantic_drift_pre": 1 - early_pres,    # drift = 1 - preservation
        "semantic_drift_post": 1 - late_pres,
        "endorsement_pre": early_end,
        "endorsement_post": late_end,
        "infection_curve": n_believing,
    }
