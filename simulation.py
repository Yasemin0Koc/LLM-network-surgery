"""main propagation loop.

big change from the old version: evaluator no longer runs inside the loop.
used to call qwen after every single message, which forced ollama to swap
models every few seconds. absolute nightmare for runtime.

now: agents run start to finish with only llama loaded, messages stored raw.
after everything finishes, evaluate_messages_batch() runs once and scores
everything. qwen loads exactly once. this was the single biggest speedup (opt #1).

each timestep:
1. every agent that has heard the rumor + believes it above threshold
   considers sharing with each neighbor
2. receiving neighbors update belief (or skip if already saturated — opt #5)
3. snapshot network state

intervention fires at a specified timestep, between snapshots.
"""

from agent import Agent
from evaluator import evaluate_messages_batch
import config


def run_simulation(
    graph,
    agents: dict[int, Agent],
    rumor: str,
    seed_node: int,
    intervention_fn=None,
    intervention_kwargs=None,
    n_timesteps: int = None,
    intervention_timestep: int = None,
    verbose: bool = False,
) -> dict:
    """run one full simulation, then batch-evaluate all messages at the end."""
    n_timesteps = n_timesteps or config.N_TIMESTEPS
    intervention_timestep = intervention_timestep or config.INTERVENTION_TIMESTEP
    intervention_kwargs = intervention_kwargs or {}

    # patient zero hears the rumor and fully believes it
    agents[seed_node].receive(rumor, sender_name="(initial source)")
    agents[seed_node].belief = 1.0

    trace = {
        "rumor": rumor,
        "seed_node": seed_node,
        "snapshots": [],
        "intervention_log": None,
        "messages": [],
    }

    for t in range(n_timesteps):
        if verbose:
            print(f"  t={t} | infected={_count_infected(agents)} "
                  f"| edges={graph.number_of_edges()}")

        if intervention_fn is not None and t == intervention_timestep:
            log = intervention_fn(graph, agents, **intervention_kwargs)
            trace["intervention_log"] = log
            if verbose:
                print(f"  >>> INTERVENTION: {log.get('name')}")

        # snapshot the active agents BEFORE this round - otherwise a message
        # could propagate two hops in one timestep which would be wrong
        active = [aid for aid, a in agents.items()
                  if a.has_heard and a.belief >= config.BELIEF_SHARE_THRESHOLD
                  and aid in graph.nodes()]

        for sender_id in active:
            sender = agents[sender_id]
            for neighbor_id in graph.neighbors(sender_id):
                if neighbor_id not in agents:
                    continue
                receiver = agents[neighbor_id]
                share, message = sender.decide_share(
                    receiver.persona["name"], rumor)
                if share and message:
                    # store raw — no evaluation yet (opt #1)
                    trace["messages"].append({
                        "t": t,
                        "from": sender_id,
                        "to": neighbor_id,
                        "message": message,
                    })
                    receiver.receive(message, sender.persona["name"])

        trace["snapshots"].append(_snapshot(agents, t))

    # batch evaluation (opt #1)
    # all llama work is done, now score everything with qwen in one pass
    trace["messages"] = evaluate_messages_batch(
        rumor, trace["messages"], verbose=verbose)

    # how many llm calls did opt #5 save across the whole run?
    trace["belief_calls_skipped"] = sum(
        a.n_belief_calls_skipped for a in agents.values())

    return trace


def _count_infected(agents) -> int:
    return sum(1 for a in agents.values()
               if a.has_heard and a.belief >= config.BELIEF_SHARE_THRESHOLD)


def _snapshot(agents, t) -> dict:
    return {
        "t": t,
        "n_heard": sum(1 for a in agents.values() if a.has_heard),
        "n_believing": sum(1 for a in agents.values() if a.belief >= 0.5),
        "mean_belief": sum(a.belief for a in agents.values()) / len(agents),
        "beliefs": {aid: a.belief for aid, a in agents.items()},
    }
