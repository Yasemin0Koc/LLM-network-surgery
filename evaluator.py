"""post-hoc batch evaluator — this is optimization #1, the big one.

old pipeline: evaluator ran INSIDE the simulation loop after every single
shared message. the problem is evaluator uses qwen and agents use llama, so
ollama was swapping models in and out of vram constantly. that was like 80%
of the runtime. genuinely painful.

new pipeline: simulation runs completely with only llama loaded, messages
stored raw. then after everything finishes, we load qwen ONCE and score
everything in one pass. huge speedup.

each message gets two scores:
- endorsement: 0 = speaker actively contradicts the rumor, 1 = fully endorses it
- preserved:   0 = rumor is unrecognizable / totally mutated, 1 = core claim intact
"""

import ollama
import re
import config


def _evaluate_one(client, original_rumor: str, message: str) -> dict:
    """score one message. assumes qwen is already loaded."""
    if not message.strip():
        return {"endorsement": 0.0, "preserved": 0.0}

    prompt = (
        f"ORIGINAL CLAIM: \"{original_rumor}\"\n\n"
        f"MESSAGE TO EVALUATE: \"{message}\"\n\n"
        f"Answer two questions about the message:\n"
        f"1. ENDORSEMENT: Does the speaker treat the original claim as true? "
        f"(0 = denies/contradicts, 5 = neutral/uncertain, 10 = fully endorses)\n"
        f"2. PRESERVATION: Is the core factual content of the original claim "
        f"recognizable in the message? "
        f"(0 = totally lost/different topic, 10 = core claim intact)\n\n"
        f"Respond with ONLY two numbers separated by a comma, like: 7,9"
    )

    try:
        resp = client.chat(
            model=config.EVALUATOR_MODEL,
            messages=[
                {"role": "system", "content": "You are a careful, neutral analyst."},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.1, "num_predict": config.MAX_TOKENS_EVAL},
        )
        text = resp["message"]["content"].strip()
    except Exception as e:
        print(f"  [warn] evaluator call failed: {e}")
        return {"endorsement": 0.0, "preserved": 0.0}

    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(nums) >= 2:
        endorsement = max(0.0, min(10.0, float(nums[0]))) / 10.0
        preserved = max(0.0, min(10.0, float(nums[1]))) / 10.0
    else:
        # qwen sometimes rambles instead of just giving numbers, default to 0
        endorsement, preserved = 0.0, 0.0

    return {"endorsement": endorsement, "preserved": preserved}


def evaluate_messages_batch(original_rumor: str, messages: list[dict],
                            verbose: bool = False) -> list[dict]:
    """score every message in one big pass.

    takes the raw message list from simulation, adds endorsement + preserved
    scores to each entry in place, returns the same list.
    qwen loads exactly once for the whole thing.
    """
    client = ollama.Client(host=config.OLLAMA_HOST)

    if verbose and messages:
        print(f"  evaluating {len(messages)} messages in one batch pass...")

    for i, m in enumerate(messages):
        result = _evaluate_one(client, original_rumor, m.get("message", ""))
        m["endorsement"] = result["endorsement"]
        m["preserved"] = result["preserved"]
        if verbose and (i + 1) % 25 == 0:
            print(f"    evaluated {i + 1}/{len(messages)}")

    return messages
