"""each node is one agent. persona + belief state + memory of what it heard.

optimizations in here:
- #4: cap output tokens on every llm call (see config.MAX_TOKENS_*)
- #5: receive() skips the belief update llm call when belief is already high
      (above config.BELIEF_SATURATION) — re-hearing something you already
      believe 90% doesn't really change anything so why waste the call

NOTE: agent doesn't call the evaluator anymore!! messages are stored raw and
evaluation happens in one big batch after the whole simulation finishes.
this was optimization #1 and it saved like hours. see evaluator.py + simulation.py
"""

import ollama
import json
import re
from dataclasses import dataclass, field
import config


@dataclass
class Agent:
    node_id: int
    persona: dict
    belief: float = 0.0          # 0 = doesn't believe, 1 = fully believes
    has_heard: bool = False
    memory: list[dict] = field(default_factory=list)
    message_log: list[str] = field(default_factory=list)
    n_belief_calls_skipped: int = 0   # just counting how much opt #5 saves us

    @property
    def system_prompt(self) -> str:
        return (
            f"You are {self.persona['name']}, {self.persona['description']} "
            f"You are interacting with people you know in your social circle. "
            f"Respond naturally and in character. Keep responses to 2-3 sentences."
        )

    def _call_llm(self, user_prompt: str, max_tokens: int) -> str:
        """one ollama call. that's it."""
        client = ollama.Client(host=config.OLLAMA_HOST)
        try:
            resp = client.chat(
                model=config.AGENT_MODEL,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={
                    "temperature": config.TEMPERATURE,
                    "num_predict": max_tokens,
                },
            )
            return resp["message"]["content"].strip()
        except Exception as e:
            # don't crash the whole 7-hour run over one bad call
            print(f"  [warn] LLM call failed for node {self.node_id}: {e}")
            return ""

    def receive(self, message: str, sender_name: str):
        """hear a message, maybe update belief.

        opt #5: already at 0.85+ belief? skip the llm call entirely.
        heard it a hundred times already, you believe it, moving on.
        """
        self.has_heard = True
        self.memory.append({"from": sender_name, "content": message})

        if self.belief >= config.BELIEF_SATURATION:
            self.n_belief_calls_skipped += 1
            return

        prompt = (
            f"Your acquaintance {sender_name} just told you:\n"
            f"\"{message}\"\n\n"
            f"How much do you believe this claim, on a scale from 0 to 10? "
            f"Reply with ONLY a single number, nothing else."
        )
        response = self._call_llm(prompt, max_tokens=config.MAX_TOKENS_BELIEF)
        score = _extract_number(response, default=0)
        new_belief = score / 10.0
        self.belief = 0.5 * self.belief + 0.5 * new_belief   # running average

    def decide_share(self, neighbor_name: str, original_rumor: str) -> tuple[bool, str]:
        """should i tell this person? if yes, what do i say?"""
        if not self.has_heard:
            return (False, "")

        prompt = (
            f"You recently heard this claim and believe it with confidence "
            f"{self.belief:.2f} (on a 0-1 scale):\n"
            f"\"{original_rumor}\"\n\n"
            f"You are about to talk to {neighbor_name}, someone you know. "
            f"Would you bring up this claim? "
            f"Respond with a JSON object: "
            f'{{"share": true_or_false, "message": "what you would say to them, '
            f'in your own words and style, or empty string if not sharing"}}'
        )
        response = self._call_llm(prompt, max_tokens=config.MAX_TOKENS_SHARE)
        share, message = _parse_share_decision(response)

        if share and message:
            self.message_log.append(message)

        return (share, message)


def _extract_number(text: str, default: float = 0.0) -> float:
    match = re.search(r"-?\d+(\.\d+)?", text)
    if match:
        try:
            return max(0.0, min(10.0, float(match.group())))
        except ValueError:
            pass
    return default


def _parse_share_decision(text: str) -> tuple[bool, str]:
    """try to parse json. if the model forgot how json works, fall back to vibes."""
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return (bool(data.get("share", False)), str(data.get("message", "")))
    except (json.JSONDecodeError, ValueError):
        pass

    # llama sometimes just says "yes" or starts with "true" — good enough
    lower = text.lower()
    if "true" in lower[:50] or "yes" in lower[:50]:
        return (True, text)
    return (False, "")
