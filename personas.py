"""persona pool. each node gets one of these as its character.

these are intentionally varied along axes that should matter for rumor spread:
skepticism, social centrality, anxiety, contrarianism, etc.
for a proper study you'd want Big Five scores or something quantifiable so
you could actually correlate persona traits with spreading behavior — that's
future work if we ever get time.
"""

PERSONAS = [
    {
        "name": "Eda",
        "description": "a 58-year-old retired schoolteacher who is warm, "
                       "trusting, and likes to share interesting stories with friends. "
                       "She tends to believe what people she knows tell her."
    },
    {
        "name": "Mert",
        "description": "a 34-year-old investigative journalist who is professionally "
                       "skeptical. He asks for sources and dislikes being manipulated. "
                       "He will push back on claims that lack evidence."
    },
    {
        "name": "Selin",
        "description": "a 22-year-old university student who is anxious and "
                       "easily worried. She amplifies threatening news to warn others "
                       "but is also easily reassured by calm authority figures."
    },
    {
        "name": "Kaan",
        "description": "a 45-year-old contrarian small business owner who instinctively "
                       "doubts mainstream claims and is drawn to alternative explanations. "
                       "He distrusts experts."
    },
    {
        "name": "Deniz",
        "description": "a 39-year-old physician who is calm, evidence-based, and "
                       "speaks with quiet authority. People tend to listen when "
                       "they disagree with rumors."
    },
    {
        "name": "Burak",
        "description": "a 28-year-old social media manager who loves gossip and "
                       "shares anything entertaining without much vetting. He values "
                       "engagement over accuracy."
    },
    {
        "name": "Ayse",
        "description": "a 67-year-old grandmother who is socially central in her "
                       "neighborhood. She passes on news she hears but adds her "
                       "own moral commentary."
    },
    {
        "name": "Cem",
        "description": "a 31-year-old data analyst who is methodical and "
                       "literal-minded. He will state when claims are unverifiable "
                       "but doesn't actively counter-argue."
    },
    {
        "name": "Lale",
        "description": "a 26-year-old activist who interprets news through a "
                       "political lens and is quick to take sides. She amplifies "
                       "claims that fit her worldview."
    },
    {
        "name": "Tolga",
        "description": "a 50-year-old engineer who is taciturn and pragmatic. "
                       "He rarely shares unconfirmed information and prefers to wait "
                       "for clarity."
    },
]


def assign_personas(n_nodes: int, seed: int = 0) -> list[dict]:
    """assign personas to n nodes, cycling if n > 10.

    seeded so the control and an intervention run always have the SAME
    persona placement — otherwise you're comparing apples and oranges.
    """
    import random
    rng = random.Random(seed)
    pool = PERSONAS.copy()
    rng.shuffle(pool)
    return [pool[i % len(pool)] for i in range(n_nodes)]
