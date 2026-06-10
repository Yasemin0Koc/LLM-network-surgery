"""all the knobs. change stuff here, not scattered across 8 different files."""

# LLM settings
AGENT_MODEL = "llama3.1:8b"      # the character nodes
EVALUATOR_MODEL = "qwen2.5:7b"   # separate model for judging belief endorsement
OLLAMA_HOST = "http://localhost:11434"
TEMPERATURE = 0.8                # crank this up for more persona variation

# output length caps (speed optimization #4)
# generation time scales with output tokens so cap short calls hard
MAX_TOKENS_BELIEF = 10           # it's literally just a number
MAX_TOKENS_SHARE = 90            # short json blob
MAX_TOKENS_EVAL = 30             # two numbers separated by a comma

# network settings
# these are the FULL experiment params. N=30 per the v2 protocol we agreed on
N_NODES = 30
GRAPH_TYPE = "watts_strogatz"    # "watts_strogatz" or "barabasi_albert"
WS_K = 4                         # mean degree for WS
WS_P = 0.2                       # rewiring prob
BA_M = 2                         # edges per new node for BA

# simulation settings
N_TIMESTEPS = 15
INTERVENTION_TIMESTEP = 5        # surgery at t=5
N_SEEDS = 5
INITIAL_INFECTED = 1

# belief dynamics
BELIEF_SHARE_THRESHOLD = 0.3     # only share if you believe it above this
BELIEF_SATURATION = 0.85         # opt #5: skip llm call if already basically
                                 # certain - saves a ton of calls late in runs

# embedding model for semantic drift
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# output
RESULTS_DIR = "results"
LOG_VERBOSE = True               # prints progress each timestep, nice to have
