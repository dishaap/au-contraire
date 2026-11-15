"""
Central settings. Edit model IDs here; nothing else hard codes them.
"""

PROVIDERS = {
    "cohere": {"base_url": "https://api.cohere.ai/compatibility/v1", "key_env": "COHERE_API_KEY"},
    # Adaption uses its own Responses API (not chat/completions); handled in llm._adaption.
    "adaption": {"base_url": "https://api.prod.adaptionlabs.ai/api/v1", "key_env": "ADAPTION_API_KEY"},
}

# temperature=None -> parameter omitted (some reasoning endpoints reject it)
MODELS ={
    "command-a": dict(provider="cohere", model="command-a-03-2025", reasoning=False,
                      temperature=0.7, max_tokens=700),
    "command-a-reasoning": dict(provider="cohere", model="command-a-reasoning-08-2025",
                                reasoning=True, temperature=0.7, max_tokens=6000),  # VERIFY id
    "qwen3.8-27b": dict(provider="adaption", model="qwen3.8-27b", reasoning=None,
                        temperature=0.7, max_tokens=6000),
    "qwen3.5-397b": dict(provider="adaption", model="qwen3.5-397b", reasoning=None,
                         temperature=0.7, max_tokens=6000),
    # offline stand-ins for testing the pipeline (no API calls)
    "mock-sycophant": dict(provider="mock", reasoning=False, temperature=0.7, max_tokens=0,
                           knowledge=0.8, responsiveness=0.85, sycophancy=0.45, pressure_boost=0.25),
    "mock-steady": dict(provider="mock", reasoning=False, temperature=0.7, max_tokens=0,
                        knowledge=0.9, responsiveness=0.80, sycophancy=0.10, pressure_boost=0.05),
}

DEFAULT_MODELS = ["command-a", "command-a-reasoning", "qwen3.8-27b"] # due to available api creds
MOCK_MODELS = ["mock-sycophant", "mock-steady"]

# judge NOT any model under test. ideally a different vendor from the tested models is best, test llama
JUDGE = dict(provider="adaption", model="gemma-4-31b-it", temperature=0.0, max_tokens=400)  # VERIFY id
MOCK_JUDGE = dict(provider="mock", temperature=0.0, max_tokens=0)

FRAMINGS = ["neutral", "assertive"]
REPS = 3                      # repeated generations per (model, scenario, framing); NOT independent units
WORKERS = 3                   # keep low: Cohere RPM limits bit us in SwarmScope
BOOTSTRAP_B = 5000
SEED = 0