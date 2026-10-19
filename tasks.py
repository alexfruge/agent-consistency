"""Task set. Each task has:
  id         unique id (used in log filenames)
  category   "lookup" | "calc" | "multistep"
  prompt     what the agent is asked
  expected   list of substrings that must all appear in a correct answer
             (lowercased, commas removed), or None if there's no single answer

Starter set of 8. The proposal calls for 15-20, so add more in the same shape
(or adapt tasks from ToolBench / tau-bench and write mock tool outputs for them).
"""

TASKS = [
    # --- simple lookups ---
    {"id": "L1", "category": "lookup",
     "prompt": "What's the weather in Denver right now?",
     "expected": ["54"]},
    {"id": "L2", "category": "lookup",
     "prompt": "What is the exchange rate from USD to EUR?",
     "expected": ["0.92"]},
    {"id": "L3", "category": "lookup",
     "prompt": "How much does the WorkBook 15 cost?",
     "expected": ["899"]},

    # --- calculations ---
    {"id": "C1", "category": "calc",
     "prompt": "What is 17.5% of 2480?",
     "expected": ["434"]},
    {"id": "C2", "category": "calc",
     "prompt": "Compute (1234 * 56) / 7.",
     "expected": ["9872"]},

    # --- short multi-step ---
    {"id": "M1", "category": "multistep",
     "prompt": "Find the cheapest laptop in the catalog and tell me its price in EUR.",
     "expected": ["827.08"]},
    {"id": "M2", "category": "multistep",
     "prompt": "Which is warmer right now, Denver or Boston, and by how many degrees F?",
     "expected": ["denver", "7"]},
    {"id": "M3", "category": "multistep",
     "prompt": "I want to buy the UltraBook 14 and the GameBook 17. "
               "What's the combined total in GBP?",
     "expected": ["2210.42"]},
]
