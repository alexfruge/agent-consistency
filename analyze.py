"""Compute consistency metrics from the JSONL logs.

Usage:
  python analyze.py            # includes semantic similarity (downloads a small model once)
  python analyze.py --no-embed # skip embeddings

Per (task, temperature) metrics:
  tool_seq_exact_rate   fraction of runs matching the most common tool-name sequence
  tool_seq_edit_dist    mean pairwise normalized Levenshtein on tool-name sequences (0 = identical)
  tool_args_exact_rate  same as exact_rate but sequences include arguments
  tool_set_jaccard      mean pairwise Jaccard of the set of tools used
  n_calls_std           std dev of number of tool calls
  answer_correct_rate   fraction containing all expected substrings (if task has them)
  answer_exact_rate     fraction matching the most common normalized final answer
  answer_sem_sim        mean pairwise cosine similarity of answer embeddings (1 = identical meaning)
"""
import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from tasks import TASKS

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
TASK_INDEX = {t["id"]: t for t in TASKS}


def levenshtein(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def norm_lev(a, b) -> float:
    return levenshtein(a, b) / max(len(a), len(b), 1)


def mode_rate(items) -> float:
    return Counter(items).most_common(1)[0][1] / len(items)


def mean_pairwise(items, fn) -> float:
    pairs = list(combinations(items, 2))
    return float(np.mean([fn(a, b) for a, b in pairs])) if pairs else float("nan")


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 1.0


def normalize(text: str) -> str:
    return " ".join(text.lower().replace(",", "").split())


def load_runs() -> pd.DataFrame:
    rows = []
    for path in sorted(DATA_DIR.glob("*.jsonl")):
        rows += [json.loads(line) for line in path.open() if line.strip()]
    if not rows:
        raise SystemExit("No logs found in data/. Run harness.py first.")
    return pd.DataFrame(rows)


def semantic_similarity(texts, model) -> float:
    emb = model.encode(texts, normalize_embeddings=True)
    sims = emb @ emb.T
    iu = np.triu_indices(len(texts), k=1)
    return float(sims[iu].mean()) if len(texts) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-embed", action="store_true")
    args = ap.parse_args()

    embedder = None
    if not args.no_embed:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")

    df = load_runs()
    ok = df[df["error"].isna()]
    out = []
    for (task_id, temp), g in ok.groupby(["task_id", "temperature"]):
        seqs = [tuple(c["name"] for c in calls) for calls in g["tool_calls"]]
        seqs_args = [tuple((c["name"], json.dumps(c["args"], sort_keys=True)) for c in calls)
                     for calls in g["tool_calls"]]
        answers = [normalize(a) for a in g["final_output"]]
        expected = TASK_INDEX[task_id]["expected"]
        row = {
            "task_id": task_id,
            "category": g["category"].iloc[0],
            "temperature": temp,
            "n_runs": len(g),
            "n_errors": int(((df.task_id == task_id) & (df.temperature == temp)
                             & df.error.notna()).sum()),
            "tool_seq_exact_rate": mode_rate(seqs),
            "tool_seq_edit_dist": mean_pairwise(seqs, norm_lev),
            "tool_args_exact_rate": mode_rate(seqs_args),
            "tool_set_jaccard": mean_pairwise(seqs, jaccard),
            "n_calls_std": float(np.std([len(s) for s in seqs])),
            "answer_exact_rate": mode_rate(answers),
            "answer_correct_rate": (np.mean([all(e in a for e in expected) for a in answers])
                                    if expected else np.nan),
            "answer_sem_sim": (semantic_similarity(list(g["final_output"]), embedder)
                               if embedder else np.nan),
        }
        out.append(row)

    summary = pd.DataFrame(out).sort_values(["task_id", "temperature"])
    RESULTS_DIR.mkdir(exist_ok=True)
    summary.to_csv(RESULTS_DIR / "per_task_summary.csv", index=False)

    metric_cols = [c for c in summary.columns
                   if c not in ("task_id", "category", "temperature", "n_runs", "n_errors")]
    by_cat = summary.groupby(["category", "temperature"])[metric_cols].mean().round(3)
    by_cat.to_csv(RESULTS_DIR / "by_category_summary.csv")

    pd.set_option("display.width", 200, "display.max_columns", None)
    print("\n=== Per task ===")
    print(summary.round(3).to_string(index=False))
    print("\n=== By category x temperature ===")
    print(by_cat)


if __name__ == "__main__":
    main()
