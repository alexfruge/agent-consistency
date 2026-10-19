"""Run each task N times per sampling condition and log every run to JSONL.

Usage:
  python harness.py --runs 3 --tasks L1 M1                     # smoke test, model default sampling
  python harness.py --runs 25                                  # full sweep, default sampling
  python harness.py --runs 25 --temps 0.0 1.0 --model <model>  # for models that allow temperature

--temps takes numbers or the word "default" (= don't pass temperature at all).
Some models (e.g. Sonnet 5.5) only allow their default, so "default" is the fallback.

API key: set ANTHROPIC_API_KEY in your shell, or put it in a .env file next to this script:
  ANTHROPIC_API_KEY=sk-ant-...

Logs go to data/<task_id>__temp<label>.jsonl (one line per run). Re-running resumes:
failed runs are dropped from the file and redone, successful runs are kept.
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage

from tasks import TASKS
from tools import ALL_TOOLS

load_dotenv(override=True)  # .env wins over any stale key already in the shell

DATA_DIR = Path("data")
DEFAULT_MODEL = os.environ.get("MODEL", "claude-sonnet-5-5")
SYSTEM_PROMPT = "You are a helpful assistant. Use the available tools when they help."


def make_llm(model: str, temperature):
    kwargs = {"model": model, "max_tokens": 1024}
    if temperature is not None:  # omit entirely for "default"
        kwargs["temperature"] = temperature
    return ChatAnthropic(**kwargs)


def preflight(model: str, temperature) -> str | None:
    """One tiny call to surface auth / unsupported-parameter errors immediately."""
    try:
        make_llm(model, temperature).invoke("Reply with OK.")
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def text_of(content) -> str:
    """Message content may be a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content
                   if isinstance(b, dict) and b.get("type") == "text")


def run_once(agent, task: dict, run_idx: int, model: str, label: str) -> dict:
    t0 = time.time()
    record = {
        "task_id": task["id"], "category": task["category"], "run_idx": run_idx,
        "model": model, "temperature": label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": task["prompt"],
    }
    try:
        result = agent.invoke({"messages": [("user", task["prompt"])]},
                              {"recursion_limit": 25})
        msgs = result["messages"]
        tool_calls = [{"name": tc["name"], "args": tc["args"]}
                      for m in msgs if isinstance(m, AIMessage) for tc in m.tool_calls]
        record.update(tool_calls=tool_calls, final_output=text_of(msgs[-1].content),
                      error=None)
    except Exception as e:  # log failures rather than crash the whole sweep
        record.update(tool_calls=[], final_output="", error=f"{type(e).__name__}: {e}")
    record["latency_s"] = round(time.time() - t0, 3)
    return record


def keep_good_runs(path: Path) -> list[dict]:
    """Load existing runs, drop errored ones, and rewrite the file with only the good ones."""
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.open() if line.strip()]
    good = [r for r in rows if not r.get("error")]
    if len(good) != len(rows):
        with path.open("w") as f:
            f.writelines(json.dumps(r) + "\n" for r in good)
        print(f"       dropped {len(rows) - len(good)} errored runs from {path.name}")
    return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--runs", type=int, default=20, help="successful runs per task per condition")
    ap.add_argument("--temps", nargs="+", default=["default"],
                    help='temperatures, or "default" to leave unset')
    ap.add_argument("--tasks", nargs="*", help="only these task ids")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    key = os.environ.get("ANTHROPIC_API_KEY", "")
    print(f"Using ANTHROPIC_API_KEY: {key[:14]}... (length {len(key)})" if key
          else "ANTHROPIC_API_KEY is not set!")

    DATA_DIR.mkdir(exist_ok=True)
    tasks = [t for t in TASKS if not args.tasks or t["id"] in args.tasks]

    for label in args.temps:
        temp = None if label == "default" else float(label)
        label = "default" if temp is None else str(temp)

        err = preflight(args.model, temp)
        if err:
            print(f"[FAIL] preflight for temperature={label}: {err}")
            print("       skipping this condition.")
            continue

        agent = create_agent(make_llm(args.model, temp), ALL_TOOLS, system_prompt=SYSTEM_PROMPT)
        for task in tasks:
            path = DATA_DIR / f"{task['id']}__temp{label}.jsonl"
            good = keep_good_runs(path)
            used = {r["run_idx"] for r in good}
            todo = [i for i in range(args.runs) if i not in used]
            if not todo:
                print(f"[skip] {path.name} already has {len(good)} good runs")
                continue
            print(f"[run ] {task['id']} temperature={label}: {len(todo)} runs")
            n_err = 0
            with ThreadPoolExecutor(max_workers=args.workers) as ex, path.open("a") as f:
                futs = [ex.submit(run_once, agent, task, i, args.model, label) for i in todo]
                for fut in as_completed(futs):
                    rec = fut.result()
                    n_err += bool(rec["error"])
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
            if n_err:
                print(f"       {n_err} runs errored (re-run to retry them)")


if __name__ == "__main__":
    main()