"""Run each task N times at each temperature and log every run to JSONL.

Usage:
  export ANTHROPIC_API_KEY=...
  python harness.py --runs 3 --temps 0.0 1.0 --tasks L1 M1     # smoke test
  python harness.py --runs 25 --temps 0.0 1.0                  # full run

Logs go to data/<task_id>__temp<T>.jsonl (one line per run). Re-running resumes:
existing runs are counted and only the missing ones are executed.
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent

from tasks import TASKS
from tools import ALL_TOOLS

DATA_DIR = Path("data")
DEFAULT_MODEL = os.environ.get("MODEL", "claude-sonnet-5-5")
SYSTEM_PROMPT = "You are a helpful assistant. Use the available tools when they help."


def build_agent(model: str, temperature: float):
    llm = ChatAnthropic(model=model, temperature=temperature, max_tokens=1024)
    return create_react_agent(llm, ALL_TOOLS, prompt=SYSTEM_PROMPT)


def text_of(content) -> str:
    """Message content may be a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content
                   if isinstance(b, dict) and b.get("type") == "text")


def run_once(agent, task: dict, run_idx: int, model: str, temperature: float) -> dict:
    t0 = time.time()
    record = {
        "task_id": task["id"], "category": task["category"], "run_idx": run_idx,
        "model": model, "temperature": temperature,
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


def count_lines(path: Path) -> int:
    return sum(1 for _ in path.open()) if path.exists() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--runs", type=int, default=20, help="runs per task per temperature")
    ap.add_argument("--temps", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--tasks", nargs="*", help="only these task ids")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    tasks = [t for t in TASKS if not args.tasks or t["id"] in args.tasks]

    for temp in args.temps:
        agent = build_agent(args.model, temp)
        for task in tasks:
            path = DATA_DIR / f"{task['id']}__temp{temp}.jsonl"
            start = count_lines(path)
            todo = range(start, args.runs)
            if not todo:
                print(f"[skip] {path.name} already has {start} runs")
                continue
            print(f"[run ] {task['id']} temp={temp}: runs {start}..{args.runs - 1}")
            with ThreadPoolExecutor(max_workers=args.workers) as ex, path.open("a") as f:
                futs = [ex.submit(run_once, agent, task, i, args.model, temp) for i in todo]
                for fut in as_completed(futs):
                    f.write(json.dumps(fut.result()) + "\n")
                    f.flush()


if __name__ == "__main__":
    main()
