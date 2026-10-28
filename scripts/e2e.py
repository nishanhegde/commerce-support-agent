"""Day 50: run the end-to-end scenario against Claude and judge it by stage.

Run: uv run python scripts/e2e.py            (reviewer approves)
     uv run python scripts/e2e.py --reject   (reviewer says no)

Live, so results vary run to run; failures.md is where the judged outcomes go.
"""

import argparse

from dotenv import load_dotenv

from support_agent.audit import format_row
from support_agent.e2e import TURNS, judge, run_scenario
from support_agent.tracing import RunTrace, format_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reject", action="store_true")
    args = parser.parse_args()
    load_dotenv()

    trace = RunTrace("e2e")
    run = run_scenario(approve=not args.reject, trace=trace)

    for text, reply in zip(TURNS, run.replies):
        print(f"customer> {text}\n\nagent> {reply}\n")
    print("--- audit trail")
    for row in run.audit_rows:
        print(format_row(row))
    print("\n--- cost and latency")
    print(format_summary(trace.summary()))
    print("\n--- stages")
    results = judge(run, approved=not args.reject)
    for stage, ok, evidence in results:
        print(f"{'PASS' if ok else 'FAIL'}  {stage:<12} {evidence}")
    print(f"\n{sum(ok for _, ok, _ in results)}/{len(results)} stages passed")


if __name__ == "__main__":
    main()
