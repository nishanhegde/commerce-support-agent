"""Day 50: run the end-to-end scenario against Claude and judge it by stage.

Run: uv run python scripts/e2e.py            (reviewer approves)
     uv run python scripts/e2e.py --reject   (reviewer says no)

Live, so results vary run to run; failures.md is where the judged outcomes go.
"""

import argparse

from dotenv import load_dotenv

from support_agent.audit import format_row
from support_agent.e2e import TURNS, judge, run_scenario


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reject", action="store_true")
    args = parser.parse_args()
    load_dotenv()

    run = run_scenario(approve=not args.reject)

    for text, reply in zip(TURNS, run.replies):
        print(f"customer> {text}\n\nagent> {reply}\n")
    print("--- audit trail")
    for row in run.audit_rows:
        print(format_row(row))
    print("\n--- stages")
    results = judge(run, approved=not args.reject)
    for stage, ok, evidence in results:
        print(f"{'PASS' if ok else 'FAIL'}  {stage:<12} {evidence}")
    print(f"\n{sum(ok for _, ok, _ in results)}/{len(results)} stages passed")


if __name__ == "__main__":
    main()
