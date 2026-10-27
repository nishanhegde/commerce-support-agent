"""Day 49: print a thread's audit trail. Every node run, and every approval.

Run: uv run python scripts/audit.py --thread a1
     uv run python scripts/audit.py --approvals          (all threads)
"""

import argparse

from support_agent.audit import AUDIT_PATH, AuditLog, format_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread")
    parser.add_argument("--approvals", action="store_true", help="approval rows only")
    args = parser.parse_args()

    with AuditLog(AUDIT_PATH) as audit:
        rows = audit.rows(args.thread, "approval" if args.approvals else None)
    for row in rows:
        print(format_row(row))
    if not rows:
        print("no audit rows" + (f" for thread {args.thread!r}" if args.thread else ""))


if __name__ == "__main__":
    main()
