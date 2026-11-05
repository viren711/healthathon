#!/usr/bin/env python3
"""Run a file of queries through the platform and report what happened.

Feeds eval/queries.txt (or any file of one-query-per-line, #-comments ignored)
through the same `answer_question` the portal uses, and prints one line per
query with its status and latency. Useful for checking a config change across
the whole query set without clicking through the UI.

    python scripts/ask_batch.py                       # eval/queries.txt, Ask mode
    python scripts/ask_batch.py --locate              # Find mode: no LLM, fast
    python scripts/ask_batch.py --limit 10 --out results.md

Ask mode costs one LLM call per query that clears both gates, so the full
50-query file takes a while on CPU. --locate needs no LLM at all.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def read_queries(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("queries", nargs="?", default="eval/queries.txt")
    ap.add_argument("--limit", type=int, help="only run the first N queries")
    ap.add_argument("--out", help="also write a markdown report here")
    ap.add_argument(
        "--locate",
        action="store_true",
        help="run the topic locator instead of question answering (no LLM)",
    )
    args = ap.parse_args()

    qpath = Path(args.queries)
    if not qpath.is_absolute():
        qpath = ROOT / qpath
    if not qpath.exists():
        print(f"no such file: {qpath}", file=sys.stderr)
        return 1

    queries = read_queries(qpath)
    if args.limit:
        queries = queries[: args.limit]
    if not queries:
        print(f"{qpath} contains no queries", file=sys.stderr)
        return 1

    print(f"{len(queries)} queries from {qpath.name}"
          f"{' (locate mode)' if args.locate else ''}\n")

    rows: list[str] = []
    counts: dict[str, int] = {}
    t_all = time.time()

    if args.locate:
        from rag.locate import locate

        for i, q in enumerate(queries, 1):
            t = time.time()
            hits = locate(q, limit=5)
            ms = int((time.time() - t) * 1000)
            top = hits[0] if hits else None
            status = "found" if hits else "none"
            counts[status] = counts.get(status, 0) + 1
            where = f"{top.book_short} {top.page_label}" if top else "-"
            print(f"[{i:>2}] {status:<6} {ms:>6}ms  {len(hits)} places  {where:<28} {q[:44]}")
            rows.append(f"| {q} | {status} | {len(hits)} | {where} | {ms} ms |")
        header = "| query | result | places | top location | time |"
    else:
        from rag.llm import LLMClient
        from rag.pipeline import answer_question

        client = LLMClient.from_config()
        ok, detail = client.health()
        if not ok:
            print(f"!! LLM backend unreachable ({detail}).", file=sys.stderr)
            print("   Start it with: bash scripts/serve_llm.sh", file=sys.stderr)
            print("   (or use --locate, which needs no model)", file=sys.stderr)
            return 1

        for i, q in enumerate(queries, 1):
            ans = answer_question(q, client=client)
            counts[ans.status] = counts.get(ans.status, 0) + 1
            cites = ",".join(str(c) for c in ans.cited) or "-"
            print(f"[{i:>2}] {ans.status:<14} {ans.elapsed_ms:>6}ms  cites[{cites}]  {q[:48]}")
            rows.append(
                f"| {q} | {ans.status} | {cites} | {len(ans.sources)} | {ans.elapsed_ms} ms |"
            )
        header = "| query | status | cited | sources | time |"

    elapsed = time.time() - t_all
    print(f"\n{'-' * 60}")
    for status, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {status:<16} {n:>3}")
    print(f"  {'total time':<16} {elapsed:>6.0f}s  ({elapsed / len(queries):.1f}s/query)")

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "\n".join(
                [
                    f"# Batch run of `{qpath.name}`",
                    "",
                    f"- Mode: {'locate' if args.locate else 'ask'}",
                    f"- {len(queries)} queries in {elapsed:.0f}s",
                    "- " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())),
                    "",
                    header,
                    "|" + "---|" * (header.count("|") - 1),
                    *rows,
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
