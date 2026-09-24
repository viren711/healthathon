#!/usr/bin/env python3
"""Stage 6 -- does retrieval work across the WHOLE corpus, not just the tongue?

`04_evaluate.py` measures Recall@k precisely, but only on tongue cancer,
because precise measurement needs PDF page numbers read off the source by
hand. That does not scale to 77 pathologies, so retrieval quality over the
rest of both books was simply unmeasured -- an uncomfortable position for a
platform that answers across all of clinical oncology.

This uses weaker but automatable ground truth: a question about prostate
cancer must retrieve chunks tagged `gu_prostate`. The tags come from the
publishers' own chapter boundaries, so nothing here is a judgement call.

    python scripts/06_eval_coverage.py
    python scripts/06_eval_coverage.py --scoped   # also test the section filter
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", default="eval/coverage.yaml")
    ap.add_argument("--out", default="eval/coverage_report.md")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--scoped",
        action="store_true",
        help="also check that restricting to the expected section still finds it",
    )
    args = ap.parse_args()

    from rag.config import load_config
    from rag.retriever import get_retriever

    cfg = load_config()
    items = yaml.safe_load((ROOT / args.questions).read_text(encoding="utf-8"))["questions"]
    retriever = get_retriever()
    scopes = retriever.available_scopes()

    unknown = sorted(
        {i.get("expect") or i.get("expect_system") for i in items} - set(scopes) - {None}
    )
    if unknown:
        print(f"!! coverage.yaml names unknown scopes: {unknown}", file=sys.stderr)
        return 1

    print(f"{len(items)} questions across {len({i.get('expect') or i.get('expect_system') for i in items})} scopes")
    print(f"index: {retriever.meta['chunk_count']:,} chunks\n")

    queries = [i["q"] for i in items]
    vecs = retriever.embed_query_batch(queries)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        all_hits = list(
            pool.map(
                lambda pair: retriever.search(pair[0], pair[1].reshape(1, -1)),
                zip(queries, vecs),
            )
        )

    top1 = hit_any = 0
    rows: list[str] = []
    misses: list[tuple[str, str, str]] = []
    by_system = Counter()
    by_system_ok = Counter()

    for item, hits in zip(items, all_hits):
        want = item.get("expect")
        want_sys = item.get("expect_system")
        key = "pathology" if want else "organ_system"
        target = want or want_sys

        got = [h.chunk.get(key) for h in hits]
        in_top1 = bool(got) and got[0] == target
        in_any = target in got
        top1 += in_top1
        hit_any += in_any
        system = target if want_sys else None
        by_system[system or target] += 1
        if in_any:
            by_system_ok[system or target] += 1

        mark = "top-1" if in_top1 else ("in top-5" if in_any else "MISS")
        rows.append(
            f"| {item['q'][:58]} | `{target}` | {mark} | "
            f"`{got[0] if got else '-'}` | {hits[0].score:.2f} |"
            if hits
            else f"| {item['q'][:58]} | `{target}` | MISS | - | - |"
        )
        if not in_any:
            misses.append((item["q"], target, got[0] if got else "-"))
        print(f"  [{mark:>8}] {item['q'][:56]:<58} -> {got[0] if got else '-'}")

    n = len(items)
    print(f"\n  correct section in top-1 : {top1}/{n} ({top1/n:.0%})")
    print(f"  correct section in top-5 : {hit_any}/{n} ({hit_any/n:.0%})")

    scoped_ok = None
    if args.scoped:
        print("\n== scoped retrieval (filter to the expected section) ==")
        scoped_ok = 0
        for item in items:
            target = item.get("expect") or item.get("expect_system")
            hits = retriever.search(item["q"], scope=target)
            ok = bool(hits) and all(
                (h.chunk.get("pathology") == target or h.chunk.get("organ_system") == target)
                for h in hits
            )
            scoped_ok += ok
            if not ok:
                print(f"  [ FAIL ] {item['q'][:56]}")
        print(f"  scoped search stayed inside its section: {scoped_ok}/{n}")

    lines = [
        "# Cross-corpus retrieval coverage",
        "",
        f"- Index: `{retriever.meta['embedding_model']}`, "
        f"{retriever.meta['chunk_count']:,} chunks",
        f"- Reranker: `{cfg['reranker']['model']}`, "
        f"rerank_candidates={cfg['retrieval']['rerank_candidates']}",
        "",
        "Ground truth is the organ-system tag, not a page number: a prostate "
        "question must retrieve chunks tagged `gu_prostate`. Weaker than "
        "Recall@k, but it covers the whole corpus instead of one disease.",
        "",
        f"- **Correct section in top-1: {top1}/{n} ({top1/n:.0%})**",
        f"- **Correct section within top-5: {hit_any}/{n} ({hit_any/n:.0%})**",
    ]
    if scoped_ok is not None:
        lines.append(f"- Scoped search stayed inside its section: {scoped_ok}/{n}")
    lines += [
        "",
        "| question | expected section | result | top hit | score |",
        "|---|---|---|---|---|",
        *rows,
        "",
    ]
    if misses:
        lines += ["## Misses", "", *[f"- `{q}` expected `{t}`, got `{g}`" for q, t, g in misses], ""]

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out}")

    # A platform claiming corpus-wide coverage should land the right section
    # for the great majority of plainly-worded questions.
    return 0 if hit_any / n >= 0.9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
