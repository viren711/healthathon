#!/usr/bin/env python3
"""Stage 4 -- measure retrieval quality and calibrate the refusal thresholds.

Runs three checks, one per obligation the platform has:

  retrieval  -- for answerable questions, does the right page come back?
  abstention -- for tongue-cancer questions the books do not cover, do we
                refuse? (swept across candidate thresholds)
  refusal    -- for off-topic questions, does the topic gate reject them?

Writes eval/report.md with a recommended min_rerank_score. Retrieval and the
topic gate need no LLM; pass --with-llm to also exercise generation and the
citation check end to end.

Retrieval for the 43 evaluation questions is embedded in one batch and then
searched across a thread pool, so a full run is seconds rather than minutes.
Because swapping the cross-encoder does not invalidate the index, --reranker
lets two candidate rerankers be compared against the same index, in parallel.

    python scripts/04_evaluate.py
    python scripts/04_evaluate.py --apply          # write thresholds back
    python scripts/04_evaluate.py --reranker BAAI/bge-reranker-base
    python scripts/04_evaluate.py --with-llm
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag import guard  # noqa: E402
from rag.config import load_config  # noqa: E402


def page_hit(hit, expect: dict[str, list[int]]) -> bool:
    """Did this chunk come from one of the pages the answer should live on?"""
    book = (hit.chunk.get("book_short") or "").lower()
    wanted = expect.get(book)
    if not wanted:
        return False
    start = hit.chunk.get("pdf_page_start", -1)
    end = hit.chunk.get("pdf_page_end", start)
    # Allow +/-1 page: a section legitimately straddles a page break.
    return any(start - 1 <= p <= end + 1 for p in wanted)


def apply_thresholds(config_path: Path, values: dict[str, float]) -> list[str]:
    """Rewrite calibrated threshold values in rag.yaml, in place.

    Edits the individual lines rather than round-tripping the YAML, because
    the comments in that file carry the reasoning for every threshold and
    PyYAML would discard all of them.
    """
    text = config_path.read_text(encoding="utf-8")
    changed: list[str] = []
    for key, value in values.items():
        pattern = re.compile(rf"^(\s*){re.escape(key)}:\s*[-+0-9.eE]+\s*$", re.MULTILINE)
        match = pattern.search(text)
        if not match:
            changed.append(f"{key}: NOT FOUND in {config_path.name}")
            continue
        old = match.group(0).strip()
        new = f"{match.group(1)}{key}: {value}"
        if old == new.strip():
            changed.append(f"{key}: unchanged ({value})")
            continue
        text = text[: match.start()] + new + text[match.end() :]
        changed.append(f"{old}  ->  {key}: {value}")
    config_path.write_text(text, encoding="utf-8")
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--with-llm", action="store_true", help="also test generation")
    ap.add_argument("--questions", default="eval/questions.yaml")
    ap.add_argument("--out", default="eval/report.md")
    ap.add_argument(
        "--reranker",
        help="cross-encoder to evaluate instead of the configured one "
        "(no reindex needed -- rerankers are independent of the embeddings)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="threads used for retrieval (default: 4)",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="write the calibrated thresholds back into config/rag.yaml",
    )
    args = ap.parse_args()

    # Must be set before the first load_config(), which is lru_cached.
    if args.reranker:
        os.environ["RERANKER_MODEL"] = args.reranker

    qs = yaml.safe_load((ROOT / args.questions).read_text(encoding="utf-8"))
    cfg = load_config()

    from rag.retriever import get_retriever

    retriever = get_retriever()
    print(f"index: {retriever.meta['chunk_count']} chunks, {retriever.meta['embedding_model']}")

    lines: list[str] = [
        "# RAG evaluation report",
        "",
        f"- Index: `{retriever.meta['embedding_model']}`, "
        f"{retriever.meta['chunk_count']} chunks, built {retriever.meta.get('built_at')}",
        f"- Reranker: `{cfg['reranker']['model']}` "
        f"({'enabled' if cfg['reranker']['enabled'] else 'disabled'})",
        f"- final_top_k: {cfg['retrieval']['final_top_k']}",
        "",
    ]

    # ---------------- retrieval on answerable questions --------------------
    print("\n== retrieval ==")
    recall_flags: list[bool] = []
    rr: list[float] = []
    answerable_top: list[float] = []
    rows: list[str] = []

    # One encode call for every question instead of 25 -- the per-call
    # overhead dominates at this batch size.
    answerable_qs = [i["q"] for i in qs["answerable"]]
    t_retr = time.time()
    answerable_vecs = retriever.embed_query_batch(answerable_qs)

    def search_one(pair):
        q, vec = pair
        return retriever.search(q, vec.reshape(1, -1))

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        all_hits = list(pool.map(search_one, zip(answerable_qs, answerable_vecs)))
    retr_ms = (time.time() - t_retr) * 1000 / max(1, len(answerable_qs))

    for item, hits in zip(qs["answerable"], all_hits):
        expect = {k.lower(): v for k, v in (item.get("expect_pages") or {}).items()}
        ranks = [i for i, h in enumerate(hits, 1) if page_hit(h, expect)]
        found = bool(ranks)
        recall_flags.append(found)
        rr.append(1.0 / ranks[0] if ranks else 0.0)
        # An empty hit list is possible (an empty index, or every candidate
        # filtered out); formatting hits[0] unconditionally would crash here.
        top = f"{hits[0].score:.3f}" if hits else "-"
        if hits:
            answerable_top.append(hits[0].score)
        rows.append(
            f"| {item['q'][:70]} | {'yes' if found else 'NO'} | "
            f"{ranks[0] if ranks else '-'} | {top} |"
        )
        print(f"  [{'ok' if found else '  '}] {item['q'][:65]}")

    if not recall_flags:
        print("no answerable questions in the eval set", file=sys.stderr)
        return 1
    recall = sum(recall_flags) / len(recall_flags)
    mrr = statistics.mean(rr)
    print(f"  retrieval: {retr_ms:.0f} ms/question "
          f"({args.workers} threads, reranker={cfg['reranker']['model']})")
    lines += [
        "## Retrieval (answerable questions)",
        "",
        f"- **Recall@{cfg['retrieval']['final_top_k']}: {recall:.2%}** "
        f"({sum(recall_flags)}/{len(recall_flags)})",
        f"- **MRR: {mrr:.3f}**",
        f"- Retrieval latency: **{retr_ms:.0f} ms/question** "
        f"(embed + FAISS + BM25 + rerank, {args.workers} threads)",
        "",
        "| question | expected page retrieved | rank | top score |",
        "|---|---|---|---|",
        *rows,
        "",
    ]

    # ---------------- topic gate on off-topic questions --------------------
    print("\n== topic gate (off-topic must be refused) ==")
    refused, leaks = 0, []
    for item in qs["off_topic"]:
        res = guard.topic_gate(item["q"])
        if res.allowed:
            leaks.append((item["q"], res.detail))
            print(f"  [LEAK] {item['q']}  {res.detail}")
        else:
            refused += 1
            print(f"  [ok  ] {item['q']}")

    lines += [
        "## Topic gate (off-topic questions)",
        "",
        f"- **Refusal accuracy: {refused}/{len(qs['off_topic'])}**",
        "",
    ]
    if leaks:
        lines += ["Leaked through (must be fixed -- these would reach the LLM):", ""]
        lines += [f"- `{q}` -> {d}" for q, d in leaks]
        lines.append("")

    # also confirm the gate does NOT reject legitimate questions
    false_refusals = [
        item["q"] for item in qs["answerable"] if not guard.topic_gate(item["q"]).allowed
    ]
    lines += [
        f"- False refusals on answerable questions: "
        f"**{len(false_refusals)}/{len(qs['answerable'])}**",
        "",
    ]
    for q in false_refusals:
        print(f"  [FALSE REFUSAL] {q}")

    # In-scope-but-not-tongue: covered by the books, so the gate must pass
    # them. This bucket is what catches a scope setting that is narrower than
    # the corpus -- the failure mode where the index holds the answer and the
    # topic gate refuses to look.
    other = qs.get("in_scope_other") or []
    other_refused = []
    if other:
        print("\n== topic gate (in-scope, non-tongue -- must NOT be refused) ==")
        for item in other:
            res = guard.topic_gate(item["q"])
            if res.allowed:
                print(f"  [ok  ] {item['q'][:62]}")
            else:
                other_refused.append((item["q"], res.reason))
                print(f"  [REFUSED] {item['q'][:56]}  <- {res.reason}")
        lines += [
            "## Topic gate (in scope, outside the tongue chapters)",
            "",
            f"- Scope setting: `{cfg['guard']['topic'].get('scope', 'tongue')}`, "
            f"corpus mode: `{cfg.get('selection', {}).get('mode', 'tongue')}`",
            f"- **Wrongly refused: {len(other_refused)}/{len(other)}**",
            "",
        ]
        if other_refused:
            lines += [
                "These are covered by the indexed books but the topic gate turned "
                "them away. Either widen `guard.topic.scope` or accept the narrower "
                "product:",
                "",
                *[f"- `{q}` -> {r}" for q, r in other_refused],
                "",
            ]

    # ---------------- topic-gate calibration -------------------------------
    # The similarity thresholds are embedder-specific, so report the actual
    # distribution rather than leaving the configured values unexamined.
    print("\n== topic-gate calibration ==")
    import numpy as np

    from rag.guard import _seed_vectors

    seeds = _seed_vectors()

    def topic_scores(q: str) -> tuple[float, float]:
        v = retriever.embed_query(q)
        return retriever.centroid_similarity(v), float(np.max(seeds @ v[0]))

    in_scope = [topic_scores(i["q"]) for i in qs["answerable"]]
    out_scope = [topic_scores(i["q"]) for i in qs["off_topic"]]

    in_seed = [s for _, s in in_scope]
    out_seed = [s for _, s in out_scope]
    in_cent = [c for c, _ in in_scope]
    out_cent = [c for c, _ in out_scope]

    if not in_seed or not out_seed:
        print(
            "calibration needs a non-empty 'answerable' AND 'off_topic' bucket",
            file=sys.stderr,
        )
        return 1

    # Widest gap between the two distributions -> midpoint is the threshold.
    margin = min(in_seed) - max(out_seed)
    suggested_seed = round((min(in_seed) + max(out_seed)) / 2, 2)
    # The centroid check is an AND-ed fallback, so it must not be stricter
    # than the weakest in-scope question; back off slightly below that.
    suggested_centroid = round(max(0.0, min(in_cent) - 0.02), 2)

    lines += [
        "## Topic-gate calibration",
        "",
        "Cosine thresholds are specific to the embedding model. These are the",
        "distributions for this index:",
        "",
        "| bucket | seed similarity (min / median / max) | centroid (min / median / max) |",
        "|---|---|---|",
        f"| answerable (in scope) | {min(in_seed):.3f} / {statistics.median(in_seed):.3f} "
        f"/ {max(in_seed):.3f} | {min(in_cent):.3f} / {statistics.median(in_cent):.3f} "
        f"/ {max(in_cent):.3f} |",
        f"| off-topic | {min(out_seed):.3f} / {statistics.median(out_seed):.3f} "
        f"/ {max(out_seed):.3f} | {min(out_cent):.3f} / {statistics.median(out_cent):.3f} "
        f"/ {max(out_cent):.3f} |",
        "",
        f"- Separation margin: **{margin:+.3f}** "
        f"({'clean' if margin > 0 else 'OVERLAPPING -- the buckets are not linearly separable'})",
        f"- Currently configured `min_seed_similarity`: "
        f"`{cfg['guard']['topic']['min_seed_similarity']}`",
        f"- **Suggested `min_seed_similarity`: `{suggested_seed}`**"
        + ("" if margin > 0 else "  (no clean split -- rely on the lexicon rules)"),
        f"- **Suggested `min_centroid_similarity`: `{suggested_centroid}`** "
        f"(just below the weakest in-scope question, {min(in_cent):.3f})",
        "",
    ]
    print(f"  in-scope  seed: {min(in_seed):.3f} .. {max(in_seed):.3f}")
    print(f"  off-topic seed: {min(out_seed):.3f} .. {max(out_seed):.3f}")
    print(f"  margin {margin:+.3f} -> suggested min_seed_similarity = {suggested_seed}")

    # ---------------- threshold sweep for abstention -----------------------
    print("\n== threshold sweep (abstention vs answering) ==")
    unans_top: list[float] = []
    for item in qs["unanswerable"]:
        gate = guard.topic_gate(item["q"])
        if not gate.allowed:
            # Refused by the topic gate -- already abstaining, score is n/a.
            unans_top.append(float("-inf"))
            continue
        hits = retriever.search(item["q"])
        unans_top.append(hits[0].score if hits else float("-inf"))

    # Broad questions retrieve lower scores than precise ones, so leaving them
    # out of the sweep is what produced a threshold that refused "causes of
    # skin cancer" while answering "risk factors for basal cell carcinoma".
    broad = qs.get("answerable_broad") or []
    broad_top: list[float] = []
    if broad:
        broad_vecs = retriever.embed_query_batch([i["q"] for i in broad])
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            broad_hits = list(
                pool.map(
                    lambda pair: retriever.search(pair[0], pair[1].reshape(1, -1)),
                    zip([i["q"] for i in broad], broad_vecs),
                )
            )
        broad_top = [h[0].score for h in broad_hits if h]

    want_answered = answerable_top + broad_top
    candidates = sorted(
        {round(x, 2) for x in want_answered + [u for u in unans_top if u > -1e9]}
    )
    sweep_rows: list[str] = []
    best = None
    # The two populations OVERLAP -- measured on this corpus, answerable
    # questions run 0.30-8.77 and unanswerable ones reach 4.75 -- so no
    # threshold separates them and a symmetric objective just splits the
    # difference, refusing real questions to catch ones layer 3 would have
    # caught anyway. Layer 2 is a cheap pre-filter; layer 3 is the guarantee.
    # So: keep at least `min_answer_rate` of the answerable questions, and
    # among thresholds that do, take the one that abstains most.
    floor = load_config()["guard"]["grounding"].get("min_answer_rate", 0.95)
    for t in candidates:
        kept = sum(1 for s in want_answered if s >= t)
        abstained = sum(1 for s in unans_top if s < t)
        ans_rate = kept / len(want_answered) if want_answered else 0
        abs_rate = abstained / len(unans_top) if unans_top else 0
        sweep_rows.append(
            f"| {t:.2f} | {ans_rate:.0%} ({kept}/{len(want_answered)}) | "
            f"{abs_rate:.0%} ({abstained}/{len(unans_top)}) |"
            + (" **" if ans_rate >= floor else " ")
            + " |"
        )
        if ans_rate >= floor and (best is None or abs_rate > best[3]):
            best = (t, abs_rate, ans_rate, abs_rate)
    if best is None and candidates:
        # Nothing meets the floor; fall back to the most permissive value so
        # the platform errs towards answering and letting layer 3 verify.
        t = candidates[0]
        best = (t, 0.0, 1.0, sum(1 for s in unans_top if s < t) / max(1, len(unans_top)))

    lines += [
        "## Grounding threshold sweep",
        "",
        "`min_rerank_score` trades answering coverage against abstention. Each row",
        "shows what that threshold would do to both buckets.",
        "",
        f"Chosen as: the most abstaining threshold that still answers at least "
        f"{floor:.0%} of the {len(want_answered)} answerable questions "
        f"({len(answerable_top)} precise + {len(broad_top)} broad). Rows marked "
        f"** meet that floor.",
        "",
        "| threshold | answerable answered | unanswerable abstained | meets floor |",
        "|---|---|---|---|",
        *sweep_rows,
        "",
    ]

    if best:
        lines += [
            f"**Recommended `guard.grounding.min_rerank_score`: `{best[0]:.2f}`** "
            f"(answers {best[2]:.0%} of answerable, abstains on {best[3]:.0%} of unanswerable)",
            "",
        ]
        print(f"\n  recommended min_rerank_score = {best[0]:.2f}")
    else:
        lines += [
            "No candidate thresholds -- every question was refused before retrieval, "
            "so there is nothing to calibrate.",
            "",
        ]

    # ---------------- write the calibration back ---------------------------
    # The thresholds are embedder- and reranker-specific, so leaving them as a
    # copy-paste instruction is how they end up stale. --apply closes the loop.
    if args.apply:
        proposed = {}
        if margin > 0:
            # Only meaningful when the two buckets actually separate. When they
            # overlap, the "midpoint" is a number between two distributions that
            # cross, and writing it would trade false refusals for leaks while
            # looking principled. The lexicon rules in guard.py carry the gate
            # in that case, and the configured values are left alone.
            proposed["min_seed_similarity"] = suggested_seed
            proposed["min_centroid_similarity"] = suggested_centroid
        else:
            print(
                "  topic-gate thresholds NOT written: the in-scope and off-topic "
                "seed distributions overlap "
                f"(in-scope min {min(in_seed):.3f} < off-topic max {max(out_seed):.3f}), "
                "so no single cutoff separates them. The lexicon rules handle "
                f"this bucket -- {refused}/{len(qs['off_topic'])} refused, "
                f"{len(false_refusals)}/{len(qs['answerable'])} false refusals."
            )
        if best:
            proposed["min_rerank_score"] = round(best[0], 2)
        changes = (
            apply_thresholds(ROOT / "config" / "rag.yaml", proposed) if proposed else []
        )
        print("\n== applied to config/rag.yaml ==")
        for c in changes:
            print(f"  {c}")
        lines += [
            "### Applied",
            "",
            "`--apply` wrote these into `config/rag.yaml`:",
            "",
            *[f"- `{c}`" for c in changes],
            "",
        ]
    elif best:
        lines += ["Re-run with `--apply` to write these into `config/rag.yaml`.", ""]

    if answerable_top:
        lines += [
            f"- Answerable top-score range: {min(answerable_top):.3f} .. {max(answerable_top):.3f} "
            f"(median {statistics.median(answerable_top):.3f})",
        ]
    finite_unans = [u for u in unans_top if u > -1e9]
    if finite_unans:
        lines += [
            f"- Unanswerable top-score range: {min(finite_unans):.3f} .. {max(finite_unans):.3f} "
            f"(median {statistics.median(finite_unans):.3f})",
            "",
        ]

    # ---------------- optional end-to-end with the LLM ---------------------
    if args.with_llm:
        print("\n== end-to-end with the LLM ==")
        from rag.pipeline import answer_question

        e2e = {"answered": 0, "not_available": 0, "out_of_scope": 0, "error": 0}
        detail_rows = []
        for bucket in ("answerable", "answerable_broad", "in_scope_other",
                       "unanswerable", "off_topic"):
            if not qs.get(bucket):
                continue
            for item in qs[bucket]:
                ans = answer_question(item["q"])
                e2e[ans.status] = e2e.get(ans.status, 0) + 1
                want = {
                    "answerable": "answered",
                    "answerable_broad": "answered",
                    "in_scope_other": "answered",
                    "unanswerable": "not_available",
                    "off_topic": "out_of_scope",
                }[bucket]
                # in_scope_other only has to avoid being refused as off-scope;
                # abstaining is acceptable there, answering is a bonus.
                ok = (
                    ans.status != "out_of_scope"
                    if bucket == "in_scope_other"
                    else ans.status == want
                )
                detail_rows.append(
                    f"| {bucket} | {item['q'][:60]} | {ans.status} | {'ok' if ok else 'MISS'} |"
                )
                print(f"  [{'ok  ' if ok else 'MISS'}] {bucket}: {item['q'][:55]} -> {ans.status}")

        lines += [
            "## End-to-end (with the LLM)",
            "",
            "| bucket | question | status | expected? |",
            "|---|---|---|---|",
            *detail_rows,
            "",
        ]

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out}")

    # Non-zero exit when a hard requirement fails, so CI can gate on it.
    return 0 if not leaks and not other_refused and recall >= 0.8 else 1


if __name__ == "__main__":
    raise SystemExit(main())
