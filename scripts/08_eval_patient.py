#!/usr/bin/env python3
"""Stage 8 -- evaluate the Patient Portal against eval/patient_questions.yaml.

No language model needed. Measures, on the patient index:

  retrieval     Recall@k: does the guide page that answers each answerable
                question appear in the top k?
  topic gate    off-topic questions refused, answerable ones admitted
  grounding     not-covered questions refused at the evidence threshold, and
                where the threshold should sit (printed; --apply writes it to
                config/patient.yaml)
  safety        every safety case routed to the right category, and the
                ordinary questions not flagged at all

All questions are embedded in one batch, and retrievals run in a thread pool
(torch releases the GIL inside its BLAS calls, so they overlap on the cores).

    python scripts/08_eval_patient.py
    python scripts/08_eval_patient.py --apply      # write the calibrated threshold
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag import guard  # noqa: E402
from rag.config import load_patient_config  # noqa: E402
from rag.patient_safety import screen  # noqa: E402
from rag.pipeline import _search_query  # noqa: E402


def page_hit(chunk: dict, expect: dict[str, list[int]]) -> bool:
    pages = expect.get(chunk.get("guide_id"))
    if not pages:
        return False
    start = chunk.get("pdf_page_start", -1)
    end = chunk.get("pdf_page_end", start)
    return any(start - 1 <= p <= end + 1 for p in pages)


def calibrate(answerable: list[float], refuse: list[float], current: float) -> tuple[float, str]:
    """Where the grounding threshold should sit, and why.

    On a set this small the useful fact is the GAP between the best
    not-covered score and the worst answerable one: any threshold inside it
    makes no errors here. If the configured value is already inside, keep it
    (it was chosen on the larger textbook set and moving it buys nothing
    measurable); otherwise move to the middle of the gap, as far from both
    sides as this evidence allows. If the two overlap, minimise errors, with
    ties going to the LOWER value -- refusing a patient's real question costs
    more than one more question reaching the citation check.
    """
    lo, hi = max(refuse), min(answerable)
    if lo < hi:
        if lo < current < hi:
            return current, f"keep {current:.2f}: inside the separating gap ({lo:.2f}, {hi:.2f})"
        mid = round((lo + hi) / 2, 2)
        return mid, f"move to {mid:.2f}: middle of the separating gap ({lo:.2f}, {hi:.2f})"
    best = None
    for t in sorted(set(round(x, 2) for x in answerable + refuse)):
        errors = sum(s < t for s in answerable) + sum(s >= t for s in refuse)
        if best is None or errors < best[1]:
            best = (t, errors)
    return best[0], f"scores overlap; {best[0]:.2f} makes {best[1]} errors on this set"


def apply_threshold(value: float) -> None:
    path = ROOT / "config" / "patient.yaml"
    text = path.read_text(encoding="utf-8")
    new, n = re.subn(
        r"(^\s*min_rerank_score:\s*)[-\d.]+", rf"\g<1>{value:.2f}", text, count=1, flags=re.M
    )
    if n != 1:
        raise SystemExit("could not find min_rerank_score in config/patient.yaml")
    path.write_text(new, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--apply", action="store_true", help="write the calibrated threshold")
    ap.add_argument("--out", default="eval/patient_report.md")
    args = ap.parse_args()

    data = yaml.safe_load((ROOT / "eval" / "patient_questions.yaml").read_text(encoding="utf-8"))
    pcfg = load_patient_config()

    from rag.retriever import get_retriever

    t0 = time.time()
    retriever = get_retriever("patient")
    print(f"patient index: {retriever.meta['chunk_count']:,} chunks, "
          f"{len(retriever.meta.get('books', []))} guides  (loaded in {time.time() - t0:.1f}s)")

    answerable = data["answerable"]
    not_covered = data["not_covered"]
    off_topic = data["off_topic"]
    questions = [a["q"] for a in answerable] + not_covered + off_topic
    # Searched exactly as the app searches: everyday words expanded.
    searched = [_search_query(q, "patient") for q in questions]
    vecs = retriever.embed_query_batch(searched)

    def run(i: int):
        q = questions[i]
        vec = vecs[i : i + 1]
        topic = guard.topic_gate(q, vec, corpus="patient")
        hits = retriever.search(searched[i], vec)
        return topic, hits

    t1 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run, range(len(questions))))
    print(f"retrieved {len(questions)} questions in {time.time() - t1:.1f}s on {args.workers} threads")

    lines: list[str] = ["# Patient Portal evaluation", ""]
    failures = 0

    # --- retrieval ------------------------------------------------------------
    n_a = len(answerable)
    found = 0
    a_scores: list[float] = []
    a_topic_refused = []
    lines += [f"## Retrieval (Recall@{args.k})", ""]
    for i, item in enumerate(answerable):
        topic, hits = results[i]
        top = hits[: args.k]
        ok = any(page_hit(h.chunk, item["expect"]) for h in top)
        found += ok
        a_scores.append(hits[0].score if hits else -99)
        if not topic.allowed:
            a_topic_refused.append(item["q"])
        best = top[0].citation if top else "-"
        lines.append(f"- [{'ok' if ok else 'MISS'}] {item['q']}  \n  top: {best} ({a_scores[-1]:.2f})")
    recall = found / n_a if n_a else 0.0
    lines += ["", f"**Recall@{args.k}: {recall:.0%} ({found}/{n_a})**", ""]
    print(f"Recall@{args.k}: {recall:.0%} ({found}/{n_a})")
    if recall < 0.8:
        failures += 1

    # --- topic gate -------------------------------------------------------------
    off_start = n_a + len(not_covered)
    leaks = [off_topic[j] for j in range(len(off_topic)) if results[off_start + j][0].allowed]
    lines += ["## Topic gate", "",
              f"- off-topic refused: {len(off_topic) - len(leaks)}/{len(off_topic)}",
              f"- answerable wrongly refused: {len(a_topic_refused)}/{n_a}"]
    lines += [f"  - LEAK: {q}" for q in leaks] + [f"  - REFUSED: {q}" for q in a_topic_refused] + [""]
    print(f"topic gate: off-topic refused {len(off_topic) - len(leaks)}/{len(off_topic)}, "
          f"answerable refused {len(a_topic_refused)}/{n_a}")
    failures += bool(leaks) + bool(a_topic_refused)

    # --- grounding threshold ----------------------------------------------------
    nc_scores = [results[n_a + j][1][0].score if results[n_a + j][1] else -99
                 for j in range(len(not_covered))]
    threshold = pcfg["guard"]["grounding"]["min_rerank_score"]
    best_t, why = calibrate(a_scores, nc_scores, threshold)
    a_below = sum(s < threshold for s in a_scores)
    nc_above = [q for q, s in zip(not_covered, nc_scores) if s >= threshold]
    lines += ["## Grounding gate", "",
              f"- answerable top score: {min(a_scores):.2f} .. {max(a_scores):.2f}",
              f"- not-covered top score: {min(nc_scores):.2f} .. {max(nc_scores):.2f}",
              f"- at the configured threshold {threshold}: {a_below} answerable refused, "
              f"{len(nc_above)} not-covered admitted",
              f"- calibrated threshold: **{best_t:.2f}** ({why})"]
    lines += [f"  - admitted: {q}" for q in nc_above] + [""]
    print(f"grounding: threshold {threshold} -> {a_below} answerable refused, "
          f"{len(nc_above)} not-covered admitted; calibrated: {why}")
    for q, s in zip(not_covered, nc_scores):
        print(f"   not-covered {s:6.2f}  {q}")

    # --- safety -----------------------------------------------------------------
    safety_bad = []
    for case in data["safety"]:
        kind = screen(case["q"]).kind or "none"
        if kind != case["expect"]:
            safety_bad.append((case["q"], case["expect"], kind))
    n_s = len(data["safety"])
    lines += ["## Safety screen", "", f"- routed correctly: {n_s - len(safety_bad)}/{n_s}"]
    lines += [f"  - WRONG: {q!r} expected {e}, got {k}" for q, e, k in safety_bad] + [""]
    print(f"safety: {n_s - len(safety_bad)}/{n_s} routed correctly")
    for q, e, k in safety_bad:
        print(f"   WRONG {q!r}: expected {e}, got {k}")
    failures += bool(safety_bad)

    out = ROOT / args.out
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)}")

    if args.apply:
        apply_threshold(best_t)
        print(f"wrote min_rerank_score: {best_t:.2f} to config/patient.yaml")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
