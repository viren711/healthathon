#!/usr/bin/env python3
"""Check the plumbing without a GPU, a full index, or a running LLM.

Builds nothing. Points the stack at a small index (built with --limit) and
exercises every path that does not need the real model server:

  * the retriever loads and returns citable hits
  * a stale index is detected instead of silently returning wrong results
  * layer 1 (topic gate) admits tongue questions and refuses off-topic ones
  * layer 2 (grounding gate) accepts/refuses on score
  * layer 3 (citation check) rejects uncited, sentinel and out-of-range answers
  * the whole pipeline runs end to end against a stubbed LLM

Usage -- build a small index first, then point at it:

    python scripts/03_build_index.py --model sentence-transformers/all-MiniLM-L6-v2 \
        --device cpu --limit 120 --out-dir /tmp/smoke_index

    EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
    INDEX_DIR=/tmp/smoke_index RERANKER_ENABLED=false \
        python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS, FAIL = "  ok  ", " FAIL "
failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{PASS if condition else FAIL}] {name}" + (f"  -- {detail}" if detail else ""))
    if not condition:
        failures.append(name)


class StubLLM:
    """Stands in for the model server so layer 3 can be tested in isolation."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def chat(self, system: str, user: str) -> str:
        self.calls += 1
        return self.reply


def main() -> int:
    from rag import guard
    from rag.config import load_config
    from rag.retriever import get_retriever

    cfg = load_config()
    print(f"index dir : {cfg['paths']['index_dir']}")
    print(f"embedder  : {cfg['embedding']['model']}")
    print(f"reranker  : {'on' if cfg['reranker']['enabled'] else 'off'}\n")

    # ---- retriever --------------------------------------------------------
    retriever = get_retriever()
    check("index loads", retriever.meta["chunk_count"] > 0, f"{retriever.meta['chunk_count']} chunks")

    hits = retriever.search("What is the lymphatic drainage of the oral tongue?")
    check("search returns hits", bool(hits), f"{len(hits)} hits")
    if hits:
        top = hits[0]
        check("hits carry a citation", bool(top.citation), top.citation)
        check("hits carry page numbers", top.chunk.get("pdf_page_start", 0) > 0,
              f"pdf p.{top.chunk.get('pdf_page_start')}")
        check("hits carry a breadcrumb", bool(top.chunk.get("breadcrumb")),
              top.chunk["breadcrumb"][:70])

    # BM25 and dense must both contribute, or the hybrid is not hybrid.
    exact = retriever.search("cisplatin")
    check("bm25 path returns results for an exact term", bool(exact))

    # ---- index mismatch detection ----------------------------------------
    from rag.retriever import IndexMismatchError, Retriever

    real_model = cfg["embedding"]["model"]
    cfg["embedding"]["model"] = "some/other-model"
    try:
        Retriever.load()
        check("stale index is rejected", False, "loaded an index built by another model")
    except IndexMismatchError:
        check("stale index is rejected", True)
    except Exception as exc:  # noqa: BLE001
        check("stale index is rejected", False, f"wrong error: {type(exc).__name__}")
    finally:
        cfg["embedding"]["model"] = real_model

    # ---- layer 1: topic gate ---------------------------------------------
    print()
    in_scope = [
        "What is the treatment for T1 oral tongue cancer?",
        "How is base of tongue carcinoma staged?",
        "What are the complications of glossectomy?",
    ]
    for q in in_scope:
        check(f"topic gate admits: {q[:46]}", guard.topic_gate(q).allowed)

    # What counts as off-topic depends on how much of the books is indexed.
    # Under scope "corpus" a prostate question is IN scope -- the books have a
    # chapter on it -- so asserting a refusal there would be asserting the bug
    # that full-book indexing exists to fix.
    scope = cfg["guard"]["topic"].get("scope", "tongue")
    always_off_topic = [
        "Write me a poem about the sea.",
        "What is the capital of France?",
        "How do I fix a Python ImportError?",
    ]
    other_sites = [
        "How do I treat prostate cancer?",
        "Explain the staging of non-small cell lung cancer.",
    ]
    for q in always_off_topic:
        res = guard.topic_gate(q)
        check(f"topic gate refuses: {q[:46]}", not res.allowed, res.reason[:60])

    for q in other_sites:
        res = guard.topic_gate(q)
        if scope == "corpus":
            check(f"topic gate admits (corpus scope): {q[:34]}", res.allowed,
                  res.reason[:60])
        else:
            check(f"topic gate refuses (tongue scope): {q[:33]}", not res.allowed,
                  res.reason[:60])

    # ---- layer 2: grounding gate -----------------------------------------
    print()
    check("grounding gate refuses with no hits", not guard.grounding_gate([]).allowed)
    if hits:
        res = guard.grounding_gate(hits)
        check("grounding gate runs on real hits", isinstance(res.allowed, bool),
              f"{res.reason} ({res.detail})")

    # ---- layer 3: citation verification ----------------------------------
    print()
    sentinel = cfg["guard"]["citation"]["sentinel"]
    cases = [
        ("accepts a properly cited answer", "The first echelon nodes are levels IB and II [1].", 3, True),
        ("rejects the abstention sentinel", sentinel, 3, False),
        ("rejects an uncited answer", "Tongue cancer is treated with surgery.", 3, False),
        ("rejects an out-of-range citation", "Doses are 70 Gy [9].", 3, False),
        ("rejects an empty answer", "   ", 3, False),
        ("accepts multiple valid citations", "Surgery [1] or radiotherapy [2][3] may be used.", 3, True),
    ]
    for name, answer, n, expected in cases:
        res = guard.verify_answer(answer, n)
        check(f"citation check {name}", res.allowed == expected, res.reason[:60])

    # ---- layer 2 responds to the threshold in both directions -------------
    # The absolute score scale depends on which reranker is loaded, so assert
    # the behaviour (threshold moves -> decision moves), not a fixed number.
    grounding_cfg = cfg["guard"]["grounding"]
    original = grounding_cfg["min_rerank_score"], grounding_cfg["min_dense_similarity"]
    try:
        grounding_cfg["min_rerank_score"] = 1e9
        grounding_cfg["min_dense_similarity"] = 1e9
        check("grounding gate refuses when the bar is impossibly high",
              not guard.grounding_gate(hits).allowed)
        grounding_cfg["min_rerank_score"] = -1e9
        grounding_cfg["min_dense_similarity"] = -1e9
        check("grounding gate admits when the bar is low",
              guard.grounding_gate(hits).allowed)
    finally:
        grounding_cfg["min_rerank_score"], grounding_cfg["min_dense_similarity"] = original

    # ---- end to end against a stubbed LLM --------------------------------
    print()
    from rag.pipeline import answer_question

    q = "What is the lymphatic drainage of the oral tongue?"

    # Generation is only reached once layer 2 passes. This smoke test runs on
    # a partial index with whatever reranker is configured, so the calibrated
    # threshold does not apply -- relax it to test the generation path itself.
    # (scripts/04_evaluate.py is what calibrates the real threshold.)
    grounding_cfg["min_rerank_score"] = -1e9
    grounding_cfg["min_dense_similarity"] = -1e9
    try:
        good = StubLLM("The oral tongue drains to the level IB and II nodes [1].")
        ans = answer_question(q, client=good)
        check("pipeline answers a grounded question", ans.status == "answered",
              f"status={ans.status}, {len(ans.sources)} sources, {ans.elapsed_ms}ms")
        check("answered result carries sources", ans.is_answer and bool(ans.sources))
        check("answered result records citations", ans.is_answer and ans.cited == [1],
              str(ans.cited))
        check("the LLM was actually called", good.calls == 1, f"{good.calls} calls")

        abstain = StubLLM(sentinel)
        ans2 = answer_question(q, client=abstain)
        check("pipeline abstains when the model emits the sentinel",
              ans2.status == "not_available", ans2.text[:60])

        uncited = StubLLM("Tongue cancer spreads to the neck nodes.")
        ans3 = answer_question(q, client=uncited)
        check("pipeline abstains on an uncited answer", ans3.status == "not_available")

        hallucinated = StubLLM("Per the 2024 trial, survival was 91% [7].")
        ans4 = answer_question(q, client=hallucinated)
        check("pipeline abstains on an out-of-range citation", ans4.status == "not_available")
    finally:
        grounding_cfg["min_rerank_score"], grounding_cfg["min_dense_similarity"] = original

    never = StubLLM("should never be called")
    # Chosen to be out of scope under BOTH settings, so this check tests the
    # short-circuit rather than the scope configuration.
    ans5 = answer_question("Write me a poem about the sea.", client=never)
    check("off-topic question is refused", ans5.status == "out_of_scope", ans5.text[:60])
    check("off-topic question never reaches the LLM", never.calls == 0,
          f"{never.calls} calls")

    # ---- topic locator (Find mode) ----------------------------------------
    print()
    from rag.locate import locate

    # A term that is certain to appear somewhere in an oncology corpus.
    places = locate("lymph node", limit=10)
    check("locator returns places", bool(places), f"{len(places)} results")
    if places:
        top = places[0]
        check("located place carries a book and a page",
              bool(top.book_short) and top.pdf_page_start > 0,
              f"{top.book_short} p{top.pdf_page_start}")
        check("located place carries a readable page label",
              bool(top.page_label), top.page_label)
        check("located place carries a snippet", bool(top.snippet.strip()))
        # Grouping is the point: one entry per section, not per chunk.
        ids = [pl.section_id for pl in places]
        check("locator groups by section (no duplicates)",
              len(ids) == len(set(ids)),
              f"{len(ids) - len(set(ids))} duplicate section(s)")
        check("locator page span is sane",
              all(pl.pdf_page_end >= pl.pdf_page_start for pl in places))
        check("locator results are ordered by score",
              all(a.score >= b.score for a, b in zip(places, places[1:])))

    check("locator ignores an empty topic", locate("") == [])

    # ---- answer length fitting ----------------------------------------------
    # A hard token cap cuts wherever the budget runs out, which lands mid-word
    # and leaves a fragment with no citation. These assert the trim only ever
    # lands on a sentence or bullet boundary.
    print()
    import re as _re2

    from rag.generate import fit_to_length

    _CLEAN_END = _re2.compile(r"""([.!?]["')\]]*(\s*\[\d+\])*|\[\d+\]|:)$""")

    prose = ("Alpha beta gamma delta [1]. Epsilon zeta eta theta [2]. "
             "Iota kappa lambda mu [1]. Nu xi omicron pi [3].")
    out, trimmed = fit_to_length(prose, 10)
    check("long prose is trimmed", trimmed and len(out.split()) <= 10,
          f"{len(out.split())} words")
    check("trimmed prose ends on a sentence boundary",
          bool(_CLEAN_END.search(out.rstrip())), repr(out[-30:]))
    check("trimmed prose keeps a citation", "[" in out)

    # Citation after the full stop must not be orphaned from its sentence.
    after = "Alpha beta gamma delta. [1] Epsilon zeta eta theta. [2]"
    out2, _ = fit_to_length(after, 6)
    check("marker after the stop stays with its sentence",
          out2.rstrip().endswith("[1]"), repr(out2))

    # Bullets usually have no terminal stop; splitting on sentences alone
    # would glue them together.
    bullets = ("Management depends on stage [1]:\n"
               "- Surgery is preferred for T1 lesions [1]\n"
               "- Radiation is an alternative [2]\n"
               "- Chemoradiation for advanced disease [3]")
    out3, _ = fit_to_length(bullets, 16)
    lines = [ln for ln in out3.split("\n") if ln.strip()]
    check("bullets are kept whole",
          all(ln.rstrip().endswith("]") or ln.rstrip().endswith(":") for ln in lines),
          repr(lines[-1]) if lines else "")
    check("no partial bullet survives the trim",
          not out3.rstrip().endswith("for") and not out3.rstrip().endswith("is"),
          repr(out3[-24:]))

    check("a short answer is returned untouched",
          fit_to_length(prose, 500) == (prose, False))
    check("an answer with no citation is never trimmed into invalidity",
          fit_to_length("one two three four five six seven eight", 3)[1] is False)

    # ---- citation integrity -------------------------------------------------
    # A citation naming the wrong chapter is worse than no citation: it looks
    # checkable and sends the reader to the wrong part of the book. This
    # caught a real bug in which 75% of DeVita chunks were labelled with
    # whichever chapter happened to be current when a hand-written chapter map
    # ran out, so every one of those citations was wrong.
    print()
    import re as _re
    from collections import Counter as _Counter

    chapters = _Counter(
        c.get("chapter") for c in retriever.chunks if c.get("book_short") == "DeVita"
    )
    check(
        "DeVita chunks span many chapters, not one sticky label",
        len(chapters) > 50,
        f"{len(chapters)} distinct chapter labels",
    )

    # Cross-check a sample against the PDF's own outline.
    try:
        import fitz

        doc = fitz.open(ROOT / "data" / "DeVita_Hellman_and_Rosenbergs_Cancer.pdf")
        starts = {}
        for level, title, page in doc.get_toc():
            if level > 2:
                continue
            m = _re.match(r"^(\d{1,3})\s+\S", _re.sub(r"^\d+\.\s*", "", title).strip())
            if m:
                starts.setdefault(int(m.group(1)), page)
        doc.close()
        ordered = sorted(starts.items(), key=lambda kv: kv[1])

        def true_chapter(page):
            prev = None
            for num, start in ordered:
                if start > page:
                    return prev
                prev = num
            return prev

        sample = [c for c in retriever.chunks if c.get("book_short") == "DeVita"][::400]
        wrong = []
        for c in sample:
            label = c.get("chapter") or ""
            m = _re.match(r"^(\d+)\s", label)
            if not m:
                continue
            expected = true_chapter(c["pdf_page_start"])
            if expected is not None and int(m.group(1)) != expected:
                wrong.append((c["pdf_page_start"], label.split(" - ")[0], expected))
        check(
            "cited chapter matches the PDF outline for that page",
            not wrong,
            f"{len(wrong)}/{len(sample)} wrong"
            + (f", e.g. p{wrong[0][0]} says ch.{wrong[0][1]} but is ch.{wrong[0][2]}" if wrong else ""),
        )
    except ImportError:
        print("[ skip ] outline cross-check (pymupdf unavailable)")

    # ---- pathology scoping -------------------------------------------------
    print()
    scopes = retriever.available_scopes()
    check("index carries a pathology taxonomy", len(scopes) > 10,
          f"{len(scopes)} selectable scopes")
    if scopes:
        every = sum(1 for c in retriever.chunks if c.get("pathology"))
        check("every chunk is tagged", every == len(retriever.chunks),
              f"{every}/{len(retriever.chunks)}")

        q = "What is the lymphatic drainage of the oral tongue?"
        wide = retriever.search(q)
        narrow = retriever.search(q, scope="hn_oral_cavity")
        check("scoped search returns hits", bool(narrow), f"{len(narrow)} hits")
        check("scoped search returns ONLY that pathology",
              all(h.chunk["pathology"] == "hn_oral_cavity" for h in narrow),
              str(sorted({h.chunk["pathology"] for h in narrow})))
        # Scoping must not degrade a question that belongs to the scope: the
        # same best chunk should surface either way.
        check("scoping does not lose the best chunk for an in-scope question",
              bool(wide) and bool(narrow)
              and abs(wide[0].score - narrow[0].score) < 1e-6,
              f"all={wide[0].score:.3f} scoped={narrow[0].score:.3f}" if wide and narrow else "")

        system = retriever.search(q, scope="head_neck")
        check("an organ-system scope also works",
              bool(system) and all(h.chunk["organ_system"] == "head_neck" for h in system))

        # Asking an out-of-scope question inside a scope must NOT answer it.
        off = retriever.search("How is prostate cancer treated?", scope="hn_oral_cavity")
        thresh = load_config()["guard"]["grounding"]["min_rerank_score"]
        check("a foreign question scores below the bar inside a scope",
              not off or off[0].score < thresh,
              f"top={off[0].score:.2f} vs threshold {thresh}" if off else "no hits")

        check("an unknown scope id falls back to the whole corpus",
              retriever.scope_rows("not_a_real_scope") is None)

    # ---- go-to-section page rendering -------------------------------------
    print()
    from rag import pdfview

    if places:
        book, page = places[0].book_short, places[0].pdf_page_start
        n_pages = pdfview.page_count(book)
        check("source PDF resolves from the book short name", n_pages > 0,
              f"{book}: {n_pages} pages")
        png = pdfview.render_page(book, page)
        # A rendered textbook page is tens of KB at minimum; a few hundred
        # bytes would mean a blank or failed raster.
        check("cited page renders to a PNG", bool(png) and len(png) > 5000,
              f"{len(png) if png else 0} bytes")
        check("out-of-range page renders nothing",
              pdfview.render_page(book, n_pages + 1000) is None)
        check("unknown book resolves to nothing",
              pdfview.resolve_book("Not A Real Book") is None)
        excerpt = pdfview.extract_pages(book, page, page + 2)
        check("page range extracts to a small PDF",
              bool(excerpt) and excerpt[:4] == b"%PDF",
              f"{len(excerpt) if excerpt else 0} bytes")

    # ---- summary ----------------------------------------------------------
    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
