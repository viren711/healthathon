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
from datetime import date
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
        self.last_system = ""

    def chat(self, system: str, user: str) -> str:
        self.calls += 1
        self.last_system = system
        return self.reply


def key_checks() -> None:
    """Several API keys: the first with credit is used, a spent one is skipped."""
    import os
    import tempfile

    from rag import keyring
    from rag.sarvam import _LiveAPI
    from rag.scite import SciteAnswer, SciteClient, SciteError

    names = ("KEY_STATE_FILE", *keyring.NAMES["sarvam"], *keyring.NAMES["scite"])
    saved = {k: os.environ.get(k) for k in names}
    try:
        state = Path(tempfile.mkdtemp()) / "key_state.json"
        os.environ["KEY_STATE_FILE"] = str(state)
        fake = dict(zip(keyring.NAMES["sarvam"], ["k-one", "k-two", "k-three", "k-four"]))
        os.environ.update(fake)

        class Spent(Exception):
            def __init__(self, status, body=""):
                self.status_code, self.body = status, body

        api = _LiveAPI()
        used: list[str] = []

        def call(spent: set[str]):
            def fn(client):
                key = next(k for k, v in api._clients.items() if v is client)
                used.append(key)
                if key in spent:
                    raise Spent(402, "insufficient credits")
                return "ok"
            return fn

        check("Sarvam keys are tried as coiffeur, key, 1, 2",
              [n for n, _ in keyring.usable("sarvam")] ==
              ["SARVAM_API_KEY_coiffeur", "SARVAM_API_KEY", "SARVAM_API_KEY_1", "SARVAM_API_KEY_2"])
        r = api._run(call({"k-one"}))
        check("an out-of-credit key hands over to the next", r == "ok" and used == ["k-one", "k-two"]
              and api.key_label == "SARVAM_API_KEY", str(used))
        used.clear()
        api._run(call({"k-one"}))
        check("the spent key is not tried again", used == ["k-two"], str(used))
        check("only a fingerprint is saved, never the key", "k-one" not in state.read_text())
        check("a plain rate limit does not retire a key",
              not keyring.exhausted("sarvam", 429, "rate limit, slow down")
              and keyring.exhausted("sarvam", 429, "insufficient credits"))
        used.clear()
        try:
            api._run(call({"k-two", "k-three", "k-four"}))
            outcome = "no error"
        except Exception as exc:  # noqa: BLE001
            outcome = getattr(exc, "status_code", None)
        used.clear()
        try:
            api._run(call(set()))
            after = "called"
        except Exception as exc:  # noqa: BLE001
            after = getattr(exc, "status_code", None)
        check("with every key spent: a 402 and no further request", outcome == 402 and after == 402 and not used,
              f"{outcome} {after} {used}")

        os.environ["SCITE_API_KEY"], os.environ["extra_scite_api"] = "s-one", "s-two"
        tried: list[str] = []

        def fake_ask(self, question, history=None, **kw):
            tried.append(self.api_key)
            if self.api_key == "s-one":
                raise SciteError("HTTP 429", 429, b'{"message":"monthly search limit"}')
            return SciteAnswer("answer", [])

        orig, SciteClient._ask = SciteClient._ask, fake_ask
        try:
            client = SciteClient("s-one")
            got = [client.ask("q"), client.ask("q")]
        finally:
            SciteClient._ask = orig
        check("Scite: the extra key answers once the first is over its monthly limit, and the first is skipped after",
              [g.content for g in got] == ["answer", "answer"] and tried == ["s-one", "s-two", "s-two"], str(tried))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def deid_checks() -> None:
    """The scite.ai guardrail (rag/deid.py): nothing identifying goes out or comes back."""
    import json

    from rag import deid
    from rag.scite import SciteClient, SciteError, SciteMCPClient

    print("De-identification guardrail")
    case = ("Mr. Ramesh Sharma (phone 9876543210, ramesh.s@gmail.com, Aadhaar 2345 6789 0123, "
            "MRN: AB123456, DOB 12/03/1968) has EGFR-mutant lung cancer. Is osimertinib effective?")
    leaks = ["Ramesh", "Sharma", "9876543210", "ramesh.s", "2345", "6789", "AB123456", "12/03/1968"]
    clean, found = deid.scrub(case)
    check("every identifier in the question is removed", not any(x in clean for x in leaks), clean)
    check("the research question itself is kept", "EGFR-mutant lung cancer" in clean and "osimertinib" in clean, clean)
    sci = "pan-cancer EGFR 10 mg HR 0.62 (95% CI 0.48-0.80) NCT01234567 after March 2019 in patients aged 18-65, median age 62, over 65 years"
    check("ordinary scientific text is left untouched", deid.scrub(sci)[0] == sci)
    check("a link in an answer is kept as a citation", "doi.org" in deid.scrub("see https://doi.org/10.1/x", keep_links=True)[0])
    check("a patient's age is generalised to a decade, 90+ and children banded",
          deid.scrub("a 54-year-old man")[0] == "a [AGE 50s] man" and "[AGE 90+]" in deid.scrub("a 93 year old")[0]
          and "[AGE under 18]" in deid.scrub("child aged 7")[0])
    check("the notice names name, age and address, not phone numbers",
          deid.describe(deid.scrub("Mr. Rao, a 54-year-old, Flat 12B, MG Road, phone 9876543210")[1]) == "name, age, address, IDs and contact details")
    check("a question that is only identifiers has nothing to send", not deid.has_content(deid.scrub("Mr. Ramesh 9876543210")[0]))

    # What really goes over the wire: capture the request body instead of sending it.
    sent: list = []
    c = SciteClient("k-test")
    def fake(method, path, payload=None):
        sent.append(payload)
        if method == "POST":
            return {"id": "t1"}
        return {"status": "SUCCESS", "content": "Contact the author at dr.x@hospital.org or 9123456789. See https://doi.org/10.1/x",
                "references": []}
    c._request = fake  # type: ignore[method-assign]
    history = [{"role": "user", "content": "Patient name is Anita Rao, 044-24123456"},
               {"role": "assistant", "content": "Earlier answer"}]
    try:
        ans = c._ask(case, history)
        wire = json.dumps(sent[0])
        check("nothing identifying is in the request sent to scite.ai",
              not any(x in wire for x in leaks + ["Anita", "Rao", "24123456"]), wire[:120])
        check("earlier turns are scrubbed too", "[NAME]" in wire and "[PHONE]" in wire)
        check("an identifier in the returned answer is removed",
              "dr.x@" not in ans.content and "9123456789" not in ans.content and "doi.org" in ans.content, ans.content)
    except Exception as exc:  # noqa: BLE001
        check("the guarded request runs", False, repr(exc))
    try:
        c._ask("Mr. Ramesh 9876543210", [])
        check("an identifiers-only question is refused, not sent", False)
    except SciteError:
        check("an identifiers-only question is refused, not sent", True)

    sent2: list = []
    m = SciteMCPClient("k-test")
    m._request = lambda payload: (sent2.append(payload) or {"result": {"content": []}})  # type: ignore[method-assign]
    try:
        m._search(case)
    except Exception:  # noqa: BLE001
        pass
    wire2 = json.dumps(sent2[0]) if sent2 else ""
    check("the literature-search path is scrubbed too", bool(wire2) and not any(x in wire2 for x in leaks), wire2[:120])


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

    # ---- Patient Portal ---------------------------------------------------
    print()
    patient_checks(StubLLM)

    # ---- Treatment Guidance -----------------------------------------------
    print()
    guidance_checks()

    # ---- Voice & languages (mock Sarvam, spends nothing) -------------------
    print()
    voice_checks(StubLLM)
    print()
    key_checks()
    print()
    deid_checks()

    # ---- summary ----------------------------------------------------------
    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


def voice_checks(stub_cls) -> None:
    """rag/sarvam.py against its offline mock, with a throwaway cache."""
    import io
    import json
    import os
    import os
    import tempfile
    import wave

    tmp = tempfile.mkdtemp(prefix="voice_smoke_")
    saved = {k: os.environ.get(k) for k in ("SARVAM_MOCK", "VOICE_CACHE_DIR", "VOICE_USAGE_LOG")}
    os.environ.update(SARVAM_MOCK="1", VOICE_CACHE_DIR=tmp,
                      VOICE_USAGE_LOG=os.path.join(tmp, "usage.jsonl"))
    try:
        from rag import sarvam
        from rag.pipeline import answer_question

        v = sarvam.SarvamVoice.from_env()
        check("mock voice client is used when SARVAM_MOCK=1", v is not None and v.mock)

        samples = {"कीमो क्या है?": "deva", "What is chemo?": "latn",
                   "কেমো কী?": "bn-IN", "ਕੀਮੋ ਕੀ ਹੈ?": "pa-IN", "કીમો શું છે?": "gu-IN",
                   "କେମୋ କ'ଣ?": "od-IN", "கீமோ என்றால் என்ன?": "ta-IN",
                   "కీమో అంటే ఏమిటి?": "te-IN", "ಕೀಮೋ ಎಂದರೇನು?": "kn-IN",
                   "കീമോ എന്താണ്?": "ml-IN", "मुझे CT scan करवाना है": "deva"}
        wrong = {t: sarvam.dominant_script(t) for t, e in samples.items()
                 if sarvam.dominant_script(t) != e}
        check("script detection names all 11 languages' scripts", not wrong, str(wrong))

        lid_before = v.api.calls["lid"]
        check("a Tamil question is detected without an API call",
              v.detect("கீமோ என்றால் என்ன?", ["en-IN", "ta-IN"]) == "ta-IN"
              and v.api.calls["lid"] == lid_before)
        check("Devanagari is Hindi for free when Marathi is not on offer",
              v.detect("कीमो क्या है?", ["en-IN", "hi-IN"]) == "hi-IN"
              and v.api.calls["lid"] == lid_before)

        # Both placements mayura:v1 produced on 2026-11-03: a marker after the
        # full stop that belongs to the NEXT sentence, and one that belongs to
        # the PREVIOUS. Only sentence order tells them apart.
        fixed = sarvam.align_markers(
            "Fatigue can happen [1]. Most effects go away [2]. Talk to your team [1][3].",
            "थकान हो सकती है [1]। [2] ज़्यादातर असर चले जाते हैं। टीम से बात करें [1][3]।")
        check("a marker the translation pushed forward goes back to its sentence",
              fixed == "थकान हो सकती है [1]। ज़्यादातर असर चले जाते हैं [2]। टीम से बात करें [1][3]।",
              fixed)
        fixed = sarvam.align_markers(
            "Radiation can convert them [4]. It can also enhance uptake [4].",
            "Radiation उन्हें convert कर सकता है। [4] यह uptake को enhance कर सकता है। [4]")
        check("a marker the translation pushed back goes back to its sentence",
              fixed == "Radiation उन्हें convert कर सकता है [4]। यह uptake को enhance कर सकता है [4]।",
              fixed)
        fixed = sarvam.align_markers(
            "It depletes the machinery. [1] The adducts need higher doses. [2]",
            "यह machinery को कम करता है। [1] Adducts को higher doses चाहिए। [2]")
        check("an English marker written after the full stop stays with its sentence",
              fixed == "यह machinery को कम करता है [1]। Adducts को higher doses चाहिए [2]।",
              fixed)
        # ...and through the whole translate path, which normalises the
        # English before the translator ever sees it.
        local = v.translate("Repair genes matter. [1] [2] So adducts drive toxicity. [1] [2]",
                            "en-IN", "hi-IN", "clinician")
        check("post-full-stop markers are attached before translating",
              local == "[hi] Repair genes matter [1][2]. So adducts drive toxicity [1][2].", local)

        answer = "Fatigue is common [1]. It gets better [2].\n\n- Rest often [1]\n- Eat well [3]"
        local = v.translate(answer, "en-IN", "hi-IN", "patient")
        check("translation keeps every citation marker",
              sorted(sarvam.MARKER.findall(local)) == sorted(sarvam.MARKER.findall(answer)),
              local)
        check("translation keeps the list layout", "\n- " in local, local)
        n = v.api.calls["translate"]
        v.translate(answer, "en-IN", "hi-IN", "patient")
        check("a repeated translation is served from the cache", v.api.calls["translate"] == n,
              f"{v.api.calls['translate'] - n} extra call(s)")

        spoken = sarvam.strip_for_speech(answer)
        check("read-aloud text has no markers and one sentence per item",
              "[" not in spoken and "Rest often." in spoken, spoken)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 16000)
        text, lang = v.transcribe(buf.getvalue())
        check("a recording is transcribed with its language", bool(text) and lang == "hi-IN",
              f"{text!r} {lang}")
        try:
            v.transcribe(b"\x00" * 100)
            check("a too-short recording is refused", False)
        except sarvam.SarvamError:
            check("a too-short recording is refused", True)

        audio, mime = v.speak(answer, "hi-IN")
        tts = v.api.calls["tts"]
        v.speak(answer, "hi-IN")
        check("read-aloud audio is cached on disk", bool(audio) and v.api.calls["tts"] == tts)

        # Two identical requests arriving together (a double rerun) pay once.
        from concurrent.futures import ThreadPoolExecutor
        before = v.api.calls["tts"]
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda _: v.speak("A brand new sentence to read aloud.", "ta-IN"), range(2)))
        check("two identical simultaneous Listen requests make one API call",
              v.api.calls["tts"] == before + 1, f"{v.api.calls['tts'] - before} calls")

        recs = [json.loads(x) for x in Path(tmp, "usage.jsonl").read_text(encoding="utf-8").splitlines()]
        tts = [r for r in recs if r["kind"] == "tts" and not r.get("cached")]
        check("a paid read-aloud is logged with model, voice, language and a text preview",
              bool(tts) and all(k in tts[-1] for k in ("model", "speaker", "lang", "chars", "text", "step")),
              str(tts[-1] if tts else None))
        v.speak("The same words, read for a doctor.", "en-IN", "clinician")
        v.speak("The same words, read for a patient.", "en-IN", "patient")
        voices = {json.loads(x)["voice"]: json.loads(x)["speaker"] for x in
                  Path(tmp, "usage.jsonl").read_text(encoding="utf-8").splitlines()
                  if '"voice"' in x and '"cached"' not in x}
        check("patients and doctors are both read to by roopa_hi_companion",
              voices == {"clinician": "roopa_hi_companion", "patient": "roopa_hi_companion"},
              str(voices))
        check("a cache hit is logged as free",
              any(r.get("cached") and r["inr"] == 0 for r in recs))
        # -- the recording limit, and the fixed "not found" messages -----------
        def wav_of(seconds: float) -> bytes:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
                w.writeframes(b"\x00\x00" * int(16000 * seconds))
            return buf.getvalue()

        from rag.sarvam import SarvamError, audio_seconds
        check("a recording's length is read from its WAV header",
              abs(audio_seconds(wav_of(12.5)) - 12.5) < 0.01)
        limit = v.cfg["limits"]["max_record_seconds"]
        check("the recording limit is 30 seconds", limit == 30, str(limit))
        before = v.api.calls["stt"]
        try:
            v.transcribe(wav_of(45), "hi-IN")
            refused = False
        except SarvamError as exc:
            refused = "limit is 30" in str(exc)
        check("a 45 s recording is refused without any request being made",
              refused and v.api.calls["stt"] == before)
        v.transcribe(wav_of(29), "hi-IN")
        check("a 29 s recording is accepted", v.api.calls["stt"] == before + 1)

        from rag.config import load_patient_config
        fixed = load_patient_config()["messages"]
        scope_tail = (" This search was limited to the guide or topic you selected -- choose "
                      "All guides to look across every NCCN patient guide.")
        paid = lambda: v.api.calls["tts"] + v.api.calls["translate"]
        for name in ("not_available", "excerpts", "out_of_scope"):
            for text in (" ".join(fixed[name].split()), " ".join(fixed[name].split()) + scope_tail):
                v.speak(v.translate(text, "en-IN", "ta-IN", "patient"), "ta-IN", "patient")
        after_first = paid()
        for name in ("not_available", "excerpts", "out_of_scope"):
            for text in (" ".join(fixed[name].split()), " ".join(fixed[name].split()) + scope_tail):
                v.speak(v.translate(text, "en-IN", "ta-IN", "patient"), "ta-IN", "patient")
        check("the fixed 'not found' messages are translated and voiced once, then served from the cache",
              after_first > 0 and paid() == after_first, f"{after_first} -> {paid()}")

        from rag.sarvam import write_usage_report
        rep = write_usage_report(Path(tmp, "usage.jsonl"), Path(tmp, "report.txt"))
        check("the report totals paid calls and counts free repeats",
              "TOTAL CREDITS USED" in rep and "served free from the cache" in rep)

        log = Path(tmp, "usage.jsonl").read_text(encoding="utf-8").splitlines()
        check("every billable call is logged with its cost",
              len(log) >= 3 and all("inr" in json.loads(x) for x in log), f"{len(log)} lines")

        real_cfg = v.cfg
        check("the daily cap is ₹500, on every day",
              sarvam.daily_budget(v.cfg, date(2026, 11, 7)) == 500
              and sarvam.daily_budget(v.cfg, date(2027, 1, 1)) == 500,
              str(v.cfg["limits"]))
        sched = {**v.cfg, "limits": {**v.cfg["limits"], "daily_budget_from": {"2026-12-01": 1000}}}
        check("a dated change to the cap still works when one is configured",
              sarvam.daily_budget(sched, date(2026, 11, 30)) == 500
              and sarvam.daily_budget(sched, date(2026, 12, 1)) == 1000)
        v.cfg = {**v.cfg, "limits": {**v.cfg["limits"], "daily_budget_inr": 0.0001,
                                     "daily_budget_from": {}}}
        try:
            v.translate("A new sentence that is not cached.", "en-IN", "ta-IN", "patient")
            check("calls are refused once the daily budget is spent", False)
        except sarvam.SarvamError as exc:
            check("calls are refused once the daily budget is spent", "budget" in str(exc))
        finally:
            v.cfg = real_cfg

        # The pipeline sees only English: a Hindi crisis message, translated,
        # is caught by the English safety screen before any model call.
        english = v.translate("मैं मरना चाहता हूँ, अब सहा नहीं जाता", "hi-IN", "en-IN",
                              "to_english")
        never = stub_cls("should not be called [1].")
        ans = answer_question(english, client=never, corpus="patient")
        check("a crisis written in Hindi is caught after translation",
              ans.status == "safety" and never.calls == 0, f"{english!r} -> {ans.status}")
    finally:
        for k, val in saved.items():
            if val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = val


def guidance_checks() -> None:
    """ASCO recommendation records, the case reader and the ranking."""
    from rag import guidance, pdfview

    if not guidance.available():
        print("(skipping Treatment Guidance checks: run scripts/09_build_asco.py)")
        return
    recs, stats, guides = guidance.load()
    check("every ASCO guideline yields recommendations",
          {g["id"] for g in guides} <= {r["guideline_id"] for r in recs},
          f"{len(recs)} recommendations from {len(guides)} guidelines")
    check("statistics are attached to recommendations",
          sum(bool(r["stats"]) for r in recs) >= 150 and len(stats) >= 500,
          f"{sum(bool(r['stats']) for r in recs)} with statistics; {len(stats)} sentences")
    case = guidance.read_case("Metastatic squamous NSCLC, PD-L1 TPS 70%, EGFR and ALK negative, "
                              "ECOG 0. First-line options?")
    check("the case reader extracts PD-L1, histology and negated drivers",
          case.pdl1 == 70 and case.histology == "squamous"
          and {"EGFR", "ALK"} <= set(case.not_biomarkers), case.summary())
    top, _ = guidance.rank(case, 3)
    check("single-agent immunotherapy ranks first for squamous NSCLC, PD-L1 >= 50%",
          top and (top[0].rec["guideline_id"], top[0].rec["number"]) == ("nsclc_nodriver", "2.1"),
          top[0].rec["statement"][:60] if top else "-")
    her2 = guidance.rank(guidance.read_case(
        "Advanced HER2-positive gastric adenocarcinoma, untreated. First-line?"), 10)[0]
    check("a HER2-positive case never gets a HER2-negative recommendation",
          her2 and not any("HER2-negative" in m.rec["statement"] for m in her2))
    none, _ = guidance.rank(guidance.read_case("Metastatic pancreatic cancer, first-line?"), 3)
    check("a cancer no guideline covers returns nothing", not none)
    check("an ASCO page renders for the viewer",
          bool(pdfview.render_page(f"asco:{top[0].rec['guideline_id']}", top[0].rec["pdf_page"])))


def patient_checks(stub_cls) -> None:
    """The patient index, its scoping, safety screen and patient prompt."""
    import json

    from rag import guard
    from rag.config import load_patient_config, patient_path
    from rag.pipeline import answer_question
    from rag.retriever import IndexMismatchError, get_retriever

    meta_path = patient_path("index_dir") / "index_meta.json"
    if not meta_path.exists():
        print("(skipping Patient Portal checks: no patient index -- run "
              "07_build_patient_corpus.py and 03_build_index.py --chunks ...)")
        return
    try:
        patient = get_retriever("patient")
    except IndexMismatchError:
        print("(skipping Patient Portal checks: this run overrides the embedding model)")
        return

    pcfg = load_patient_config()
    n_guides = len(pcfg["guides"])
    check("patient index covers every registered guide",
          len(patient.meta.get("books", [])) == n_guides,
          f"{len(patient.meta.get('books', []))}/{n_guides}")
    check("every patient chunk names its guide, topic area and citation",
          all(c.get("guide_id") and c.get("pathology") == c["guide_id"]
              and c.get("organ_system") == c.get("category") and c.get("citation")
              for c in patient.chunks),
          f"{len(patient.chunks):,} chunks")
    check("the two indexes share one loaded embedder",
          get_retriever()._embedder is patient._embedder)

    hits = patient.search("What are the signs of mouth cancer?", scope="mouth")
    check("a guide scope returns only that guide",
          bool(hits) and all(h.chunk["guide_id"] == "mouth" for h in hits),
          hits[0].citation if hits else "no hits")
    hits = patient.search("How can I manage tiredness?", scope="supportive_care")
    check("a topic-area scope returns only that area",
          bool(hits) and all(h.chunk["category"] == "supportive_care" for h in hits))

    check("patient topic gate refuses an off-topic question",
          not guard.topic_gate("What is the recipe for chocolate cake?", corpus="patient").allowed)
    check("patient topic gate admits a patient question",
          guard.topic_gate("How can I cope with distress during treatment?", corpus="patient").allowed)

    stub = stub_cls("Distress is a normal part of having cancer [1].")
    ans = answer_question("What is distress?", client=stub, corpus="patient")
    check("pipeline answers from the patient index with the patient prompt",
          ans.status == "answered" and "NCCN Guidelines for Patients" in stub.last_system
          and ans.sources and ans.sources[0].book_short.startswith("NCCN"),
          f"{ans.status}; source {ans.sources[0].citation if ans.sources else '-'}")

    stub = stub_cls("should never be called [1].")
    ans = answer_question("I want to die, I can't cope anymore", client=stub, corpus="patient")
    check("a crisis question gets the crisis message and no generated answer",
          ans.status == "safety" and stub.calls == 0 and ans.alerts
          and ans.alerts[0]["kind"] == "crisis")

    # An emergency notice must not stop the answer: the question still goes
    # through retrieval and generation, with the notice shown above it.
    stub = stub_cls("A fever during chemotherapy can be serious [1].")
    ans = answer_question("What should I do if I get a fever during chemotherapy?",
                          client=stub, corpus="patient")
    check("an emergency question is flagged AND still answered",
          ans.alerts and ans.alerts[0]["kind"] == "emergency" and stub.calls >= 1,
          ans.status)

    # A model that abstains on a well-covered question must not leave the
    # patient with nothing: the guides' own passages are shown instead. A
    # question the guides do not cover still ends in "not available".
    stub = stub_cls("ANSWER_NOT_AVAILABLE")
    ans = answer_question("What is palliative care?", client=stub, corpus="patient")
    check("an abstaining model on a covered question falls back to the guides' passages",
          ans.status == "excerpts" and 1 <= len(ans.sources) <= 2
          and all(s.book_short.startswith("NCCN") for s in ans.sources),
          f"{ans.status}; {len(ans.sources)} passages")
    ans = answer_question("Which hospital in Mumbai is the best for cancer surgery?",
                          client=stub, corpus="patient")
    check("an uncovered question still ends in 'not available'",
          ans.status == "not_available", ans.status)

    stub = stub_cls("Cisplatin is a platinum agent [1].")
    ans = answer_question("What is the mechanism of action of cisplatin?", client=stub)
    check("the Knowledge Assistant still uses the clinician prompt",
          "NCCN Guidelines for Patients" not in stub.last_system and not ans.alerts)

    from rag import pdfview

    png = pdfview.render_page("NCCN · Mouth Cancer", 8)
    check("a patient guide page renders from its source label",
          bool(png) and len(png) > 5000, f"{len(png) if png else 0} bytes")

    library = json.loads(patient_path("library_file").read_text(encoding="utf-8"))
    mouth = next(g for g in library["guides"] if g["id"] == "mouth")
    kinds = {s["kind"] for ch in mouth["chapters"] for s in ch["sections"]}
    check("library has every guide, with FAQs, key points, questions and glossary",
          len(library["guides"]) == n_guides
          and {"faq", "key_points", "questions"} <= kinds and len(mouth["glossary"]) > 20,
          f"mouth: {sorted(kinds)}, {len(mouth['glossary'])} terms")


if __name__ == "__main__":
    raise SystemExit(main())
