#!/usr/bin/env python3
"""Stage 10 -- evaluate Treatment Guidance against eval/guidance_cases.yaml.

No language model. Measures:

  matching    does a recommendation that answers each case land in the top 3?
  reading     does the case reader extract the facts each case states?
  coverage    no result for a cancer the guidelines do not cover
  verbatim    every recommendation and every statistic is a span of its PDF's
              text (after whitespace normalisation) -- the property the page
              promises when it says "quoted, not generated"

    python scripts/10_eval_guidance.py
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag import guidance  # noqa: E402


def _builder():
    spec = importlib.util.spec_from_file_location("build_asco", ROOT / "scripts" / "09_build_asco.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s)


def verbatim_failures(entry: dict) -> list[str]:
    """Statements and statistics of one guideline that are not in its PDF."""
    import pymupdf

    b = _builder()
    text = _squash(b._flat(b._clean_pages(pymupdf.open(ROOT / entry["path"])))[0])
    recs, stats, _ = guidance.load()
    bad = []
    for r in recs:
        if r["guideline_id"] == entry["id"] and _squash(r["statement"]) not in text:
            bad.append(f"rec {r['number']}: {r['statement'][:70]}")
    for s in stats:
        if s["guideline_id"] == entry["id"] and _squash(s["text"]) not in text:
            bad.append(f"stat p{s['pdf_page']}: {s['text'][:70]}")
    return bad


def main() -> int:
    data = yaml.safe_load((ROOT / "eval" / "guidance_cases.yaml").read_text(encoding="utf-8"))
    failures = 0

    # --- matching and reading ---------------------------------------------------
    hits = total = read_ok = read_n = none_ok = none_n = 0
    for c in data["cases"]:
        case = guidance.read_case(c["text"])
        top, _ = guidance.rank(case, k=3)
        if c.get("none"):
            none_n += 1
            ok = not top
            none_ok += ok
            print(f"[{'ok' if ok else 'FAIL'}] (not covered) {c['text'][:70]}")
            continue
        total += 1
        got = [(m.rec["guideline_id"], m.rec["number"].rstrip("+")) for m in top]
        ok = any(g in c["expect"] and n in c["expect"][g] for g, n in got)
        hits += ok
        print(f"[{'ok' if ok else 'MISS'}] {c['text'][:70]}\n       top: {got}")
        for field, want in (c.get("read") or {}).items():
            read_n += 1
            have = getattr(case, field)
            good = set(want) <= set(have) if isinstance(want, list) else have == want
            read_ok += good
            if not good:
                print(f"       READ {field}: expected {want}, got {have}")

    for c in data.get("tuned_after_heldout", []):  # round 1, now tuning data
        top, _ = guidance.rank(guidance.read_case(c["text"]), k=3)
        got = [(m.rec["guideline_id"], m.rec["number"].rstrip("+")) for m in top]
        ok = any(g in c["expect"] and n in c["expect"][g] for g, n in got)
        hits += ok
        total += 1
        print(f"[{'ok' if ok else 'MISS'}] {c['text'][:70]}\n       top: {got}")

    held = held_n = 0
    for c in data.get("heldout", []):
        top, _ = guidance.rank(guidance.read_case(c["text"]), k=3)
        got = [(m.rec["guideline_id"], m.rec["number"].rstrip("+")) for m in top]
        ok = any(g in c["expect"] and n in c["expect"][g] for g, n in got)
        held += ok
        held_n += 1
        print(f"[{'ok' if ok else 'MISS'}] (held out) {c['text'][:70]}\n       top: {got}")

    print(f"\nmatching: expected recommendation in top 3 for {hits}/{total} cases ({hits / total:.0%})")
    if held_n:
        print(f"round 2:  {held}/{held_n} ({held / held_n:.0%}) -- its first, untouched run "
              "scored 5/10; see eval/guidance_cases.yaml")
    print(f"reading:  {read_ok}/{read_n} stated facts extracted")
    print(f"coverage: {none_ok}/{none_n} uncovered cases returned nothing")
    failures += hits / total < 0.8
    failures += read_ok < read_n
    failures += none_ok < none_n

    # --- verbatim -----------------------------------------------------------------
    entries = yaml.safe_load((ROOT / "config" / "asco_guidelines.yaml").read_text(encoding="utf-8"))["guidelines"]
    with ProcessPoolExecutor() as pool:
        bad = [x for result in pool.map(verbatim_failures, entries) for x in result]
    recs, stats, _ = guidance.load()
    print(f"verbatim: {len(recs) + len(stats) - len(bad)}/{len(recs) + len(stats)} "
          f"statements and statistics found word for word in their PDF")
    for b in bad[:10]:
        print("   NOT VERBATIM", b)
    failures += bool(bad)

    out = ROOT / "eval" / "guidance_report.json"
    out.write_text(json.dumps({"top3": f"{hits}/{total}", "heldout_top3": f"{held}/{held_n}", "reading": f"{read_ok}/{read_n}",
                               "uncovered": f"{none_ok}/{none_n}", "not_verbatim": len(bad)}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
