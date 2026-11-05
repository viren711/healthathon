#!/usr/bin/env python3
"""Stage 7 -- build the Patient Portal corpus from the NCCN Guidelines for Patients.

Reads the 76 guides listed in config/patient_guides.yaml and writes, to the
paths in config/patient.yaml:

  sections.jsonl  one record per guide section ("What is staging?",
                  "Key points", ...), with its chapter, kind and pages
  library.json    the Learn tab's reading material, all of it NCCN's own
                  text: per guide, its FAQs (sections whose heading is a
                  question), key points, questions to ask, glossary, and
                  every other section by chapter. No language model involved.
  chunks.jsonl    retrieval chunks for the Ask tab, in the same record shape
                  as the textbook chunks so rag/ can search them unchanged

Embed them with the existing index builder:

    python scripts/07_build_patient_corpus.py
    python scripts/03_build_index.py --chunks corpus/patient/chunks.jsonl \\
        --out-dir artifacts/index_patient

Parallelism: one process per guide for parsing (the PDF read, sectioning and
glossary extraction all happen in that one pass, biggest guides first so no
core idles at the end), then a second pool for chunking with the tokenizer
loaded once per worker. --workers caps both.

How a guide is sectioned. 68 of the 76 PDFs carry an outline whose level-2
entries are the guide's sections, many phrased as the question a patient
would ask. The other 8 have no outline, but every page carries a breadcrumb
running header ("Initial treatment  »  Options for limited-stage cancer"),
and a change of breadcrumb is a change of section. Front matter (cover,
supporters, contents) is dropped by starting at the first section; back
matter (NCCN contributors, cancer centers, index) by excluding those titles.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

# Back matter and front matter that is not reading material.
EXCLUDE_TITLES = {
    "nccn contributors",
    "nccn cancer centers",
    "index",
    "contents",
    "table of contents",
    "supporters",
    "share with us",
    "take our survey",
    "about the nccn guidelines for patients",
    "chapter contents",  # synthetic: a chapter opener page listing its sections
}

HEADING_MIN_SIZE = 12.5   # body text is 11 pt; section headings 13-20 pt
CHAPTER_LINE = re.compile(r"^\d{1,2}[ \s]+\S")
PAGE_NUMBER = re.compile(r"^\s*(?:\d{1,3}|e\d{1,3})\s*$")
EDITION_LINE = re.compile(r",\s*20\d\d\s*$")


def norm(text: str) -> str:
    """Lowercase alphanumerics only -- for matching outline titles to headings."""
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    return re.sub(r"[^a-z0-9]+", "", text)


def tidy(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip()


def section_kind(title: str) -> str:
    t = norm(title)
    if t == "keypoints":
        return "key_points"
    if t.startswith("questionstoask"):
        return "questions"
    if t == "wordstoknow":
        return "glossary"
    if title.strip().endswith("?"):
        return "faq"
    return "body"


# ---------------------------------------------------------------------------
# Page parsing (runs in worker processes)
# ---------------------------------------------------------------------------


def _line_record(line: dict) -> dict | None:
    spans = [s for s in line["spans"] if s["text"].strip()]
    if not spans:
        return None
    text = "".join(s["text"] for s in line["spans"])
    first = spans[0]["text"].strip()
    # NCCN bullets are a lower-case "h" set in a dingbat font, in a span of
    # its own: "h Red or white patches inside the mouth".
    bullet = first == "h" and len(spans) > 1
    if bullet:
        text = "".join(s["text"] for s in line["spans"][line["spans"].index(spans[0]) + 1 :])
    return {
        "text": tidy(text),
        "size": max(s["size"] for s in spans),
        "bold": all(s["flags"] & 16 for s in spans),
        "bullet": bullet,
        "y0": line["bbox"][1],
        "y1": line["bbox"][3],
    }


def parse_page(page) -> dict:
    """One page -> header facts plus a list of content elements in order.

    Elements are {"kind": heading|para|bullet, "text", "bold", "size"}.
    Wrapped lines are joined into paragraphs using the vertical gap between
    lines, which is far more reliable than guessing from punctuation.
    """
    lines = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            rec = _line_record(line)
            if rec and rec["text"]:
                lines.append(rec)

    # --- running header: page number, "NCCN Guidelines for Patients(R)",
    # "<Title>, <year>", breadcrumb. Order varies by edition, so strip by
    # pattern from the top rather than by position.
    printed = None
    breadcrumb = None
    i = 0
    while i < len(lines) and i < 8:
        t = lines[i]["text"]
        if PAGE_NUMBER.match(t):
            printed = printed or t.strip()
        elif "NCCN Guidelines for Patients" in t or "NCCN GUIDELINES FOR PATIENTS" in t:
            pass
        elif EDITION_LINE.search(t) and lines[i]["size"] < HEADING_MIN_SIZE + 1:
            pass
        elif "»" in t or (CHAPTER_LINE.match(t) and lines[i]["size"] < 16):
            breadcrumb = t
        else:
            break
        i += 1
    body = lines[i:]

    elements: list[dict] = []
    prev = None
    for ln in body:
        is_heading = (
            ln["size"] >= HEADING_MIN_SIZE and not ln["bullet"] and len(ln["text"]) < 120
        )
        kind = "heading" if is_heading else ("bullet" if ln["bullet"] else "para")
        if elements and prev is not None and not ln["bullet"]:
            last = elements[-1]
            # Continue the previous element when this line sits directly under
            # it in the same style: a wrapped paragraph or heading line, or a
            # bullet's continuation lines (which carry no bullet glyph).
            continues = last["kind"] == kind or (last["kind"] == "bullet" and kind == "para")
            gap = ln["y0"] - prev["y1"]
            same_style = abs(ln["size"] - prev["size"]) < 0.5 and ln["bold"] == prev["bold"]
            # Large headings are set with looser leading than body text.
            limit = (1.3 if kind == "heading" else 0.6) * ln["size"]
            if continues and same_style and -2 <= gap < limit:
                sep = "" if last["text"].endswith("-") and last["text"][-2:-1].isalpha() else " "
                last["text"] = (last["text"] + sep + ln["text"]).strip()
                prev = ln
                continue
        elements.append(
            {"kind": kind, "text": ln["text"], "bold": ln["bold"], "size": round(ln["size"], 1)}
        )
        prev = ln
    return {"printed": printed, "breadcrumb": breadcrumb, "elements": elements}


def _breadcrumb_parts(crumb: str | None) -> list[str]:
    if not crumb:
        return []
    crumb = re.sub(r"^\d{1,2}[ \s]+", "", tidy(crumb))
    return [p.strip() for p in crumb.split("»") if p.strip()]


def synthetic_toc(pages: list[dict]) -> list[tuple[int, str, int]]:
    """An outline rebuilt from breadcrumb running headers.

    "Chapter » Section » Subsection" on a page means each component that was
    not on the previous page starts here: the chapter at level 1, the rest at
    level 2. Pages with no breadcrumb are chapter openers (a mini contents
    list) or back matter; the first heading tells which.
    """
    toc: list[tuple[int, str, int]] = []
    prev: list[str] = []
    started = False
    last_l1 = None
    for pno, pg in enumerate(pages, start=1):
        parts = _breadcrumb_parts(pg["breadcrumb"])
        if parts:
            started = True
            if not prev or parts[0] != prev[0]:
                toc.append((1, parts[0], pno))
                last_l1 = parts[0]
            for depth, part in enumerate(parts[1:], start=1):
                if depth >= len(prev) or prev[depth] != part:
                    toc.append((2, part, pno))
            prev = parts
            continue
        if not started:
            continue  # cover, supporters, contents
        first = next((e["text"] for e in pg["elements"] if e["kind"] == "heading" and len(e["text"]) > 3), "")
        if norm(first) == "wordstoknow" or norm(first) in {norm(x) for x in EXCLUDE_TITLES}:
            title = first
        else:
            title = "Chapter contents"
        if title != last_l1:
            toc.append((1, title, pno))
            last_l1 = title
        prev = []
    return toc


def add_back_matter(toc: list, pages: list[dict]) -> list:
    """Make sure the glossary and back matter start sections of their own.

    Several outlines stop at the last "Questions to ask", so without this the
    final section would run on through Words to know, NCCN Contributors and
    the Index. Their pages are recognised by their first heading.
    """
    have = {norm(title) for _, title, _ in toc}
    wanted = {"wordstoknow"} | {norm(x) for x in EXCLUDE_TITLES}
    extra = []
    for pno, pg in enumerate(pages, start=1):
        first = next((e["text"] for e in pg["elements"] if e["kind"] == "heading" and len(e["text"]) > 3), "")
        key = norm(first)
        if key in wanted and key not in have:
            extra.append((1, first, pno))
            have.add(key)
    return sorted(toc + extra, key=lambda e: e[2])


def _render(elements: list[dict]) -> str:
    """Elements -> light Markdown: paragraphs, '- ' bullets, bold sub-headings."""
    out: list[str] = []
    for el in elements:
        text = el["text"].strip()
        if not text:
            continue
        if el["kind"] == "bullet":
            out.append(f"- {text}")
        elif el["kind"] == "heading":
            out.append(f"\n**{text}**\n")
        else:
            out.append(f"\n{text}\n")
    md = "\n".join(out)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def _glossary(elements: list[dict]) -> list[dict]:
    """'Words to know': a bold term line followed by its plain definition."""
    terms: list[dict] = []
    for el in elements:
        if el["kind"] == "heading" and norm(el["text"]) == "wordstoknow":
            continue
        # Bold is what marks a term; some guides set the whole glossary in
        # heading-sized type, so the element kind cannot be relied on.
        if el["bold"] and el["kind"] != "bullet" and len(el["text"]) < 80:
            terms.append({"term": el["text"].strip(), "definition": ""})
        elif terms and not el["bold"]:
            d = terms[-1]["definition"]
            terms[-1]["definition"] = (d + " " + el["text"]).strip()
    return [t for t in terms if t["definition"]]


def parse_guide(guide: dict) -> dict:
    """Read one guide and return its sections and glossary. Worker entry point."""
    import fitz  # PyMuPDF; imported here so the parent need not load it

    doc = fitz.open(ROOT / guide["path"])
    try:
        pages = [parse_page(doc[i]) for i in range(doc.page_count)]
        # Some outlines are stored out of page order (Throat, Bladder); the
        # walk below needs them in reading order. sorted() is stable, so
        # entries sharing a page keep their outline order.
        toc = sorted(
            ((lvl, tidy(title), page) for lvl, title, page in doc.get_toc() if lvl <= 2),
            key=lambda e: e[2],
        )
    finally:
        doc.close()

    # A flat stream of (pdf_page, element) so a section can span pages.
    stream: list[tuple[int, dict]] = []
    page_start_idx: list[int] = []
    for pno, pg in enumerate(pages, start=1):
        page_start_idx.append(len(stream))
        for el in pg["elements"]:
            stream.append((pno, el))

    # --- section boundaries -------------------------------------------------
    # The 8 guides without an outline get one built from their breadcrumbs,
    # so both kinds go through the same heading-locating code below.
    has_outline = bool(toc)
    if not toc:
        toc = synthetic_toc(pages)
    toc = add_back_matter(toc, pages)

    # Each entry: (title, level, start index into stream, excluded?)
    entries: list[tuple[str, int, int, bool]] = []
    excluded_titles = {norm(t) for t in EXCLUDE_TITLES}
    last_pos = 0
    skipping = False
    for lvl, title, page in toc:
        if not 1 <= page <= len(pages):
            continue
        if lvl == 1:
            skipping = norm(title) in excluded_titles
        excluded = skipping or norm(title) in excluded_titles
        lo = page_start_idx[page - 1]
        hi = page_start_idx[page] if page < len(pages) else len(stream)
        pos = lo
        target = norm(title)
        for j in range(max(lo, last_pos), hi):
            el = stream[j][1]
            h = norm(el["text"])
            # Either may be the longer: a wrapped heading can be split into
            # two elements, and an outline title can be abbreviated.
            if el["kind"] == "heading" and len(h) >= 4 and target and (
                h.startswith(target) or target.startswith(h)
            ):
                pos = j
                break
        pos = max(pos, last_pos)
        entries.append((title, lvl, pos, excluded))
        last_pos = pos

    # --- materialise sections ----------------------------------------------
    # In guides that carry breadcrumbs, every content page has one; the pages
    # without are chapter openers (a mini contents list), blanks and back
    # matter -- measured, none of them over 120 words. Their text would only
    # add "5 What is distress? 6 How is distress managed?" noise to a section.
    has_crumbs = any(pg["breadcrumb"] for pg in pages)
    sections: list[dict] = []
    glossary: list[dict] = []
    chapter = None
    for k, (title, lvl, start, excluded) in enumerate(entries):
        end = entries[k + 1][2] if k + 1 < len(entries) else len(stream)
        if lvl == 1 and not re.fullmatch(r"(?i)chapter\s+\d+", title):
            chapter = title
        if excluded or end <= start:
            continue
        chunk = stream[start:end]
        is_glossary = section_kind(title) == "glossary"
        if has_crumbs and not is_glossary:
            chunk = [(pno, el) for pno, el in chunk if pages[pno - 1]["breadcrumb"]]
        # Drop the heading that opened the section; the title carries it.
        if chunk and chunk[0][1]["kind"] == "heading" and norm(chunk[0][1]["text"]).startswith(norm(title)[:20]):
            chunk = chunk[1:]
        elements = [el for _, el in chunk]
        if not elements:
            continue
        first_page = chunk[0][0]
        last_page = chunk[-1][0]
        crumb = next(
            (_breadcrumb_parts(pages[p - 1]["breadcrumb"]) for p in range(first_page, last_page + 1)
             if pages[p - 1]["breadcrumb"]),
            [],
        )
        chap = crumb[0] if crumb else chapter
        # Outlines that number chapters ("Chapter 1") name them only in the
        # breadcrumb, so take the name from there.
        if re.fullmatch(r"(?i)chapter\s+\d+", title):
            title = chap or title
        # Three guides have no breadcrumbs either; their chapter opener's own
        # heading is the only name it has.
        if re.fullmatch(r"(?i)chapter\s+\d+", title):
            heading = next(
                (el["text"] for el in elements if el["kind"] == "heading" and len(el["text"]) > 3),
                None,
            )
            title = heading or "Introduction"
            chapter = chap = title
        kind = section_kind(title)
        if kind == "glossary":
            glossary.extend(
                dict(t, page=first_page) for t in _glossary(elements)
            )
            continue
        text = _render(elements)
        if len(text) < 40:
            continue
        sections.append(
            {
                "section_id": f"{guide['id']}_p{first_page:03d}_{k:03d}",
                "guide_id": guide["id"],
                "title": title,
                "chapter": chap or title,
                "kind": kind,
                "pdf_page_start": first_page,
                "pdf_page_end": last_page,
                "printed_page_start": pages[first_page - 1]["printed"],
                "printed_page_end": pages[last_page - 1]["printed"],
                "bullets": [el["text"] for el in elements if el["kind"] == "bullet"],
                "text": text,
            }
        )
    return {
        "guide_id": guide["id"],
        "sections": sections,
        "glossary": glossary,
        "pages": len(pages),
        "outline": has_outline,
    }


# ---------------------------------------------------------------------------
# Chunking (runs in worker processes)
# ---------------------------------------------------------------------------

_W: dict = {}


def _load_stage2():
    """Import scripts/02_build_chunks.py (its name is not a valid identifier)."""
    spec = importlib.util.spec_from_file_location("stage2", ROOT / "scripts" / "02_build_chunks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _init_chunker(model_name: str, chunk_cfg: dict, doc_prefix: str, guides: dict) -> None:
    stage2 = _load_stage2()
    _W["stage2"] = stage2
    _W["count_tokens"], _ = stage2.load_tokenizer(model_name)
    _W["cfg"] = chunk_cfg
    _W["prefix"] = doc_prefix
    _W["guides"] = guides


def split_units(stage2, text: str, count_tokens, max_tokens: int) -> list[str]:
    """Paragraph units, with oversized bullet lists broken at item boundaries.

    Stage 2 splits an oversized unit on sentence ends, which a list of drug
    names does not have -- it would go out as one 1,300-token chunk. Lists
    are therefore regrouped line by line here, before packing.
    """
    out: list[str] = []
    for unit in stage2.split_paragraphs(text):
        if count_tokens(unit) <= max_tokens or "\n" not in unit:
            out.append(unit)
            continue
        buf: list[str] = []
        n = 0
        for line in unit.split("\n"):
            ln = count_tokens(line)
            if buf and n + ln > max_tokens:
                out.append("\n".join(buf))
                buf, n = [], 0
            buf.append(line)
            n += ln
        if buf:
            out.append("\n".join(buf))
    return out


def citation(guide: dict, sec: dict) -> str:
    page = sec.get("printed_page_start") or sec["pdf_page_start"]
    return f"NCCN Patients: {guide['title']} ({guide['year']}), {sec['title']}, p.{page}"


def chunk_section(sec: dict) -> list[dict]:
    stage2 = _W["stage2"]
    count_tokens = _W["count_tokens"]
    cfg = _W["cfg"]
    guide = _W["guides"][sec["guide_id"]]
    label = guide_label(guide)
    breadcrumb = " > ".join([label, sec["chapter"], sec["title"]] if sec["chapter"] != sec["title"]
                            else [label, sec["title"]])
    rows = []
    units = split_units(stage2, sec["text"], count_tokens, cfg["max_tokens"])
    for i, text in enumerate(stage2.pack(units, count_tokens, cfg)):
        n = count_tokens(text)
        if n < cfg["min_tokens"]:
            continue
        rows.append(
            {
                "chunk_id": hashlib.sha1(f"{sec['section_id']}|{i}|{text[:120]}".encode()).hexdigest()[:16],
                "section_id": sec["section_id"],
                "text": text,
                "embed_text": f"{_W['prefix']}{breadcrumb}\n\n{text}",
                "breadcrumb": breadcrumb,
                "book": f"NCCN Guidelines for Patients: {guide['title']} ({guide['year']})",
                "book_short": label,
                "chapter": sec["chapter"],
                "section_path": [sec["title"]],
                "pdf_page_start": sec["pdf_page_start"],
                "pdf_page_end": sec["pdf_page_end"],
                "printed_page_start": sec["printed_page_start"],
                "printed_page_end": sec["printed_page_end"],
                "tier": "patient",
                "citation": citation(guide, sec),
                "is_table": False,
                "n_tokens": n,
                # The retriever scopes on these two keys; for the patient
                # index they mean "guide" and "topic area".
                "pathology": guide["id"],
                "organ_system": guide["category"],
                "guide_id": guide["id"],
                "guide_title": guide["title"],
                "year": guide["year"],
                "category": guide["category"],
                "section_kind": sec["kind"],
                "clinical_pathologies": guide.get("pathologies") or [],
            }
        )
    return rows


def glossary_chunks(guide: dict, terms: list[dict], count_tokens, cfg: dict, prefix: str) -> list[dict]:
    """Pack a guide's glossary into a few chunks, so "what does adjuvant mean?"
    is answerable and cites the guide's own definition."""
    rows: list[dict] = []
    label = guide_label(guide)
    buf: list[str] = []
    n = 0
    pages: list[int] = []

    def flush():
        nonlocal buf, n, pages
        if not buf:
            return
        text = "\n".join(buf)
        sec = {"title": "Words to know", "pdf_page_start": min(pages), "printed_page_start": None}
        rows.append(
            {
                "chunk_id": hashlib.sha1(f"{guide['id']}|gloss|{text[:120]}".encode()).hexdigest()[:16],
                "section_id": f"{guide['id']}_glossary_{len(rows):02d}",
                "text": text,
                "embed_text": f"{prefix}{label} > Words to know\n\n{text}",
                "breadcrumb": f"{label} > Words to know",
                "book": f"NCCN Guidelines for Patients: {guide['title']} ({guide['year']})",
                "book_short": label,
                "chapter": "Words to know",
                "section_path": ["Words to know"],
                "pdf_page_start": min(pages),
                "pdf_page_end": max(pages),
                "printed_page_start": None,
                "printed_page_end": None,
                "tier": "patient",
                "citation": citation(guide, sec),
                "is_table": False,
                "n_tokens": n,
                "pathology": guide["id"],
                "organ_system": guide["category"],
                "guide_id": guide["id"],
                "guide_title": guide["title"],
                "year": guide["year"],
                "category": guide["category"],
                "section_kind": "glossary",
                "clinical_pathologies": guide.get("pathologies") or [],
            }
        )
        buf, n, pages = [], 0, []

    for t in terms:
        line = f"{t['term']}: {t['definition']}"
        tn = count_tokens(line)
        if buf and n + tn > cfg["max_tokens"]:
            flush()
        buf.append(line)
        n += tn
        pages.append(t["page"])
    flush()
    return rows


def guide_label(guide: dict) -> str:
    """Short source label shown on source cards, e.g. 'NCCN · Mouth Cancer'."""
    return f"NCCN · {guide['title']}"


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                    help="processes for parsing and chunking (default: all cores)")
    ap.add_argument("--guide", help="only this guide id (for checking the parser)")
    ap.add_argument("--config", default="config/patient.yaml")
    args = ap.parse_args()

    pcfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    rag = yaml.safe_load((ROOT / "config" / "rag.yaml").read_text(encoding="utf-8"))
    registry = yaml.safe_load((ROOT / pcfg["paths"]["guides"]).read_text(encoding="utf-8"))
    guides = {g["id"]: g for g in registry["guides"]}
    todo = [g for g in registry["guides"] if not args.guide or g["id"] == args.guide]
    if not todo:
        print(f"unknown guide: {args.guide}", file=sys.stderr)
        return 1
    missing = [g["path"] for g in todo if not (ROOT / g["path"]).exists()]
    if missing:
        print("missing PDFs (copy sources/NCCN into data/nccn):\n  " + "\n  ".join(missing), file=sys.stderr)
        return 1

    # Biggest first: equal-sized tasks would leave cores idle at the end.
    todo.sort(key=lambda g: -g["pages"])
    workers = max(1, min(args.workers, len(todo)))
    t0 = time.time()
    print(f"parsing {len(todo)} guides ({sum(g['pages'] for g in todo):,} pages) on {workers} workers")
    results: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(parse_guide, todo):
            results[res["guide_id"]] = res
            print(f"  {res['guide_id']:<30} {len(res['sections']):>3} sections "
                  f"{len(res['glossary']):>3} terms  {'outline' if res['outline'] else 'breadcrumbs'}",
                  flush=True)
    print(f"parsed in {time.time() - t0:.1f}s")

    # Registry order, so outputs are deterministic regardless of scheduling.
    order = [g["id"] for g in registry["guides"] if g["id"] in results]
    sections = [s for gid in order for s in results[gid]["sections"]]

    out_dir = ROOT / pcfg["paths"]["corpus_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with (ROOT / pcfg["paths"]["sections_file"]).open("w", encoding="utf-8") as fh:
        for s in sections:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    # --- library.json: the Learn tab, NCCN text only -----------------------
    library = {"categories": registry["categories"], "guides": []}
    for gid in order:
        g = guides[gid]
        secs = results[gid]["sections"]
        chapters: list[dict] = []
        for s in secs:
            if not chapters or chapters[-1]["title"] != s["chapter"]:
                chapters.append({"title": s["chapter"], "sections": []})
            chapters[-1]["sections"].append(
                {k: s[k] for k in ("section_id", "title", "kind", "text", "bullets",
                                   "pdf_page_start", "pdf_page_end", "printed_page_start")}
            )
        library["guides"].append(
            {
                "id": gid,
                "title": g["title"],
                "year": g["year"],
                "category": g["category"],
                "pathologies": g.get("pathologies") or [],
                "label": guide_label(g),
                "pages": results[gid]["pages"],
                "chapters": chapters,
                "glossary": results[gid]["glossary"],
                "counts": dict(Counter(s["kind"] for s in secs)),
            }
        )
    (ROOT / pcfg["paths"]["library_file"]).write_text(
        json.dumps(library, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    # --- chunks --------------------------------------------------------------
    t1 = time.time()
    model_name = rag["embedding"]["model"]
    prefix = rag["embedding"].get("doc_prefix", "")
    chunk_cfg = pcfg["chunking"]
    print(f"chunking {len(sections):,} sections on {workers} workers (tokenizer: {model_name})")
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_chunker,
        initargs=(model_name, chunk_cfg, prefix, guides),
    ) as pool:
        chunk_lists = list(pool.map(chunk_section, sections, chunksize=16))
    rows = [r for lst in chunk_lists for r in lst]

    stage2 = _load_stage2()
    count_tokens, _ = stage2.load_tokenizer(model_name)
    for gid in order:
        rows.extend(glossary_chunks(guides[gid], results[gid]["glossary"], count_tokens, chunk_cfg, prefix))

    with (ROOT / pcfg["paths"]["chunks_file"]).open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"chunked in {time.time() - t1:.1f}s")

    # --- report ----------------------------------------------------------------
    kinds = Counter(s["kind"] for s in sections)
    by_cat = Counter(guides[r["guide_id"]]["category"] for r in rows)
    no_faq = [gid for gid in order if not any(s["kind"] == "faq" for s in results[gid]["sections"])]
    no_gloss = [gid for gid in order if not results[gid]["glossary"]]
    print()
    print(f"guides     : {len(order)}")
    print(f"sections   : {len(sections):,}  {dict(kinds)}")
    print(f"glossary   : {sum(len(results[g]['glossary']) for g in order):,} terms")
    print(f"chunks     : {len(rows):,}  by topic area {dict(by_cat)}")
    tokens = sorted(r["n_tokens"] for r in rows)
    if tokens:
        print(f"tokens     : median {tokens[len(tokens) // 2]}, max {tokens[-1]}")
    if no_faq:
        print(f"no FAQ-style sections: {', '.join(no_faq)}")
    if no_gloss:
        print(f"no glossary found    : {', '.join(no_gloss)}")
    print(f"\nwrote {pcfg['paths']['sections_file']}, {pcfg['paths']['library_file']}, "
          f"{pcfg['paths']['chunks_file']}")
    print("next: python scripts/03_build_index.py --chunks corpus/patient/chunks.jsonl "
          "--out-dir artifacts/index_patient")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
