#!/usr/bin/env python3
"""Stage 1 -- turn the source textbooks into a clean, sectioned corpus.

Reads the PDFs listed in config/books.yaml and writes readable Markdown
sections plus a JSONL manifest and an audit report, to the paths in
config/rag.yaml.

`selection.mode` decides how much is kept: "full" keeps both whole books
(minus index, bibliography and blank pages), "tongue" keeps only
tongue-cancer content. See docs/pipeline.md.

CPU only. Page text extraction is spread across worker processes, so the run
is bounded by the slowest core rather than by a single one.

    python scripts/01_extract_corpus.py
    python scripts/01_extract_corpus.py --book gunderson   # one book only
    python scripts/01_extract_corpus.py --limit-pages 200  # quick smoke test
    python scripts/01_extract_corpus.py --workers 4        # cap parallelism
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path

import fitz  # PyMuPDF
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Page:
    """One PDF page after extraction and classification."""

    pdf_page: int  # 1-based, matches a PDF viewer's page box
    text: str
    kind: str = "body"  # body | index | references | figure_only | blank
    printed_page: str | None = None
    chapter_num: int | None = None
    chapter_title: str | None = None
    tongue_hits: int = 0
    support_hits: int = 0
    in_window: bool = False


@dataclass
class Section:
    """A contiguous run of text under one heading, the unit we store and cite."""

    book_id: str
    book_title: str
    book_short: str
    chapter_num: int | None
    chapter_title: str | None
    section_path: list[str]
    text: str
    pdf_page_start: int
    pdf_page_end: int
    printed_page_start: str | None = None
    printed_page_end: str | None = None
    tier: str = "context"  # core | context
    reason: str = ""
    section_id: str = ""
    char_count: int = 0
    has_table: bool = False

    def breadcrumb(self) -> str:
        """Human-readable trail used as the chunk prefix and in source cards."""
        parts = [self.book_short]
        if self.chapter_num is not None:
            parts.append(f"ch.{self.chapter_num} {self.chapter_title or ''}".strip())
        parts.extend(self.section_path)
        return " > ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

# Trailing superscript citation runs that PyMuPDF flattens into the prose.
# The pattern to picture is a sentence ending, then reference numbers with no
# space, then the next sentence: "...drain to the deep nodes.12 Lymphatic" or
# "...are squamous carcinomas.7,31 The results". We drop the digits but keep
# the sentence punctuation. (Illustration written for this comment rather
# than quoted, so the repository carries no text from the source books.)
_CITATION_RUN = re.compile(r"(?<=[a-z\)\]])([.,;:])\s?\d{1,3}(?:[-,–]\d{1,3})*(?=\s|$)")

# Superscript markers fused onto the preceding word, e.g. "Rouviere15 described".
# Only stripped when the word is >=4 letters and not an all-caps acronym, so
# that real tokens like p16, HPV16, T1, N2, CD8 and 5FU survive untouched.
_CITATION_ATTACHED = re.compile(r"\b([A-Za-zÀ-ÿ]{4,})(\d{1,3}(?:[,–-]\d{1,3})*)\b")


def _drop_attached_citation(m: re.Match) -> str:
    word = m.group(1)
    return word if not word.isupper() else m.group(0)

_HYPHEN_BREAK = re.compile(r"(\w)[-‐‑]\n(\w)")
_MULTI_BLANK = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")

_DASHES = {"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "--"}
_QUOTES = {"‘": "'", "’": "'", "“": '"', "”": '"'}


def normalise(text: str) -> str:
    """Unicode-normalise and flatten typographic characters."""
    text = unicodedata.normalize("NFKC", text)
    for src, dst in {**_DASHES, **_QUOTES}.items():
        text = text.replace(src, dst)
    text = text.replace(" ", " ").replace("ﬁ", "fi").replace("ﬂ", "fl")
    return text


def clean_page_text(text: str, strip_patterns: list[re.Pattern]) -> str:
    """Strip watermarks, rejoin hyphenated line breaks, drop citation runs."""
    for pat in strip_patterns:
        text = pat.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _CITATION_RUN.sub(r"\1", text)
    text = _CITATION_ATTACHED.sub(_drop_attached_citation, text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


_LABEL_FRAGMENT = re.compile(r"^[A-Z0-9][A-Z0-9./%-]{0,4}$")


def drop_figure_labels(lines: list[str]) -> list[str]:
    """Remove graphic labels lifted out of figure artwork ("A", "B", "TSD").

    These are short, all-caps, standalone lines that PyMuPDF picks up from
    the figure itself. Only lines sitting within five lines of a figure
    caption are dropped, so short table cells elsewhere are left alone.
    """
    fig_at = [i for i, ln in enumerate(lines) if _FIG_HEAD.match(ln)]
    if not fig_at:
        return lines
    near = {j for i in fig_at for j in range(i - 5, i + 6)}
    return [
        ln
        for i, ln in enumerate(lines)
        if not (i in near and _LABEL_FRAGMENT.match(ln.strip()))
    ]


def reflow(text: str) -> str:
    """Undo the PDF's hard line wrapping so sections read as real paragraphs.

    A line is joined to the previous one unless the previous line ends a
    sentence, or either line is a table/figure caption -- captions and table
    rows must keep their own breaks or the table becomes unreadable.
    """
    out: list[str] = []
    for raw in drop_figure_labels(text.splitlines()):
        line = raw.rstrip()
        if not line.strip():
            out.append("")
            continue
        structural = bool(_TABLE_HEAD.match(line) or _FIG_HEAD.match(line))
        if structural and out and out[-1].strip():
            # Blank line before a caption so Stage 2 sees the table as its own
            # block and never splits a caption away from its rows.
            out.append("")
        if (
            out
            and out[-1]
            and not structural
            and not _TABLE_HEAD.match(out[-1])
            and not _FIG_HEAD.match(out[-1])
            # Previous line did not finish a sentence -> same paragraph.
            and not out[-1].rstrip().endswith((".", "!", "?", ":", ";"))
            and line[0].islower()
        ):
            out[-1] = out[-1].rstrip() + " " + line.lstrip()
        else:
            out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Page classification
# ---------------------------------------------------------------------------

# "Somatostatinomas, 1363" / "for glottic larynx, 573, 573t" / "irinotecan, 279f, 280t"
_INDEX_ENTRY = re.compile(r"^\s*\S.*,\s*\d{1,4}[a-z]?(\s*[-–,]\s*\d{1,4}[a-z]?)*\s*$")
# "12. Smith J, Jones A, et al. Lancet 2010;376(9742):687-697."
_REF_ENTRY = re.compile(r"^\s*\d{1,3}\.\s+[A-Z]")
_REF_TAIL = re.compile(r"\b(19|20)\d{2};\d+")


def classify_page(text: str, cfg: dict) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "blank"

    index_like = sum(1 for ln in lines if _INDEX_ENTRY.match(ln))
    if index_like / len(lines) >= cfg["index_line_ratio"]:
        return "index"

    ref_like = sum(1 for ln in lines if _REF_ENTRY.match(ln))
    # Continuation lines of a bibliography entry are indented and carry a
    # journal/year token, so count them too before taking the ratio.
    ref_like += sum(1 for ln in lines if _REF_TAIL.search(ln) and not _REF_ENTRY.match(ln))
    if ref_like / len(lines) >= cfg["reference_line_ratio"]:
        return "references"

    prose = sum(len(ln) for ln in lines if len(ln.split()) >= 6)
    if prose < cfg["min_prose_chars"]:
        return "figure_only"

    return "body"


def find_running_lines(pages: list[str], min_pages: int) -> set[str]:
    """Lines appearing verbatim on many pages are headers/footers, not content."""
    counter: Counter[str] = Counter()
    for text in pages:
        seen = set()
        for ln in text.splitlines()[:3] + text.splitlines()[-3:]:
            s = ln.strip()
            if 3 < len(s) < 90:
                seen.add(s)
        counter.update(seen)
    return {line for line, n in counter.items() if n >= min_pages}


# ---------------------------------------------------------------------------
# Lexicon matching
# ---------------------------------------------------------------------------


def build_lexicon_regex(terms: list[str]) -> re.Pattern:
    escaped = sorted((re.escape(t) for t in terms), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Heading detection / section splitting
# ---------------------------------------------------------------------------

# A heading is a short standalone line, mostly title-case or all-caps, with no
# terminal sentence punctuation. Both books format headings this way.
_ALL_CAPS = re.compile(r"^[A-Z][A-Z0-9 ,'\-/&():]{3,70}$")
_TITLE_CASE = re.compile(r"^[A-Z][A-Za-z0-9]*(?:[ \-/&'(),:]+[A-Za-z0-9][A-Za-z0-9]*)*[.]?$")
_TABLE_HEAD = re.compile(r"^\s*(TABLE|eTABLE|Table)\s+\d+", re.IGNORECASE)
_FIG_HEAD = re.compile(r"^\s*(eFig|Fig|FIGURE|Figure)\.?\s*\d+", re.IGNORECASE)


def heading_level(line: str) -> int | None:
    """Return 1 for a major (ALL CAPS) heading, 2 for a minor one, else None."""
    s = line.strip()
    if not (3 < len(s) <= 75):
        return None
    if s.endswith((".", ";", ",", ":")) and not _ALL_CAPS.match(s):
        return None
    words = s.split()
    if not (1 <= len(words) <= 10):
        return None
    if _TABLE_HEAD.match(s) or _FIG_HEAD.match(s):
        return None
    # A line ending in a digit is usually a stray running header remnant.
    if s[-1].isdigit():
        return None
    if _ALL_CAPS.match(s) and s.upper() == s:
        return 1
    if _TITLE_CASE.match(s):
        # Reject lines that read as sentences: a lower-case function word
        # early on plus many words means it is prose, not a heading.
        lowers = sum(1 for w in words if w[0].islower())
        if lowers <= max(1, len(words) // 3):
            return 2
    return None


def split_into_sections(pages: list[Page], book: dict) -> list[Section]:
    """Walk the pages in order, cutting a new section at each heading."""
    sections: list[Section] = []
    path: list[str] = []
    buf: list[str] = []
    start_page = pages[0].pdf_page if pages else 0
    start_printed = pages[0].printed_page if pages else None
    cur_chap_num = pages[0].chapter_num if pages else None
    cur_chap_title = pages[0].chapter_title if pages else None
    end_page = start_page
    end_printed = start_printed
    has_table = False

    def flush() -> None:
        nonlocal buf, has_table
        # end_page is whatever page we are currently on, so a section that is
        # cut by a heading at the top of a new page still ends on the page it
        # actually finished on -- never before its own start page.
        body = reflow("\n".join(buf)).strip()
        if body:
            sections.append(
                Section(
                    book_id=book["id"],
                    book_title=book["title"],
                    book_short=book["short"],
                    chapter_num=cur_chap_num,
                    chapter_title=cur_chap_title,
                    section_path=list(path) or ["(chapter body)"],
                    text=body,
                    pdf_page_start=start_page,
                    pdf_page_end=max(end_page, start_page),
                    printed_page_start=start_printed,
                    printed_page_end=end_printed,
                    has_table=has_table,
                )
            )
        buf = []
        has_table = False

    prev_page: int | None = None
    for page in pages:
        # A gap in page numbers means the pages either side are unrelated --
        # the whole-book sweep pulls in isolated pages hundreds of pages from
        # the last kept one. Without a break here the open section swallows
        # them and reports a span like "p.1110-3671", so every chunk in it
        # cites a page its text did not come from.
        if prev_page is not None and page.pdf_page > prev_page + 1:
            flush()
            path = []
            start_page = page.pdf_page
            start_printed = page.printed_page
        prev_page = page.pdf_page

        # Advance the end marker before emitting anything from this page, so a
        # flush triggered mid-page records this page as the end, not the last.
        end_page = page.pdf_page
        end_printed = page.printed_page

        # A chapter change forces a section break and resets the heading path.
        # Note the comparison does NOT skip None: a page that belongs to no
        # known chapter must clear the current one, not inherit it. Before
        # this, once a chapter was set it was never unset, so in full-book
        # mode three quarters of DeVita's chunks were labelled with whichever
        # chapter happened to be current when the hand-written map ran out --
        # every one of those citations named the wrong chapter.
        if page.chapter_num != cur_chap_num:
            flush()
            cur_chap_num = page.chapter_num
            cur_chap_title = page.chapter_title
            path = []
            start_page = page.pdf_page
            start_printed = page.printed_page

        for line in page.text.splitlines():
            if _TABLE_HEAD.match(line):
                has_table = True
            level = heading_level(line)
            if level is not None and buf:
                flush()
                start_page = page.pdf_page
                start_printed = page.printed_page
            if level == 1:
                path = [line.strip()]
                continue
            if level == 2:
                path = (path[:1] if path else []) + [line.strip()]
                continue
            buf.append(line)

    flush()
    return sections


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def in_any_window(page_no: int, windows: list[dict]) -> bool:
    return any(w["start"] <= page_no <= w["end"] for w in windows)


def chapter_map_from_outline(pdf_path: Path, total_pages: int) -> list[dict]:
    """Chapter number -> page range, read from the PDF's embedded outline.

    Both books ship a full outline, which is the publisher's own structure and
    covers every chapter. A hand-written `chapter_map` in books.yaml only ever
    covered the head & neck chapters, which was fine while only those pages
    were kept and silently wrong the moment the corpus went full-book.

    Returns [] when the PDF has no usable outline, so the caller can fall back.
    """
    doc = fitz.open(pdf_path)
    try:
        starts: dict[int, tuple[int, str]] = {}
        for level, title, page in doc.get_toc():
            if level > 2:
                continue
            clean = re.sub(r"^\d+\.\s*", "", title).strip()
            m = re.match(r"^(\d{1,3})\s+(\S.*)$", clean)
            if m:
                # First occurrence wins: the outline repeats a chapter number
                # for each of its subsections.
                starts.setdefault(int(m.group(1)), (page, m.group(2).strip()))
    finally:
        doc.close()

    ordered = sorted(starts.items(), key=lambda kv: kv[1][0])
    out: list[dict] = []
    for i, (num, (start, title)) in enumerate(ordered):
        end = ordered[i + 1][1][0] - 1 if i + 1 < len(ordered) else total_pages
        out.append({"num": num, "title": title, "start": start, "end": end})
    return out


def _read_page_block(task: tuple[str, int, int]) -> tuple[int, list[str]]:
    """Read pages [start, end) of a PDF and return their normalised text.

    Runs in a worker process. PyMuPDF documents cannot cross a process
    boundary, so each worker opens the file itself; MuPDF mmaps the pages it
    touches, which is why opening the 310 MB Gunderson PDF twelve times is
    cheap rather than twelve times the memory.
    """
    pdf_path, start, end = task
    doc = fitz.open(pdf_path)
    try:
        # "text" mode uses MuPDF's reading order, which handles Gunderson's
        # two-column layout correctly. Never use a layout-preserving mode here.
        return start, [normalise(doc[i].get_text("text")) for i in range(start, end)]
    finally:
        doc.close()


def read_pages_parallel(pdf_path: Path, total: int, workers: int) -> list[str]:
    """Extract `total` pages of text in page order, across `workers` processes."""
    if workers <= 1 or total < 64:
        doc = fitz.open(pdf_path)
        try:
            return [normalise(doc[i].get_text("text")) for i in range(total)]
        finally:
            doc.close()

    # Several blocks per worker: the head & neck chapters are far denser than
    # the front matter, so equal-sized blocks would leave cores idle at the end.
    block = max(16, total // (workers * 4))
    tasks = [
        (str(pdf_path), start, min(start + block, total))
        for start in range(0, total, block)
    ]

    raw: list[str] = [""] * total
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for start, texts in pool.map(_read_page_block, tasks):
            raw[start : start + len(texts)] = texts
            done += len(texts)
            print(f"    ...read {done}/{total} pages", end="\r", flush=True)
    print(f"    ...read {done}/{total} pages", flush=True)
    return raw


def extract_book(
    book: dict,
    cfg: dict,
    lex: dict,
    limit_pages: int | None,
    workers: int = 1,
    all_pages: bool = False,
) -> tuple[list[Page], dict]:
    pdf_path = ROOT / book["path"]
    if not pdf_path.exists():
        raise FileNotFoundError(f"missing source PDF: {pdf_path}")

    strip_patterns = [re.compile(p) for p in book.get("strip_patterns", [])]
    tongue_re = build_lexicon_regex(lex["tongue"])
    support_re = build_lexicon_regex(lex["support"])

    header_re = (
        re.compile(book["running_header_regex"]) if book.get("running_header_regex") else None
    )
    footer_re = (
        re.compile(book["running_footer_regex"]) if book.get("running_footer_regex") else None
    )
    page_re = (
        re.compile(book["printed_page_regex"]) if book.get("printed_page_regex") else None
    )
    chapter_map = book.get("chapter_map", [])
    if book.get("use_pdf_outline", True):
        derived = chapter_map_from_outline(pdf_path, book.get("pages", 10**6))
        if derived:
            chapter_map = derived
            print(f"    chapter map: {len(derived)} chapters from the PDF outline")
        elif chapter_map:
            print("    chapter map: PDF has no outline, using books.yaml")

    windows = book.get("windows", [])
    excludes = book.get("exclude_windows", [])
    pc = cfg["page_classifier"]
    sweep_min = cfg["selection"]["sweep_min_tongue_hits"]

    # --- pass 1: raw text in reading order (parallel) ----------------------
    doc = fitz.open(pdf_path)
    total = doc.page_count if limit_pages is None else min(doc.page_count, limit_pages)
    doc.close()
    raw = read_pages_parallel(pdf_path, total, workers)

    running = find_running_lines(raw, pc["running_line_min_pages"])

    # --- pass 2: classify, tag chapter/printed page, score relevance -------
    pages: list[Page] = []
    stats: dict = {
        "total_pages": total,
        "kinds": Counter(),
        "kept": 0,
        "dropped_excluded": 0,
        "dropped_not_selected": 0,
        "kept_by_window": 0,
        "kept_by_sweep": 0,
        "sweep_pages": [],
        "chapters": defaultdict(lambda: {"pages": 0, "range": [10**9, 0], "title": None}),
    }

    cur_chap_num: int | None = None
    cur_chap_title: str | None = None

    for i, text in enumerate(raw):
        page_no = i + 1
        printed: str | None = None
        lines = text.splitlines()

        # Chapter + printed page from the running header/footer (Gunderson).
        # Reading order puts these on two adjacent lines, in either order:
        #   "CHAPTER 38 Oral Cavity" / "611"      (recto)
        #   "612" / "SECTION III Disease Sites"   (verso)
        # so match the label line, then take the bare number beside it.
        header_lines: set[int] = set()
        for j, probe in enumerate(lines[:4]):
            m = header_re.match(probe) if header_re else None
            if m:
                cur_chap_num = int(m.group("num"))
                cur_chap_title = m.group("title").strip()
                header_lines.add(j)
            elif footer_re and footer_re.match(probe):
                header_lines.add(j)
            else:
                continue
            for k in (j + 1, j - 1):
                if 0 <= k < len(lines) and page_re and page_re.match(lines[k]):
                    printed = page_re.match(lines[k]).group("page")
                    header_lines.add(k)
                    break
            break

        # Drop the running header/footer lines so they never reach the corpus.
        if header_lines:
            lines = [ln for j, ln in enumerate(lines) if j not in header_lines]

        # Books without a running header (DeVita) use explicit page ranges.
        if chapter_map:
            hit = next(
                (c for c in chapter_map if c["start"] <= page_no <= c["end"]), None
            )
            cur_chap_num = hit["num"] if hit else None
            cur_chap_title = hit["title"] if hit else None

        body = "\n".join(ln for ln in lines if ln.strip() not in running)
        body = clean_page_text(body, strip_patterns)
        kind = classify_page(body, pc)
        stats["kinds"][kind] += 1

        page = Page(
            pdf_page=page_no,
            text=body,
            kind=kind,
            printed_page=printed,
            chapter_num=cur_chap_num,
            chapter_title=cur_chap_title,
            tongue_hits=len(tongue_re.findall(body)),
            support_hits=len(support_re.findall(body)),
            in_window=in_any_window(page_no, windows),
        )

        if in_any_window(page_no, excludes):
            stats["dropped_excluded"] += 1
            continue
        if kind in ("index", "references", "blank"):
            continue
        if kind == "figure_only":
            # Keep only the captions -- they carry real information
            # (of the form "Fig. 12.3 <what the figure shows>").
            caps = [ln for ln in body.splitlines() if _FIG_HEAD.match(ln) or _TABLE_HEAD.match(ln)]
            if not caps:
                continue
            page.text = "\n".join(caps)

        if all_pages:
            # Full-book mode: every page of real prose is kept. Index,
            # bibliography and blank pages have already been dropped above --
            # they are noise in any mode, not just the tongue-focused one.
            selected = True
            stats["kept_by_window" if page.in_window else "kept_by_sweep"] += 1
        else:
            selected = False
            if page.in_window:
                selected = True
                stats["kept_by_window"] += 1
            elif page.tongue_hits >= sweep_min:
                # Whole-book sweep: relevant material outside the H&N chapters.
                selected = True
                stats["kept_by_sweep"] += 1
                stats["sweep_pages"].append(page_no)

            if not selected:
                stats["dropped_not_selected"] += 1
                continue

        stats["kept"] += 1
        if cur_chap_num is not None:
            c = stats["chapters"][cur_chap_num]
            c["pages"] += 1
            c["title"] = cur_chap_title
            c["range"][0] = min(c["range"][0], page_no)
            c["range"][1] = max(c["range"][1], page_no)
        pages.append(page)

    return pages, stats


# ---------------------------------------------------------------------------
# Tiering and output
# ---------------------------------------------------------------------------


def tier_sections(
    sections: list[Section], lex: dict, min_chars: int, all_pages: bool = False
) -> list[Section]:
    tongue_re = build_lexicon_regex(lex["tongue"])
    support_re = build_lexicon_regex(lex["support"])
    kept: list[Section] = []

    for sec in sections:
        sec.char_count = len(sec.text)
        if sec.char_count < min_chars:
            continue
        # The heading trail matters as much as the body: a section titled
        # "Oral Tongue" whose prose only says "the lesion" is still core.
        heading = " ".join(sec.section_path)
        t_hits = len(tongue_re.findall(heading + " " + sec.text))
        s_hits = len(support_re.findall(heading + " " + sec.text))
        # Density, not raw count: one stray "tongue" in a 12k-character
        # laryngectomy section does not make that section about the tongue.
        t_needed = max(2, sec.char_count // 2000)

        if tongue_re.search(heading):
            sec.tier = "core"
            sec.reason = f"tongue term in heading ({heading})"
        elif t_hits >= t_needed:
            sec.tier = "core"
            sec.reason = f"tongue lexicon x{t_hits} (>= {t_needed} for {sec.char_count} chars)"
        elif t_hits >= 1 or s_hits >= 2:
            sec.tier = "context"
            sec.reason = f"tongue x{t_hits}, support lexicon x{s_hits}"
        elif all_pages:
            # Full-book mode keeps everything. The tier still records how
            # tongue-relevant the section is, so retrieval can prefer core
            # material, but nothing is discarded for being off-topic.
            sec.tier = "general"
            sec.reason = "full-book mode: not tongue-related"
        else:
            continue

        slug = slugify(" ".join(sec.section_path[-2:]) or "body")
        sec.section_id = f"{sec.book_id}_p{sec.pdf_page_start:04d}_{slug}"[:90]
        kept.append(sec)

    return kept


def slugify(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", text.lower()).strip("-")
    return re.sub(r"-{2,}", "-", s)[:48] or "section"


def write_sections(
    sections: list[Section], out_dir: Path, only_book: str | None = None
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Section ids are prefixed with the book id, so a single-book run clears
    # only its own output. Wiping "*.md" here would delete the other book's
    # sections, silently halving the corpus.
    pattern = f"{only_book}_*.md" if only_book else "*.md"
    for f in out_dir.glob(pattern):
        f.unlink()

    seen: Counter[str] = Counter()
    for sec in sections:
        seen[sec.section_id] += 1
        name = sec.section_id
        if seen[sec.section_id] > 1:
            name = f"{name}-{seen[sec.section_id]}"
        front = {
            "section_id": name,
            "book": sec.book_title,
            "book_short": sec.book_short,
            "chapter": (
                f"{sec.chapter_num} - {sec.chapter_title}" if sec.chapter_num else None
            ),
            "section_path": sec.section_path,
            "pdf_page_start": sec.pdf_page_start,
            "pdf_page_end": sec.pdf_page_end,
            "printed_page_start": sec.printed_page_start,
            "printed_page_end": sec.printed_page_end,
            "tier": sec.tier,
            "reason": sec.reason,
            "has_table": sec.has_table,
        }
        body = (
            "---\n"
            + yaml.safe_dump(front, sort_keys=False, allow_unicode=True)
            + "---\n\n"
            + f"# {sec.breadcrumb()}\n\n"
            + sec.text
            + "\n"
        )
        (out_dir / f"{name}.md").write_text(body, encoding="utf-8")
        sec.section_id = name


def write_manifest(sections: list[Section], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for sec in sections:
            row = asdict(sec)
            row["breadcrumb"] = sec.breadcrumb()
            row.pop("text")  # text lives in the .md files
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(
    all_stats: dict[str, dict], sections: list[Section], path: Path
) -> None:
    lines: list[str] = [
        "# Tongue-cancer corpus extraction report",
        "",
        "Audit trail for Stage 1. Check this before building the index: if the",
        "page accounting below looks wrong, the index will be wrong too.",
        "",
    ]

    for book_id, st in all_stats.items():
        lines += [
            f"## {book_id}",
            "",
            f"- Pages read: **{st['total_pages']}**",
            f"- Pages kept: **{st['kept']}** "
            f"({st['kept_by_window']} from chapter windows, "
            f"{st['kept_by_sweep']} from the whole-book tongue sweep)",
            f"- Dropped, inside an excluded window: {st['dropped_excluded']}",
            f"- Dropped, not tongue-relevant: {st['dropped_not_selected']}",
            "",
            "Page types across the whole book:",
            "",
            "| type | pages |",
            "|---|---|",
        ]
        for kind, n in sorted(st["kinds"].items(), key=lambda kv: -kv[1]):
            lines.append(f"| {kind} | {n} |")
        lines.append("")

        if st["chapters"]:
            lines += ["Chapters covered:", "", "| ch | title | pages kept | pdf range |", "|---|---|---|---|"]
            for num in sorted(st["chapters"]):
                c = st["chapters"][num]
                lines.append(
                    f"| {num} | {c['title'] or ''} | {c['pages']} | {c['range'][0]}-{c['range'][1]} |"
                )
            lines.append("")

        sweep = st["sweep_pages"]
        if sweep:
            lines += [
                f"Pages pulled in by the whole-book sweep (outside the H&N chapters): "
                f"{compact_ranges(sweep)}",
                "",
            ]

    by_book = Counter(s.book_short for s in sections)
    by_tier = Counter(s.tier for s in sections)
    chars = sum(s.char_count for s in sections)
    lines += [
        "## Sections written",
        "",
        f"- Total sections: **{len(sections)}**",
        f"- Total characters: **{chars:,}**",
        "- By tier: " + ", ".join(f"{k}={v}" for k, v in by_tier.most_common()),
        "- By book: " + ", ".join(f"{k}={v}" for k, v in by_book.most_common()),
        "",
        "### Largest core sections",
        "",
        "| section | pages | chars |",
        "|---|---|---|",
    ]
    core = sorted(
        (s for s in sections if s.tier == "core"), key=lambda s: -s.char_count
    )[:25]
    for s in core:
        lines.append(
            f"| {s.breadcrumb()} | {s.pdf_page_start}-{s.pdf_page_end} | {s.char_count:,} |"
        )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def compact_ranges(nums: list[int]) -> str:
    """[1,2,3,7,8] -> '1-3, 7-8'"""
    if not nums:
        return "(none)"
    out, start, prev = [], nums[0], nums[0]
    for n in nums[1:]:
        if n != prev + 1:
            out.append(f"{start}-{prev}" if start != prev else str(start))
            start = n
        prev = n
    out.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(out)


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--book", help="only process this book id (devita|gunderson)")
    ap.add_argument("--limit-pages", type=int, help="stop after N pages (smoke test)")
    ap.add_argument("--config", default="config/books.yaml")
    ap.add_argument("--rag-config", default="config/rag.yaml")
    ap.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="processes used to read PDF pages (default: all cores)",
    )
    ap.add_argument(
        "--all-pages",
        action="store_true",
        help="index the ENTIRE books, not just tongue-cancer content. Index, "
        "bibliography and blank pages are still dropped. Overrides "
        "selection.mode in rag.yaml.",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    rag = yaml.safe_load((ROOT / args.rag_config).read_text(encoding="utf-8"))
    lex = cfg["lexicon"]

    all_pages = args.all_pages or rag.get("selection", {}).get("mode") == "full"
    print(f"mode: {'FULL BOOKS' if all_pages else 'tongue-cancer only'}")

    corpus_dir = ROOT / rag["paths"]["corpus_dir"]
    sections_dir = ROOT / rag["paths"]["sections_dir"]

    all_sections: list[Section] = []
    all_stats: dict[str, dict] = {}

    for book in cfg["books"]:
        if args.book and book["id"] != args.book:
            continue
        print(
            f"[{book['id']}] reading {book['path']} with {args.workers} workers ...",
            flush=True,
        )
        pages, stats = extract_book(
            book, cfg, lex, args.limit_pages, args.workers, all_pages
        )
        all_stats[book["id"]] = stats
        print(f"[{book['id']}] kept {stats['kept']} pages -> splitting sections", flush=True)

        secs = split_into_sections(pages, book)
        secs = tier_sections(
            secs, lex, cfg["selection"]["min_section_chars"], all_pages
        )
        print(
            f"[{book['id']}] {len(secs)} sections "
            f"(core={sum(1 for s in secs if s.tier == 'core')})",
            flush=True,
        )
        all_sections.extend(secs)

    if not all_sections:
        print("no sections extracted -- check the page windows in books.yaml", file=sys.stderr)
        return 1

    write_sections(all_sections, sections_dir, only_book=args.book)
    write_manifest(all_sections, ROOT / rag["paths"]["manifest_file"])
    write_report(all_stats, all_sections, corpus_dir / "corpus_report.md")
    if args.book:
        print(
            f"\nNOTE: --book {args.book} was given. The other book's sections were "
            "left in place, but the manifest and report now describe only this "
            "book. Re-run without --book before building an index.",
            file=sys.stderr,
        )

    print()
    print(f"wrote {len(all_sections)} sections -> {sections_dir}")
    print(f"report -> {corpus_dir / 'corpus_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
