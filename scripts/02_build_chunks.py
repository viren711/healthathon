#!/usr/bin/env python3
"""Stage 2 -- turn the isolated sections into retrieval chunks.

Reads the section files written by Stage 1 (paths come from
config/rag.yaml) and emits chunks.jsonl, one JSON object per chunk, carrying the
full citation trail so the app can show a real source card.

Two rules matter here:
  * chunks never cross a section boundary, so a chunk is always about one
    thing and its citation is always exact;
  * detected tables are emitted whole, because a staging table split
    mid-row retrieves as nonsense.

Sections are independent of one another, so the work is spread across worker
processes; results are re-sorted by section id afterwards so the output file
is byte-identical regardless of how many workers ran.

    python scripts/02_build_chunks.py
    python scripts/02_build_chunks.py --workers 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

_TABLE_LINE = re.compile(r"^\s*(TABLE|eTABLE|Table)\s+\d+", re.IGNORECASE)


def load_tokenizer(model_name: str):
    """Use the embedding model's own tokenizer so the budget is honest.

    Falls back to a whitespace estimate when transformers is not installed,
    which keeps Stage 2 runnable on a machine with no ML stack.
    """
    try:
        import logging

        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
        # The tokenizer is used only to COUNT tokens -- nothing is ever fed to
        # the model here -- so its "sequence longer than max_seq_length, this
        # will result in indexing errors" warning is noise, and it fires on
        # every oversized paragraph before the packer splits it.
        logging.getLogger("transformers.tokenization_utils_base").setLevel(
            logging.ERROR
        )
        return lambda s: len(tok.encode(s, add_special_tokens=False)), "hf"
    except Exception as exc:  # noqa: BLE001 - any failure means fall back
        print(f"  ! tokenizer unavailable ({type(exc).__name__}), using word estimate")
        # ~1.3 subword tokens per whitespace word for English clinical prose.
        return lambda s: int(len(s.split()) * 1.3) + 1, "estimate"


def parse_section(path: Path) -> tuple[dict, str]:
    """Split a section .md into its YAML front matter and its body."""
    raw = path.read_text(encoding="utf-8")
    if not raw.startswith("---\n"):
        raise ValueError(f"{path.name}: missing front matter")
    _, front, body = raw.split("---\n", 2)
    meta = yaml.safe_load(front)
    # Drop the "# breadcrumb" title line; it is rebuilt for the embed text.
    body = re.sub(r"^\s*#\s+.*\n", "", body, count=1).strip()
    return meta, body


def split_paragraphs(text: str) -> list[str]:
    """Paragraph-ish units. Tables are kept as one unit each."""
    units: list[str] = []
    buf: list[str] = []
    in_table = False

    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        starts_table = bool(_TABLE_LINE.match(block))
        if starts_table:
            if buf:
                units.append("\n\n".join(buf))
                buf = []
            units.append(block)
            in_table = True
            continue
        if in_table and len(block.split()) < 12:
            # Short block right after a table caption is still table body.
            units[-1] += "\n\n" + block
            continue
        in_table = False
        buf.append(block)

    if buf:
        units.append("\n\n".join(buf))
    return units


def pack(units: list[str], count_tokens, cfg: dict) -> list[str]:
    """Greedily pack paragraph units up to the token budget, with overlap."""
    max_t = cfg["max_tokens"]
    overlap_t = cfg["overlap_tokens"]
    table_max = cfg["table_max_tokens"]

    chunks: list[str] = []
    cur: list[str] = []
    cur_tokens = 0

    def flush() -> None:
        nonlocal cur, cur_tokens
        if cur:
            chunks.append("\n\n".join(cur).strip())
            cur, cur_tokens = [], 0

    for unit in units:
        n = count_tokens(unit)
        is_table = bool(_TABLE_LINE.match(unit))

        if is_table:
            # A table goes out whole and alone, so its caption always travels
            # with its rows. If it blows past the ceiling it is split rather
            # than truncated -- dropping rows would silently lose content.
            flush()
            if n <= table_max:
                chunks.append(unit)
            else:
                chunks.extend(split_long(unit, count_tokens, max_t, overlap_t))
            continue

        if n > max_t:
            # A single oversized paragraph: split on sentences.
            flush()
            chunks.extend(split_long(unit, count_tokens, max_t, overlap_t))
            continue

        if cur_tokens + n > max_t:
            flush()
            # Carry the tail of the previous chunk forward as overlap so a
            # fact spanning the seam is still retrievable from either side.
            if chunks:
                tail = tail_tokens(chunks[-1], count_tokens, overlap_t)
                if tail:
                    cur, cur_tokens = [tail], count_tokens(tail)

        cur.append(unit)
        cur_tokens += n

    flush()
    return [c for c in chunks if c.strip()]


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def split_long(text: str, count_tokens, max_t: int, overlap_t: int) -> list[str]:
    sents = _SENT.split(text)
    out: list[str] = []
    cur: list[str] = []
    n = 0
    for s in sents:
        sn = count_tokens(s)
        if n + sn > max_t and cur:
            out.append(" ".join(cur))
            tail = tail_tokens(out[-1], count_tokens, overlap_t)
            cur, n = ([tail] if tail else []), (count_tokens(tail) if tail else 0)
        cur.append(s)
        n += sn
    if cur:
        out.append(" ".join(cur))
    return out


def tail_tokens(text: str, count_tokens, budget: int) -> str:
    """Last complete sentences of `text` fitting inside `budget` tokens."""
    sents = _SENT.split(text)
    out: list[str] = []
    n = 0
    for s in reversed(sents):
        sn = count_tokens(s)
        if n + sn > budget:
            break
        out.insert(0, s)
        n += sn
    return " ".join(out)


def citation_label(meta: dict) -> str:
    """Short human-readable source label, e.g. 'Gunderson, ch.38 Oral Cavity, p.613'."""
    bits = [meta.get("book_short") or meta.get("book", "")]
    if meta.get("chapter"):
        bits.append(f"ch.{meta['chapter']}")
    printed = meta.get("printed_page_start")
    if printed:
        bits.append(f"p.{printed}")
    bits.append(f"PDF p.{meta['pdf_page_start']}")
    return ", ".join(b for b in bits if b)


# --- worker plumbing -------------------------------------------------------
# Loading the HF tokenizer costs ~1s, so each worker loads it once at start-up
# rather than once per section. These globals are populated by _init_worker in
# the child process; the parent never touches them.
_W: dict = {}


def _init_worker(model_name: str, chunk_cfg: dict, doc_prefix: str) -> None:
    count_tokens, _ = load_tokenizer(model_name)
    _W["count_tokens"] = count_tokens
    _W["chunk_cfg"] = chunk_cfg
    _W["doc_prefix"] = doc_prefix


def chunk_section(
    path: Path, count_tokens=None, chunk_cfg: dict | None = None, doc_prefix: str = ""
) -> tuple[list[dict], int]:
    """Turn one section file into its chunks. Returns (rows, skipped_short)."""
    count_tokens = count_tokens or _W["count_tokens"]
    chunk_cfg = chunk_cfg if chunk_cfg is not None else _W["chunk_cfg"]
    doc_prefix = doc_prefix or _W.get("doc_prefix", "")

    meta, body = parse_section(path)
    if not body.strip():
        return [], 0

    units = split_paragraphs(body)
    breadcrumb = " > ".join(
        [meta.get("book_short", "")]
        + ([f"ch.{meta['chapter']}"] if meta.get("chapter") else [])
        + list(meta.get("section_path") or [])
    )

    rows: list[dict] = []
    skipped_short = 0
    for i, text in enumerate(pack(units, count_tokens, chunk_cfg)):
        n_tokens = count_tokens(text)
        # Tables get a lower floor than prose (a caption plus a couple of
        # rows is still a usable source), but a bare stub caption is not.
        floor = 20 if _TABLE_LINE.match(text) else chunk_cfg["min_tokens"]
        if n_tokens < floor:
            skipped_short += 1
            continue
        chunk_id = hashlib.sha1(
            f"{meta['section_id']}|{i}|{text[:120]}".encode("utf-8")
        ).hexdigest()[:16]
        rows.append(
            {
                "chunk_id": chunk_id,
                "section_id": meta["section_id"],
                "text": text,
                # The breadcrumb is embedded with the chunk so that a short
                # passage like "The first echelon nodes are level IB and II"
                # still carries "Oral Tongue" into the vector.
                "embed_text": f"{doc_prefix}{breadcrumb}\n\n{text}",
                "breadcrumb": breadcrumb,
                "book": meta.get("book"),
                "book_short": meta.get("book_short"),
                "chapter": meta.get("chapter"),
                "section_path": meta.get("section_path"),
                "pdf_page_start": meta.get("pdf_page_start"),
                "pdf_page_end": meta.get("pdf_page_end"),
                "printed_page_start": meta.get("printed_page_start"),
                "printed_page_end": meta.get("printed_page_end"),
                "tier": meta.get("tier"),
                "citation": citation_label(meta),
                "is_table": bool(_TABLE_LINE.match(text)),
                "n_tokens": n_tokens,
            }
        )
    return rows, skipped_short


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/rag.yaml")
    ap.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="processes used to chunk sections (default: all cores)",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    sections_dir = ROOT / cfg["paths"]["sections_dir"]
    out_path = ROOT / cfg["paths"]["chunks_file"]

    files = sorted(sections_dir.glob("*.md"))
    if not files:
        print(f"no sections in {sections_dir} -- run 01_extract_corpus.py first", file=sys.stderr)
        return 1

    chunk_cfg = cfg["chunking"]
    model_name = cfg["embedding"]["model"]
    doc_prefix = cfg["embedding"].get("doc_prefix", "")

    rows: list[dict] = []
    skipped_short = 0
    workers = max(1, min(args.workers, len(files)))

    if workers == 1:
        count_tokens, tok_kind = load_tokenizer(model_name)
        print(f"tokenizer: {tok_kind} ({model_name})")
        results = [
            chunk_section(p, count_tokens, chunk_cfg, doc_prefix) for p in files
        ]
    else:
        print(f"chunking {len(files)} sections across {workers} workers "
              f"(tokenizer: {model_name})")
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(model_name, chunk_cfg, doc_prefix),
        ) as pool:
            results = list(pool.map(chunk_section, files, chunksize=8))

    # `files` is already sorted and pool.map preserves input order, so the
    # output is deterministic no matter how many workers ran.
    for section_rows, skipped in results:
        rows.extend(section_rows)
        skipped_short += skipped

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    tokens = [r["n_tokens"] for r in rows]
    core = sum(1 for r in rows if r["tier"] == "core")
    tables = sum(1 for r in rows if r["is_table"])
    print()
    print(f"sections in : {len(files)}")
    print(f"chunks out  : {len(rows)}  (core={core}, context={len(rows) - core}, tables={tables})")
    if tokens:
        print(
            f"tokens/chunk: min={min(tokens)} median={sorted(tokens)[len(tokens) // 2]} "
            f"max={max(tokens)} mean={sum(tokens) // len(tokens)}"
        )
    print(f"dropped as too short: {skipped_short}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
