#!/usr/bin/env python3
"""Stage 5 -- tag every chunk with its organ system and pathology.

Adds two fields, `organ_system` and `pathology`, to the chunk records, using
the chapter page ranges in the PDFs' own outlines and the map in
config/organ_systems.yaml.

This deliberately does NOT re-embed anything. Tagging changes metadata, not
`embed_text`, so every vector in the index stays exactly as valid as it was --
and re-embedding 15,866 chunks costs 75 minutes on this CPU for no benefit.
The script therefore rewrites both the corpus chunks and the copy sitting
next to the index, and recomputes the per-pathology centroids from the
existing embeddings.npy. It runs in seconds.

    python scripts/05_tag_pathologies.py
    python scripts/05_tag_pathologies.py --write-corpora   # per-system files
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def chapter_ranges(book_path: Path, last_page: int) -> dict[int, tuple[int, int]]:
    """Map chapter number -> (first_page, last_page) from the PDF outline.

    The outline is the publisher's own, so this needs no heuristics. Where a
    chapter number appears more than once (the outline repeats it for
    subsections) the first occurrence wins.
    """
    import fitz

    doc = fitz.open(book_path)
    try:
        starts: dict[int, int] = {}
        for level, title, page in doc.get_toc():
            if level > 2:
                continue
            clean = re.sub(r"^\d+\.\s*", "", title).strip()
            m = re.match(r"^(\d+)\s+(\S.*)$", clean)
            if m:
                starts.setdefault(int(m.group(1)), page)
    finally:
        doc.close()

    ordered = sorted(starts.items(), key=lambda kv: kv[1])
    out: dict[int, tuple[int, int]] = {}
    for i, (num, start) in enumerate(ordered):
        end = ordered[i + 1][1] - 1 if i + 1 < len(ordered) else last_page
        out[num] = (start, end)
    return out


def build_page_map(cfg: dict, books_cfg: dict) -> tuple[dict, dict]:
    """(book_id, page) -> pathology id, plus pathology metadata.

    Also enforces that no two pathologies claim the same chapter; silently
    overlapping tags would make a filtered corpus quietly wrong.
    """
    ranges = {}
    for book in books_cfg["books"]:
        ranges[book["id"]] = chapter_ranges(ROOT / book["path"], book["pages"])

    page_to_path: dict[tuple[str, int], str] = {}
    meta: dict[str, dict] = {}
    claimed: dict[tuple[str, int], str] = {}
    clashes: list[str] = []

    for system in cfg["systems"]:
        for path in system["pathologies"]:
            meta[path["id"]] = {
                "label": path["label"],
                "system": system["id"],
                "system_label": system["label"],
                "refine_from_devita_45_core": bool(
                    path.get("refine_from_devita_45_core")
                ),
            }
            for book_id, chapters in (path.get("chapters") or {}).items():
                for ch in chapters:
                    key = (book_id, ch)
                    if key in claimed:
                        clashes.append(
                            f"{book_id} ch.{ch} claimed by both "
                            f"{claimed[key]} and {path['id']}"
                        )
                        continue
                    claimed[key] = path["id"]
                    span = ranges.get(book_id, {}).get(ch)
                    if span is None:
                        clashes.append(f"{book_id} ch.{ch} not found in the PDF outline")
                        continue
                    for page in range(span[0], span[1] + 1):
                        page_to_path[(book_id, page)] = path["id"]

    if clashes:
        for c in clashes:
            print(f"  !! {c}", file=sys.stderr)
        raise SystemExit("organ_systems.yaml has overlapping or unknown chapters")

    fb = cfg["fallback"]
    meta[fb["id"]] = {
        "label": fb["label"],
        "system": fb["system"],
        "system_label": fb["system_label"],
        "refine_from_devita_45_core": False,
    }
    return page_to_path, meta


def tag_rows(rows: list[dict], page_to_path: dict, meta: dict, fallback: str) -> None:
    """Assign organ_system / pathology to every chunk, in place."""
    # DeVita's single head-and-neck chapter has no internal chapter boundary
    # to split the tongue out with, so Stage 1's tongue tiering stands in.
    refine_target = next(
        (pid for pid, m in meta.items() if m["refine_from_devita_45_core"]), None
    )

    for row in rows:
        book = (row.get("book_short") or "").lower()
        page = row.get("pdf_page_start") or 0
        pid = page_to_path.get((book, page))

        if pid is None:
            # A chunk can start on the last page of one chapter and run into
            # the next; try its end page before giving up on it.
            pid = page_to_path.get((book, row.get("pdf_page_end") or page))

        if pid is None:
            pid = fallback

        if (
            refine_target
            and book == "devita"
            and row.get("tier") == "core"
            and meta.get(pid, {}).get("system") == "head_neck"
        ):
            pid = refine_target

        row["pathology"] = pid
        row["organ_system"] = meta[pid]["system"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/organ_systems.yaml")
    ap.add_argument(
        "--write-corpora",
        action="store_true",
        help="also write corpus/by_system/<system>/<pathology>.jsonl for audit",
    )
    args = ap.parse_args()

    from rag.config import load_config, path_of

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    books_cfg = yaml.safe_load((ROOT / "config" / "books.yaml").read_text(encoding="utf-8"))
    rag_cfg = load_config()

    print("reading chapter ranges from the PDF outlines...")
    page_to_path, meta = build_page_map(cfg, books_cfg)
    print(f"  {len(meta) - 1} pathologies over {len(page_to_path):,} mapped pages")

    fallback = cfg["fallback"]["id"]
    targets = [ROOT / rag_cfg["paths"]["chunks_file"]]
    index_chunks = path_of("index_dir") / "chunks.jsonl"
    if index_chunks.exists():
        targets.append(index_chunks)

    counts = Counter()
    for target in targets:
        if not target.exists():
            print(f"  skip (missing): {target}")
            continue
        rows = read_jsonl(target)
        tag_rows(rows, page_to_path, meta, fallback)
        write_jsonl(target, rows)
        counts = Counter(r["pathology"] for r in rows)
        print(f"  tagged {len(rows):,} chunks -> {target.relative_to(ROOT)}")

    # --- per-pathology centroids, from the embeddings we already have ------
    emb_path = path_of("index_dir") / "embeddings.npy"
    if emb_path.exists() and index_chunks.exists():
        import numpy as np

        vectors = np.load(emb_path)
        rows = read_jsonl(index_chunks)
        if len(rows) == len(vectors):
            groups = defaultdict(list)
            for i, row in enumerate(rows):
                groups[row["pathology"]].append(i)
            centroids = {}
            for pid, idx in groups.items():
                c = vectors[idx].mean(axis=0)
                c /= np.linalg.norm(c) + 1e-12
                centroids[pid] = c.astype("float32").tolist()
            (path_of("index_dir") / "pathology_centroids.json").write_text(
                json.dumps(centroids), encoding="utf-8"
            )
            print(f"  wrote {len(centroids)} per-pathology centroids "
                  f"(no re-embedding needed)")
        else:
            print(f"  !! embeddings ({len(vectors)}) and chunks ({len(rows)}) disagree; "
                  "skipping centroids", file=sys.stderr)

    # --- taxonomy manifest the UI reads ------------------------------------
    taxonomy = {
        "systems": [
            {
                "id": s["id"],
                "label": s["label"],
                "pathologies": [
                    {
                        "id": p["id"],
                        "label": p["label"],
                        "chunks": counts.get(p["id"], 0),
                    }
                    for p in s["pathologies"]
                ],
            }
            for s in cfg["systems"]
        ]
        + [
            {
                "id": cfg["fallback"]["system"],
                "label": cfg["fallback"]["system_label"],
                "pathologies": [
                    {
                        "id": fallback,
                        "label": cfg["fallback"]["label"],
                        "chunks": counts.get(fallback, 0),
                    }
                ],
            }
        ],
        "total_chunks": sum(counts.values()),
    }
    out = path_of("index_dir") / "taxonomy.json"
    out.write_text(json.dumps(taxonomy, indent=2), encoding="utf-8")
    print(f"  wrote {out.relative_to(ROOT)}")

    # --- optional per-system corpus files ----------------------------------
    if args.write_corpora and index_chunks.exists():
        rows = read_jsonl(index_chunks)
        by_path = defaultdict(list)
        for row in rows:
            by_path[row["pathology"]].append(row)
        base = ROOT / "corpus" / "by_system"
        for pid, rs in by_path.items():
            write_jsonl(base / meta[pid]["system"] / f"{pid}.jsonl", rs)
        print(f"  wrote {len(by_path)} per-pathology corpora -> corpus/by_system/")

    print("\nchunks per organ system:")
    by_sys = Counter()
    for pid, n in counts.items():
        by_sys[meta[pid]["system_label"]] += n
    for label, n in by_sys.most_common():
        print(f"  {label:<38} {n:>6,}")
    empty = [p for p in counts if counts[p] == 0]
    for s in cfg["systems"]:
        for p in s["pathologies"]:
            if counts.get(p["id"], 0) == 0:
                empty.append(p["id"])
    if empty:
        print(f"\n  note: {len(set(empty))} pathologies matched no chunks: "
              f"{', '.join(sorted(set(empty))[:8])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
