#!/usr/bin/env python3
"""Stage 9 -- turn the ASCO guidelines into recommendation records.

For each guideline PDF in config/asco_guidelines.yaml, extract

  recommendations  one record per rated recommendation statement: its number,
                   verbatim text, evidence quality, strength, version, pages
                   and case tags (setting, biomarkers, drugs, prior therapy)
  statistics       verbatim sentences that carry trial figures (HR, median
                   survival, response rates...), each with its page and the
                   recommendation section it sits under

and attach statistics to recommendations. Nothing is paraphrased: every
statement and statistic is a span of the PDF's own text, so a figure on the
page can never differ from the figure shown.

One process per guideline (largest first); each reads its PDF once.

    python scripts/09_build_asco.py
    python scripts/09_build_asco.py --workers 4
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from bisect import bisect_right
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.guidance_lexicon import (  # noqa: E402
    biomarkers_in,
    drugs_in,
    negated_biomarkers_in,
    prior_therapy_in,
    settings_in,
)

OUT_DIR = ROOT / "corpus" / "asco"

# "(Type: Evidence based, benefits outweigh harms; Evidence quality: Moderate;
#  Strength of recommendation: Strong)" and the shorter form without Type.
# Journal-layout PDFs hyphenate across lines ("recom- mendation").
# Some put a qualifier first ("(75% of cells...; Evidence quality: High; ...)")
# and some rate "N/A".
RATING = re.compile(
    r"\(\s*(?:[^();]{0,160};\s*)?(?:Type:\s*(?P<type>[^;)]*?)\s*;\s*)?"
    r"Evidence\s+quality:\s*(?P<evidence>[A-Za-z/ ]+?)\s*;\s*"
    r"Strength\s+of\s+recom-?\s*mendation:\s*(?P<strength>[A-Za-z/ ]+?)\s*\)",
    re.I | re.S,
)
# "2.10. For patients..." / "1.1 Neoadjuvant..." -- a recommendation number
# at the start of a statement.
# "1.9.*" marks a recommendation updated in this version.
NUMBER = re.compile(r"(?<![\w.])(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\.?\*?\s+(?=[A-Z])")
# A recommendation HEADING ("Recommendation 1.1." alone on its line), not an
# inline cross-reference such as "see Recommendation 1.2".
REC_MARK = re.compile(r"(?m)^[ \t]*Recommendations?\s+(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\b[^\n]{0,45}$")

STAT_WORD = re.compile(
    r"\b(HR|hazard ratio|ORR|response rate|median|survival|PFS|OS|DFS|EFS|CI)\b", re.I
)
STAT_NUM = re.compile(r"\d+(?:\.\d+)?\s?(?:%|months?|mo\b)|\bP\s*[<=≤]|\b(?:HR|OR|RR)\b[,:= ]+\d|95% CI")
TRIAL = re.compile(r"\b[A-Z][A-Z0-9]{3,}(?:-[A-Za-z0-9]+)*\b")
TRIAL_STOP = {
    "NSCLC", "ASCO", "CHEMOTHERAPY", "PD-L1", "EGFR", "HER2", "BRAF", "MEDIAN", "RECOMMENDATION",
    "RECOMMENDATIONS", "TABLE", "FIGURE", "NCCN", "FDA", "EMA", "CNS", "ECOG", "NGS", "IHC",
    "MET", "RET", "ALK", "ROS1", "NTRK", "KRAS", "BRCA", "BRCA1", "BRCA2", "HRD", "PARP",
    "AND", "THE", "WITH", "FOR", "STAGE", "PSA", "ADT", "ARPI", "CDK4", "TKI", "ICI",
}
HEADING_KINDS = {
    "driver": r"\b(?:EGFR|exon|ALK|ROS1|BRAF|MET\b|RET\b|NTRK|HER2|KRAS|BRCA|HRD|MSI|dMMR|FGFR|IDH|CLDN|PIK3CA|ESR1|L858R|driver)",
    "histology": r"squamous|adenocarcinoma",
    "platinum": r"platinum[- ](?:sensitive|resistant)|\bPSOC\b|\bPROC\b",
    "pdl1": r"PD-?L1|\bTPS\b",
    "ps": r"performance status|\bPS\b|ECOG",
    "line": r"first[- ]line|second[- ]line|subsequent|later[- ]line",
    "stage": r"\bstage\b",
}
ABBREV = ("vs.", "et al.", "i.e.", "e.g.", "ie.", "eg.", "Fig.", "Figs.", "No.", "approx.", "Dr.")


def _clean_pages(doc) -> list[str]:
    """Page texts without running headers and footers.

    A line that repeats (modulo digits) on a large share of pages is page
    furniture, not content: copyright lines, the guideline's title and
    version, the page number.
    """
    # Journal PDFs set "fi"/"fl" as ligatures ("ﬁrst-line"), which no pattern matches.
    pages = [p.get_text().replace("\ufb01", "fi").replace("\ufb02", "fl") for p in doc]
    seen: Counter[str] = Counter()
    for t in pages:
        for line in {re.sub(r"\d+", "#", ln.strip()) for ln in t.splitlines() if ln.strip()}:
            if len(line) < 120:
                seen[line] += 1
    furniture = {k for k, n in seen.items() if n >= max(4, 0.35 * len(pages))}
    out = []
    for t in pages:
        keep = [
            ln for ln in t.splitlines()
            if re.sub(r"\d+", "#", ln.strip()) not in furniture and not re.fullmatch(r"\s*\d{1,3}\s*", ln)
        ]
        out.append("\n".join(keep))
    return out


def _flat(pages: list[str]) -> tuple[str, list[int]]:
    starts, buf, pos = [], [], 0
    for t in pages:
        starts.append(pos)
        buf.append(t)
        pos += len(t) + 1
    return "\n".join(buf), starts


def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _version(doc, title: str, fname: str) -> str:
    m = re.search(r"v\d{4}\.\d+(?:\.\d+)?", "\n".join(doc[i].get_text() for i in range(min(3, len(doc)))))
    if m:
        return m.group(0)
    y = re.search(r"(20\d\d)", fname)
    return f"JCO {y.group(1)}" if y else ""


def _norm(s: str) -> str:
    return _ws(s).lower()


def extract_recommendations(gid: str, text: str, starts: list[int]) -> list[dict]:
    recs: list[dict] = []
    prev_end = 0
    last_number = None
    headers: dict[str, str] = {}
    other: list[str] = []
    last_prefix = None
    for m in RATING.finditer(text):
        region = text[prev_end : m.start()]
        # Start of this statement: the last recommendation number in the region.
        nums = list(NUMBER.finditer(region))
        number, begin = None, 0
        if nums:
            number, begin = nums[-1].group(1), nums[-1].end()
        else:
            marks = list(REC_MARK.finditer(region))
            if marks:
                number = marks[-1].group(1)
                begin = marks[-1].end()
            elif last_number:
                number = last_number.rstrip("+") + "+"  # a further statement under the same recommendation
        statement = region[begin:]
        if len(statement) > 900:  # a long run with no number: keep the last sentence group
            cut = statement[-900:]
            k = re.search(r"(?<=[.)])\s+(?=[A-Z])", cut)
            statement = cut[k.end():] if k else cut
        statement = _ws(statement)
        statement = re.sub(r"^\(reprinted here for clarity\)\s*", "", statement)
        statement = re.sub(r"^[.\s]+|^Qualifying statements?\.?\s*", "", statement)
        lead = re.match(r"(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\.\*?\s+(.*)", statement)
        if lead:  # a statement starting in lower case ("1.5. or amivantamab...")
            number, statement = lead.group(1), lead.group(2)
        statement = re.sub(r"^\((?:New|Updated|Revised)\)\s*", "", statement)
        if statement.endswith("?") or re.match(r"(What|Which|How|Should)\b[^.]*\?", statement):
            prev_end = m.end()
            continue  # the guideline's question, not a recommendation
        if re.match(r"(See |The ASCO guideline|Refer to )", statement):
            prev_end = m.end()
            continue  # a pointer to another section, not a recommendation
        if not number and len(statement) > 450:
            continue  # front matter swallowed with no recommendation number: not a statement
        # Context: the short heading lines above the number ("Non-squamous Cell
        # Carcinoma", "TPS >= 50%", "ROS1") are sticky -- they govern every
        # statement until a heading of the same kind replaces them -- and the
        # lead-in sentence ("Clinicians should offer any of the following:").
        if nums:
            # A new numbered section ("2.3" after "2.2.x") asks a new question:
            # the previous section's line of therapy no longer applies.
            prefix = number.rsplit(".", 1)[0] if number else ""
            if prefix.split(".")[0] != (last_prefix or "").split(".")[0]:
                headers.clear()  # a new clinical question: its own headings follow
            elif prefix != last_prefix:
                headers.pop("line", None)
            last_prefix = prefix
            # Rejoin words hyphenated across lines ("pa-\ntients").
            block = re.sub(r"-\n\s*(?=[a-z])", "", region[: nums[-1].start()])
            for ln in block.splitlines():
                ln = ln.strip()
                # "4. What is the recommended treatment of patients with recurrent
                # PROC?" -- the guideline's numbered question opens a new context.
                if re.match(r"\d{1,2}\.\s+(What|Which|How|In|For|Should|Is|Are)\b", ln):
                    headers.clear()
                    for k, pat in HEADING_KINDS.items():
                        hit = re.search(pat, ln, re.I)
                        if hit and k != "line":
                            headers[k] = hit.group(0)
                    q = re.search(r"\b(first|second|subsequent|later)[- ]line\b", ln, re.I)
                    if q:
                        headers["line"] = q.group(0)
                    continue
                # The guideline's own question ("What are the most effective
                # second-line treatment options...") sets the line of therapy.
                q = re.search(r"\b(first|second|subsequent|later)[- ]line\b", ln, re.I)
                if q and re.match(r"(?:Clinical Question \d+:\s*)?(What|Which|How|In patients|For patients)\b", ln):
                    headers["line"] = q.group(0)
                    continue
                if not (3 <= len(ln) <= 70) or ln.endswith((".", ",", ")")) or not (ln[0].isupper() or ln[0].isdigit()):
                    continue
                if ln.startswith(("Qualifying", "Recommendation", "Table", "TABLE", "Figure")):
                    continue
                if len(re.findall(r"[A-Za-z]", ln)) < 3:
                    continue  # citation numbers, page furniture
                key = next((k for k, pat in HEADING_KINDS.items() if re.search(pat, ln, re.I)), None)
                if key is None:
                    if len(ln.split()) > 6:
                        continue  # a wrapped prose line, not a heading
                    other.append(ln)
                    del other[:-2]
                else:
                    headers[key] = ln
        # Only headings of a known kind: free-standing short lines in journal
        # layouts are as often section titles ("Methods") as table headings.
        context = " · ".join(headers.values())
        heading = context
        prev_end = m.end()
        if len(statement) < 25:
            continue
        last_number = number or last_number
        recs.append({
            "guideline_id": gid,
            "number": number or "",
            "heading": heading,
            "statement": statement,
            "type": _ws(m.group("type") or ""),
            "evidence": _ws(m.group("evidence")).capitalize(),
            "strength": _ws(m.group("strength")).capitalize(),
            "pdf_page": bisect_right(starts, m.start()),
            "span": (m.start(), m.end()),
        })
    # A summary box often repeats statements that the body then numbers: drop
    # the NUMBERLESS copy. (Identical regimens under different headings are
    # different recommendations and are kept.)
    numbered = {_norm(r["statement"])[-90:] for r in recs if r["number"] and not r["number"].endswith("+")}
    kept = [
        r for r in recs
        if r["number"] and not r["number"].endswith("+") or _norm(r["statement"])[-90:] not in numbered
    ]
    # The same numbered statement printed twice (summary table, then body):
    # keep the first.
    seen, out = set(), []
    for r in kept:
        key = (r["number"], re.sub(r"[^a-z]", "", _norm(r["statement"]))[:80])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


# Guidelines that print numbered statements WITHOUT the "(Evidence quality...;
# Strength...)" parenthesis -- the communication guideline (each statement
# followed by its panel agreement, "96%") and the ctDNA guideline ("1. Outside
# of a clinical trial, ctDNA testing should...").
UNRATED_START = re.compile(r"(?m)^[ \t]*(\d{1,2}(?:\.\d{1,2})?)\.\s+(?=[A-Za-z])")
UNRATED_STOP = re.compile(r"(?m)^[ \t]*(?:\d{1,2}(?:\.\d{1,2})?\.\s+[A-Za-z]|[a-h]\.\s|\d{2,3}%\s*$|TABLE\b|Qualifying)")


def extract_unrated(gid: str, text: str, starts: list[int]) -> list[dict]:
    recs = []
    for m in UNRATED_START.finditer(text):
        rest = text[m.end():]
        stop = UNRATED_STOP.search(rest)
        body = rest[: stop.start() if stop else 700]
        statement = _ws(body)
        if re.search(r"\bet al\b|\(\w+ \d{4}\)", statement):
            continue  # a reference-list entry
        statement = re.sub(r"\s*\((?:Good Practice Statement|Evidence quality:[^)]*)\)\s*$", "", statement)
        if not (40 <= len(statement) <= 700) or not re.search(r"\b(should|may|recommend|offer|is not|are not)\b", statement):
            continue
        agree = re.match(r"\s*(\d{2,3})%", rest[stop.start():] if stop else "")
        strength = re.search(r"\b(Strong|Conditional|Weak)\b", body)
        recs.append({
            "guideline_id": gid, "number": m.group(1), "heading": "", "statement": statement,
            "type": "", "evidence": "",
            "strength": strength.group(1) if strength else (f"Panel agreement {agree.group(1)}%" if agree else ""),
            "pdf_page": bisect_right(starts, m.start()), "span": (m.start(), m.end()),
        })
    seen, out = set(), []
    for r in recs:  # a statement printed twice (summary and body): keep the first
        k = _norm(r["statement"])[:90]
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def extract_statistics(gid: str, text: str, starts: list[int]) -> list[dict]:
    """Verbatim sentences carrying trial figures, with the recommendation
    section each sits under."""
    marks = [(m.start(), m.group(1)) for m in REC_MARK.finditer(text)]
    mark_pos = [p for p, _ in marks]
    guarded = text
    for ab in ABBREV:
        guarded = guarded.replace(ab, ab.replace(".", "․"))
    out, pos = [], 0
    for sent in re.split(r"(?<=[.])(?:\d{1,3}(?:[,–-]\d{1,3})*)?\s+(?=[A-Z])", guarded):
        start = text.find(sent.replace("․", "."), pos) if sent else -1
        s = sent.replace("․", ".")
        if start < 0:
            start = pos
        pos = start + len(s)
        flat = _ws(s)
        if not (40 <= len(flat) <= 700):
            continue
        if "Evidence quality" in flat or "Strength of" in flat:
            continue
        if not (STAT_WORD.search(flat) and STAT_NUM.search(flat)):
            continue
        i = bisect_right(mark_pos, start) - 1
        out.append({
            "guideline_id": gid,
            "text": flat,
            "pdf_page": bisect_right(starts, start),
            "section": marks[i][1] if i >= 0 else "",
            "trials": sorted({t for t in TRIAL.findall(flat) if t not in TRIAL_STOP and not t.isdigit()}),
            "drugs": drugs_in(flat),
            "biomarkers": biomarkers_in(flat),
        })
    return out


def attach(recs: list[dict], stats: list[dict], max_stats: int = 4) -> None:
    for r in recs:
        mine, base = [], r["number"].rstrip("+")
        for i, s in enumerate(stats):
            if s["guideline_id"] != r["guideline_id"]:
                continue
            if base and s["section"] == base:
                mine.append((3.0 + 0.1 * bool(re.search(r"\bHR\b|hazard", s["text"])), i, "section"))
                continue
            shared = set(r["drugs"]) & set(s["drugs"])
            if not shared:
                continue
            if r["biomarkers"] and s["biomarkers"] and not set(r["biomarkers"]) & set(s["biomarkers"]):
                continue
            mine.append((len(shared) + 0.1 * bool(re.search(r"\bHR\b|hazard", s["text"])), i, "drug match"))
        mine.sort(key=lambda x: -x[0])
        r["stats"] = [{"i": i, "basis": basis} for _, i, basis in mine[:max_stats]]


def process(entry: dict) -> dict:
    import pymupdf

    t0 = time.time()
    path = ROOT / entry["path"]
    doc = pymupdf.open(path)
    pages = _clean_pages(doc)
    text, starts = _flat(pages)
    gid = entry["id"]
    version = _version(doc, entry["title"], path.name)
    recs = extract_recommendations(gid, text, starts)
    if len(recs) < 5:  # a guideline that mostly does not print ratings
        recs = [r for r in recs if r["number"]]
        have = {r["number"] for r in recs}
        recs += [r for r in extract_unrated(gid, text, starts) if r["number"] not in have]
    stats = extract_statistics(gid, text, starts)
    for r in recs:
        r.pop("span")
        ctx = f"{r['heading']} {r['statement']}"
        r["version"] = version
        r["guideline_title"] = entry["title"]
        r["cancer"] = entry["cancer"]
        # The statement's own words decide a marker's polarity; headings only
        # add markers the statement does not mention (a stale "1p19q co-deleted"
        # heading must not override "1p19q non-codeleted" in the statement).
        st_pos, st_neg = set(biomarkers_in(r["statement"])), set(negated_biomarkers_in(r["statement"]))
        hd_pos, hd_neg = set(biomarkers_in(r["heading"])), set(negated_biomarkers_in(r["heading"]))
        r["biomarkers"] = sorted(st_pos | (hd_pos - st_neg))
        r["not_biomarkers"] = sorted(st_neg | (hd_neg - st_pos))
        r["drugs"] = drugs_in(r["statement"])
        r["settings"] = settings_in(ctx)
        r["prior"] = prior_therapy_in(r["statement"])
    return {
        "id": gid, "pages": len(doc), "version": version, "recs": recs, "stats": stats,
        "secs": round(time.time() - t0, 1), "size": path.stat().st_size,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    entries = yaml.safe_load((ROOT / "config" / "asco_guidelines.yaml").read_text(encoding="utf-8"))["guidelines"]
    entries.sort(key=lambda e: -(ROOT / e["path"]).stat().st_size)
    t0 = time.time()
    import os

    workers = args.workers or os.cpu_count() or 4
    with ProcessPoolExecutor(max_workers=min(workers, len(entries))) as pool:
        results = list(pool.map(process, entries))
    by_id = {r["id"]: r for r in results}
    ordered = [by_id[e["id"]] for e in sorted(entries, key=lambda e: e["id"])]

    all_stats, all_recs = [], []
    for r in ordered:
        offset = len(all_stats)
        for s in r["stats"]:
            s["id"] = len(all_stats)
            all_stats.append(s)
        for k, rec in enumerate(r["recs"]):
            rec["rec_id"] = f"{rec['guideline_id']}:{rec['number'] or 'x'}:{k}"
            all_recs.append(rec)
        attach(r["recs"], all_stats[offset:], 4)
        for rec in r["recs"]:
            for st in rec["stats"]:
                st["i"] += offset  # local -> global statistic id

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "recommendations.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in all_recs) + "\n", encoding="utf-8")
    (OUT_DIR / "statistics.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in all_stats) + "\n", encoding="utf-8")

    print(f"{len(entries)} guidelines parsed in {time.time() - t0:.1f}s")
    print(f"{'guideline':16s} {'pp':>3s} {'version':>10s} {'recs':>5s} {'with stats':>10s} {'stat sentences':>14s}")
    for r in ordered:
        n = len(r["recs"])
        w = sum(bool(x["stats"]) for x in r["recs"])
        print(f"{r['id']:16s} {r['pages']:3d} {r['version']:>10s} {n:5d} {w:10d} {len(r['stats']):14d}")
    print(f"total: {len(all_recs)} recommendations, {len(all_stats)} statistic sentences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
