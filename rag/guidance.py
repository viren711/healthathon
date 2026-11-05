"""Treatment Guidance: match a clinical case to ASCO guideline recommendations.

Two steps, neither involving a language model:

  read_case(text)   a pasted case summary -> the facts that decide which
                    recommendations apply (cancer, setting, biomarkers, PD-L1,
                    histology, drugs already given) plus the question asked.
                    Shown to the doctor as editable fields before anything is
                    searched, so a misreading is visible and correctable.
  rank(case)        the recommendations for that cancer, scored on agreement
                    with the case, with hard exclusions on contradiction, then
                    ordered by that score, the strength of recommendation and
                    the quality of evidence. Every score comes with the reasons
                    behind it, which the page shows.

Records come from scripts/09_build_asco.py (corpus/asco/). Statements and
statistics are verbatim spans of the guideline PDFs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import yaml

from rag.config import ROOT
from rag.guidance_lexicon import biomarkers_in, negated_biomarkers_in, settings_in, therapies_in

CORPUS = ROOT / "corpus" / "asco"
CROSS = "Any cancer (cross-cutting)"

# Everyday and abbreviated names -> the `cancer` value in config/asco_guidelines.yaml.
CANCERS: dict[str, str] = {
    "Lung cancer (NSCLC)": r"\bNSCLC\b|non[- ]small[- ]cell|lung (?:adeno|squamous|cancer|carcinoma)|adenocarcinoma of the lung",
    "Breast cancer": r"\bbreast\b",
    "Ovarian cancer": r"\bovar\w+|fallopian|primary peritoneal",
    "Multiple myeloma": r"myeloma|\bMM\b|plasma cell",
    "Melanoma": r"melanoma",
    "Thyroid cancer": r"thyroid|papillary carcinoma|anaplastic|medullary",
    "Gastroesophageal cancer": r"gastr\w+|esophag\w+|oesophag\w+|\bGEJ\b|gastroesophageal|stomach",
    "Biliary tract cancer": r"biliary|cholangio\w+|gallbladder|ampullary",
    "Prostate cancer": r"prostat\w+|\bmCRPC\b|\bCRPC\b",
    "Brain tumor (adult glioma)": r"glioma|astrocytoma|oligodendroglioma|glioblastoma",
    "Neuroendocrine tumor (GEP-NET)": r"neuroendocrine|carcinoid|\bNETs?\b|panNET",
}

# Cross-cutting guidelines join the results when the case mentions their subject.
CROSS_TRIGGERS: dict[str, str] = {
    "pregnancy": r"pregnan\w+|gestation|trimester",
    "wbc_gf": r"neutropeni\w+|\bG-?CSF\b|filgrastim|pegfilgrastim|growth factor",
    "ctdna": r"\bctDNA\b|liquid biopsy|cell-free DNA|\bcfDNA\b",
    "communication": r"break(?:ing)? (?:the )?(?:bad )?news|how (?:do I|to|should I) (?:tell|discuss|talk)|communicat\w+|goals of care",
}

# Biomarkers that define a treatment group: a recommendation written for one of
# them does not apply to a case known to carry a different one.
DRIVERS = {"EGFR", "EGFR exon 20 insertion", "ALK", "ROS1", "BRAF", "MET", "RET", "NTRK", "HER2", "KRAS", "BRCA",
           "MSI-H/dMMR", "IDH", "1p19q", "FGFR", "CLDN18.2", "ESR1", "PIK3CA"}

# Tags of which a tumour has at most one: a recommendation for one member does
# not apply to a case stated to have another.
EXCLUSIVE = [
    {"WHO grade 2", "WHO grade 3", "WHO grade 4"},
    {"Platinum-sensitive", "Platinum-resistant"},
    {"Castration-resistant", "mHSPC"},
    {"Transplant-eligible", "Transplant-ineligible"},
    {"Squamous (ESCC)", "Adenocarcinoma (GEJ/gastric)"},
    {"Anaplastic thyroid (ATC)", "Differentiated thyroid (DTC)", "Medullary thyroid (MTC)"},
]

SUBTYPES = {"EGFR exon 20 insertion": "EGFR"}

STRENGTH = {"strong": 3, "moderate": 2, "conditional": 1, "weak": 1}
EVIDENCE = {"high": 4, "moderate": 3, "intermediate": 3, "low": 2, "very low": 1, "insufficient": 0}

STOPWORDS = set(
    "a an and are as at be by can do does for from has have how i in is it its of on or "
    "should the their this to was what when which who will with patient patients "
    "treatment therapy options option best next line cancer".split()
)


@dataclass
class Case:
    cancer: str = ""
    settings: list[str] = field(default_factory=list)
    biomarkers: list[str] = field(default_factory=list)
    not_biomarkers: list[str] = field(default_factory=list)  # stated absent
    pdl1: int | None = None        # tumour proportion score, %
    histology: str = ""            # "squamous" | "non-squamous" | ""
    prior: list[str] = field(default_factory=list)   # drugs already given
    ps: int | None = None          # ECOG performance status
    pregnant: bool = False
    question: str = ""
    text: str = ""                 # the free text, for the cross-cutting triggers

    def summary(self) -> str:
        bits = [self.cancer or "cancer type not given"]
        bits += self.settings
        if self.histology:
            bits.append(self.histology)
        bits += self.biomarkers
        if self.pdl1 is not None:
            bits.append(f"PD-L1 {self.pdl1}%")
        if self.ps is not None:
            bits.append(f"ECOG {self.ps}")
        if self.prior:
            bits.append("prior: " + ", ".join(self.prior))
        if self.pregnant:
            bits.append("pregnant")
        return " · ".join(bits)


@dataclass
class Match:
    rec: dict[str, Any]
    score: float
    reasons: list[str]
    stats: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def available() -> bool:
    return (CORPUS / "recommendations.jsonl").exists()


@lru_cache(maxsize=1)
def load() -> tuple[list[dict], list[dict], list[dict]]:
    recs = [json.loads(l) for l in (CORPUS / "recommendations.jsonl").read_text(encoding="utf-8").splitlines() if l]
    stats = [json.loads(l) for l in (CORPUS / "statistics.jsonl").read_text(encoding="utf-8").splitlines() if l]
    guides = yaml.safe_load((ROOT / "config" / "asco_guidelines.yaml").read_text(encoding="utf-8"))["guidelines"]
    return recs, stats, guides


def cancers() -> list[str]:
    """Cancer types with at least one guideline, cross-cutting last."""
    seen = []
    for g in load()[2]:
        if g["cancer"] not in seen and g["cancer"] != CROSS:
            seen.append(g["cancer"])
    return sorted(seen)


def guideline(gid: str) -> dict:
    return next(g for g in load()[2] if g["id"] == gid)


def source_label(gid: str) -> str:
    """The name rag/pdfview resolves to this guideline's PDF."""
    return f"ASCO · {guideline(gid)['title']}"


# ---------------------------------------------------------------------------
# Reading a case
# ---------------------------------------------------------------------------


def _question(text: str) -> str:
    sentences = re.split(r"(?<=[.?!])\s+", text.strip())
    asked = [s for s in sentences if s.endswith("?")]
    if asked:
        return asked[-1]
    cue = [s for s in sentences if re.match(r"(what|which|how|should|is|are|can|when)\b", s, re.I)]
    return cue[-1] if cue else ""


def read_case(text: str) -> Case:
    """The facts in a case summary that decide which recommendations apply.

    Deliberately literal: it reads what is written and nothing else, so it
    can be wrong only in ways the doctor can see in the form and fix.
    """
    t = text or ""
    case = Case(text=t)
    case.question = _question(t)
    history = t.replace(case.question, " ") if case.question else t

    for name, pat in CANCERS.items():
        if re.search(pat, t, re.I):
            case.cancer = name
            break

    # Negated mentions ("EGFR negative", "no ALK") do not count: the case says
    # the tumour does NOT carry them.
    case.biomarkers = [b for b in biomarkers_in(t) if b != "PD-L1"]
    case.not_biomarkers = [b for b in negated_biomarkers_in(t) if b != "PD-L1"]

    m = re.search(r"PD-?L1[^0-9<>≥%]{0,25}(?:TPS[^0-9<>≥]{0,5})?([<>≥]?)\s*(\d{1,3})\s*%", t, re.I)
    if m:
        value = int(m.group(2))
        case.pdl1 = 0 if m.group(1) == "<" and value <= 1 else value
    elif re.search(r"PD-?L1\s*(?:TPS\s*)?(?:negative|<\s*1|below 1|less than 1)", t, re.I):
        case.pdl1 = 0

    if re.search(r"non[- ]?squamous|adenocarcinoma", t, re.I):
        case.histology = "non-squamous"
    elif re.search(r"squamous", t, re.I):
        case.histology = "squamous"

    m = re.search(r"\b(?:ECOG|PS|performance status)\s*(?:of|=|:)?\s*([0-4])\b", t, re.I)
    if m:
        case.ps = int(m.group(1))

    # "No prior BRAF therapy", "never had a MET inhibitor", "IO-naive": named to
    # say they were NOT given.
    negated = " ".join(m.group(0) for m in re.finditer(
        r"(?:\bno(?:t)?\s+(?:prior\s+|previous\s+)?|never\s+(?:had|received|treated with)\s+(?:an?\s+)?|"
        r"without\s+(?:prior\s+)?)[\w\s/-]{0,40}?(?:therapy|inhibitor|TKI|\w+(?:mab|nib|lib|ib))\b|"
        r"\b[\w/-]+[- ]na[iï]ve\b", history, re.I))
    case.prior = sorted(set(therapies_in(history)) - set(therapies_in(negated)))
    case.pregnant = bool(re.search(CROSS_TRIGGERS["pregnancy"], t, re.I))

    settings = [s for s in settings_in(t) if s != "supportive"]
    progressed = re.search(r"progress\w*|relapse\w*|refractory|recurr\w+|after|failed|resistan\w+|previously treated|pre-?treated", history, re.I)
    if case.prior and progressed and not {"second-line", "later-line"} & set(settings):
        settings.append("second-line")
    if "first-line" in settings and case.prior and progressed:
        settings.remove("first-line")
    case.settings = settings
    return case


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def _tps_range(context: str) -> tuple[float, float] | None:
    """The PD-L1 range a recommendation's heading restricts it to."""
    c = context.replace("≥", ">=")
    if re.search(r"TPS\s*>=\s*50|>=\s*50\s*%", c):
        return (50, 101)
    if re.search(r"1\s*%?\s*[-–]\s*49", c):
        return (1, 50)
    if re.search(r"TPS\s*<\s*1|<\s*1\s*%|negative PD-?L1", c, re.I):
        return (0, 1)
    return None


def _words(text: str) -> set[str]:
    """Content words, plurals folded ("symptoms" == "symptom")."""
    out = set()
    for w in re.findall(r"[a-z][a-z0-9-]{2,}", text.lower()):
        if w in STOPWORDS:
            continue
        out.add(w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w)
    return out


def score(rec: dict, case: Case) -> tuple[float | None, list[str]]:
    """Agreement between one recommendation and the case, or None if it is
    ruled out. Reasons are short phrases a clinician can check."""
    context = rec.get("heading", "")
    whole = f"{context} {rec['statement']}"
    s, why = 0.0, []

    # A recommendation for marker-NEGATIVE disease ("HER2-negative") does not
    # apply to a case that carries the marker, and vice versa.
    clash = set(rec.get("not_biomarkers", [])) & set(case.biomarkers)
    if clash:
        return None, [f"for {', '.join(sorted(clash))}-negative disease"]
    rec_bm = set(rec.get("biomarkers", [])) - {"PD-L1"}
    for group in EXCLUSIVE:
        r_g, c_g = rec_bm & group, set(case.biomarkers) & group
        if r_g and c_g:
            if r_g & c_g:
                s += 2
                why.append(", ".join(sorted(r_g & c_g)))
            else:
                return None, [f"for {', '.join(sorted(r_g))}"]
    if rec_bm & DRIVERS & set(case.not_biomarkers):
        return None, [f"for {', '.join(sorted(rec_bm & DRIVERS & set(case.not_biomarkers)))}-positive disease"]
    rec_drivers = rec_bm & DRIVERS
    case_bm = set(case.biomarkers)
    # Subtypes treated differently from their parent marker: an exon 20
    # insertion recommendation is not for an exon 19 deletion.
    for sub, parent in SUBTYPES.items():
        if sub in rec_drivers and parent in case_bm and sub not in case_bm:
            return None, [f"for {sub} only"]
        if sub in case_bm and parent in rec_drivers and sub not in rec_drivers and rec.get("guideline_id") == "nsclc_driver" and re.search(r"exon ?19|L858R", context):
            return None, ["for exon 19 deletion / L858R"]
    if rec_drivers:
        shared = rec_drivers & case_bm
        if shared:
            s += 3 * len(shared)
            why.append("matches " + ", ".join(sorted(shared)))
        elif case_bm & DRIVERS:
            return None, [f"written for {', '.join(sorted(rec_drivers))}"]
        else:
            s -= 1.5
            why.append(f"applies only if {', '.join(sorted(rec_drivers))} is present")
    elif case_bm & DRIVERS and rec["guideline_id"] == "nsclc_nodriver":
        return None, ["for tumours without a driver alteration"]

    rng = _tps_range(context)
    if rng and case.pdl1 is not None:
        if rng[0] <= case.pdl1 < rng[1]:
            s += 2
            why.append(f"PD-L1 {case.pdl1}% is in this group")
        else:
            return None, ["different PD-L1 group"]

    if case.histology:
        if re.search(r"non[- ]?squamous", context, re.I):
            if case.histology != "non-squamous":
                return None, ["non-squamous histology"]
            s += 1
        elif re.search(r"\bsquamous", context, re.I):
            if case.histology != "squamous":
                return None, ["squamous histology"]
            s += 1

    prior = set(case.prior)
    rec_prior = rec.get("prior", {"has": [], "not": []})
    if prior & set(rec_prior.get("not", [])):
        return None, ["for patients who have NOT had " + ", ".join(sorted(prior & set(rec_prior["not"])))]
    needs = set(rec_prior.get("has", []))
    had = prior & needs
    missing = needs - prior
    if had:
        s += min(3.0, 1.5 * len(had))
        why.append("after " + ", ".join(sorted(had)))
    if missing and (had or prior):
        # "progressed on osimertinib AND platinum chemotherapy" for a case
        # that has had only osimertinib: it is for a later line.
        s -= 1.5 * len(missing - {"chemotherapy"} if "platinum" in missing else missing)
        why.append("assumes prior " + ", ".join(sorted(missing)))

    rec_set = set(rec.get("settings", []))
    if "first-line" in rec_set and re.search(r"progress\w*|refractory|after (?:first|prior)|previously (?:received|treated)", rec["statement"], re.I):
        rec_set = (rec_set - {"first-line"}) | {"second-line"}  # "progressed on first-line X"
    case_set = set(case.settings)
    if rec_set & case_set:
        s += 2
        why.append(", ".join(sorted(rec_set & case_set)))
    first = {"first-line"}
    later = {"second-line", "later-line"}
    if (case_set & later and rec_set & first and not rec_set & later) or (
        case_set & first and rec_set & later and not rec_set & first
    ):
        s -= 3
        why.append("different line of therapy")

    # Curative-intent (resected) vs advanced disease are different questions.
    periop = {"adjuvant", "neoadjuvant"}
    if case_set & {"metastatic", "locally advanced"} and not case_set & periop and rec_set & periop:
        s -= 2.5
        why.append("for resected / perioperative disease")
    elif case_set & periop and rec_set & {"metastatic"} and not rec_set & periop:
        s -= 2.5
        why.append("for advanced disease")

    # Everything this recommendation offers has already been given.
    offered = set(rec.get("drugs", [])) - needs
    if offered and offered <= prior:
        s -= 3
        why.append("already given: " + ", ".join(sorted(offered)))

    if case.question:
        overlap = _words(case.question) & _words(whole)
        # "which chemotherapy?" is answered by "carboplatin": compare classes too.
        overlap |= (set(therapies_in(case.question)) & set(therapies_in(rec["statement"]))) - prior
        if overlap:
            s += min(2.0, 0.5 * len(overlap))
            why.append("question: " + ", ".join(sorted(overlap)[:4]))

    # Recommendations about TESTING matter less once the biomarkers are known.
    if case.biomarkers and re.search(r"\btesting\b|\btest(?:ed)?\b|assay|should be (?:universally )?accessible", rec["statement"], re.I) \
            and not re.search(r"\boffer\w*\b.*\b(?:\w+mab|\w+nib|chemotherapy)", rec["statement"], re.I):
        s -= 2
        why.append("about biomarker testing")

    # The case's own clinical terms ("flushing", "diarrhea"), lightly weighted:
    # they separate statements within one guideline, never across cancers. A
    # drug already given is history, not a match.
    shared = (_words(case.text) - _words(case.question) - {p.lower() for p in prior}) & _words(rec["statement"])
    if shared:
        s += min(1.5, 0.25 * len(shared))

    if re.search(r"insufficient (?:evidence|data)|no recommendation (?:for or against|can be made)", rec["statement"], re.I):
        s -= 1.5
        why.append("no recommendation either way")

    if re.search(r"\bshould not\b|\bnot recommended\b|\bnot be offered\b|no survival advantage", rec["statement"], re.I):
        s -= 1
        why.append("recommends AGAINST")

    if case.ps is not None and re.search(r"\bPS\s*(?:of\s*)?2|performance status (?:of )?2|poor performance", whole, re.I):
        s += 1.5 if case.ps >= 2 else -2
    return s, why


def rank(case: Case, k: int = 5) -> tuple[list[Match], list[Match]]:
    """(top matches, cross-cutting matches). Ties break on strength, then
    evidence quality, then the guideline's own order."""
    recs, stats, guides = load()
    by_gid = {g["id"]: g for g in guides}
    pool = [r for r in recs if by_gid[r["guideline_id"]]["cancer"] == case.cancer]
    scored = []
    for order, r in enumerate(pool):
        sc, why = score(r, case)
        if sc is None:
            continue
        strength = STRENGTH.get(str(r.get("strength", "")).lower(), 0)
        evidence = EVIDENCE.get(str(r.get("evidence", "")).lower(), 0)
        scored.append((sc, strength, evidence, -order, r, why))
    scored.sort(key=lambda x: x[:4], reverse=True)
    top = [Match(r, sc, why, _stats(r, stats)) for sc, _, _, _, r, why in scored[:k]]

    cross = []
    for gid, pat in CROSS_TRIGGERS.items():
        if re.search(pat, case.text, re.I) and gid in by_gid:
            words = _words(case.text)
            cands = [r for r in recs if r["guideline_id"] == gid]
            cands.sort(key=lambda r: -len(words & _words(r["statement"])))
            cross += [Match(r, 0, ["mentioned in the case"], _stats(r, stats)) for r in cands[:2]]
    return top, cross


def _stats(rec: dict, stats: list[dict]) -> list[dict]:
    return [{**stats[s["i"]], "basis": s["basis"]} for s in rec.get("stats", [])]


# Figures worth drawing the eye to inside a verbatim statistic sentence.
FIGURE = re.compile(
    r"(\b(?:HR|OR|RR)\b[,:=]?\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s?(?:%|months?|mo\b)|"
    r"95% CI,?\s*[\d.]+\s*(?:to|-|–)\s*[\d.]+|\bP\s*[<=≤]\s*\.?\d+(?:\.\d+)?)"
)


def highlight(sentence: str) -> str:
    """Markdown with the figures in bold; the text itself is unchanged."""
    safe = sentence.replace("$", "\\$").replace("*", "\\*").replace("_", "\\_")
    return FIGURE.sub(lambda m: f"**{m.group(0)}**", safe)
