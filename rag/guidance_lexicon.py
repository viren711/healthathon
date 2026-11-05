"""One vocabulary for describing a case, shared by the guideline parser
(scripts/09_build_asco.py) and the case reader (rag/guidance.py).

If the parser tags a recommendation "EGFR" and the case reader tags a case
"EGFR", they match; two vocabularies would drift apart. Everything here is a
plain regular expression -- no model -- so the same text always gives the same
tags and a mis-tag can be fixed by editing one line.
"""

from __future__ import annotations

import re

# canonical name -> pattern. Order does not matter.
BIOMARKERS: dict[str, str] = {
    "EGFR": r"\bEGFR\b|exon ?19 deletion|L858R|exon ?20 insertion|T790M",
    "EGFR exon 20 insertion": r"exon ?20 insertion",
    "ALK": r"\bALK\b",
    "ROS1": r"\bROS1\b",
    "BRAF": r"\bBRAF(?:\s?V600E?)?\b",
    "MET": r"\bMET(?:\s+exon\s*14)?\b(?! ?\w*therapy)",
    "RET": r"\bRET\b",
    "NTRK": r"\bNTRK\b",
    "HER2": r"\bHER-?2\b",
    "KRAS": r"\bKRAS(?:\s?G12C)?\b",
    "PD-L1": r"\bPD-?\(?L\)?-?1\b",
    "BRCA": r"\bBRCA\s?[12]?\b|\bgBRCA\b|\btBRCA\b",
    "HRD": r"\bHRD\b|homologous recombination",
    "MSI-H/dMMR": r"\bMSI-?H(?:igh)?\b|\bdMMR\b|microsatellite instab\w+|mismatch repair[- ]deficien\w+",
    "Anaplastic thyroid (ATC)": r"\bATC\b|anaplastic thyroid",
    "Differentiated thyroid (DTC)": r"\bDTC\b|differentiated thyroid|radioiodine[- ]refractory|\bRAIR\b",
    "Medullary thyroid (MTC)": r"\bMTC\b|medullary",
    "TMB-H": r"\bTMB-?H\b|tumou?r mutational burden",
    "IDH": r"\bIDH[ -]?(?:mutant|1|2)?\b",
    "1p19q": r"\b1p/?19q\b",
    "MGMT": r"\bMGMT\b",
    "ER/PR positive": r"\b(?:ER|HR)[- ]?(?:positive|\+)\b|hormone receptor[- ]positive",
    "PIK3CA": r"\bPIK3CA\b",
    "ESR1": r"\bESR1\b",
    "CLDN18.2": r"\bCLDN ?18\.2\b|claudin ?18\.2",
    "FGFR": r"\bFGFR\w*\b",
    "IDH1": r"\bIDH1\b",
    "HER2-low": r"\bHER2[- ]low\b",
    "del(17p)/high-risk": r"del\(?17p\)?|high-risk cytogenetic\w*",
    "CD38": r"\bCD38\b|daratumumab|isatuximab",
    "BCMA": r"\bBCMA\b",
    "RAS wild-type": r"\bRAS wild[- ]type\b",
    "Platinum-sensitive": r"platinum[- ]sensitive|\bPSOC\b",
    "Platinum-resistant": r"platinum[- ]resistant|\bPROC\b",
    "Transplant-eligible": r"transplant[- ]eligible|candidates? for (?:ASCT|transplant)",
    "Transplant-ineligible": r"transplant[- ]ineligible|not (?:a )?candidates? for (?:ASCT|transplant)|ineligible for (?:ASCT|transplant)",
    "Squamous (ESCC)": r"\bESCC\b|(?:esophageal|oesophageal) squamous",
    "Adenocarcinoma (GEJ/gastric)": r"gastr\w*(?:/GEJ)? adenocarcinoma|gastroesophageal adenocarcinoma|GEJ adenocarcinoma|esophageal adenocarcinoma",
    "WHO grade 2": r"\bgrade (?:2|II)\b",
    "WHO grade 3": r"\bgrade (?:3|III)\b",
    "WHO grade 4": r"\bgrade (?:4|IV)\b",
    "Castration-resistant": r"castration[- ]resistant|\bmCRPC\b",
    "mHSPC": r"\bmHSPC\b|castration[- ]sensitive|hormone[- ]sensitive",
}

SETTINGS: dict[str, str] = {
    "first-line": r"first[- ]line|front[- ]?line|\binitial (?:therapy|treatment)|previously untreated|treatment[- ]na[iï]ve|newly diagnosed",
    "second-line": r"second[- ]line|\bsecond-?line\b|progress\w* (?:on|after|following)|after progression",
    "later-line": r"third[- ]line|subsequent[- ](?:therapy|line|treatment|lines?)|later[- ]line|relapsed|(?<!iodine[- ])(?<!iodine )refractory",
    "adjuvant": r"(?<!neo)\badjuvant\b|\bresected\b|post-?operative|after (?:surgery|resection)",
    "neoadjuvant": r"neoadjuvant|preoperative",
    "maintenance": r"\bmaintenance\b",
    "recurrence": r"\brecurren\w+|\brelapse\w*",
    "metastatic": r"\bmetasta\w+|stage IV|advanced",
    "locally advanced": r"locally advanced|unresectable",
    "early stage": r"stage (?:I|II|III)\b|early[- ]stage|resectable",
    "surveillance": r"surveillance|follow-?up",
    "symptom": r"symptom|carcinoid syndrome|diarrh\w+|flushing",
    "supportive": r"neutropenia|growth factor|G-?CSF|supportive",
}

_SUFFIX = (
    r"[a-z]{3,}(?:tinib|ciclib|parib|mab|zomib|lisib|rafenib|metinib|degib|platin|taxel|"
    r"rubicin|stat|fenib|nib|lutamide|relix|tide|vir|tamab|ximab|zumab|limab|tecan)"
)
_EXTRA = (
    "pemetrexed gemcitabine capecitabine fluorouracil irinotecan oxaliplatin temozolomide "
    "lomustine procarbazine vincristine tamoxifen letrozole anastrozole exemestane fulvestrant "
    "abiraterone enzalutamide apalutamide darolutamide lenalidomide thalidomide pomalidomide "
    "dexamethasone cyclophosphamide melphalan lutetium octreotide lanreotide telotristat "
    "everolimus doxorubicin etoposide docetaxel paclitaxel cabazitaxel radium ipilimumab "
    "nivolumab pembrolizumab relatlimab carboplatin cisplatin bevacizumab ramucirumab "
    "trastuzumab zolbetuximab durvalumab tremelimumab atezolizumab cemiplimab dostarlimab "
    "elranatamab teclistamab talquetamab belantamab selinexor venetoclax ixazomib carfilzomib"
).split()
_DRUG = re.compile(rf"\b(?:{_SUFFIX}|{'|'.join(_EXTRA)})\b", re.I)
_NOT_DRUG = {"stat", "tide", "vir", "nib", "mab", "imab", "adjuvant"}

# Drug classes, so "progressed on EGFR TKI" in a guideline meets "progressed on
# osimertinib" in a case. A class is matched by its name or any member.
THERAPY_CLASSES: dict[str, str] = {
    "platinum": r"platinum|carboplatin|cisplatin|oxaliplatin",
    "chemotherapy": r"chemotherap\w*|carboplatin|cisplatin|pemetrexed|docetaxel|paclitaxel|gemcitabine|"
                    r"cabazitaxel|capecitabine|fluorouracil|FOLFOX|FOLFIRI|irinotecan|etoposide",
    "immunotherapy": r"immunotherap\w*|immune checkpoint|anti-PD-?\(?L\)?-?1|pembrolizumab|nivolumab|"
                     r"atezolizumab|cemiplimab|durvalumab|dostarlimab|ipilimumab",
    "EGFR TKI": r"EGFR[- ]TKI|osimertinib|gefitinib|erlotinib|afatinib|dacomitinib|lazertinib|"
                r"third-generation TKI",
    "ALK inhibitor": r"ALK inhibitor|ALK TKI|alectinib|brigatinib|lorlatinib|crizotinib|ceritinib|ensartinib",
    "BRAF therapy": r"BRAF(?:/MEK)? (?:therapy|inhibitor)|dabrafenib|encorafenib|vemurafenib",
    "PARP inhibitor": r"PARP ?i(?:nhibitor)?\w*|olaparib|niraparib|rucaparib|talazoparib",
    "ARPI": r"\bARPI\b|androgen receptor pathway|abiraterone|enzalutamide|apalutamide|darolutamide",
    "taxane": r"taxane|docetaxel|cabazitaxel|paclitaxel",
    "anti-CD38": r"anti-CD38|daratumumab|isatuximab",
    "IMiD": r"\bIMiD|immunomodulatory|lenalidomide|pomalidomide|thalidomide",
    "proteasome inhibitor": r"proteasome inhibitor|bortezomib|carfilzomib|ixazomib",
    "trastuzumab": r"trastuzumab",
    "bevacizumab": r"bevacizumab|anti-VEGF",
}


def therapies_in(text: str) -> list[str]:
    """Named drugs plus the classes they (or the text) belong to."""
    classes = [k for k, p in THERAPY_CLASSES.items() if re.search(p, text, re.I)]
    return sorted(set(drugs_in(text)) | set(classes))


PRIOR = re.compile(
    r"(?:previously (?:received|treated with|exposed to)|prior (?:treatment with|therapy with|use of)?|"
    r"progress\w*(?: disease)? (?:on|after|following|with)|after (?:treatment with|therapy with)?|"
    r"have (?:not )?received|who (?:have )?(?:not )?received|(?:refractory|resistant) to)\s+(?P<what>[^.;(),]{3,90})",
    re.I,
)


def prior_therapy_in(statement: str) -> dict[str, list[str]]:
    """Therapies a statement says the patient HAS or HAS NOT had before.

    {"has": [...], "not": [...]}, as drug names and drug classes:
    "progressed on osimertinib and platinum chemotherapy" ->
    has: [chemotherapy, EGFR TKI, osimertinib, platinum].
    """
    has, no = [], []
    for m in PRIOR.finditer(statement):
        window = m.group(0)
        # "...progressed on X, clinicians may offer Y": stop before the offer.
        what = re.split(r"\b(?:clinicians|the panel|should|may|consider)\b", m.group("what"), flags=re.I)[0]
        names = therapies_in(what)
        if not names:
            continue
        (no if re.search(r"\bnot\b|\bno prior|\bwithout\b|na[iï]ve", window, re.I) else has).extend(names)
    return {"has": sorted(set(has)), "not": sorted(set(no))}


# "HER2-negative", "EGFR wild-type", "without ALK": the text names the marker
# to say the tumour does NOT carry it.
_NEGATED_AFTER = re.compile(r"^[\s-]*(?:\w+\s+){0,2}?(?:negative|neg\b|wild[- ]?type|-ve\b|not detected|non-?amplified|non-?\s?co-?\s?deleted|low\b)", re.I)
_NEGATED_BEFORE = re.compile(r"(?:\bno|\bnegative for|\bwithout|\babsence of|other than|besides|excluding)\s+(?:\w+\s+){0,2}$", re.I)


def _mentions(text: str) -> dict[str, list[bool]]:
    """canonical marker -> one flag per mention, True when that mention is negated."""
    out: dict[str, list[bool]] = {}
    for k, p in BIOMARKERS.items():
        for m in re.finditer(p, text, re.I if k != "MET" else 0):
            neg = bool(_NEGATED_AFTER.match(text[m.end(): m.end() + 30])
                       or _NEGATED_BEFORE.search(text[max(0, m.start() - 30): m.start()]))
            out.setdefault(k, []).append(neg)
    return out


def biomarkers_in(text: str) -> list[str]:
    """Markers the text says ARE present (at least one non-negated mention)."""
    return [k for k, flags in _mentions(text).items() if not all(flags)]


def negated_biomarkers_in(text: str) -> list[str]:
    """Markers the text only ever mentions as absent ("HER2-negative")."""
    return [k for k, flags in _mentions(text).items() if all(flags)]


def settings_in(text: str) -> list[str]:
    return [k for k, p in SETTINGS.items() if re.search(p, text, re.I)]


def drugs_in(text: str) -> list[str]:
    return sorted({m.group(0).lower() for m in _DRUG.finditer(text)} - _NOT_DRUG)
