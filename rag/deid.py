"""De-identification guardrail for text that leaves this machine.

The Research Platform sends questions to scite.ai. Before anything is sent,
`scrub()` replaces personal and patient identifiers with neutral tags such as
[PHONE] or [NAME], and the same scrub is applied to the answer that comes back,
so an identifier cannot travel in either direction.

What it catches (pattern based, no model, deterministic):
  names introduced by a title (Mr, Mrs, Dr, Shri ...) or by a cue ("patient name
  is ...", "named ...", "my father ..."); a patient's age, generalised to a decade
  ("54-year-old" -> [AGE 50s]) so the clinical picture survives, while ranges and
  thresholds ("aged 18-65", "over 65") are left alone; street addresses and PIN
  codes; full dates (birth, visit); and, though rarely relevant to research,
  email addresses, phone numbers, Aadhaar/PAN, record/ABHA numbers, IP addresses
  and web links (questions only).

What it cannot catch, and no pattern list can: a bare name with no title or
cue ("Ramesh has stage III lung cancer"), a hospital or village name, or an
identifier written in an unusual way. It lowers the risk; it does not remove
the need to keep patient details out of research questions. The page says so.

It is deliberately not switchable from the UI or .env: a guardrail that can be
turned off by configuration is not one.
"""

from __future__ import annotations

import re
from collections import Counter

_MONTH = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
_TITLE = r"(?i:mr|mrs|ms|miss|mx|dr|prof|shri|smt|sri|master)\.?"
_WORD = r"(?:[A-Z][a-z]+(?:[-'][A-Z][a-z]+)?|[A-Z]{3,})"
_NAME = rf"{_WORD}(?:\s+{_WORD}){{0,2}}"

# A patient's own age: "54-year-old", "54 y/o", "aged 54", "age: 54". Not a population
# range, threshold or statistic: "aged 18-65", "18 to 65 years old", "over 65", "65 years
# and older", "median age 62".
_AGE_BEFORE = "".join(f"(?<!{w}\\s)" for w in
                      ("over", "above", "under", "below", "than", "to", "and", "median", "mean", "average", "or"))
_AGE = (r"(?i)(?<![<>≥≤+\d–-])" + _AGE_BEFORE + r"\b(?:(?P<a>\d{1,3})[- ]?(?:years?|yrs?)[- ]?old"
        r"|(?P<c>\d{1,3})\s?(?:y/o|yo)\b|(?:aged?|age:)\s*(?P<b>\d{1,3}))\b"
        r"(?!\s*(?:[–-]|to\b)\s*\d)(?!\s*(?:years?\s+)?(?:and\s+|or\s+)?(?:older|above|over|younger|\+))")

# (label, pattern, replacement). Order matters: identifiers first, names last.
# A replacement that is a function keeps the cue words and tags only the name.
_RULES: list[tuple[str, re.Pattern[str], object]] = [
    ("email", re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"), "[EMAIL]"),
    ("link", re.compile(r"(?i)\b(?:https?://|www\.)\S+"), "[LINK]"),
    ("Aadhaar number", re.compile(r"(?<!\d)[2-9]\d{3}[\s-]\d{4}[\s-]\d{4}(?!\d)|(?<!\d)[2-9]\d{11}(?!\d)"), "[ID]"),
    ("ID number", re.compile(
        r"(?i)\b(?:mrn|uhid|ip\s*no|op\s*no|ipd|opd|abha(?:\s*(?:id|no|number))?|aadh?aa?r(?:\s*(?:id|no|number|card))?"
        r"|reg(?:istration)?\.?\s*(?:no|number|id)|patient\s*(?:id|no|number)|hospital\s*(?:id|no|number)"
        r"|case\s*(?:id|no|number)|passport(?:\s*(?:no|number))?)\b\.?\s*(?:no\.?|number|id|#)?\s*[:#\-]?\s*"
        r"(?=[A-Z0-9/-]*\d)[A-Z0-9][A-Z0-9/-]{3,}(?:[\s-]\d{4}){0,2}"), "[ID]"),
    ("PAN number", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[ID]"),
    ("phone number", re.compile(
        r"(?<![\w.])(?:\+?91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}(?![\w])"      # Indian mobile
        r"|(?<!\w)\+\d{1,3}[\s-]?\(?\d{1,4}\)?[\s-]?\d{3,4}[\s-]?\d{3,5}(?!\w)"  # international
        r"|(?<!\w)0\d{2,4}[\s-]\d{6,8}(?!\w)"), "[PHONE]"),                 # landline with STD code
    ("IP address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IP]"),
    ("date", re.compile(
        rf"\b\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b"
        rf"|\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}\.?,?\s+\d{{2,4}}\b"
        rf"|\b{_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b"), "[DATE]"),
    ("age", re.compile(_AGE), lambda m: _age_band(int(m.group("a") or m.group("b") or m.group("c")), m.group(0))),
    ("address", re.compile(
        r"(?i)\b(?:pin(?:\s*code)?|postal\s*code|zip)\s*[:\-]?\s*\d{5,6}\b"
        r"|\b(?:flat|house|plot|door|h\.?\s*no)\.?\s*(?:no\.?)?\s*[\w/-]*\d[\w/-]*[,\s]+[\w\s,.-]{0,60}?"
        r"(?:road|street|nagar|colony|lane|marg|sector|society|apartments?|layout)\b"), "[ADDRESS]"),
    ("name", re.compile(rf"\b{_TITLE}\s+{_NAME}"), "[NAME]"),
    ("name", re.compile(
        rf"(?i:(?P<cue>patient(?:'s)?\s+name|name\s+of\s+(?:the\s+)?patient|my\s+name\s+is|name\s+is|named|called|name\s*:))"
        rf"(?P<gap>\s*[:\-]?\s*)(?P<who>{_NAME})"),
        lambda m: f"{m.group('cue')}{m.group('gap')}[NAME]"),
    ("name", re.compile(
        rf"(?P<cue>(?i:\b(?:my|his|her|our)\s+(?:father|mother|wife|husband|son|daughter|brother|sister|uncle|aunt|friend|neighbou?r))\s+)"
        rf"(?P<who>{_NAME})"),
        lambda m: f"{m.group('cue')}[NAME]"),
]

def _age_band(age: int, original: str) -> str:
    """'54-year-old' -> '[AGE 50s]'; 90 and over -> '[AGE 90+]'. Keeps the clinical
    context (a patient in their fifties) without the exact age."""
    if age >= 90:
        return "[AGE 90+]"
    if age < 18:
        return "[AGE under 18]"
    return f"[AGE {age // 10 * 10}s]"


_TAG = re.compile(r"\[(?:NAME|PHONE|EMAIL|LINK|ID|IP|DATE|ADDRESS|AGE[^\]]*)\]")


def scrub(text: str, *, keep_links: bool = False) -> tuple[str, Counter]:
    """Return (clean text, what was removed as {label: count}).

    `keep_links=True` leaves web links alone: used on answers, where a link is
    a citation (a DOI), not an identifier.
    """
    found: Counter = Counter()
    if not text:
        return text or "", found
    for label, pattern, repl in _RULES:
        if keep_links and label == "link":
            continue
        text, n = pattern.subn(repl, text)  # type: ignore[arg-type]
        if n:
            found[label] += n
    return text, found


def has_content(clean: str) -> bool:
    """True if something is left once the tags are removed: a question that was
    nothing but identifiers has nothing to send."""
    words = re.findall(r"[A-Za-z]{3,}", _TAG.sub(" ", clean))
    return len(words) >= 3


_SHOWN = {"name": "name", "age": "age", "address": "address", "date": "dates"}


def describe(found: Counter) -> str:
    """'name, age, address' for the notice shown to the user. Contact numbers and
    ID numbers are removed too, but are not research-relevant enough to list one by
    one: they appear as 'IDs and contact details'."""
    shown = [_SHOWN[k] for k in _SHOWN if k in found]
    if any(k not in _SHOWN for k in found):
        shown.append("IDs and contact details")
    return ", ".join(shown)
