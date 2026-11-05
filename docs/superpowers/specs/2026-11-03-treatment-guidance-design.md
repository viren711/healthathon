# Treatment Guidance (ASCO) — design

Status: approved in conversation 2026-11-03 (sections 1–4), implemented on the
user's instruction without a separate spec-review stop. Not committed.

## Goal
A clinician describes a case; the page lists the most relevant ASCO guideline
recommendations, each with its strength and evidence quality, the guideline
version, a page link, and the statistics the guideline itself prints.

## Decisions (user)
- Approach A: recommendation-centred. Records are parsed from the guidelines;
  ranking is rule-based; the language model writes nothing on this page.
- Statistics: only what the sources print, verbatim sentences. None if absent.
- Input: short structured form + free-text box. A pasted case summary is read by
  rules (no model), shown as editable fields, and the doctor confirms.

## Data (`scripts/09_build_asco.py` -> `corpus/asco/`)
- One record per recommendation: guideline id/title/version, number, statement,
  evidence quality, strength, type, pages, case tags (cancer, setting,
  biomarkers, prior therapy).
- Statistic sentences: verbatim sentences carrying trial figures, with page,
  trial names, drugs; attached to a recommendation by its section number, else
  by shared drug/trial names (labelled as such).
- Parallel per PDF. Output gitignored (copyrighted).

## Matching (`rag/guidance.py`)
- Rule-based case reader: cancer, setting, biomarkers, prior therapy, question.
- Score: biomarker/setting/prior-therapy agreement; hard exclusions on conflict
  ("not previously received X" vs a case that has). Order by score, then
  strength, then evidence quality.

## Page (`app/views/treatment_guidance.py`, `/guidance`)
- Form + free text -> confirm -> ranked cards with statement, rating, version,
  page viewer, statistics. "No ASCO guideline covers this" + closest textbook
  passages (marked background) when nothing matches. Fixed decision-support note.

## Out of scope
Scite trial lookups, model-written plans, indexing ASCO into the retrieval index.

## Tests
`scripts/10_eval_guidance.py`: parser coverage per guideline, every statistic
sentence verbatim in its PDF, labelled cases (expected recommendation in top 3),
case-reader field recovery. Smoke/UI checks for the page.
