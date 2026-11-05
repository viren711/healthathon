# Treatment Guidance

The platform's fifth tool, for **clinicians**. A doctor picks a cancer,
answers two or three questions, and sees the recommendations from the **ASCO
clinical practice guidelines** that fit best, ranked, each with its strength
of recommendation, evidence quality, why it fits, and (under "Details") the
guideline version, the statistics the guideline itself prints, and the PDF
page.

**The flow** (`app/views/treatment_guidance.py`):

1. **Describe the case, or pick the cancer.** The page starts with a text box
   and six example cases (each checked: the guideline's own answer comes out
   first). Typing a case fills in the cancer and the questions below, which
   stay editable. Or skip the text and tap one of the 11 covered cancers.
   A line under the title lists the covered cancers and says any other is not
   covered.
2. **A few questions, all optional.** The situation first (starting
   treatment, first treatment stopped working, after surgery, cancer has come
   back ...), then the one or two facets that most of *that cancer's*
   recommendations depend on, chosen from the data, not hard-coded: tumour
   details such as biomarkers or transplant eligibility, PD-L1 level,
   histology, or prior drugs. Lung gets situation, biomarkers and histology;
   neuroendocrine tumours only the situation, because nothing else
   separates their recommendations.
3. **Best matches, at once.** The top three appear as soon as a cancer is
   picked, with a fit bar, and re-rank after every answer; the next five sit
   behind "Show more". A cancer-less case, or one with answers that rule
   everything out, says so instead of guessing.

The same rules-based reader fills in the questions from typed text, and any
answer it fills in that is not one of the usual questions is shown rather than
acting unseen. All 26 evaluation cases give the expected answer in the top 3
through this page, as they do through the engine. A recommendation with no
printed trial figures simply shows none.

No language model is involved. Every statement and every figure on the page is
a verbatim span of a guideline PDF, and a test checks that.

Page: `app/views/treatment_guidance.py`, at `/guidance`.

---

## Source material

19 ASCO guidelines in `data/asco/` (copyrighted, gitignored like the books),
registered in `config/asco_guidelines.yaml` with the cancer each one covers:

| cancer | guidelines |
|---|---|
| Lung (NSCLC) | stage IV with driver alterations; without driver alterations (both living, v2026.3.4) |
| Breast | HR-positive HER2-negative stage I–III medical therapy; follow-up and surveillance |
| Ovarian | PARP inhibitors; systemic treatment of recurrence |
| Multiple myeloma | ASCO living guideline; ASCO–Ontario Health |
| Melanoma, thyroid, gastroesophageal, biliary, prostate (mCRPC), adult glioma, GEP-NET | one each |
| Cross-cutting | ctDNA testing, WBC growth factors, cancer in pregnancy, patient–clinician communication |

Cross-cutting guidelines are shown under "Also relevant" when a case mentions
their subject (pregnancy, neutropenia/G-CSF, ctDNA, breaking news).

---

## Build: `scripts/09_build_asco.py`

One process per guideline; all 19 in about **1 second**.

| output | what it is |
|---|---|
| `corpus/asco/recommendations.jsonl` | 443 records: number, verbatim statement, evidence quality, strength, type, page, version, context headings, case tags |
| `corpus/asco/statistics.jsonl` | 821 verbatim sentences carrying trial figures (HR, median OS/PFS, ORR, CI, P), with page, trial names and drugs |

### Recommendations

Most guidelines end each statement with a rating, e.g. *"(Evidence quality:
Moderate; Strength of recommendation: Strong)"*. The parser finds every rating
and takes the statement back to its recommendation number. Things that had to
be handled, each found by checking output:

- **Context headings.** In the lung guidelines the regimen alone ("Pembrolizumab
  + carboplatin + pemetrexed") means nothing without the table headings above
  it ("Non-squamous · TPS 1%–49%"). Short heading lines are kept as sticky
  context by kind (driver, PD-L1, histology, line of therapy, stage), and a
  new heading of the same kind replaces the old one. The guideline's own
  numbered questions ("Clinical Question 2: … second-line …") set the line of
  therapy and start a fresh context.
- **The same regimen under different headings** is a different recommendation
  and is kept; a numberless copy of a numbered statement (summary boxes) is
  dropped, as is a numbered statement printed twice.
- **Ligatures** ("ﬁrst-line" in journal PDFs) are normalised; words hyphenated
  across lines are rejoined in headings.
- Ratings of **"N/A"** and ratings with a qualifier first ("(75% of cells…;
  Evidence quality: High; …)") are recognised.
- Two guidelines print **no ratings** (communication; ctDNA): their numbered
  statements are taken instead, with the panel agreement or strength where
  printed.
- **Polarity.** "HER2-negative", "1p19q non-codeleted", "other than BRCA" name a
  marker to say it is absent. Tags record both present and absent markers,
  and the statement's own wording overrides a heading's.

### Statistics

Sentences with a statistic word (HR, median, survival, ORR, PFS, OS, CI…) and a
figure (a percentage, months, a P value, an HR). Split only at sentence ends —
splitting at ";" left fragments like "HR, 0.19) and…". Each is attached to a
recommendation by its **section** (the "Recommendation 1.9" heading it sits
under) or, failing that, by **shared drug names**, and the page labels which.

---

## Case reader and ranking: `rag/guidance.py`

`read_case(text)` is literal: it reads what is written. Cancer type (synonyms
such as "cholangiocarcinoma" → biliary), setting and line of therapy,
biomarkers present and tested negative, PD-L1 TPS, histology, ECOG, pregnancy,
drugs and drug classes already given — minus those negated ("no prior BRAF
therapy", "IO-naive") — and the question. A history of treatment with
progression implies a later line.

The page shows these as fields. **Nothing is searched until the doctor
confirms**, and a cancer no guideline covers leaves the field unset with a
warning, so a pancreatic case can never be searched as biliary.

`rank(case)` scores each recommendation for that cancer:

| rule | effect |
|---|---|
| shared driver biomarker | +3 each |
| recommendation needs a driver the case does not have, when the case names its drivers | excluded |
| marker polarity clash (HER2-negative regimen, HER2-positive case) | excluded |
| mutually exclusive groups (WHO grade; platinum-sensitive/resistant; transplant-eligible/ineligible; ESCC/adenocarcinoma; thyroid subtype) | excluded on mismatch |
| PD-L1 range or histology in the heading disagrees | excluded |
| "for patients who have not received X" and X was given | excluded |
| required prior therapy present / missing | + / − per item |
| line of therapy agrees / conflicts; resected vs advanced | +2 / −3; −2.5 |
| everything offered was already given | −3 |
| "should not", "not recommended" | −1, and badged "Recommends against" |
| testing recommendation when biomarkers are known; "insufficient evidence" | −2; −1.5 |
| question words, drug classes and case terms shared with the statement | small, capped |

Ties break on strength, then evidence quality, then the guideline's own order.
The reasons are shown on every card ("matches EGFR · after EGFR TKI,
osimertinib · second-line").

---

## Results (`scripts/10_eval_guidance.py`, `eval/guidance_cases.yaml`)

| check | result |
|---|---|
| statements and statistics found word for word in their PDF | **1,264 / 1,264** |
| facts extracted by the case reader (11 labelled) | **11 / 11** |
| cancer not covered → nothing returned | **1 / 1** |
| expected recommendation in the top 3, **new cases scored once, untouched** | **5 / 10** |
| expected recommendation in the top 3, cases used while tuning | 35 / 35 |

The first number to trust is the **5/10**. Labels were read off the guideline
statements, not off the matcher's output. Two rounds of held-out cases each
scored 5/10 on first contact; each round's misses exposed general gaps (negated
prior therapy, "MSI-high", transplant eligibility, ESCC vs adenocarcinoma,
"PD-L1 below 1", a heading overriding a statement's polarity), which were fixed
and the cases folded into the tuning set. One miss was arguably a labelling
choice: for "progressed on lenvatinib" the guideline also recommends adding
pembrolizumab (3.2), which the matcher ranked first; the label listed only
cabozantinib (1.2) and was not changed after the fact.

Examples that now come out right: first-line EGFR exon 19 → osimertinib +
chemotherapy or amivantamab + lazertinib (1.1); progressed on osimertinib →
platinum chemotherapy with amivantamab (2.1); after osimertinib and platinum →
datopotamab deruxtecan (2.3); exon 20 insertion → sunvozertinib (1.7); squamous
NSCLC with PD-L1 70% → single-agent PD-(L)1 inhibitor (2.1); HER2-positive
gastric → HER2-positive regimens only.

## Known limits

- **Rules, not understanding.** About half of genuinely new cases get the
  expected recommendation in the top 3 on first contact. Remaining known
  misses: survivorship/follow-up cases drawn to treatment guidelines; HRR
  genes other than BRCA (ATM) not in the vocabulary; a PD-L1 <1% NSCLC case
  ranking palliative-care referral first. The reasons line makes a wrong
  match visible, and the doctor sees the guideline text, not a conclusion.
- Statistics attached "by drug name" are related figures from the same
  guideline, not necessarily the trial behind that recommendation; the page
  labels them so.
- Some PDF glyphs extract oddly and are shown as extracted (e.g. "PLD 1
  carboplatin" where the PDF prints "+").
- Coverage is the 19 guidelines. Anything else gets "not covered", plus
  optional textbook background marked as older and not a recommendation.
- The guidelines are copyrighted; this runs locally. Publishing it would need
  ASCO's permission.

## When the guidelines have nothing: the textbooks

If the ASCO guidelines here cannot answer (the pasted text names a cancer none of
the 19 guidelines covers, or no recommendation matches the picked answers), the
page asks the oncology textbooks the same question, through the same pipeline the
Knowledge Assistant uses (retrieval, the three refusal layers, every sentence
cited). The answer appears under **From the textbooks** with a green notice that
it is **not from the ASCO guidelines** and where it does come from, its sources
listed by book, chapter and page, and a button that opens the same question in the
Knowledge Assistant. If the textbooks do not cover it either, it says so.

- The question is the doctor's own text, or, when they only picked a cancer, a
  question built from the cancer and the answers picked.
- The answer is cached for the session, so a rerun does not ask again.
- It needs the language model (about half a minute; the first use also loads the
  textbook index). A covered case never calls it: ASCO recommendations stay
  quoted, not generated.
- Checked live on "Metastatic pancreatic adenocarcinoma, untreated. First-line?":
  no ASCO guideline here covers pancreatic cancer, and the textbooks returned a
  cited answer (DeVita, ch. 55) in 185 s on a cold start.
