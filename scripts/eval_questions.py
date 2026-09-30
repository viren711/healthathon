#!/usr/bin/env python3
"""Answer a question set from a Word file and write the results to a spreadsheet.

Each question goes through the same `answer_question` the portal uses, so all
three refusal layers apply: a question the books do not answer comes back as
"Not in book", never as a guess.

    python scripts/eval_questions.py                        # test/Questions .docx
    python scripts/eval_questions.py --limit 3              # first three only
    python scripts/eval_questions.py --build-only           # re-make the sheet, no model calls

Answers are checkpointed to a JSONL file as each one finishes, so an
interrupted run resumes where it stopped (on CPU a full run takes hours).
Delete the checkpoint, or pass --fresh, to start over.

The model is any preset in config/models.ini. The default, qwen3.5-4b, gave
more precise, better-reasoned answers than the portal's qwen2.5-3b-instruct on
these multi-part clinical questions, at ~1.5x the time per question.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STATUS_LABEL = {
    "answered": "Answered",
    "not_available": "Not in book",
    "out_of_scope": "Not in book (outside the books' scope)",
    "error": "Error",
}


def read_docx_questions(path: Path) -> list[tuple[int, str]]:
    """Numbered questions from a .docx, one per paragraph ("12. text")."""
    xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8")
    out: list[tuple[int, str]] = []
    for para in re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S):
        text = "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", para)).strip()
        text = (text.replace("&amp;", "&").replace("&lt;", "<")
                .replace("&gt;", ">").replace("&quot;", '"').replace("&apos;", "'"))
        if not text:
            continue
        m = re.match(r"^(\d+)[.)]\s*(.+)$", text)
        if m:
            out.append((int(m.group(1)), m.group(2).strip()))
        else:
            out.append((len(out) + 1, text))
    return out


def make_client(model: str):
    from rag.llm import LLMClient

    client = LLMClient.from_config()
    client.model = model
    # Qwen3.x reasons before answering unless told not to; the hidden
    # reasoning would eat the answer's token budget. The writer config
    # already records which models need this.
    try:
        from writer.studio import load_writer_config

        for m in load_writer_config().get("models", []):
            if m["id"] == model and m.get("extra_body"):
                client.extra_body = m["extra_body"]
    except Exception:  # noqa: BLE001 - writer config is optional here
        pass
    return client


def references(ans) -> list[str]:
    """The sources the answer actually cites, in citation-number order."""
    by_number = {s.number: s for s in ans.sources}
    return [f"[{n}] {by_number[n].citation}" for n in ans.cited if n in by_number]


def run(questions, checkpoint: Path, model: str) -> None:
    from rag.pipeline import answer_question

    done = set()
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["number"])
    todo = [(n, q) for n, q in questions if n not in done]
    print(f"{len(questions)} questions, {len(done)} already answered, {len(todo)} to go")
    if not todo:
        return

    client = make_client(model)
    ok, detail = client.health()
    if not ok:
        raise SystemExit(f"model server unreachable ({detail}); start it with ./start")

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    for i, (n, q) in enumerate(todo, 1):
        try:
            ans = answer_question(q, client=client)
            row = {
                "number": n,
                "question": q,
                "status": ans.status,
                "response": ans.text,
                "references": references(ans),
                "elapsed_s": round(ans.elapsed_ms / 1000, 1),
                "truncated": bool(getattr(ans, "truncated", False)),
            }
        except Exception as exc:  # noqa: BLE001 - one failure must not end the run
            row = {"number": n, "question": q, "status": "error",
                   "response": f"{type(exc).__name__}: {exc}", "references": [],
                   "elapsed_s": 0.0, "truncated": False}
        row["model"] = model
        with checkpoint.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        eta = (time.time() - t_all) / i * (len(todo) - i) / 60
        print(f"[{n:>3}] {row['status']:<14} {row['elapsed_s']:>6.0f}s  "
              f"refs {len(row['references'])}  (~{eta:.0f} min left)  {q[:50]}",
              flush=True)


def build_sheet(checkpoint: Path, out: Path, source_name: str) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.table import Table, TableStyleInfo

    rows = {}
    for line in checkpoint.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["number"]] = r  # a later line for the same question wins
    rows = [rows[k] for k in sorted(rows)]

    font = "Arial"
    head = Font(name=font, bold=True, color="FFFFFF")
    body = Font(name=font, size=10)
    wrap = Alignment(wrap_text=True, vertical="top")
    thin = Side(style="thin", color="D0D0D0")
    border = Border(top=thin, bottom=thin, left=thin, right=thin)
    fills = {
        "answered": PatternFill("solid", fgColor="E2F0D9"),
        "not_available": PatternFill("solid", fgColor="FCE4D6"),
        "out_of_scope": PatternFill("solid", fgColor="FCE4D6"),
        "error": PatternFill("solid", fgColor="F4CCCC"),
    }

    wb = Workbook()
    ws = wb.active
    ws.title = "Results"
    headers = ["#", "Question", "Response", "In book?", "References", "Time (s)"]
    widths = [5, 55, 90, 16, 48, 9]
    ws.append(headers)
    for c, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(c)].width = w
        cell = ws.cell(row=1, column=c)
        cell.font = head
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.alignment = Alignment(vertical="center", wrap_text=True)

    for r in rows:
        answered = r["status"] == "answered"
        ws.append([
            r["number"],
            r["question"],
            r["response"] if answered or r["status"] == "error" else "Not in book",
            "Yes" if answered else ("Error" if r["status"] == "error" else "No — not in book"),
            "\n".join(r["references"]) if r["references"] else "—",
            r["elapsed_s"],
        ])
        i = ws.max_row
        for c in range(1, len(headers) + 1):
            cell = ws.cell(row=i, column=c)
            cell.font = body
            cell.alignment = wrap
            cell.border = border
        ws.cell(row=i, column=4).fill = fills.get(r["status"], PatternFill())
        ws.cell(row=i, column=6).number_format = "0"
        # Row height so the answer is readable without resizing: ~95 chars
        # per line in the Response column at this width.
        lines = max(len(str(r["response"])) // 95 + str(r["response"]).count("\n") + 1,
                    len(r["references"]), 2)
        ws.row_dimensions[i].height = min(409, 13 * lines)

    last = ws.max_row
    ws.freeze_panes = "C2"
    if last > 1:
        table = Table(displayName="Results", ref=f"A1:F{last}")
        table.tableStyleInfo = TableStyleInfo(name="TableStyleLight1", showRowStripes=False)
        ws.add_table(table)

    # --- Summary (formulas over the Results sheet, so it follows any edits) ---
    sm = wb.create_sheet("Summary")
    sm.column_dimensions["A"].width = 34
    sm.column_dimensions["B"].width = 14
    sm.column_dimensions["C"].width = 12
    rng = f"Results!$D$2:$D${max(last, 2)}"
    items = [
        ("Questions", f"=COUNTA(Results!$A$2:$A${max(last, 2)})", None),
        ("Answered from the books", f'=COUNTIF({rng},"Yes")', "pct"),
        ("Not in book (refused)", f'=COUNTIF({rng},"No*")', "pct"),
        ("Errors", f'=COUNTIF({rng},"Error")', "pct"),
        ("Average time per question (s)", f"=IFERROR(AVERAGE(Results!$F$2:$F${max(last, 2)}),0)", None),
        ("Total run time (min)", f"=SUM(Results!$F$2:$F${max(last, 2)})/60", None),
    ]
    sm.append(["Measure", "Value", "Share"])
    for c in range(1, 4):
        sm.cell(row=1, column=c).font = head
        sm.cell(row=1, column=c).fill = PatternFill("solid", fgColor="1F4E78")
    for label, formula, share in items:
        sm.append([label, formula, "=IFERROR(B{r}/$B$2,0)".format(r=sm.max_row + 1) if share else None])
        r = sm.max_row
        for c in range(1, 4):
            sm.cell(row=r, column=c).font = body
        sm.cell(row=r, column=2).number_format = "0" if r <= 5 else "0.0"
        sm.cell(row=r, column=3).number_format = "0.0%"

    model = rows[0]["model"] if rows else "?"
    notes = [
        "",
        "About this run",
        f"Questions: {source_name}",
        f"Model: {model} (4-bit, CPU, llama.cpp), via the portal's full pipeline",
        "Sources: DeVita's Cancer (11th ed.) and Gunderson & Tepper's Clinical Radiation Oncology",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "How to read it",
        "In book? = Yes: the answer passed all three checks and every sentence cites a source.",
        "In book? = No: the books do not answer it well enough — the platform refuses rather than guesses.",
        "References: [n] matches the [n] markers in the response; book, chapter, printed page and PDF page.",
        "Answers are machine-generated from the textbooks and must be checked by a clinician.",
    ]
    for line in notes:
        sm.append([line])
        cell = sm.cell(row=sm.max_row, column=1)
        cell.font = Font(name=font, size=10, bold=line in ("About this run", "How to read it"))

    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    print(f"wrote {out}  ({len(rows)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("questions", nargs="?", default="test/Questions .docx")
    ap.add_argument("--model", default="qwen3.5-4b", help="a preset name from config/models.ini")
    ap.add_argument("--out", default="test/questions_results.xlsx")
    ap.add_argument("--checkpoint", default="eval/questions_results.jsonl")
    ap.add_argument("--limit", type=int, help="only the first N questions")
    ap.add_argument("--fresh", action="store_true", help="discard the checkpoint and start over")
    ap.add_argument("--build-only", action="store_true", help="only rebuild the sheet from the checkpoint")
    args = ap.parse_args()

    qpath = Path(args.questions)
    qpath = qpath if qpath.is_absolute() else ROOT / qpath
    checkpoint = ROOT / args.checkpoint
    out = ROOT / args.out

    if not args.build_only:
        questions = read_docx_questions(qpath)
        if args.limit:
            questions = questions[: args.limit]
        if args.fresh and checkpoint.exists():
            checkpoint.unlink()
        run(questions, checkpoint, args.model)
    if not checkpoint.exists():
        print("nothing answered yet", file=sys.stderr)
        return 1
    build_sheet(checkpoint, out, qpath.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
