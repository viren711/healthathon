"""Safety screen for Patient Portal questions, run before anything else.

A patient question can carry something no retrieval answer should be the
first response to: a crisis, or symptoms that need care now. This checks for
those with plain patterns -- no language model, so it is instant, cannot be
talked out of its job, and behaves the same every time. Patterns and messages
live in config/patient.yaml under `safety`.

Three outcomes, strongest first:

  crisis             self-harm or suicidal language. The crisis message is the
                     whole response; no answer is generated. The Distress guide
                     is offered alongside it.
  emergency          symptoms that can be an emergency during cancer care. The
                     message is shown first, and the question is still answered.
  personal_decision  "should I stop my chemo?" -- answered in general terms,
                     with a reminder that the decision belongs to the care team.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from rag.config import load_patient_config

ORDER = ("crisis", "emergency", "personal_decision")


@dataclass
class SafetyResult:
    kind: str | None = None  # None when nothing matched
    message: str = ""
    guide: str | None = None  # a guide id worth offering, e.g. "distress"

    @property
    def blocks_answer(self) -> bool:
        return self.kind == "crisis"


@lru_cache(maxsize=1)
def _patterns() -> dict[str, re.Pattern]:
    cfg = load_patient_config()["safety"]
    return {
        kind: re.compile(r"\b(?:" + "|".join(cfg[kind]["patterns"]) + r")", re.IGNORECASE)
        for kind in ORDER
    }


def screen(question: str) -> SafetyResult:
    """The strongest safety category the question falls into, if any."""
    cfg = load_patient_config()["safety"]
    text = (question or "").replace("’", "'")
    for kind in ORDER:
        if _patterns()[kind].search(text):
            return SafetyResult(
                kind=kind,
                message=" ".join(cfg[kind]["message"].split()),
                guide=cfg[kind].get("guide"),
            )
    return SafetyResult()
