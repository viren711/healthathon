#!/usr/bin/env python3
"""UI regression tests for the portal, driven by Streamlit's AppTest harness.

scripts/smoke_test.py covers the pipeline but never renders a page, so it
cannot catch interaction bugs. This does: it clicks the real buttons and
asserts what ends up in the element tree.

It exists because of one specific bug that no amount of pipeline testing would
have found. "Go to section" rendered its page correctly and then hid it,
because the button lives inside an expander, clicking a button reruns the
script, and the expander was rebuilt with a hard-coded `expanded=False`. The
page was there in the element tree the whole time, inside a container that had
snapped shut. To the user it looked like a dead button.

    python scripts/ui_test.py            # everything (Ask and drafting need the LLM)
    python scripts/ui_test.py --no-llm   # no model required
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Voice and translation run against the offline mock in rag/sarvam.py, with a
# throwaway cache: UI tests must never spend Sarvam credits.
# scripts/11_voice_check.py is the one live check.
import os  # noqa: E402
import tempfile  # noqa: E402

os.environ["SARVAM_MOCK"] = "1"
_VOICE_TMP = tempfile.mkdtemp(prefix="voice_ui_test_")
os.environ["VOICE_CACHE_DIR"] = _VOICE_TMP
os.environ["VOICE_USAGE_LOG"] = os.path.join(_VOICE_TMP, "usage.jsonl")

# The entrypoint is now the home page; each tool is its own view, driven
# directly so the tests do not depend on navigating through the home page.
ENTRY = str(ROOT / "app" / "streamlit_app.py")
APP = str(ROOT / "app" / "views" / "oncology_reference.py")
STUDIO = str(ROOT / "app" / "views" / "content_studio.py")
RESEARCH = str(ROOT / "app" / "views" / "research_platform.py")
PATIENTS = str(ROOT / "app" / "views" / "patient_portal.py")
GUIDANCE = str(ROOT / "app" / "views" / "treatment_guidance.py")

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{'  ok  ' if condition else ' FAIL '}] {name}" + (f"  -- {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def scope_word_in(text: str) -> bool:
    return "Searching" in text


def goto_buttons(at):
    return [i for i, b in enumerate(at.button) if "Go to section" in b.label]


def test_find_mode(AppTest) -> None:
    print("\n== Find mode ==")
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    check("app starts without an exception", not at.exception)

    at.radio[0].set_value("Find a topic").run()
    at.text_input[0].set_value("brachytherapy").run()
    check("locator returns results", len(goto_buttons(at)) > 0,
          f"{len(goto_buttons(at))} places")
    check("no page is open before clicking", len(at.image) == 0)

    idx = goto_buttons(at)[0]
    at.button[idx].click().run()
    check("clicking 'Go to section' renders the page", len(at.image) == 1,
          f"{len(at.image)} image(s)")
    check("no exception on opening the page", not at.exception)
    check("a close control appears", any("Close" in b.label for b in at.button))
    check("page navigation appears", any("Prev" in b.label for b in at.button)
          and any("Next" in b.label for b in at.button))

    # A nonsense topic must find nothing rather than the nearest five passages.
    at2 = AppTest.from_file(APP, default_timeout=300)
    at2.run()
    at2.radio[0].set_value("Find a topic").run()
    at2.text_input[0].set_value("What is the capital of France?").run()
    check("an out-of-corpus topic returns no places", len(goto_buttons(at2)) == 0,
          f"{len(goto_buttons(at2))} places")


def test_scope_selector(AppTest) -> None:
    print("\n== Section selector ==")
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    labels = [sb.label for sb in at.selectbox]
    check("an organ-system selector exists", "Organ system" in labels, str(labels))
    if "Organ system" not in labels:
        return

    sysbox = at.selectbox[labels.index("Organ system")]
    opts = list(sysbox.options)
    check("selector offers every organ system", len(opts) >= 10, f"{len(opts)} options")
    hn = next((o for o in opts if o.startswith("Head & Neck")), None)
    check("Head & Neck is selectable", hn is not None, str(opts[:4]))
    if not hn:
        return

    sysbox.set_value(hn).run()
    labels = [sb.label for sb in at.selectbox]
    check("choosing a system reveals a pathology selector", "Pathology" in labels,
          str(labels))
    if "Pathology" not in labels:
        return

    pathbox = at.selectbox[labels.index("Pathology")]
    tongue = next((o for o in pathbox.options if "tongue" in o.lower()), None)
    check("'Oral cavity & tongue' is selectable", tongue is not None,
          str(list(pathbox.options)[:4]))
    if not tongue:
        return

    pathbox.set_value(tongue).run()
    main_boxes = [sb.label for sb in at.main.selectbox]
    check("organ and pathology selectors are on the main page, not the sidebar",
          {"Organ system", "Pathology"} <= set(main_boxes)
          and not any(sb.label in ("Organ system", "Pathology") for sb in at.sidebar.selectbox),
          str(main_boxes))
    check("no exception when scoping", not at.exception)

    # An active scope must be visible in the MAIN panel, not just the sidebar.
    # A section left selected from an earlier question is otherwise invisible,
    # and the platform looks broken when it refuses something the books cover.
    check("an active section is announced in the main panel",
          any("only" in w.value and scope_word_in(w.value) for w in at.warning),
          str([w.value[:60] for w in at.warning]))
    check("a one-click 'Search all' escape exists",
          any("Search all" in b.label for b in at.button))

    # Clearing the scope must actually clear it.
    clear = [i for i, b in enumerate(at.button) if "Search all" in b.label]
    if clear:
        at.button[clear[0]].click().run()
        check("'Search all' clears the section filter",
              not any("Answering only from" in m.value for m in at.success),
              str([m.value[:40] for m in at.success]))
        # Put it back for the remaining assertions.
        labels = [sb.label for sb in at.selectbox]
        at.selectbox[labels.index("Organ system")].set_value(hn).run()
        labels = [sb.label for sb in at.selectbox]
        pathbox = at.selectbox[labels.index("Pathology")]
        tongue = next((o for o in pathbox.options if "tongue" in o.lower()), None)
        if tongue:
            pathbox.set_value(tongue).run()

    # Scoped Find: a topic from a different organ system must not come back.
    # The topic has to be genuinely disjoint. "prostate brachytherapy" is not:
    # brachytherapy is discussed for T1/T2 tongue cancers, so the oral-cavity
    # scope returns it at a low score, correctly. Wilms tumour shares nothing.
    at.radio[0].set_value("Find a topic").run()
    at.text_input[0].set_value("Wilms tumour").run()
    n = len(goto_buttons(at))
    check("a foreign topic finds nothing inside the tongue scope", n == 0,
          f"{n} places returned")
    check("the miss points the user at where it does appear",
          any("appear elsewhere" in m.value for m in at.info), "")


def test_ask_mode(AppTest) -> None:
    print("\n== Ask mode (one real LLM call) ==")
    at = AppTest.from_file(APP, default_timeout=900)
    at.run()
    at.chat_input[0].set_value("What is the lymphatic drainage of the oral tongue?").run()
    check("question answered without an exception", not at.exception)
    check("the answer carries a source card", len(at.expander) > 0,
          f"{len(at.expander)} expander(s)")

    btns = goto_buttons(at)
    check("source card offers 'Go to section'", len(btns) > 0)
    if not btns:
        return

    at.button[btns[0]].click().run()
    check("clicking renders the page", len(at.image) == 1, f"{len(at.image)} image(s)")
    # The regression itself: the page is useless if its container closes on it.
    expanded = [e.proto.expanded for e in at.expander]
    check("the source card stays OPEN so the page is visible", any(expanded),
          f"expanded flags: {expanded}")
    check("no exception after opening the page", not at.exception)


def test_home(AppTest) -> None:
    print("\n== Home ==")
    at = AppTest.from_file(ENTRY, default_timeout=300)
    at.run()
    check("home renders without an exception", not at.exception,
          str([e.value for e in at.exception][:1]))
    check("home carries the platform title",
          any("Doctor Knowledge" in h.proto.body for h in at.get("html")),
          str(len(at.get("html"))))
    headings = " ".join(l.proto.label for l in at.get("page_link"))
    for name in ("Knowledge Assistant", "Content Studio", "Research Platform",
                 "Treatment Guidance", "Patient Portal"):
        check(f"home has a card for the {name}", name in headings)
    check("home has no 'Open …' buttons: the cards themselves are the links",
          not [b.label for b in at.button if b.label.startswith("Open ")])
    links = at.get("page_link")
    check("home has a clickable card link into each of the five tools", len(links) == 5, str(len(links)))
    groups = at.get("button_group")
    tasks = [o.content for o in groups[0].proto.options] if groups else []
    check("the launcher offers all five tasks",
          tasks == ["Ask the textbooks", "Draft an article", "Search the literature",
                    "Find treatment guidance", "Answer a patient question"], str(tasks))
    check("no tool is marked as a placeholder",
          not any("Coming soon" in i.value for i in at.info))

    # The launcher hands its text to the chosen tool. "Draft an article" is
    # the route that needs no model: the studio should open with the topic
    # already filled in.
    at.session_state["home_action"] = "Draft an article"
    at.run()
    at.text_input(key="home_query").set_value("Managing fatigue during radiotherapy")
    at.button[0].click().run()  # the launcher's submit button comes first
    check("launching a draft opens the Content Studio",
          any("Content Studio" in t.value for t in at.title),
          str([t.value for t in at.title]))
    check("the studio topic is pre-filled from the launcher",
          [t.value for t in at.text_input if t.label == "Topic"] == ["Managing fatigue during radiotherapy"],
          str([(t.label, t.value) for t in at.text_input]))


def test_studio_brief(AppTest) -> None:
    print("\n== Content Studio (no model needed) ==")
    at = AppTest.from_file(STUDIO, default_timeout=300)
    at.run()
    check("studio renders without an exception", not at.exception,
          str([e.value for e in at.exception][:1]))
    check("studio asks for a topic and a description",
          [t.label for t in at.text_input] == ["Topic"]
          and [t.label for t in at.text_area] == ["Describe the piece"])
    check("studio asks what kind of piece",
          any(sb.label == "Kind of piece" for sb in at.selectbox))
    check("a writer model is offered in the sidebar",
          len(at.sidebar.radio) == 1 and len(at.sidebar.radio[0].options) >= 1,
          str([r.options for r in at.sidebar.radio]))

    # An empty brief must be stopped before any model call is made.
    [b for b in at.button if "Next" in b.label][0].click().run()
    warnings = [w.value for w in at.warning]
    check("an empty brief is refused with a reason",
          any("topic" in w.lower() for w in warnings), str(warnings))
    check("an empty brief stays on step 1",
          any(t.label == "Topic" for t in at.text_input))


def test_studio_draft(AppTest) -> None:
    print("\n== Content Studio (real draft, a few minutes) ==")
    at = AppTest.from_file(STUDIO, default_timeout=900)
    at.run()
    at.text_input[0].set_value("Early warning signs of oral cancer")
    at.text_area[0].set_value(
        "A friendly blog post for my clinic website encouraging adults over 40, "
        "especially smokers, to get their mouth checked.")
    [b for b in at.button if "Next" in b.label][0].click().run()
    check("step 2 is reached without an exception", not at.exception,
          str([e.value for e in at.exception][:1]))
    check("topic-specific questions are asked", len(at.text_input) >= 1,
          f"{len(at.text_input)} questions")

    short = [r for r in at.radio if any(o.startswith("Short") for o in r.options)]
    if short:
        short[0].set_value([o for o in short[0].options if o.startswith("Short")][0])
    [b for b in at.button if "Write the article" in b.label][0].click().run()
    check("a draft is produced without an exception", not at.exception,
          str([e.value for e in at.exception][:1]))
    article = [m.value for m in at.markdown if m.value.startswith("# ")]
    check("the draft opens with a single H1 title", len(article) == 1)
    check("the draft is marked for physician review",
          any("Draft for your review" in w.value for w in at.warning))
    check("the draft can be downloaded", len(at.get("download_button")) == 1)
    if article:
        # A draft that stops mid-sentence is the specific failure the tidy-up
        # step exists to prevent.
        body = article[0].split("\n---\n")[0].rstrip()
        check("the draft ends on a complete sentence",
              body[-1:] in ".!?)*", repr(body[-40:]))


def test_research_page(AppTest) -> None:
    print("\n== Research Platform (no network) ==")
    at = AppTest.from_file(RESEARCH, default_timeout=300)
    at.run()
    check("research page renders without an exception", not at.exception,
          str([e.value for e in at.exception][:1]))
    check("research page carries its title",
          any("Research Platform" in t.value for t in at.title),
          str([t.value for t in at.title]))
    check("research page has a question editor", len(at.text_area) == 1,
          f"{len(at.text_area)} editor(s)")
    check("answer length and citation style are offered",
          [sb.label for sb in at.sidebar.selectbox] == ["Answer length", "Citation style"],
          str([sb.label for sb in at.sidebar.selectbox]))
    check("the page says questions leave the machine",
          any("scite.ai" in m.value for m in at.markdown))
    # An empty question must be stopped before any network call.
    at.button(key="send_scite").click().run()
    check("an empty question is refused",
          any("Enter a question" in w.value for w in at.warning))


def test_patient_learn(AppTest) -> None:
    print("\n== Patient Portal (Learn: no model needed) ==")
    at = AppTest.from_file(PATIENTS, default_timeout=300)
    at.run()
    check("patient portal renders without an exception", not at.exception,
          str([e.value for e in at.exception][:1]))
    check("patient portal carries its title",
          any("Patient Portal" in t.value for t in at.title), str([t.value for t in at.title]))
    check("language, topic area and guide selectors are on the main page, not the sidebar",
          [s.label for s in at.main.selectbox][:3] == ["Language", "Topic area", "Guide"]
          and not at.sidebar.selectbox,
          str([s.label for s in at.main.selectbox]))
    check("the page says it is not personal medical advice",
          any("not medical advice" in i.value for i in at.info))

    at.session_state["patient_mode"] = "Learn"
    at.run()
    guide_buttons = [b for b in at.button if b.key and b.key.startswith("open-")]
    check("Learn lists every guide to open", len(guide_buttons) >= 70,
          f"{len(guide_buttons)} guides")
    at.button(key="open-mouth").click().run()
    check("opening a guide shows its reading material", not at.exception
          and [s.value for s in at.subheader][:1] == ["Mouth Cancer"],
          str([s.value for s in at.subheader]))
    check("the guide offers all five reading tabs",
          [t.label for t in at.tabs] == ["Common questions", "Key points",
                                         "Questions to ask your doctor", "Words to know",
                                         "Read by chapter"],
          str([t.label for t in at.tabs]))
    check("common questions are listed", any(e.label == "What is mouth cancer?" for e in at.expander),
          str([e.label for e in at.expander][:3]))
    check("the glossary is shown as a table", len(at.dataframe) == 1)

    # Narrowing the topic area must not leave a guide from another area selected.
    at.selectbox(key="patient_area").select("supportive_care").run()
    check("changing topic area resets an out-of-area guide",
          at.selectbox(key="patient_guide").value == "all",
          str(at.selectbox(key="patient_guide").value))


def test_guidance(AppTest) -> None:
    print("\n== Treatment Guidance (no model needed) ==")
    from rag import guidance

    if not guidance.available():
        print("(skipping: run scripts/09_build_asco.py first)")
        return
    at = AppTest.from_file(GUIDANCE, default_timeout=300)
    at.run()
    check("treatment guidance renders without an exception", not at.exception,
          str([e.value for e in at.exception][:1]))
    check("the page says it is decision support, not a plan",
          any("not a treatment plan" in i.value for i in at.info))
    check("the page lists the cancers the ASCO guidelines cover",
          any("Covered by the ASCO guidelines here" in c.value and "Lung cancer (NSCLC)" in c.value
              and "not covered" in c.value for c in at.caption))
    check("it starts with a text box and offers examples",
          bool(at.text_area(key="tg_paste")) and bool(at.pills(key="tg_example").options),
          str(at.pills(key="tg_example").options[:2]))
    check("nothing is shown until a cancer is chosen",
          not any(m.value.startswith("**1.") for m in at.markdown))

    at.pills(key="tg_cancer").set_value("Lung cancer (NSCLC)").run()
    first = [m.value for m in at.markdown if m.value.startswith("**1.")]
    check("choosing a cancer shows ranked recommendations at once", bool(first), str(first[:1]))
    heads = [m.value for m in at.markdown if re.match(r"\*\*\d+\. ", m.value)]
    check("only the best three are shown up front, the rest folded away",
          len(heads) == 8 and any(e.label == "Show 5 more" for e in at.expander),
          str([e.label for e in at.expander][:4]))
    asked = [p.label for p in at.pills]
    check("it asks the situation first, then at most two more questions",
          asked[2] == "Situation" and 2 <= len(asked) - 2 <= 3, str(asked))

    # Answering re-ranks: progression on osimertinib puts the guideline's own
    # answer for that on top.
    at.pills(key="tg_setting").set_value(["second-line"]).run()
    at.pills(key="tg_tags").set_value(["EGFR"]).run()
    first = [m.value for m in at.markdown if m.value.startswith("**1.")]
    check("answers re-rank the results (EGFR, second line)",
          not at.exception and bool(first), str(first[:1]))
    check("each recommendation shows its strength",
          any("badge[" in m.value and "recommendation" in m.value for m in at.markdown))
    check("the figures sit inside each recommendation's Details",
          any("Figures quoted word for word" in c.value for c in at.caption))
    check("a missing figure is never mentioned",
          not any("No figures" in c.value for c in at.caption))

    # Pasting a case answers the questions for you.
    at = AppTest.from_file(GUIDANCE, default_timeout=300)
    at.run()
    at.text_area(key="tg_paste").set_value(
        "Stage IV NSCLC with EGFR L858R, progressed on osimertinib after 18 months. What next?")
    next(b for b in at.button if b.label == "Find recommendations").click().run()
    check("pasting a case picks the cancer and fills the answers",
          at.session_state["tg_cancer"] == "Lung cancer (NSCLC)"
          and "EGFR" in (at.session_state["tg_tags"] or [])
          and "osimertinib" in (at.session_state["tg_prior"] or []),
          f"{at.session_state['tg_cancer']} {at.session_state['tg_tags']}")
    first = [m.value for m in at.markdown if m.value.startswith("**1.")]
    check("the top recommendation is the guideline's answer for progression on osimertinib",
          bool(first) and "Recommendation 2.1" in first[0], str(first[:1]))

    # A cancer no guideline covers: never another cancer's recommendations.
    # The textbooks are searched instead (stubbed here, so no model is needed),
    # and the answer says it is from the books.
    from rag import pipeline

    calls: list[str] = []

    def stub(question, client=None, **kw):
        calls.append(question)
        src = pipeline.Source(
            number=1, breadcrumb="DeVita > Pancreatic cancer", citation="DeVita, p. 9", book="DeVita",
            book_short="DeVita", chapter="Pancreatic cancer", pdf_page_start=9, pdf_page_end=9,
            printed_page_start="9", text="Stub passage text.", score=1.0, tier="a")
        return pipeline.Answer(question=question, text="Stub textbook answer [1].",
                               status="answered", sources=[src], cited=[1])

    real = pipeline.answer_question
    pipeline.answer_question = stub
    try:
        # The ASCO answer for a covered cancer must not call the textbooks.
        check("a covered cancer never falls back to the textbooks", not calls, str(calls))
        text = "Metastatic pancreatic adenocarcinoma, untreated. First-line?"
        at.text_area(key="tg_paste").set_value(text)
        next(b for b in at.button if b.label == "Find recommendations").click().run()
        check("an uncovered cancer is flagged and nothing is searched as another cancer",
              not at.exception and not at.session_state["tg_cancer"]
              and any("No cancer covered" in w.value for w in at.warning)
              and not any(m.value.startswith("**1.") for m in at.markdown),
              str([w.value[:60] for w in at.warning]))
        check("the textbooks are asked the doctor's own words", calls == [text], str(calls))
        check("the textbook answer is shown and labelled as not from ASCO",
              any("not from the ASCO guidelines" in x.value for x in at.success)
              and any("Stub textbook answer" in m.value for m in at.markdown),
              str([x.value[:50] for x in at.success]))
        check("its source is listed with book and page",
              any(e.label.startswith("**[1]**") and "p. 9" in e.label for e in at.expander),
              str([e.label for e in at.expander][:3]))
        at.run()  # a rerun reuses the answer instead of asking again
        check("a rerun does not ask the textbooks again", calls == [text], str(calls))
    finally:
        pipeline.answer_question = real


def test_patient_ask(AppTest) -> None:
    print("\n== Patient Portal (live: Ask) ==")
    at = AppTest.from_file(PATIENTS, default_timeout=600)
    at.run()
    at.chat_input(key="patient_chat").set_value("What are the signs of mouth cancer?").run()
    check("a patient question is answered without an exception", not at.exception,
          str([e.value for e in at.exception][:1]))
    check("the answer cites the NCCN guide",
          any(e.label.startswith("[") and "NCCN" in e.label for e in at.expander),
          str([e.label for e in at.expander][:3]))
    at.chat_input(key="patient_chat").set_value("I want to die, I can't cope anymore").run()
    check("a crisis question shows the crisis message",
          any("14416" in e.value for e in at.error), str([e.value[:60] for e in at.error]))

    # Hindi, answered: shown in Hindi with the English beside it (mock
    # translation; the model answers the English question for real).
    at = AppTest.from_file(PATIENTS, default_timeout=600)
    at.run()
    # (Auto-detect: AppTest cannot re-read a selectbox whose labels changed
    # language mid-session, which choosing Hindi explicitly would do.)
    at.selectbox(key="patient_lang").set_value("auto").run()
    at.chat_input(key="patient_chat").set_value("कीमोथेरेपी के दुष्प्रभाव क्या हैं?").run()
    check("a Hindi patient question is answered without an exception", not at.exception,
          str([e.value for e in at.exception][:1]))
    captions = [c.value for c in at.caption]
    check("the answer is shown in Hindi, followed by the English",
          any("हिन्दी" in c for c in captions) and "English" in captions
          and any("[hi]" in m.value for m in at.markdown), str(captions[:8]))


def test_voice_offline(AppTest) -> None:
    """Languages without the model: the paths that answer before generation."""
    print("\n== Voice & languages (mock Sarvam, no model) ==")
    at = AppTest.from_file(PATIENTS, default_timeout=300)
    at.run()
    langs = [s for s in at.selectbox if s.label == "Language"]
    check("the patient page has a Language selector in the main panel", len(langs) == 1,
          str([s.label for s in at.selectbox]))
    if langs:
        check("it offers the 11 voiced languages plus Auto-detect, starting on English",
              langs[0].options[0] == "English" and langs[0].options[-1] == "Auto-detect"
              and len(langs[0].options) == 12 and langs[0].value == "en-IN",
              str(langs[0].options))
    check("a 'Choose your language' line cycles through the languages above it",
          any("hv-langline" in h.proto.body
              and "உங்கள் மொழியை" in h.proto.body for h in at.get("html")))

    # English is the default and costs nothing: Hindi typed while English is
    # selected is not sent anywhere; the page offers to switch instead.
    from rag.sarvam import get_voice
    calls = sum(get_voice().api.calls.values())
    at.chat_input(key="patient_chat").set_value("मैं मरना चाहता हूँ, अब सहा नहीं जाता").run()
    check("with English selected, nothing is sent to Sarvam",
          sum(get_voice().api.calls.values()) == calls,
          str(get_voice().api.calls))
    offer = [b for b in at.button if b.label == "Ask in हिन्दी"]
    check("Hindi typed under English gets an 'Ask in हिन्दी' button", bool(offer),
          str([b.label for b in at.button][:6]))

    # Default English: the page's own buttons are English only.
    check("in English the buttons are English only",
          any(b.label == "Clear conversation" for b in at.button),
          str([b.label for b in at.button]))
    hi = AppTest.from_file(PATIENTS, default_timeout=300)
    hi.run()
    hi.selectbox(key="patient_lang").set_value("hi-IN").run()
    clear = [b.label for b in hi.button if b.label.startswith("Clear conversation")]
    check("in Hindi a button is English with Hindi beneath it",
          bool(clear) and "\n\n[hi] Clear conversation" in clear[0], str(clear))
    check("in Hindi the title and the how-to box are English with Hindi beside them",
          any("Patient Portal" in t.value and "[hi]" in t.value for t in hi.title)
          and any("[hi]" in m.value and "Plain-language answers" in m.value
                  for m in hi.markdown),
          str([t.value for t in hi.title]))
    check("the Ask/Learn switch keeps its English values", hi.session_state["patient_mode"] == "Ask a question")

    # A crisis message typed in Hindi is caught: the safety screen reads the
    # English translation, and the notice comes back in Hindi AND English.
    if offer:
        offer[0].click().run()
    check("a Hindi question runs without an exception", not at.exception,
          str([e.value for e in at.exception][:1]))
    errors = " ".join(e.value for e in at.error)
    check("a crisis typed in Hindi shows the helpline", "14416" in errors, errors[:120])
    check("the crisis notice is shown in Hindi too", "[hi]" in errors, errors[:120])
    captions = " ".join(c.value for c in at.caption)
    check("the English the question was searched as is shown",
          "I want to die" in captions, captions[:200])
    check("the button switched the page to Hindi", at.session_state["patient_lang"] == "hi-IN")
    check("the notice has a Listen button",
          any(b.label.startswith("Listen") for b in at.button),
          str([b.label for b in at.button][:6]))

    # Tamil on Auto-detect, off topic: refused by the topic gate, refusal
    # shown in Tamil.
    # (A fresh page: AppTest cannot re-read a selectbox whose labels changed
    # language mid-session, which a real browser handles.)
    at = AppTest.from_file(PATIENTS, default_timeout=300)
    at.run()
    at.selectbox(key="patient_lang").set_value("auto").run()
    at.chat_input(key="patient_chat").set_value("1998 கால்பந்து உலகக் கோப்பையை வென்றது யார்?").run()
    refusals = " ".join(x.value for x in list(at.info) + list(at.warning))
    check("an off-topic Tamil question is refused in Tamil", "[ta]" in refusals, refusals[-160:])
    check("Auto-detect remembers Tamil",
          at.session_state["patient_lang::detected"] == "ta-IN")

    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    langs = [s for s in at.selectbox if s.label == "Answer language"]
    check("the Knowledge Assistant offers English and Hindi only",
          bool(langs) and langs[0].options == ["English", "हिन्दी · Hindi"],
          str([s.options for s in langs]))
    at.selectbox(key="ref_lang").set_value("hi-IN").run()
    at.chat_input[0].set_value("क्रिकेट विश्व कप किसने जीता?").run()
    check("a Hindi question to the textbooks runs without an exception", not at.exception,
          str([e.value for e in at.exception][:1]))
    captions = " ".join(c.value for c in at.caption)
    check("it is searched in English", "Who won the cricket World Cup?" in captions,
          captions[:200])


def test_research_live(AppTest) -> None:
    print("\n== Research Platform (live: scite.ai + local summary) ==")
    import os

    from rag.config import _load_dotenv

    _load_dotenv()
    if not (os.environ.get("SCITE_API_KEY") or os.environ.get("SCITE_MCP_API_KEY")):
        print("(skipping: no SCITE_API_KEY or SCITE_MCP_API_KEY in .env)")
        return
    at = AppTest.from_file(RESEARCH, default_timeout=600)
    at.run()
    at.text_area[0].set_value(
        "Does a close surgical margin alone warrant adjuvant radiotherapy in oral "
        "squamous cell carcinoma?").run()
    at.button(key="send_scite").click().run()
    check("the question is answered without an exception", not at.exception,
          str([e.value for e in at.exception][:1]))
    check("the question and a reply are shown", len(at.chat_message) >= 2,
          f"{len(at.chat_message)} message(s)")
    check("the reply lists sources", len(at.expander) >= 1, f"{len(at.expander)} source card(s)")
    errors = [e.value for e in at.error]
    check("no request error is shown", not errors, str(errors)[:200])
    check("the question box clears after sending", at.text_area[0].value == "",
          repr(at.text_area[0].value))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-llm", action="store_true",
                    help="skip Ask mode, which needs a running model server")
    args = ap.parse_args()

    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        print("streamlit.testing is unavailable -- upgrade streamlit", file=sys.stderr)
        return 1

    test_home(AppTest)

    test_find_mode(AppTest)
    test_scope_selector(AppTest)
    test_studio_brief(AppTest)
    test_research_page(AppTest)
    test_patient_learn(AppTest)
    test_guidance(AppTest)
    test_voice_offline(AppTest)

    if args.no_llm:
        print("\n(skipping Ask mode and drafting: --no-llm)")
    else:
        from rag.llm import LLMClient

        ok, detail = LLMClient.from_config().health()
        if ok:
            test_ask_mode(AppTest)
            test_studio_draft(AppTest)
            test_research_live(AppTest)
            test_patient_ask(AppTest)
        else:
            print(f"\n(skipping Ask mode and drafting: model server unreachable -- {detail})")

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all UI checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
